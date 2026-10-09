# Reference corpus integration using Subjective and Objective

Integrate the reference patient corpus into `scripts/run_med_copilot.py` so retrieval and reranking compare the target patient's Subjective and Objective (S/O) against reference S/O. Keep each reference patient's Assessment and Plan (A/P) available for final generation. The target supplies S/O, and the final model generates A/P.

The current runner imports `HybridRetriever` and `CrossEncoderReranker` from [upstream/create_embeddings.py](../upstream/create_embeddings.py). Those classes embed, score, and rerank reference S/O/A, and load relative cache files under `med_copilot/upstream/`. Generating S/O vectors alone does not change those behaviors. Implement the adaptations below in project-owned files; preserve upstream source.

All paths below are relative to the repository root. The builder and adapter are files to create as part of this integration.

## 1 Prepare the reference corpus

Save the full reference corpus to:

```text
med_copilot/data/patient_corpus/reference_patients.json
```

Use a JSON array of records with this schema:

```json
[
  {
    "patient_id": "patient-001",
    "subjective": "Reported symptoms and relevant history...",
    "objective": "Examination findings and investigations...",
    "assessment": "Reference patient's diagnosis and assessment...",
    "plan": "Reference patient's investigations, treatment, and follow-up..."
  }
]
```

Require a nonblank string patient ID and string values for all four clinical fields. Use `""` for a missing clinical section, and reject records with no usable S/O. Preserve deterministic record ordering. If a patient has multiple reference cases, retain their shared patient ID and distinguish cases using their corpus row or a stable case ID.

The data team guarantees patient-level separation between reference and evaluation patients. Keep evaluation labels outside inference inputs. Raw Synthea files need conversion into these fields before this loader can consume them; retain their source paths or full original text separately if needed.

## 2 Create the offline embedding builder

Create `scripts/build_reference_embeddings.py`. Use the pretrained `SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")`; no fine-tuning or OpenAI calls are needed to generate reference embeddings.

For each record, construct the embedding input as:

```python
text = record["subjective"] + " " + record["objective"]
```

Keep Assessment, Plan, and patient IDs out of the embedding input. Initialize BioClinicalBERT before importing FAISS to preserve the runner's Apple Silicon workaround.

The following builder handles records whose combined S/O fits the encoder. It stops on oversized records so that later findings are not silently truncated. Steps 3 and 4 provide code to extend this same file with passage chunking and validated artifact persistence. Apply those extensions before building the full corpus.

```python
from pathlib import Path
import hashlib
import json

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "med_copilot/data/patient_corpus/reference_patients.json"
destination = ROOT / "med_copilot/data/patient_corpus/so"
source_bytes = source.read_bytes()
records = json.loads(source_bytes)
if not isinstance(records, list) or not records:
    raise ValueError("Reference corpus must be a nonempty JSON array")

texts = []
for row, record in enumerate(records):
    if not isinstance(record, dict):
        raise ValueError(f"Corpus row {row} must be an object")
    patient_id = record.get("patient_id")
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError(f"Corpus row {row} needs a patient_id")
    for field in ("subjective", "objective", "assessment", "plan"):
        if not isinstance(record.get(field), str):
            raise ValueError(f"Corpus row {row} needs a string {field}")
    text = record["subjective"] + " " + record["objective"]
    if not text.strip():
        raise ValueError(f"Corpus row {row} has empty S/O")
    texts.append(text)

model_name = "emilyalsentzer/Bio_ClinicalBERT"
model = SentenceTransformer(model_name)

# Initialize the model before importing FAISS.
import faiss

for row, text in enumerate(texts):
    token_ids = model.tokenizer(
        text, add_special_tokens=True, truncation=False
    )["input_ids"]
    if len(token_ids) > model.max_seq_length:
        raise ValueError(f"Corpus row {row} requires passage chunking")

vectors = model.encode(
    texts,
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=True,
)
vectors = np.ascontiguousarray(vectors, dtype=np.float32)
if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
    raise ValueError("Embeddings must be finite and nonzero")
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
assert vectors.shape[0] == index.ntotal == len(records)

mapping = [
    {"vector_row": row, "corpus_row": row, "patient_id": record["patient_id"]}
    for row, record in enumerate(records)
]
metadata = {
    "representation": "SO",
    "corpus_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "model": model_name,
    "max_seq_length": model.max_seq_length,
    "embedding_dimension": vectors.shape[1],
    "vector_count": index.ntotal,
    "chunking": None,
}

destination.mkdir(parents=True, exist_ok=True)
np.save(destination / "embeddings.npy", vectors)
faiss.write_index(index, str(destination / "index.faiss"))
(destination / "index_rows.json").write_text(
    json.dumps(mapping, indent=2), encoding="utf-8"
)
(destination / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
```

After creating the builder, run from the repository root:

```bash
python scripts/build_reference_embeddings.py
```

The environment needs `sentence-transformers`, NumPy, and FAISS, and access to the pretrained model weights. L2-normalized vectors with `IndexFlatIP` give cosine-similarity scores. Preserve the exact model, resolved model revision, pooling, and encoding settings for reproducibility and query encoding. Extend the metadata with those details before using the artifacts for experiments.

## 3 Handle long records with passages

The implementation below uses overlapping token windows and checks each resulting passage again after extracting its original text. It preserves character offsets into the constructed S/O string, covers the entire string, and uses the encoder's actual `model.max_seq_length`, including special tokens. Section or sentence boundary preferences can be added later without changing the artifact schema.

For this path, save:

```text
med_copilot/data/patient_corpus/so/chunks.json
```

Put this function in `med_copilot/assets/so_reference_artifacts.py`, alongside the functions in Step 4. The builder and retriever will share it:

```python
def chunk_so(text, tokenizer, max_seq_length, overlap_tokens=64):
    if not tokenizer.is_fast:
        raise ValueError("Chunking needs a fast tokenizer with character offsets")
    budget = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap_tokens < budget:
        raise ValueError("Overlap must be smaller than the content token budget")
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True,
    )
    offsets = [(a, b) for a, b in encoded["offset_mapping"] if b > a]
    if not offsets:
        raise ValueError("S/O contains no usable tokens")

    passages = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        char_start = 0 if start == 0 else offsets[start][0]
        while end > start:
            char_end = len(text) if end == len(offsets) else offsets[end][0]
            passage = text[char_start:char_end]
            token_count = len(tokenizer(
                passage, add_special_tokens=True, truncation=False
            )["input_ids"])
            if token_count <= max_seq_length:
                break
            # A substring can tokenize differently at its boundaries.
            end -= 1
        if end == start:
            raise ValueError("A passage cannot fit the encoder token budget")
        passages.append({
            "start": char_start, "end": char_end,
            "text": passage, "token_count": token_count,
        })
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap_tokens)
    return passages
```

Replace Step 2's token-length rejection loop with this block. It creates one or more passages for every record, so short and long records use the same mapping:

```python
import sys

sys.path.insert(0, str(ROOT))
from med_copilot.assets.so_reference_artifacts import chunk_so

overlap_tokens = 64
chunks = []
for row, (record, text) in enumerate(zip(records, texts)):
    for number, passage in enumerate(chunk_so(
        text, model.tokenizer, model.max_seq_length, overlap_tokens
    )):
        chunks.append({
            "chunk_id": f"{row}:{number}",
            "patient_id": record["patient_id"],
            "corpus_row": row,
            **passage,
        })
```

In the `model.encode(...)` call, replace the first argument `texts` with `[chunk["text"] for chunk in chunks]`. Keep float32 conversion, finite/nonzero checks, L2 normalization, and `IndexFlatIP` construction. Replace the count assertion and `mapping = ...` block with:

```python
assert vectors.shape[0] == index.ntotal == len(chunks)
mapping = [
    {
        "vector_row": row,
        "chunk_id": chunk["chunk_id"],
        "corpus_row": chunk["corpus_row"],
        "patient_id": chunk["patient_id"],
    }
    for row, chunk in enumerate(chunks)
]
```

The `start` and `end` offsets are Python string character positions, with an exclusive end. Each FAISS/vector row now resolves to a passage; vector count equals passage count rather than patient or record count. Step 4 saves and validates this mapping and the passage text.

The retrieval adapter must aggregate passage matches into distinct patient candidates and recover their original full reference records. Define and record the aggregation rule, such as maximum passage score per patient. If multiple cases belong to a patient, retain the winning case identity for generation. Apply length handling to target S/O at query time as well.

Reranking has a separate length budget: use the cross-encoder tokenizer to check each target/reference pair, including special tokens. Split long inputs into fitting passage pairs and aggregate their scores. Do not assume that passages fitting BioClinicalBERT also fit a cross-encoder pair.

## 4 Save and verify the artifacts

Keep the S/O experiment artifacts together:

```text
med_copilot/data/patient_corpus/so/embeddings.npy
med_copilot/data/patient_corpus/so/index.faiss
med_copilot/data/patient_corpus/so/index_rows.json
med_copilot/data/patient_corpus/so/metadata.json
med_copilot/data/patient_corpus/so/chunks.json
med_copilot/data/patient_corpus/so/corpus.json      # exact corpus snapshot
```

Verify that vector, FAISS, and mapping counts agree; every index row resolves to the expected corpus record or passage; and vectors are finite and normalized. The metadata must identify the exact corpus hash, representation `SO`, model revision/settings, ordered mapping, and chunking settings.

Create `med_copilot/assets/so_reference_artifacts.py` with the following code and append the `chunk_so` function from Step 3 to that same file. If `med_copilot/assets/` does not exist, create it and an empty `__init__.py` there. This helper stages output files, writes metadata last, and verifies file hashes, corpus identity, encoder settings, counts, normalization, passage coverage, and row mappings when loading. Call its save/load functions only after initializing BioClinicalBERT, because they import FAISS.

```python
from pathlib import Path
import hashlib
import io
import json
import os
import tempfile

import numpy as np

FILES = ("corpus.json", "chunks.json", "index_rows.json",
         "embeddings.npy", "index.faiss")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _validate(records, chunks, mapping, vectors, index):
    import faiss

    count = len(chunks)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not count:
        raise ValueError("Expected a nonempty float32 embedding matrix")
    if vectors.shape[0] != count or index.ntotal != count or len(mapping) != count:
        raise ValueError("Vector, index, passage, and mapping counts differ")
    if index.d != vectors.shape[1] or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Index dimension or similarity metric differs")
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Embeddings must be finite and L2 normalized")
    for start in range(0, count, 1024):
        size = min(1024, count - start)
        if not np.allclose(index.reconstruct_n(start, size), vectors[start:start + size]):
            raise ValueError("FAISS vectors differ from embeddings.npy")

    seen_ids = set()
    coverage = {row: [] for row in range(len(records))}
    for position, chunk in enumerate(chunks):
        row = chunk["corpus_row"]
        if type(row) is not int or not 0 <= row < len(records):
            raise ValueError("Passage has an invalid corpus row")
        record = records[row]
        original = record["subjective"] + " " + record["objective"]
        begin, end = chunk["start"], chunk["end"]
        if (type(begin) is not int or type(end) is not int or
                not 0 <= begin < end <= len(original) or
                chunk["text"] != original[begin:end]):
            raise ValueError("Passage text or source offsets differ")
        if chunk["patient_id"] != record["patient_id"] or chunk["chunk_id"] in seen_ids:
            raise ValueError("Passage identity is inconsistent or duplicated")
        seen_ids.add(chunk["chunk_id"])
        expected = {
            "vector_row": position, "chunk_id": chunk["chunk_id"],
            "corpus_row": row, "patient_id": chunk["patient_id"],
        }
        if mapping[position] != expected:
            raise ValueError("Index row does not resolve to its passage")
        coverage[row].append((begin, end))
    for row, spans in coverage.items():
        covered = 0
        for begin, end in sorted(spans):
            if begin > covered:
                raise ValueError("Passages omit part of the S/O text")
            covered = max(covered, end)
        original = records[row]["subjective"] + " " + records[row]["objective"]
        if covered != len(original):
            raise ValueError("Passages do not cover the complete S/O text")


def save_artifacts(destination, corpus_bytes, records, chunks, mapping,
                   vectors, index, encoding):
    import faiss

    if json.loads(corpus_bytes) != records:
        raise ValueError("Corpus snapshot differs from the encoded records")
    _validate(records, chunks, mapping, vectors, index)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary)
        (staging / "corpus.json").write_bytes(corpus_bytes)
        (staging / "chunks.json").write_bytes(_json_bytes(chunks))
        (staging / "index_rows.json").write_bytes(_json_bytes(mapping))
        np.save(staging / "embeddings.npy", vectors)
        faiss.write_index(index, str(staging / "index.faiss"))
        metadata = {
            "schema_version": 1, "representation": "SO",
            "corpus_sha256": _sha(corpus_bytes), "encoding": encoding,
            "vector_count": index.ntotal, "embedding_dimension": index.d,
            "file_sha256": {name: _sha((staging / name).read_bytes()) for name in FILES},
        }
        (staging / "metadata.json").write_bytes(_json_bytes(metadata))
        for name in FILES:
            os.replace(staging / name, destination / name)
        # Publish the manifest after all payload files have been replaced.
        os.replace(staging / "metadata.json", destination / "metadata.json")


def load_artifacts(destination, corpus_path, expected_encoding):
    import faiss

    destination = Path(destination)
    metadata = json.loads((destination / "metadata.json").read_bytes())
    if metadata["schema_version"] != 1 or metadata["representation"] != "SO":
        raise ValueError("Unsupported artifact schema or representation")
    if _json_bytes(metadata["encoding"]) != _json_bytes(expected_encoding):
        raise ValueError("Encoder or chunking settings differ; rebuild artifacts")
    if _sha(Path(corpus_path).read_bytes()) != metadata["corpus_sha256"]:
        raise ValueError("Configured corpus changed; rebuild artifacts")
    # Load each payload once, then validate and deserialize those same bytes.
    payloads = {name: (destination / name).read_bytes() for name in FILES}
    for name, data in payloads.items():
        if _sha(data) != metadata["file_sha256"][name]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if _sha(payloads["corpus.json"]) != metadata["corpus_sha256"]:
        raise ValueError("Corpus snapshot hash differs")
    records = json.loads(payloads["corpus.json"])
    chunks = json.loads(payloads["chunks.json"])
    mapping = json.loads(payloads["index_rows.json"])
    vectors = np.load(io.BytesIO(payloads["embeddings.npy"]), allow_pickle=False)
    index = faiss.deserialize_index(np.frombuffer(payloads["index.faiss"], dtype=np.uint8))
    _validate(records, chunks, mapping, vectors, index)
    if (metadata["vector_count"] != index.ntotal or
            metadata["embedding_dimension"] != index.d):
        raise ValueError("Metadata counts or dimensions differ")
    return {
        "records": records, "chunks": chunks, "mapping": mapping,
        "vectors": vectors, "index": index, "metadata": metadata,
    }
```

In `scripts/build_reference_embeddings.py`, remove the old `metadata = ...` block and everything from `destination.mkdir(...)` through the old save calls. Replace them with the following. This assumes Step 3's `chunks` and `mapping` are already constructed:

```python
import sys
from importlib.metadata import version

sys.path.insert(0, str(ROOT))
from med_copilot.assets.so_reference_artifacts import save_artifacts

resolved_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
if not resolved_revision:
    raise ValueError("Use a model with a resolved, pinned Hugging Face revision")
encoding = {
    "model": model_name,
    "model_revision": resolved_revision,
    "max_seq_length": model.max_seq_length,
    "pooling": model[1].get_config_dict(),
    "normalize_embeddings": True,
    "chunking": {"method": "token_windows", "overlap_tokens": overlap_tokens},
    "sentence_transformers_version": version("sentence-transformers"),
    "transformers_version": version("transformers"),
}
save_artifacts(
    destination, source_bytes, records, chunks, mapping, vectors, index, encoding
)
```

Pin the model used by the builder with `SentenceTransformer(model_name, revision="<resolved commit SHA>")`. The resolved SHA in metadata identifies which weights were used. The retriever must initialize that same revision and compute its own expected encoding settings using the dictionary above; pass those settings to `load_artifacts(...)`. Do not merely copy the stored settings as the expected value, because that would not detect a runtime configuration change.

The project-owned retrieval adapter can then load the bundle after model initialization:

```python
from med_copilot.assets.so_reference_artifacts import load_artifacts

bundle = load_artifacts(artifact_dir, corpus_path, expected_encoding)
index = bundle["index"]
chunks = bundle["chunks"]
records = bundle["records"]
# A valid FAISS result position i resolves to chunks[i], then
# records[chunks[i]["corpus_row"]] for the full reference SOAP record.
```

Here `artifact_dir` is the `med_copilot/data/patient_corpus/so/` directory, `corpus_path` is the configured `reference_patients.json`, and `expected_encoding` is the runtime encoder/chunking configuration. Missing files and malformed data raise errors; they do not fall back to a different index.

Rebuild the complete artifact set when the corpus, ordering, representation, model, or chunking settings change. The staging code prevents a failed encoding or validation step from replacing an existing bundle. File replacement is not a single atomic transaction: a reader during publication may encounter a hash mismatch and should stop until the build finishes. Hash validation rejects partially published or mixed bundles. The saved corpus snapshot preserves the records that produced the index.

## 5 Implement reference retrieval and reranking

Create `med_copilot/assets/so_reference_retrieval.py`, exposing project-owned retriever and reranker classes. Use [upstream/create_embeddings.py](../upstream/create_embeddings.py) as the behavioral reference and preserve its source.

The implementation must:

- Load the corpus, S/O vectors, FAISS index, mapping, and metadata through explicit paths, verifying that they belong together.
- Use reference S/O for embedding inputs, keyword-overlap scoring, and cross-encoder pairs. Assessment and Plan must not influence matching.
- Encode target evidence using the same encoder and normalization settings. Preserve the existing keyword-based query strategy if retaining upstream hybrid behavior; extract terms from supplied S/O without diagnostic inference.
- Retain the runner's initial hybrid weight of `alpha=0.5`, top-20 retrieval, top-5 reranking, and selection of the first ranked reference unless deliberately changing the experiment.
- Handle passage-to-patient aggregation if chunking is used.
- Ignore FAISS positions below zero, validate positions against the mapping, and cap requested counts where appropriate for small corpora.
- Return an empty list when no eligible references are found, allowing the runner's existing fallback to continue.
- Retain patient identity and scores. Format the selected reference's full S/O/A/P for final generation, using the existing SOAP example style where possible.

Reference A/P remains available as generation evidence even though it is excluded from retrieval and reranking text.

Copy the following block into `med_copilot/assets/so_reference_retrieval.py`. It uses the shared `chunk_so` and `load_artifacts` functions from Steps 3 and 4. The encoder, cross-encoder, and OpenAI client are supplied by the runner, so this module does not initialize models or make API calls on import.

The aggregation rule is maximum hybrid passage score per patient. If a patient has multiple reference cases, the winning passage selects the case supplied to reranking and generation. Reranking uses the maximum score across fitting target/reference passage pairs. The upstream dominant-phrase attribution display is omitted; reference S/O/A/P, patient identity, and scores are retained.

```python
from importlib.metadata import version
from itertools import islice, product
import json

import numpy as np

from .so_reference_artifacts import chunk_so, load_artifacts


def extract_keywords(client, text, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": (
                'Copy at most 16 medically relevant terms verbatim from the text. '
                'Do not infer diagnoses or add terms. Return JSON: {"terms": []}. '
                'An empty list is valid.\n\nText:\n' + text
            ),
        }],
    )
    terms = json.loads(response.choices[0].message.content)["terms"]
    if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
        raise ValueError("Expected a list of keyword strings")
    # Enforce copied spans and remove duplicates before embedding or scoring.
    return list(dict.fromkeys(
        term.strip() for term in terms
        if term.strip() and term.strip() in text
    ))[:16]


class HybridRetriever:
    def __init__(self, corpus_path, artifact_dir, encoder, keyword_extractor,
                 alpha=0.5, overlap_tokens=64,
                 model_name="emilyalsentzer/Bio_ClinicalBERT"):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one")
        self.encoder = encoder
        self.keyword_extractor = keyword_extractor
        self.alpha = alpha
        self.overlap_tokens = overlap_tokens
        revision = getattr(encoder[0].auto_model.config, "_commit_hash", None)
        if not revision:
            raise ValueError("Encoder needs a resolved model revision")
        encoding = {
            "model": model_name,
            "model_revision": revision,
            "max_seq_length": encoder.max_seq_length,
            "pooling": encoder[1].get_config_dict(),
            "normalize_embeddings": True,
            "chunking": {"method": "token_windows", "overlap_tokens": overlap_tokens},
            "sentence_transformers_version": version("sentence-transformers"),
            "transformers_version": version("transformers"),
        }
        bundle = load_artifacts(artifact_dir, corpus_path, encoding)
        self.index = bundle["index"]
        self.chunks = bundle["chunks"]
        self.records = bundle["records"]

    def search(self, query, topk=20):
        if topk <= 0 or not query.strip() or not self.index.ntotal:
            return []
        keywords = self.keyword_extractor(query)
        # Average normalized keyword vectors, or full S/O passage vectors
        # when no keywords were extracted. Split long terms before encoding.
        texts = [part["text"] for text in (keywords or [query]) for part in chunk_so(
            text, self.encoder.tokenizer, self.encoder.max_seq_length,
            self.overlap_tokens,
        )]
        vectors = self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
        )
        query_vector = np.asarray(vectors, dtype=np.float32).mean(axis=0, keepdims=True)
        norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_vector).all() or not np.isfinite(norm) or norm == 0:
            raise ValueError("Query embedding must be finite and nonzero")
        query_vector = np.ascontiguousarray(query_vector / norm, dtype=np.float32)
        count = min(self.index.ntotal, topk * 8)
        scores, positions = self.index.search(query_vector, count)
        best = {}
        for score, position in zip(scores[0], positions[0]):
            if not 0 <= position < len(self.chunks):
                continue
            chunk = self.chunks[position]
            hits = sum(term.lower() in chunk["text"].lower() for term in keywords)
            lexical = hits / len(keywords) if keywords else 0.0
            hybrid = self.alpha * float(score) + (1 - self.alpha) * lexical
            patient_id = chunk["patient_id"]
            if patient_id not in best or hybrid > best[patient_id]["hybrid_score"]:
                best[patient_id] = {
                    "patient_id": patient_id,
                    "corpus_row": chunk["corpus_row"],
                    "chunk_id": chunk["chunk_id"],
                    "record": self.records[chunk["corpus_row"]],
                    "embedding_score": float(score),
                    "keyword_score": lexical,
                    "hybrid_score": hybrid,
                }
        return sorted(best.values(), key=lambda item: item["hybrid_score"], reverse=True)[:topk]


class CrossEncoderReranker:
    def __init__(self, cross_encoder):
        self.model = cross_encoder
        self.ranked = []

    def rerank(self, query, candidates, topk=5):
        self.ranked = []
        if topk <= 0 or not candidates or not query.strip():
            return []
        tokenizer = self.model.tokenizer
        limit = self.model.max_length
        # Give each side half the pair's content budget, reserving special tokens.
        budget = (limit - tokenizer.num_special_tokens_to_add(pair=True)) // 2
        if budget <= 0:
            raise ValueError("Cross-encoder pair budget is too small")
        single_limit = budget + tokenizer.num_special_tokens_to_add(pair=False)
        overlap = min(32, budget - 1)
        query_parts = [part["text"] for part in chunk_so(
            query, tokenizer, single_limit, overlap,
        )]
        ranked = []
        for candidate in candidates:
            record = candidate["record"]
            reference = record["subjective"] + " " + record["objective"]
            reference_parts = [part["text"] for part in chunk_so(
                reference, tokenizer, single_limit, overlap,
            )]
            pairs = product(query_parts, reference_parts)
            best_score = float("-inf")
            # Batch the Cartesian product instead of storing every pair at once.
            while batch := list(islice(pairs, 32)):
                for left, right in batch:
                    if len(tokenizer(left, right, truncation=False)["input_ids"]) > limit:
                        raise ValueError("Cross-encoder pair exceeds its token budget")
                scores = np.asarray(self.model.predict(batch)).reshape(-1)
                if len(scores) != len(batch) or not np.isfinite(scores).all():
                    raise ValueError("Expected one finite cross-encoder score per pair")
                best_score = max(best_score, float(scores.max()))
            ranked.append({**candidate, "rerank_score": best_score})
        self.ranked = sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:topk]
        results = []
        for number, candidate in enumerate(self.ranked, 1):
            record = candidate["record"]
            results.append(
                f"Result {number}\nPatient ID: {candidate['patient_id']}\n"
                f"Subjective:\n{record['subjective']}\n"
                f"Objective:\n{record['objective']}\n"
                f"Assessment:\n{record['assessment']}\n"
                f"Plan:\n{record['plan']}\n"
                f"Hybrid score: {candidate['hybrid_score']:.4f}\n"
                f"Rerank score: {candidate['rerank_score']:.4f}\n"
            )
        return results
```

For example, once the artifacts and both helper modules exist, the retrieval-stage calls in the runner would become:

```python
from sentence_transformers import CrossEncoder
from med_copilot.assets.so_reference_retrieval import (
    HybridRetriever, CrossEncoderReranker, extract_keywords,
)

# Initialize this encoder with the same pinned revision used by the builder.
retriever = HybridRetriever(
    corpus_path=PATIENT_CORPUS,
    artifact_dir=MED / "data" / "patient_corpus" / "so",
    encoder=_preloaded_bioclinicalbert,
    keyword_extractor=lambda text: extract_keywords(client, text),
    alpha=0.5,
)
candidates = retriever.search(s_o, topk=20)
results = []
if candidates:
    reranker = CrossEncoderReranker(CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512,
    ))
    results = reranker.rerank(s_o, candidates, topk=5)
retrieved_info = results[0] if results else "No reference patient available."
```

`s_o` is the runner's target S/O string. Keep the runner's existing empty-corpus check before constructing the retriever, since an empty corpus does not have a built artifact bundle. The `ranked` attribute retains selected patient identities and numeric scores for saving retrieval metadata. Keyword extraction makes one additional OpenAI call per search, as in the existing hybrid pipeline. Model/API/artifact errors propagate; empty candidate lists return the normal no-reference fallback.

Passage reranking compares all fitting passage pairs for each candidate, so runtime grows with target and reference length even though memory is bounded by the batch size. Limiting the passage pairs would be a separate retrieval policy. This module handles encoder lengths; final-generation prompt length still needs the runner's own budget policy.

## 6 Wire the adapter into the runner

In [scripts/run_med_copilot.py](../../scripts/run_med_copilot.py):

1. Set `PATIENT_CORPUS = MED / "data" / "patient_corpus" / "reference_patients.json"`.
2. Replace the upstream retrieval imports with the classes from `med_copilot.assets.so_reference_retrieval`.
3. Pass the corpus and S/O artifact paths explicitly to the project-owned loader. Ensure its result types match the runner's candidate list and formatted reference list.
4. Remove the retrieval-stage `os.chdir(UPSTREAM)` once relative artifacts are unused. Remove the upstream import-path dependency if no remaining imports require it; preserve the model-before-FAISS initialization order.
5. Keep question generation based on target S/O. Pass the selected reference's complete S/O/A/P into `{example}` in `EVALUATE_TEMPLATE_KEYINFO`.
6. Preserve the no-reference behavior: empty corpus, retrieval, or reranking results yield `No reference patient available.` and generation continues with target S/O and guideline evidence. Retrieval exceptions remain failures.

Index preparation alone does not complete integration: the runner must load the new artifacts and use S/O at every reference-matching stage.

## 7 Verify the integration

Every runner change needs corresponding coverage in [tests/test_run_med_copilot.py](../../tests/test_run_med_copilot.py). Preserve the existing full-pipeline tests and mock external models and APIs for offline runs.

Cover corpus loading, explicit artifact paths, index-to-record mapping, stale-cache rejection, small and empty corpora, empty candidate/reranking lists, and failures. Verify that changing reference A/P does not change the text supplied to matching stages, while the selected reference's A/P still appears in the final prompt. If chunking is used, cover end-of-record evidence and aggregation into distinct patients.

From the repository root, run:

```bash
python tests/test_run_med_copilot.py
```

Also perform a retrieval integration check with the real encoder and FAISS: verify a known reference is recovered, that its index row resolves to the correct patient, and that long inputs are handled according to the recorded policy. Report live retrieval and generation checks separately from the mocked suite.
