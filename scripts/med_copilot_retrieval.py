"""S/O-only matching, with full reference SOAP retained for generation.

Import after the runner initializes BioClinicalBERT, before FAISS is loaded.
The upstream query strategy, phrase attribution, and model defaults are reused;
reference embeddings, keyword scoring, and cross-encoder inputs use matching_text.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from create_embeddings import (
    CONTRIBUTION_THRESHOLD,
    CrossEncoderReranker as _UpstreamReranker,
    HybridRetriever as _UpstreamRetriever,
    LLMKeywordExtractor,
)


def matching_text(record):
    """Preserve the supplied S/O strings, excluding all other record fields."""
    for field in ("subjective", "objective"):
        if not isinstance(record.get(field), str):
            raise ValueError(f"Reference {field} must be a string")
    return f'Subjective: {record["subjective"]}\nObjective: {record["objective"]}'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_vectors(vectors):
    vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if vectors.ndim != 2 or not all(vectors.shape) or not np.isfinite(vectors).all():
        raise ValueError("Expected a nonempty matrix of finite embeddings")
    with np.errstate(over="ignore"):
        norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Embedding vector norms must be finite and nonzero")
    faiss.normalize_L2(vectors)
    return vectors


class _SOEmbeddingRetriever:
    """Load a validated S/O cache, or rebuild it without using legacy artifacts."""

    def __init__(self, texts, model_name, encoder=None):
        self.model = encoder if encoder is not None else SentenceTransformer(model_name)
        self.emb_path = "embeddings.npy"
        self.index_path = "index.faiss"
        self.metadata_path = "matching.json"
        # A/P is deliberately absent from this identity: changing the payload
        # must not change matching. Ordered texts bind index rows to corpus rows.
        try:
            revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        except (AttributeError, IndexError, KeyError, TypeError):
            revision = None
        identity = {
            "representation": "SO",
            "text_format": "labeled-so-v1",
            "model": model_name,
            "model_revision": revision,
            "max_seq_length": getattr(self.model, "max_seq_length", None),
            "record_count": len(texts),
            "matching_text_sha256": hashlib.sha256(
                json.dumps(texts, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
        if not self._load(identity):
            self.embeddings = _unit_vectors(self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=True,
            ))
            if self.embeddings.shape[0] != len(texts):
                raise ValueError("Expected one reference embedding per corpus row")
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            self._save(identity)

    def _load(self, identity):
        paths = [Path(self.emb_path), Path(self.index_path), Path(self.metadata_path)]
        if not all(path.is_file() for path in paths):
            return False
        try:
            metadata = json.loads(paths[2].read_text(encoding="utf-8"))
            if metadata["identity"] != identity:
                return False
            if metadata["sha256"] != {path.name: _sha256(path) for path in paths[:2]}:
                return False
            embeddings = np.load(self.emb_path, allow_pickle=False)
            index = faiss.read_index(self.index_path)
            if (embeddings.ndim != 2 or embeddings.shape[0] != identity["record_count"]
                    or embeddings.shape[1] != metadata["dimension"]
                    or index.ntotal != len(embeddings) or index.d != embeddings.shape[1]
                    or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all()
                    or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4)):
                return False
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return False
        self.embeddings, self.index = embeddings, index
        return True

    def _save(self, identity):
        # Publish metadata last. A failed replacement cannot make a partial
        # bundle look valid on the next run because both file hashes are checked.
        with tempfile.TemporaryDirectory(prefix=".so-index-", dir=Path.cwd()) as temporary:
            staging = Path(temporary)
            np.save(staging / self.emb_path, self.embeddings)
            faiss.write_index(self.index, str(staging / self.index_path))
            metadata = {
                "identity": identity,
                "dimension": self.embeddings.shape[1],
                "sha256": {name: _sha256(staging / name)
                           for name in (self.emb_path, self.index_path)},
            }
            (staging / self.metadata_path).write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            for name in (self.emb_path, self.index_path, self.metadata_path):
                os.replace(staging / name, name)


class HybridRetriever(_UpstreamRetriever):
    """Reuse upstream query/scoring utilities with an explicit S/O reference view."""

    def __init__(self, df, alpha=0.5, model_name="emilyalsentzer/Bio_ClinicalBERT",
                 llm_model="gpt-4o-mini", llm_api_key=None, encoder=None):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one")
        if df.empty:
            raise ValueError("Reference corpus must not be empty")
        self.df = df.copy()
        records = self.df.to_dict(orient="records")
        self.matching_texts = [matching_text(record) for record in records]
        if any(not (record["subjective"].strip() or record["objective"].strip())
               for record in records):
            raise ValueError("Reference records must have nonempty S/O")
        for record in records:
            for field in ("assessment", "plan"):
                if not isinstance(record.get(field), str):
                    raise ValueError(f"Reference {field} must be a string")
        self.alpha = alpha
        self.embedder = _SOEmbeddingRetriever(self.matching_texts, model_name, encoder)
        self.keyword_extractor = LLMKeywordExtractor(
            model=llm_model, api_key=llm_api_key or os.getenv("OPENAI_API_KEY"),
        )

    def search(self, query, topk=20):
        self.ranked_candidates = None
        if topk <= 0 or not query.strip():
            self.ranked_candidates = []
            if getattr(self, "on_ranking", None) is not None:
                self.on_ranking([])
            return []
        keywords = self.keyword_extractor.extract(query)
        query_vector = (self.build_query_embedding(keywords) if keywords else
                        self.embedder.model.encode([query], convert_to_numpy=True))
        query_vector = _unit_vectors(query_vector)
        scores, positions = self.embedder.index.search(
            query_vector, min(topk * 8, len(self.df)),
        )
        candidates, seen = [], set()
        for position, score in zip(positions[0], scores[0]):
            row = int(position)
            if row in seen or not 0 <= row < len(self.df):
                continue
            seen.add(row)
            record = self.df.iloc[row]
            case_text = self.matching_texts[row]
            keyword_score = self.compute_keyword_overlap_score(keywords, case_text)
            candidates.append({
                "idx": row,
                **{field: record[field] for field in
                   ("subjective", "objective", "assessment", "plan")},
                "case_text": case_text,
                "display_text": (
                    f'Subjective:\n{record["subjective"]}\n\n'
                    f'Objective:\n{record["objective"]}\n\n'
                    f'Assessment:\n{record["assessment"]}\n\n'
                    f'Plan:\n{record["plan"]}\n'
                ),
                "embedding_score": float(score),
                "keyword_score": float(keyword_score),
                "hybrid_score": self.alpha * float(score) + (1 - self.alpha) * keyword_score,
            })
        candidates = sorted(candidates, key=lambda item: item["hybrid_score"], reverse=True)[:topk]
        self.ranked_candidates = candidates
        if getattr(self, "on_ranking", None) is not None:
            self.on_ranking(candidates)
        # Full-SOAP phrase attribution is display/generation evidence only. It
        # runs after hybrid selection and is never an input to either ranker.
        for candidate in candidates:
            all_terms, dominant_terms = self.compute_case_phrase_contributions(
                query_embedding=query_vector, case_text=candidate["display_text"],
                threshold=CONTRIBUTION_THRESHOLD,
            )
            candidate["term_contributions"] = all_terms
            candidate["dominant_terms"] = dominant_terms
        return candidates


class CrossEncoderReranker(_UpstreamReranker):
    """Score only S/O pairs and format the winning candidates' full SOAP."""

    def rerank(self, query, candidates, topk=5):
        self.ranked_candidates = None
        if topk <= 0 or not candidates or not query.strip():
            self.ranked_candidates = []
            if getattr(self, "on_ranking", None) is not None:
                self.on_ranking([])
            return []
        pairs = [(query, matching_text(candidate)) for candidate in candidates]
        scores = np.asarray(self.model.predict(pairs)).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError("Expected one finite reranking score per candidate")
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:topk]
        self.ranked_candidates = ranked
        if getattr(self, "on_ranking", None) is not None:
            self.on_ranking(ranked)
        results = []
        for number, record in enumerate(ranked, 1):
            info = f"""
Result {number}
Subjective:
{record['subjective']}
Objective:
{record['objective']}
Assessment:
{record['assessment']}
Plan:
{record['plan']}
Rerank score: {record['rerank_score']:.4f}

Dominant medical concepts (contribution >= {CONTRIBUTION_THRESHOLD}):
"""
            for term in record["dominant_terms"]:
                info += f" - {term['original_text']} ({term['contribution']:.4f})\n"
            results.append(info)
        return results
