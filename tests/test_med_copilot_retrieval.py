"""Offline S/O matching tests: fake models/API/FAISS, real scoring and cache I/O.

Run: python tests/test_med_copilot_retrieval.py
The fake FAISS implementation also makes these tests usable without model packages.
"""

import ast
import copy
from contextlib import chdir
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "scripts/med_copilot_retrieval.py"
UPSTREAM = ROOT / "med_copilot/upstream/create_embeddings.py"


class FakeEncoder:
    max_seq_length = 512

    def __init__(self, revision="test-revision"):
        self.revision = revision
        self.calls = []

    def __getitem__(self, key):
        return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(
            _commit_hash=self.revision,
        )))

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.asarray([[1, 3 * text.lower().count("cough"),
                            3 * text.lower().count("fever"),
                            3 * text.lower().count("headache")]
                           for text in texts], dtype=np.float32)


class FakeIndex:
    def __init__(self, dimension):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.concatenate([self.vectors, vectors])

    def search(self, query, count):
        scores = query @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :count]
        return np.take_along_axis(scores, order, axis=1), order


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [float("cough" in reference.lower()) for _, reference in pairs]


def load_retrieval_module(keyword_client=None):
    """Load production adapters and actual upstream utilities without live models."""
    faiss_module = ModuleType("faiss")

    def normalize(vectors):
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    def write_index(index, path):
        Path(path).write_text(json.dumps({"dimension": index.d,
                                        "vectors": index.vectors.tolist()}))

    def read_index(path):
        saved = json.loads(Path(path).read_text())
        index = FakeIndex(saved["dimension"])
        index.add(np.asarray(saved["vectors"], dtype=np.float32))
        return index

    faiss_module.normalize_L2 = normalize
    faiss_module.IndexFlatIP = FakeIndex
    faiss_module.write_index = write_index
    faiss_module.read_index = read_index
    if keyword_client is None:
        keyword_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=AssertionError("Unexpected API request")),
        )))
    namespace = {
        "np": np, "pd": pd, "faiss": faiss_module, "json": json, "os": SimpleNamespace(),
        "OpenAI": lambda api_key=None: keyword_client,
        "CrossEncoder": lambda name: FakeCrossEncoder(),
        "CONTRIBUTION_THRESHOLD": 0.85,
        "NLP": Mock(side_effect=AssertionError("Unexpected NLP invocation")),
    }
    # Execute the actual inherited utility methods; constructors with external
    # dependencies are overridden by production adapters or replaced above.
    tree = ast.parse(UPSTREAM.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"HybridRetriever", "CrossEncoderReranker", "LLMKeywordExtractor"}]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(UPSTREAM), "exec"), namespace)
    upstream_module = ModuleType("create_embeddings")
    upstream_module.__dict__.update(namespace)
    sentence_module = ModuleType("sentence_transformers")
    sentence_module.SentenceTransformer = Mock(return_value=FakeEncoder())
    specification = importlib.util.spec_from_file_location("so_retrieval_under_test", RETRIEVAL)
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, {"faiss": faiss_module, "create_embeddings": upstream_module,
                                  "sentence_transformers": sentence_module}):
        specification.loader.exec_module(module)
    return module


class SOMatchingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cwd = chdir(self.directory)
        self.cwd.__enter__()
        self.addCleanup(self.cwd.__exit__, None, None, None)
        self.module = load_retrieval_module()
        self.encoder = FakeEncoder()
        self.records = [
            {"note_id": "DISTRACTOR", "subjective": "headache", "objective": "normal",
             "assessment": "cough diagnosis ONLY_IN_A", "plan": "cough treatment ONLY_IN_P"},
            {"note_id": "MATCH", "subjective": "cough café\nDay 2", "objective": "SpO₂ 98%",
             "assessment": "REFERENCE_DIAGNOSIS", "plan": "REFERENCE_PLAN\nFollow-up"},
        ]

    def retriever(self, records=None, encoder=None, **kwargs):
        retriever = self.module.HybridRetriever(
            pd.DataFrame(self.records if records is None else records),
            encoder=self.encoder if encoder is None else encoder, **kwargs,
        )
        retriever.keyword_extractor = Mock()
        retriever.keyword_extractor.extract.return_value = ["cough"]
        retriever.compute_case_phrase_contributions = Mock(return_value=([], []))
        return retriever

    def metadata(self):
        return json.loads((self.directory / "matching.json").read_text())

    def test_matching_text_uses_only_so_and_preserves_whitespace_unicode_and_empty_sections(self):
        record = {"subjective": " café\n ", "objective": "", "assessment": "PRIVATE_A",
                  "plan": "PRIVATE_P", "id": "PRIVATE_ID"}
        original = copy.deepcopy(record)
        self.assertEqual(self.module.matching_text(record), "Subjective:  café\n \nObjective: ")
        self.assertEqual(record, original)
        for field in ("subjective", "objective"):
            for bad in (None, 7, [], {}):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    self.module.matching_text({**record, field: bad})
            with self.assertRaises(ValueError):
                self.module.matching_text({name: value for name, value in record.items() if name != field})

    def test_build_uses_so_only_keeps_full_payload_and_writes_normalized_mapped_artifacts(self):
        frame = pd.DataFrame(self.records)
        original = frame.copy(deep=True)
        retriever = self.module.HybridRetriever(frame, encoder=self.encoder)
        expected = [self.module.matching_text(record) for record in self.records]
        self.assertEqual(self.encoder.calls, [expected])
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(retriever.df, original)
        self.assertEqual(set(path.name for path in self.directory.iterdir()),
                         {"embeddings.npy", "index.faiss", "matching.json"})
        vectors = np.load(self.directory / "embeddings.npy")
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        self.assertEqual(retriever.embedder.index.ntotal, len(self.records))
        self.assertEqual(self.metadata()["identity"]["representation"], "SO")
        self.assertEqual(self.metadata()["identity"]["record_count"], 2)
        self.assertNotIn("ONLY_IN_A", json.dumps(self.metadata()))

    def test_constructor_rejects_invalid_corpus_and_weight_before_writing_artifacts(self):
        for records, alpha in [([], 0.5), (self.records, -0.1), (self.records, 1.1),
                               ([{**self.records[0], "subjective": 7}], 0.5),
                               ([{**self.records[0], "assessment": None}], 0.5),
                               ([{**self.records[0], "plan": None}], 0.5),
                               ([{**self.records[0], "subjective": " ", "objective": ""}], 0.5)]:
            with self.subTest(records=records, alpha=alpha), self.assertRaises(ValueError):
                self.retriever(records, alpha=alpha)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_constructor_without_supplied_encoder_uses_requested_model(self):
        with patch.object(self.module, "SentenceTransformer", return_value=self.encoder) as factory:
            retriever = self.module.HybridRetriever(pd.DataFrame(self.records), model_name="test-model")
        factory.assert_called_once_with("test-model")
        self.assertIs(retriever.embedder.model, self.encoder)
        self.assertEqual(retriever.keyword_extractor.model, "gpt-4o-mini")

    def test_reuses_cache_after_ap_changes_without_reembedding_or_stale_generation_payload(self):
        self.retriever()
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        changed = copy.deepcopy(self.records)
        changed[1].update(assessment="NEW_A", plan="NEW_P")
        fresh_encoder = FakeEncoder()
        retriever = self.retriever(changed, encoder=fresh_encoder)
        self.assertEqual(fresh_encoder.calls, [])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})
        candidate = retriever.search("cough")[0]
        self.assertEqual(candidate["idx"], 1)
        self.assertEqual(candidate["assessment"], "NEW_A")
        self.assertEqual(candidate["plan"], "NEW_P")

    def test_rebuilds_cache_when_so_row_order_model_or_encoder_revision_changes(self):
        self.retriever()
        cases = [
            ([{**self.records[0], "objective": "fever"}, self.records[1]], {}, FakeEncoder()),
            (list(reversed(self.records)), {}, FakeEncoder()),
            (self.records, {"model_name": "other-encoder"}, FakeEncoder()),
            (self.records, {}, FakeEncoder(revision="different-revision")),
        ]
        for records, options, encoder in cases:
            with self.subTest(options=options, revision=encoder.revision):
                before = self.metadata()["identity"]
                retriever = self.retriever(records, encoder=encoder, **options)
                self.assertEqual(encoder.calls, [[self.module.matching_text(row) for row in records]])
                self.assertNotEqual(before, self.metadata()["identity"])
                self.assertEqual(retriever.search("cough")[0]["subjective"],
                                 next(row["subjective"] for row in records if "cough" in row["subjective"]))

    def test_missing_manifest_legacy_or_tampered_artifacts_are_rebuilt(self):
        for filename in ("matching.json", "embeddings.npy", "index.faiss"):
            with self.subTest(filename=filename):
                self.retriever()
                (self.directory / filename).write_bytes(b"LEGACY_SOA_OR_CORRUPT")
                encoder = FakeEncoder()
                self.retriever(encoder=encoder)
                self.assertEqual(len(encoder.calls), 1)
                self.assertEqual(self.metadata()["identity"]["representation"], "SO")
                (self.directory / filename).unlink()
                encoder = FakeEncoder()
                self.retriever(encoder=encoder)
                self.assertEqual(len(encoder.calls), 1)

    def test_cache_with_invalid_counts_dimensions_norms_or_nan_is_rebuilt_even_with_matching_hashes(self):
        mutations = [
            lambda vectors: vectors[:1],
            lambda vectors: vectors[:, :2],
            lambda vectors: vectors * 2,
            lambda vectors: vectors * np.nan,
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.retriever()
                path = self.directory / "embeddings.npy"
                np.save(path, mutation(np.load(path)))
                metadata = self.metadata()
                metadata["sha256"]["embeddings.npy"] = self.module._sha256(path)
                (self.directory / "matching.json").write_text(json.dumps(metadata))
                encoder = FakeEncoder()
                self.retriever(encoder=encoder)
                self.assertEqual(len(encoder.calls), 1)

    def test_bad_model_embeddings_fail_without_publishing_cache(self):
        for vectors in (np.zeros((2, 4)), np.ones((1, 4)), np.full((2, 4), np.nan),
                        np.full((2, 4), 1e30), np.ones(4)):
            with self.subTest(shape=vectors.shape):
                encoder = FakeEncoder()
                encoder.encode = Mock(return_value=vectors)
                with self.assertRaises(ValueError):
                    self.retriever(encoder=encoder)
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_write_failure_propagates_cleans_staging_and_next_start_recovers(self):
        with patch.object(self.module.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.retriever()
        self.assertEqual(list(self.directory.iterdir()), [])
        self.retriever()
        self.assertEqual(self.metadata()["identity"]["representation"], "SO")

    def test_interrupted_cache_replacement_is_detected_and_rebuilt(self):
        self.retriever()
        changed = [{**self.records[0], "subjective": "fever"}, self.records[1]]
        original_replace = self.module.os.replace

        def fail_index(source, destination):
            if str(destination) == "index.faiss":
                raise OSError("interrupted replacement")
            return original_replace(source, destination)

        with patch.object(self.module.os, "replace", side_effect=fail_index):
            with self.assertRaisesRegex(OSError, "interrupted replacement"):
                self.retriever(changed)
        self.assertFalse(any(path.is_dir() for path in self.directory.iterdir()))
        encoder = FakeEncoder()
        self.retriever(changed, encoder=encoder)
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(self.metadata()["sha256"], {
            name: self.module._sha256(self.directory / name)
            for name in ("embeddings.npy", "index.faiss")
        })

    def test_search_ignores_ap_for_dense_and_keyword_scores_but_retains_full_soap(self):
        original = copy.deepcopy(self.records)
        retriever = self.retriever()
        candidates = retriever.search("cough", topk=20)
        self.assertEqual([item["idx"] for item in candidates], [1, 0])
        self.assertEqual([item["keyword_score"] for item in candidates], [1, 0])
        self.assertIn("REFERENCE_DIAGNOSIS", candidates[0]["display_text"])
        self.assertIn("REFERENCE_PLAN\nFollow-up", candidates[0]["display_text"])
        self.assertNotIn("ONLY_IN_A", candidates[1]["case_text"])
        self.assertNotIn("ONLY_IN_P", candidates[1]["case_text"])
        self.assertEqual(self.records, original)
        expected = 0.5 * candidates[0]["embedding_score"] + 0.5
        self.assertAlmostEqual(candidates[0]["hybrid_score"], expected)
        self.assertEqual(retriever.compute_case_phrase_contributions.call_count, 2)
        self.assertIn("REFERENCE_PLAN", retriever.compute_case_phrase_contributions.call_args_list[0]
                      .kwargs["case_text"])

    def test_search_empty_keywords_uses_full_target_and_multiple_keywords_use_mean_embedding(self):
        retriever = self.retriever()
        for keywords, expected in [([], ["cough fever"]), (["cough", "fever"], ["cough", "fever"])]:
            with self.subTest(keywords=keywords):
                retriever.keyword_extractor.extract.return_value = keywords
                candidates = retriever.search("cough fever", topk=1)
                self.assertEqual(len(candidates), 1)
                self.assertEqual(self.encoder.calls[-1], expected)
                if not keywords:
                    self.assertEqual(candidates[0]["keyword_score"], 0)

    def test_search_small_corpus_empty_query_nonpositive_topk_and_invalid_positions(self):
        retriever = self.retriever(self.records[:1])
        original_search = retriever.embedder.index.search
        retriever.embedder.index.search = Mock(wraps=original_search)
        self.assertEqual(len(retriever.search("cough", topk=20)), 1)
        self.assertEqual(retriever.embedder.index.search.call_args.args[1], 1)
        retriever.keyword_extractor.extract.reset_mock()
        for query, count in [("", 20), ("  ", 20), ("cough", 0), ("cough", -1)]:
            self.assertEqual(retriever.search(query, count), [])
        retriever.keyword_extractor.extract.assert_not_called()
        retriever.embedder.index.search = Mock(return_value=(
            np.array([[1, 1, 1, 1]]), np.array([[-1, 99, 0, 0]]),
        ))
        self.assertEqual([item["idx"] for item in retriever.search("cough")], [0])

    def test_search_failures_propagate_instead_of_becoming_no_reference(self):
        for stage in ("keyword", "embedding", "index", "attribution"):
            with self.subTest(stage=stage):
                retriever = self.retriever(encoder=FakeEncoder())
                error = RuntimeError(stage)
                if stage == "keyword":
                    retriever.keyword_extractor.extract.side_effect = error
                elif stage == "embedding":
                    retriever.embedder.model.encode = Mock(side_effect=error)
                elif stage == "index":
                    retriever.embedder.index.search = Mock(side_effect=error)
                else:
                    retriever.compute_case_phrase_contributions.side_effect = error
                with self.assertRaisesRegex(RuntimeError, stage):
                    retriever.search("cough")

    def test_reranker_scores_so_only_and_formats_original_soap_preserving_stable_ties(self):
        candidates = self.retriever().search("cough")
        reranker = self.module.CrossEncoderReranker()
        reranker.model.predict = Mock(return_value=[0.9, 0.9])
        results = reranker.rerank("TARGET_SO", candidates, topk=1)
        self.assertEqual(reranker.model.predict.call_args.args[0], [
            ("TARGET_SO", self.module.matching_text(candidate)) for candidate in candidates
        ])
        self.assertEqual([item["rerank_score"] for item in candidates], [0.9, 0.9])
        self.assertEqual(len(results), 1)
        for field in ("subjective", "objective", "assessment", "plan"):
            self.assertIn(self.records[1][field], results[0])
        self.assertIn("Rerank score: 0.9000", results[0])

    def test_ap_changes_cannot_change_matching_or_reranking_but_change_generation_example(self):
        query = "cough"
        candidates = self.retriever().search(query)
        reranker = self.module.CrossEncoderReranker()
        before = reranker.rerank(query, candidates)
        changed = copy.deepcopy(self.records)
        for record in changed:
            record.update(assessment="fever cough NEW_A", plan="headache cough NEW_P")
        after_candidates = self.retriever(changed).search(query)
        after = reranker.rerank(query, after_candidates)
        fields = ("idx", "embedding_score", "keyword_score", "hybrid_score", "rerank_score")
        self.assertEqual([{field: item[field] for field in fields} for item in candidates],
                         [{field: item[field] for field in fields} for item in after_candidates])
        self.assertEqual(reranker.model.calls[0], reranker.model.calls[1])
        self.assertNotEqual(before, after)
        self.assertIn("NEW_A", after[0])
        self.assertIn("NEW_P", after[0])

    def test_reranker_empty_inputs_bad_predictions_and_model_failure(self):
        reranker = self.module.CrossEncoderReranker()
        reranker.model.predict = Mock()
        for query, candidates, count in [("query", [], 5), ("", self.records, 5),
                                          ("query", self.records, 0)]:
            self.assertEqual(reranker.rerank(query, candidates, count), [])
        reranker.model.predict.assert_not_called()
        candidates = self.retriever().search("cough")
        for scores in ([1], [1, np.nan], [1, np.inf]):
            reranker.model.predict.return_value = scores
            with self.assertRaises(ValueError):
                reranker.rerank("query", candidates)
        reranker.model.predict.side_effect = RuntimeError("model failure")
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            reranker.rerank("query", candidates)


if __name__ == "__main__":
    unittest.main()
