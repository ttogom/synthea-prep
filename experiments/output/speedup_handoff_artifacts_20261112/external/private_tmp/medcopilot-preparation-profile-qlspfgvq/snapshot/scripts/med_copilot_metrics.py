"""Singleton lexical metrics and bounded, reusable BERTScore inference.

One TextMetricEvaluator belongs to one process. Calls are serialized; construct
it inside a future scoring worker rather than transferring a live Torch model.
No work or results are retained between score_pairs() calls except the model.
"""

from dataclasses import dataclass
import math
from numbers import Real
import os
from threading import Lock

from bert_score import BERTScorer
import nltk
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
import torch


METRIC_NAMES = ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1")
BATCH_SIZE = 64
BERT_SETTINGS = {
    "model_type": "roberta-large", "num_layers": 17, "lang": "en",
    "idf": False, "rescale_with_baseline": False, "use_fast_tokenizer": False,
    "all_layers": False, "device": None, "nthreads": 4,
}


def resolved_device():
    """Use bert-score's unchanged CUDA/CPU default, never implicitly Apple MPS."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class MetricPair:
    """An explicit caller identity and a single reference/candidate pair."""

    key: tuple
    reference: str
    candidate: str


@dataclass(frozen=True)
class MetricScore:
    """Exactly one complete metric payload or an ordinary field-local failure."""

    key: tuple
    metrics: dict | None = None
    error: Exception | None = None


def validate_metrics(metrics):
    """Copy and validate the six finite scalar metrics before committing a field."""
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise ValueError("Evaluator must return exactly the six upstream metric keys")
    values = {}
    for name in METRIC_NAMES:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Evaluator metric {name} must be a finite real number")
        values[name] = float(value)
    return values


def lexical_metrics(reference, candidate):
    """Preserve upstream singleton BLEU/METEOR/ROUGE, including tokenization."""
    bleu = sacrebleu.corpus_bleu([candidate], [[reference]]).score / 100
    meteor = meteor_score.single_meteor_score(
        nltk.word_tokenize(reference), nltk.word_tokenize(candidate))
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = rouge.score(reference, candidate)
    return {
        "BLEU": bleu,
        "METEOR": meteor,
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }


class TextMetricEvaluator:
    """Lazy model reuse and real batches; yields results in input identity order.

    The finite batch API accepts MetricPair objects and yields MetricScore objects.
    Ordinary inference failures fall back to singleton requests on the SAME model
    to isolate bad pairs. Interruptions propagate; the caller owns checkpointing.
    Failed initialization is retried on a later call, never once per input pair.
    """

    def __init__(self, *, bert_scorer_factory=None, lexical=None):
        self._factory = bert_scorer_factory or BERTScorer
        self._lexical = lexical or lexical_metrics
        self._bert = None
        self._pid = os.getpid()
        self._lock = Lock()

    def score_pairs(self, pairs):
        """Score finite chunks of 64; exhaust or close the iterator to release its lock."""
        if os.getpid() != self._pid:
            raise RuntimeError("Create TextMetricEvaluator inside its owning scoring process")
        pairs = list(pairs)
        keys = set()
        for pair in pairs:
            if (not isinstance(pair, MetricPair) or not isinstance(pair.key, tuple)
                    or not isinstance(pair.reference, str) or not isinstance(pair.candidate, str)):
                raise TypeError("Expected MetricPair with a tuple key and string texts")
            if pair.key in keys:
                raise ValueError("MetricPair identities must be unique")
            keys.add(pair.key)
        if not pairs:
            return
        with self._lock:
            if self._bert is None:
                try:
                    # Assign only after successful construction; failures are not cached.
                    self._bert = self._factory(**BERT_SETTINGS)
                except Exception as exc:
                    for pair in pairs:
                        yield MetricScore(pair.key, error=exc)
                    return
            for start in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[start:start + BATCH_SIZE]
                for pair, value in zip(batch, self._bert_values(batch)):
                    if isinstance(value, Exception):
                        yield MetricScore(pair.key, error=value)
                        continue
                    try:
                        metrics = validate_metrics({**self._lexical(pair.reference, pair.candidate),
                                                    "BERTScore_F1": value})
                    except Exception as exc:
                        yield MetricScore(pair.key, error=exc)
                    else:
                        yield MetricScore(pair.key, metrics=metrics)

    def _bert_values(self, pairs):
        try:
            _, _, values = self._bert.score(
                [pair.candidate for pair in pairs], [pair.reference for pair in pairs],
                batch_size=BATCH_SIZE, verbose=False)
            if hasattr(values, "tolist"):
                values = values.tolist()
            if not isinstance(values, (list, tuple)) or len(values) != len(pairs):
                raise ValueError("BERTScore must return one F1 value per pair")
            return [float(value) if not isinstance(value, bool) and isinstance(value, Real)
                    and math.isfinite(value) else ValueError("BERTScore F1 must be a finite real number")
                    for value in values]
        except Exception as exc:
            if len(pairs) == 1:
                return [exc]
            # Do not reload the model or retry successful batches. A single bad
            # input/backend batch failure must not discard unrelated fields.
            return [self._bert_values([pair])[0] for pair in pairs]
