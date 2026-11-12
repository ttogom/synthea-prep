"""Metric-runtime double for existing field routing and recovery regressions.

The production adapter's batching/inference is tested separately with fake
BERTScorers; these tests retain their original per-field failure injection.
"""

from types import SimpleNamespace


class SingletonMetricDouble:
    def __init__(self, evaluate):
        self.evaluate = evaluate

    def score_pairs(self, pairs):
        for pair in pairs:
            try:
                metrics = self.evaluate([pair.reference], [pair.candidate], lang="en")
            except Exception as exc:
                yield SimpleNamespace(key=pair.key, metrics=None, error=exc)
            else:
                yield SimpleNamespace(key=pair.key, metrics=metrics, error=None)
