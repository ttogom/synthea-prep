"""Compare each agentic variant with its fixed counterpart using paired two-sided tests."""

import argparse
from pathlib import Path
import sys

# Direct execution must resolve repository modules before importing them.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.summarize_experiment_results import RESULTS, load_scored_results
from scripts.analysis.table_statistics import summarize_agentic_counterpart_comparisons
from scripts.plan_judge_v2.inputs import sha256_file


OUTPUT_NAME = 'agentic_vs_nonagentic_significance.csv'


def generate_agentic_comparisons(results_path=ROOT / RESULTS):
    """Validate the protocol-2 input and save six tests without changing other artifacts."""
    results_path = Path(results_path).expanduser().resolve()
    fingerprint = sha256_file(results_path)
    results = load_scored_results(results_path)
    if (not {'plan1_protocol_version', 'plan1_rank', 'diagnosis1_status'} <= set(results)
            or not results.plan1_protocol_version.eq(2).all() or not results.plan1_rank.eq(1).all()):
        raise ValueError('Require protocol-2 rank-1 judgments and diagnosis statuses')
    summary = summarize_agentic_counterpart_comparisons(results)
    if fingerprint != sha256_file(results_path):
        raise RuntimeError('The source CSV changed during analysis')
    destination = results_path.parent / 'analysis' / OUTPUT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False, na_rep='')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path, default=ROOT / RESULTS)
    args = parser.parse_args(argv)
    summary = generate_agentic_comparisons(args.results_csv)
    destination = args.results_csv.expanduser().resolve().parent / 'analysis' / OUTPUT_NAME
    print(f'Saved {len(summary)} paired two-sided comparisons to {destination}')


if __name__ == '__main__':
    main()
