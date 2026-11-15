"""Render the four report tables from three saved statistics CSVs.

Run after summarize_experiment_results.py --csv-only. By default all four
experiment artifacts and report copies are updated. Use --table to select one,
or --analysis-only to leave the report untouched. No statistical tests or judging are run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys

import pandas as pd

# Direct script execution must resolve project modules before importing them.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.table_rendering import (
    render_main_results_table,
    render_generator_retrieval_gains_table,
    render_group_retrieval_gains_table,
    render_history_gains_over_others_table,
)


DEFAULT_ANALYSIS = ROOT / 'experiments/output/2026-11-12_022701PST__test_so__main/analysis'
DEFAULT_REPORT_DIR = ROOT.parent / 'CSCI-566-Midterm-Report/figures'
TABLES = {
    'main_results': render_main_results_table,
    'generator_retrieval_gains': render_generator_retrieval_gains_table,
    'group_retrieval_gains': render_group_retrieval_gains_table,
    'history_gains_over_others': render_history_gains_over_others_table,
}

# Multiple layouts can share one statistics CSV; never duplicate computed results.
TABLE_INPUTS = {name: name for name in TABLES}
TABLE_INPUTS['history_gains_over_others'] = 'group_retrieval_gains'


def generate_tables(analysis_dir=DEFAULT_ANALYSIS, table_names=None, report_dir=None):
    """Save selected TeX files, optional report copies, and byte-level provenance.

    Validate every selected CSV before changing any table. A single-table refresh
    preserves the other tables' existing export-manifest entries.
    """
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve() if report_dir is not None else None
    names = list(TABLES) if table_names is None else ([table_names] if isinstance(table_names, str) else list(table_names))
    if not names or len(set(names)) != len(names) or set(names) - set(TABLES):
        raise ValueError('Select distinct table names from: ' + ', '.join(TABLES))
    prepared = {}
    for name in names:
        source = analysis_dir / f'{TABLE_INPUTS[name]}.csv'
        snapshot = source.read_bytes()
        scores = pd.read_csv(StringIO(snapshot.decode('utf-8-sig')))
        prepared[name] = (source, snapshot, TABLES[name](scores).encode('utf-8'))
    manifest_path = analysis_dir / 'table_exports_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = previous.get('tables', {})
    implementations = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name('table_rendering.py'))
    }
    outputs = {}
    for name, (source, snapshot, snippet) in prepared.items():
        destination = analysis_dir / 'tables' / f'{name}.tex'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(snippet)
        record = {
            'input_csv': str(source), 'input_csv_sha256': hashlib.sha256(snapshot).hexdigest(),
            'table_path': str(destination), 'table_sha256': hashlib.sha256(snippet).hexdigest(),
            'implementation_sha256': implementations,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        outputs[name] = {'table': destination}
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_copy = report_dir / f'{name}.tex'
            report_copy.write_bytes(snippet)
            record.update(report_path=str(report_copy), report_sha256=record['table_sha256'])
            outputs[name]['report'] = report_copy
        records[name] = record
    if report_dir is not None:
        main = report_dir.parent / 'main.tex'
        if main.exists():
            main.touch()
    manifest = {
        'statistics': 'Uses saved means, intervals, coverage and adjusted p-values; no statistics recomputed.',
        'tables': records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument('--table', choices=tuple(TABLES), action='append', help='Select one table; repeat to select several.')
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument('--analysis-only', action='store_true', help='Write experiment TeX files without updating report copies.')
    args = parser.parse_args(argv)
    outputs = generate_tables(args.analysis_dir, args.table, None if args.analysis_only else args.report_dir)
    for name, paths in outputs.items():
        print(f"Saved {name}: {paths['table']}")
        if 'report' in paths:
            print(f"Updated report copy: {paths['report']}")
    return outputs


if __name__ == '__main__':
    main()
