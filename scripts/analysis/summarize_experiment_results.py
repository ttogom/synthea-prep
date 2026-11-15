"""Generate saved statistics for the three tables used in the report.

The older diagnosis-only summary helper APIs below are retained from the
committed version. The CLI generates only the three selected report tables.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version

# Direct script execution must resolve project modules before importing them.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.analysis.render_tables import generate_tables
from scripts.analysis.table_statistics import (
    summarize_main_results,
    summarize_generator_retrieval_gains,
    summarize_group_retrieval_gains,
)


RESULTS = "experiments/output/2026-11-12_022701PST__test_so__main/experiment_results.csv"


DECIMAL_PLACES = 4


MODEL_LABELS = {
    "gpt_4o_mini": "GPT-4o mini",
    "ollama_gemma3_4b": "Gemma 3 4B",
    "ollama_llama3_2_3b": "Llama 3.2 3B",
    "ollama_ministral_3_3b": "Ministral 3 3B",
}


VARIANT_LABELS = {
    "llm_only": "Baselines",
    "guidelines": "Baselines + Guideline RAG",
    "patient_retrieval": "Baselines + Sim. Patient Retrieval",
    "patient_retrieval_guidelines": "Baselines + Guideline RAG + Sim. Patient Retrieval",
}


SCORES = {
    "biolord_score_diagnosis1": "biolord",
    "bertf1_diagnosis1": "bertf1",
}


CONDITION_GROUPS = {
    "history": {
        "chronic obstructive pulmonary disease": "Chronic obstructive pulmonary disease",
        "essential hypertension": "Essential hypertension",
    },
    "acute": {
        "streptococcal sore throat": "Streptococcal sore throat",
        "acute viral pharyngitis": "Acute viral pharyngitis",
        "acute bacterial sinusitis": "Acute bacterial sinusitis",
        "acute infective cystitis": "Acute infective cystitis",
    },
    "control": {
        "chronic congestive heart failure": "Chronic congestive heart failure",
    },
}


CONDITION_VARIANT_LABELS = {
    "llm_only": "Baseline",
    "guidelines": "Guideline retrieval",
    "patient_retrieval": "Similar-patient retrieval",
    "patient_retrieval_guidelines": "Guideline and similar-patient retrieval",
}


def _load_results(results_path, include_conditions=False):
    identity = ["id", "model", "variant"]
    labels = ["bin", "condition_label"] if include_conditions else []
    results = pd.read_csv(results_path, usecols=[*identity, *labels, *SCORES])
    if results.empty:
        raise ValueError("The results CSV contains no patient rows.")
    if results[[*identity, *labels]].isna().any().any():
        raise ValueError("Patient ID, model, variant, and requested labels must be populated.")
    if results.duplicated(identity).any():
        raise ValueError("Results must have unique patient/model/variant combinations.")
    unknown_models = set(results["model"]) - set(MODEL_LABELS)
    unknown_variants = set(results["variant"]) - set(VARIANT_LABELS)
    if unknown_models or unknown_variants:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}; variants: {sorted(unknown_variants)}")
    results = results.rename(columns=SCORES)
    metrics = list(SCORES.values())
    results[metrics] = results[metrics].apply(pd.to_numeric, errors="raise")
    if results[metrics].isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Scores must be finite or missing.")
    return results


def _summarize(results, group_columns, order):
    metrics = list(SCORES.values())
    grouped = results.groupby(group_columns, sort=False)
    summary = grouped[metrics].describe()
    statistics = {"25%": "q25", "50%": "median", "75%": "q75"}
    summary.columns = [f"{metric}_{statistics.get(stat, stat)}" for metric, stat in summary.columns]
    summary = summary.reindex(order)
    summary.insert(0, "n_total", grouped.size().reindex(order, fill_value=0))
    summary.insert(0, "model_label", summary.index.get_level_values("model").map(MODEL_LABELS))
    summary.insert(0, "variant_label", summary.index.get_level_values("variant").map(VARIANT_LABELS))
    for metric in metrics:
        summary[f"{metric}_count"] = summary[f"{metric}_count"].fillna(0).astype("int64")
        summary[f"{metric}_missing"] = summary["n_total"] - summary[f"{metric}_count"]
    return summary


def summarize_results(results_path):
    results = _load_results(results_path)
    order = pd.MultiIndex.from_product([VARIANT_LABELS, MODEL_LABELS], names=["variant", "model"])
    return _summarize(results, ["variant", "model"], order)


def summarize_conditions(results_path):
    results = _load_results(results_path, include_conditions=True)
    condition_bins = {condition: group for group, conditions in CONDITION_GROUPS.items() for condition in conditions}
    if not results["condition_label"].map(condition_bins).eq(results["bin"]).all():
        raise ValueError("Condition labels must match the history, acute, and control groups.")
    labels_per_case = results.groupby("id")[["bin", "condition_label"]].nunique()
    if labels_per_case.gt(1).any().any():
        raise ValueError("Each patient case must have consistent labels across models and variants.")

    condition_keys = ["model", "variant", "bin", "condition_label"]
    condition_order = pd.MultiIndex.from_tuples(
        [(model, variant, group, condition) for model in MODEL_LABELS for variant in VARIANT_LABELS
         for group, conditions in CONDITION_GROUPS.items() for condition in conditions],
        names=condition_keys,
    )
    conditions = _summarize(results, condition_keys, condition_order).reset_index()
    conditions["level"] = "condition"
    group_keys = ["model", "variant", "bin"]
    group_order = pd.MultiIndex.from_product([MODEL_LABELS, VARIANT_LABELS, CONDITION_GROUPS], names=group_keys)
    groups = _summarize(results, group_keys, group_order).reset_index()
    groups["level"] = "group"
    groups["condition_label"] = ""

    keys = ["model", "bin", "level", "condition_label", "variant"]
    order = []
    for model in MODEL_LABELS:
        for group, conditions_in_group in CONDITION_GROUPS.items():
            for condition in [*conditions_in_group, ""]:
                level = "condition" if condition else "group"
                order.extend((model, group, level, condition, variant) for variant in VARIANT_LABELS)
    summary = pd.concat([conditions, groups], ignore_index=True).set_index(keys)
    return summary.reindex(pd.MultiIndex.from_tuples(order, names=keys))


def select_best_variants(summary):
    summary = summary.sort_index()
    rows = []
    for group, conditions in CONDITION_GROUPS.items():
        for condition in conditions:
            for model, model_label in MODEL_LABELS.items():
                choices = summary.loc[(model, group, "condition", condition)].reindex(VARIANT_LABELS)
                available = choices["biolord_mean"].dropna()
                best_variant = available.idxmax() if not available.empty else None
                row = choices.loc[best_variant].to_dict() if best_variant is not None else dict.fromkeys(choices.columns)
                row.pop("variant_label")
                row["best_variant_label"] = CONDITION_VARIANT_LABELS.get(best_variant)
                row.update(bin=group, condition_label=condition, model=model, model_label=model_label,
                           best_variant=best_variant)
                rows.append(row)
    selected = pd.DataFrame(rows).set_index(["bin", "condition_label", "model"])
    selected.insert(1, "best_variant", selected.pop("best_variant"))
    selected.insert(2, "best_variant_label", selected.pop("best_variant_label"))
    count_columns = ["n_total", *[f"{metric}_{stat}" for metric in SCORES.values() for stat in ("count", "missing")]]
    selected[count_columns] = selected[count_columns].astype("Int64")
    return selected


def _format_mean(value, best=None):
    if pd.isna(value):
        return "--"
    text = f"{value:.{DECIMAL_PLACES}f}"
    return rf"\textbf{{{text}}}" if best is not None and value == best else text


def render_diagnosis_table(summary):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Diagnosis results on Synthea Subjective and Objective clinical notes.}",
        r"\label{tab:diagnosis}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Diagnosis Scores}} \\",
        r"\cmidrule(lr){2-3}",
        r" & BioLORD & BERTScore-F1 \\",
    ]
    for variant, variant_label in VARIANT_LABELS.items():
        best_means = summary.loc[variant, ["biolord_mean", "bertf1_mean"]].max()
        heading = rf"\textbf{{{variant_label}}}"
        if variant == "patient_retrieval_guidelines":
            heading = r"\shortstack{\textbf{Baselines + Guideline RAG}\\\textbf{+ Sim. Patient Retrieval}}"
        lines.extend([
            r"\midrule",
            rf"\multicolumn{{3}}{{c}}{{{heading}}} \\",
            r"\midrule",
        ])
        for model, model_label in MODEL_LABELS.items():
            means = summary.loc[(variant, model), ["biolord_mean", "bertf1_mean"]]
            values = [_format_mean(value, best_means[metric]) for metric, value in means.items()]
            lines.append(rf"{model_label} & {values[0]} & {values[1]} \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def render_condition_table(summary):
    summary = summary.sort_index()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"
        r"p{0.075\textwidth}p{0.225\textwidth}p{0.15\textwidth}p{0.225\textwidth}rrr@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Group} & \multirow{2}{*}{Condition} & \multirow{2}{*}{Model} & "
        r"\multirow{2}{*}{Best Variant} & \multirow{2}{*}{n} & "
        r"\multicolumn{2}{c}{\textbf{Diagnosis Scores}} \\",
        r"\cmidrule(lr){6-7}",
        r" & & & & & BioLORD & BERTScore-F1 \\",
        r"\midrule",
    ]
    for group, conditions in CONDITION_GROUPS.items():
        for condition_index, (condition, condition_label) in enumerate(conditions.items()):
            best_means = summary.loc[(group, condition), ["biolord_mean", "bertf1_mean"]].max()
            for model_index, (model, model_label) in enumerate(MODEL_LABELS.items()):
                row = summary.loc[(group, condition, model)]
                group_cell = ""
                condition_cell = ""
                if condition_index == 0 and model_index == 0:
                    group_cell = rf"\multirow[t]{{{len(conditions) * len(MODEL_LABELS)}}}{{0.075\textwidth}}{{{group.title()}}}"
                if model_index == 0:
                    condition_cell = rf"\multirow[t]{{{len(MODEL_LABELS)}}}{{0.225\textwidth}}{{{condition_label}}}"
                variant = CONDITION_VARIANT_LABELS.get(row["best_variant"], "--")
                n = "--" if pd.isna(row["n_total"]) else str(int(row["n_total"]))
                values = [_format_mean(row[metric], best_means[metric]) for metric in ("biolord_mean", "bertf1_mean")]
                cells = [group_cell, condition_cell, model_label, variant, n, *values]
                lines.append(" & ".join(cells) + r" \\")
            if condition_index < len(conditions) - 1:
                lines.append(r"\cmidrule(lr){2-7}")
        if group != "control":
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\caption{Diagnosis results by condition. For each model and condition, "
        r"the variant with the highest mean BioLORD is shown.}",
        r"\label{tab:diagnosis_by_condition}",
        r"\par\smallskip",
        r"{\scriptsize Bold marks the highest displayed score per condition.\par}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


_UUID_SUFFIX = re.compile(r"([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$")


_RECORD_ARTIFACT = re.compile(r"\(\s*record\s+artifact\s*\)", re.IGNORECASE)


REQUIRED_SCORE_COLUMNS = (
    "id", "model", "variant", "bin", "condition_label", "plan_label",
    "biolord_score_diagnosis1", "plan1_score", "plan1_status", "plan1_reason",
)


def load_scored_results(results_path):
    """Load raw rank-1 outcomes; a missing judgement is never a zero score."""
    results = pd.read_csv(results_path)
    missing = set(REQUIRED_SCORE_COLUMNS) - set(results)
    if missing:
        raise ValueError(f"Missing table result columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("The table results CSV is empty.")
    keys = ["id", "model", "variant"]
    if results[[*keys, "bin", "condition_label"]].isna().any().any():
        raise ValueError("Case identity and condition/group labels must be populated.")
    if results.duplicated(keys).any():
        raise ValueError("table results require unique case/model/variant keys.")
    if set(results["model"]) - set(MODEL_LABELS) or set(results["variant"]) - set(VARIANT_LABELS):
        raise ValueError("table results contain unknown models or variants.")
    condition_groups = {
        condition: group for group, conditions in CONDITION_GROUPS.items() for condition in conditions
    }
    if not results["condition_label"].map(condition_groups).eq(results["bin"]).all():
        raise ValueError("Condition labels do not match their expected groups.")
    results["plan_label"] = results["plan_label"].fillna("").astype(str)
    if results.groupby("id")[["bin", "condition_label", "plan_label"]].nunique().gt(1).any().any():
        raise ValueError("Each case must have consistent condition, group, and reference labels.")
    results["patient_uuid"] = results["id"].astype(str).map(
        lambda value: match.group(1).lower() if (match := _UUID_SUFFIX.search(value)) else value
    )
    results["diagnosis"] = pd.to_numeric(results["biolord_score_diagnosis1"], errors="raise")
    plan_score = pd.to_numeric(results["plan1_score"], errors="raise")
    results["plan"] = plan_score.where(results["plan1_status"].eq("scored"))
    # Reference availability is a case property, independent of generation/judge failures.
    reference = results["plan_label"].str.strip()
    results["plan_eligible"] = (
        reference.ne("") & ~reference.str.casefold().isin(["none recorded", '["none recorded"]'])
        & ~reference.str.contains(_RECORD_ARTIFACT)
    )
    results.loc[~results["plan_eligible"], "plan"] = np.nan
    for metric in ("diagnosis", "plan"):
        values = results[metric].dropna()
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"{metric} scores must be finite and within [0, 1], or missing.")
    return results


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_table_statistics(results_path, output_dir=None, bootstrap_samples=1000, seed=20261114):
    """Generate only the CSVs required by the three report tables.

    Read source and run metadata without changing them. Snapshot fingerprints
    connect the saved statistics to the experiment; abort if the CSV changes
    while calculations are running. Never publish incomplete eligible cohorts.
    """
    results_path = Path(results_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else results_path.parent / 'analysis'
    source_hash = _sha256(results_path)
    results = load_scored_results(results_path)
    run_path = results_path.with_name('run.json')
    run_bytes = run_path.read_bytes() if run_path.exists() else None
    run = json.loads(run_bytes) if run_bytes is not None else {}
    judge = run.get('plan_evaluation', {}).get('judge_settings', {})
    unvalidated = any(judge.get(field) == 'experimental_unvalidated'
                      for field in ('validation_status', 'reporting'))
    if unvalidated:
        results['plan'] = np.nan
    summaries = {
        'main_results': summarize_main_results(results, bootstrap_samples, seed),
        'generator_retrieval_gains': summarize_generator_retrieval_gains(results, include_model_average=True),
        'group_retrieval_gains': summarize_group_retrieval_gains(results),
    }
    if source_hash != _sha256(results_path):
        raise RuntimeError('The source CSV changed during analysis; rerun from a stable export.')
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in summaries.items():
        path = output_dir / f'{name}.csv'
        frame.to_csv(path, index=False, na_rep='')
        outputs.append({'path': str(path), 'sha256': _sha256(path)})
    source_jsonl = results_path.with_name('results.jsonl')
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_csv': str(results_path), 'source_csv_sha256': source_hash,
        'source_results_newer_than_csv': source_jsonl.exists() and source_jsonl.stat().st_mtime > results_path.stat().st_mtime,
        'run_metadata_path': str(run_path) if run_bytes is not None else None,
        'run_metadata_sha256': hashlib.sha256(run_bytes).hexdigest() if run_bytes is not None else None,
        'unvalidated_plan_scores_excluded': unvalidated,
        'rank': 1, 'case_weighting': 'Equal weight per condition example',
        'rows': len(results), 'condition_examples': results['id'].nunique(),
        'unique_patients': results['patient_uuid'].nunique(),
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy_version},
        'test_policy': {
            'null': 'mean(retrieval - no retrieval) <= 0', 'alternative': 'greater',
            'test': 'Paired CR1 patient-clustered t test', 'degrees_of_freedom': 'patient clusters minus 1',
            'thresholds': {'*': 'adjusted p < 0.05', '**': 'adjusted p < 0.01'},
            'missing_tests': 'No stars; retain their place in the planned Holm family',
        },
        'main_results': {
            'holm_family': 'Three retrieval configurations within each generator/outcome',
            'mean_intervals': 'Pointwise 95% patient-cluster percentile bootstrap',
            'bootstrap_samples': bootstrap_samples, 'seed': seed,
        },
        'generator_retrieval_gains': {
            'holm_family': 'Two single-source configurations across all generators and their average, per outcome',
            'average': 'Average paired scores equally across the fixed generators within each case before patient clustering',
            'family_sizes': summaries['generator_retrieval_gains'].groupby('metric')['family_size'].first().to_dict(),
        },
        'group_retrieval_gains': {
            'holm_family': 'Nine tests: three retrieval configurations by three groups, within each generator/outcome',
            'scope': 'Within-group improvement; not a test comparing gains between groups',
            'plan_history_scope': 'Hypertension only; COPD reference Plans are unavailable',
        },
        'coverage_policy': 'Keep individual means only for fully scored eligible arms; suppress differences and inference if any eligible pair is missing. Unavailable Plan references are excluded.',
        'implementation_sha256': {
            str(path.resolve()): _sha256(path)
            for path in (Path(__file__), Path(__file__).with_name('table_statistics.py'))
        },
        'outputs': outputs,
    }
    (output_dir / 'table_statistics_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path, default=ROOT / RESULTS)
    parser.add_argument('--csv-only', action='store_true', help='Regenerate the three statistics CSVs without rendering TeX.')
    args = parser.parse_args(argv)
    manifest = generate_table_statistics(args.results_csv)
    analysis_dir = args.results_csv.expanduser().resolve().parent / 'analysis'
    print(f'Saved statistics and provenance to {analysis_dir}')
    if not args.csv_only:
        # All imports remain at module scope; this renderer only consumes the saved CSVs.
        generate_tables(analysis_dir)
    return manifest


if __name__ == '__main__':
    main()
