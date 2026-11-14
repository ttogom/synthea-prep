from pathlib import Path

import pandas as pd


RESULTS = "experiments/output/2026-11-12_022701PST__test_so__main/experiment_results.csv"
ROOT = Path(__file__).resolve().parents[2]
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
        "chronic obstructive pulmonary disease": "COPD",
        "essential hypertension": "Essential hypertension",
    },
    "acute": {
        "streptococcal sore throat": "Strep. sore throat",
        "acute viral pharyngitis": "Acute viral pharyngitis",
        "acute bacterial sinusitis": "Acute bact. sinusitis",
        "acute infective cystitis": "Acute infective cystitis",
    },
    "control": {
        "chronic congestive heart failure": "Chronic CHF",
    },
}
VARIANT_SHORT_LABELS = {
    "llm_only": "None",
    "guidelines": "Guide.",
    "patient_retrieval": "Patient",
    "patient_retrieval_guidelines": "Both",
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
                row["best_variant_label"] = row.pop("variant_label")
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
        r"\begin{table}[t]",
        r"\centering",
        r"\fontsize{8}{9}\selectfont",
        r"\setlength{\tabcolsep}{0.3pt}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}"
        r"p{0.128\columnwidth}p{0.195\columnwidth}p{0.155\columnwidth}p{0.13\columnwidth}rrr@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Group} & \multirow{2}{*}{Condition} & \multirow{2}{*}{Model} & "
        r"\multirow{2}{*}{\shortstack{Best\\Variant}} & \multirow{2}{*}{n} & "
        r"\multicolumn{2}{c}{\textbf{Diagnosis Scores}} \\",
        r"\cmidrule(lr){6-7}",
        r" & & & & & BioLORD & \shortstack{BERT\\Score-F1} \\",
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
                    group_cell = rf"\multirow[t]{{{len(conditions) * len(MODEL_LABELS)}}}{{0.128\columnwidth}}{{{group.title()}}}"
                if model_index == 0:
                    label = condition_label.replace("hypertension", r"hyper\-tension")
                    condition_cell = rf"\multirow[t]{{{len(MODEL_LABELS)}}}{{0.195\columnwidth}}{{{label}}}"
                variant = VARIANT_SHORT_LABELS.get(row["best_variant"], "--")
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
        r"{\scriptsize None: no retrieval; Guide.: guidelines; Patient: similar-patient retrieval; "
        r"Both: both sources. Bold marks the highest displayed score per condition.\par}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main():
    results_path = ROOT / RESULTS
    summary = summarize_results(results_path)
    table = render_diagnosis_table(summary)
    all_condition_summary = summarize_conditions(results_path)
    condition_summary = select_best_variants(all_condition_summary)
    condition_table = render_condition_table(condition_summary)
    output_dir = results_path.parent / "analysis"
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "diagnosis_summary.csv"
    table_path = tables_dir / "diagnosis_by_model_variant.tex"
    condition_summary_path = output_dir / "diagnosis_by_condition_summary.csv"
    all_condition_summary_path = output_dir / "diagnosis_by_condition_all_variants.csv"
    condition_table_path = tables_dir / "diagnosis_by_condition.tex"
    summary.to_csv(summary_path, na_rep="")
    table_path.write_text(table, encoding="utf-8")
    condition_summary.to_csv(condition_summary_path, na_rep="")
    all_condition_summary.to_csv(all_condition_summary_path, na_rep="")
    condition_table_path.write_text(condition_table, encoding="utf-8")
    print(f"Saved diagnosis summary to {summary_path}")
    print(f"Saved LaTeX table to {table_path}")
    print(f"Saved condition summary to {condition_summary_path}")
    print(f"Saved all-variant condition/group summary to {all_condition_summary_path}")
    print(f"Saved condition table to {condition_table_path}")


if __name__ == "__main__":
    main()
