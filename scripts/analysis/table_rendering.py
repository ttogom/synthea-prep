"""LaTeX layouts for main results and retrieval gains by generator and group.

Renderers consume saved statistics CSV schemas; they never rescore predictions
or recompute significance. Formatting and approved captions live here.
"""

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "gpt_4o_mini": "GPT-4o mini", "ollama_gemma3_4b": "Gemma 3 4B",
    "ollama_llama3_2_3b": "Llama 3.2 3B", "ollama_ministral_3_3b": "Ministral 3 3B",
}


METRICS = ("diagnosis", "plan")


GROUPS = ("history", "acute", "control")


ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex(value):
    return "".join(ESCAPES.get(character, character) for character in str(value))


def _truth(value):
    if isinstance(value, str):
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
        raise ValueError(f"Invalid coverage_complete value: {value!r}")
    return bool(value) if pd.notna(value) else False


def _models(summary):
    present = set(summary["model"])
    return [model for model in MODEL_LABELS if model in present] + sorted(present - set(MODEL_LABELS))


def render_main_results_table(scores):
    """Render main-results means, mean CIs, and baseline-improvement significance stars."""
    columns = {"model", "variant", "metric", "mean_score", "coverage_complete",
               "mean_ci_low", "mean_ci_high", "baseline_delta", "p_value_holm"}
    if columns - set(scores):
        raise ValueError(f"Missing score columns: {sorted(columns - set(scores))}")
    data = scores.copy()
    if data.empty or data.duplicated(["model", "variant", "metric"]).any():
        raise ValueError("Overall score identities must be present and unique.")
    data["coverage_complete"] = data["coverage_complete"].map(_truth)
    numeric = ["mean_score", "mean_ci_low", "mean_ci_high", "baseline_delta", "p_value_holm"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    valid = data.loc[data["coverage_complete"], "mean_score"]
    if not np.isfinite(valid).all() or not valid.between(0, 1).all():
        raise ValueError("Complete means must be finite scores in [0, 1].")
    bounds = data[["mean_ci_low", "mean_ci_high"]]
    if (bounds.isna().any(axis=1) != bounds.isna().all(axis=1)).any():
        raise ValueError("Mean CI bounds must both be available or both missing.")
    observed_bounds = bounds.dropna()
    if (not np.isfinite(observed_bounds).all().all()
            or not observed_bounds.ge(0).all().all() or not observed_bounds.le(1).all().all()
            or (observed_bounds["mean_ci_low"] > observed_bounds["mean_ci_high"]).any()):
        raise ValueError("Require ordered mean CI bounds in [0, 1].")
    if not data["p_value_holm"].dropna().between(0, 1).all():
        raise ValueError("Adjusted p-values must lie in [0, 1].")
    variants = (
        ("llm_only", "No retrieval"),
        ("patient_retrieval", "Sim. patient retrieval"),
        ("guidelines", "Guideline retrieval"),
        ("patient_retrieval_guidelines", "Both"),
    )
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllclc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Generator Model} & \multirow{2}{*}{Configuration} & \multicolumn{2}{c}{Diagnoses} & \multicolumn{2}{c}{Treatment Plans} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Mean & 95\% CI & Mean & 95\% CI \\", r"\midrule",
    ]
    for model_index, model in enumerate(_models(data)):
        if model_index:
            lines.append(r"\midrule")
        best = {
            metric: data.loc[data["model"].eq(model) & data["metric"].eq(metric)
                             & data["coverage_complete"], "mean_score"].max()
            for metric in METRICS
        }
        for index, (variant, label) in enumerate(variants):
            cells = [_tex(MODEL_LABELS.get(model, model)) if index == 0 else "", label]
            for metric in METRICS:
                row = data.loc[data["model"].eq(model) & data["variant"].eq(variant)
                               & data["metric"].eq(metric)]
                if len(row) != 1:
                    raise ValueError(f"Missing overall mean for {model}/{variant}/{metric}.")
                row = row.iloc[0]
                if not row["coverage_complete"]:
                    cells += ["--", "--"]
                    continue
                stars = ""
                if variant != "llm_only" and row["baseline_delta"] > 0:
                    pvalue = row["p_value_holm"]
                    stars = "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
                suffix = "^{" + stars + "}" if stars else ""
                mean = f"{row['mean_score']:.4f}"
                if np.isclose(row["mean_score"], best[metric], rtol=0, atol=1e-12):
                    mean = rf"\mathbf{{{mean}}}"
                cells.append(f"${mean}{suffix}$")
                cells.append(r"{\scriptsize " + f"$[{row['mean_ci_low']:.2f}, {row['mean_ci_high']:.2f}]$" + "}"
                             if pd.notna(row["mean_ci_low"]) else "--")
            lines.append(" & ".join(cells) + (r" \\[2pt]" if index == 0 else r" \\"))
    lines += [
        r"\bottomrule", r"\end{tabular*}",
        r"\captionsetup{justification=raggedright,singlelinecheck=false}",
        '\\caption{Mean diagnosis and treatment-plan scores by generator and retrieval configuration. Both scores range from 0 to 1; higher scores are better. $^{*}p<0.05$, $^{**}p<0.01$: significant improvement over "No retrieval" configuration within the same generator and outcome.}',
        r"\label{tab:main-results}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_generator_retrieval_gains_table(summary):
    """Render two single-column panels with equal-generator average tests.

    Accept the saved baseline-significance CSV schema. An omitted outcome or
    incomplete eligible-score coverage is displayed as --, never as zero.
    Round displayed gains only; significance uses the unrounded adjusted p-value.
    """
    required = {"model", "metric", "variant", "baseline_variant", "mean_difference",
                "coverage_complete", "p_value_holm", "family_size", "n_generators_averaged"}
    if required - set(summary):
        raise ValueError(f"Missing gain columns: {sorted(required - set(summary))}")
    data = summary.copy()
    keys = ["model", "metric", "variant"]
    if data.empty or data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError("Gain identities must be populated and unique.")
    if not set(data.metric).issubset(METRICS):
        raise ValueError("Unknown gain outcome.")
    variants = (("patient_retrieval", "Similar-patient retrieval"),
                ("guidelines", "Guideline retrieval"))
    if not set(data.variant).issubset(dict(variants)) or not data.baseline_variant.eq("llm_only").all():
        raise ValueError("Gains must compare the two retrieval sources with No retrieval.")
    models = [model for model in _models(data) if model != "model_average"]
    if not models:
        raise ValueError("At least one generator is required.")
    models.append("model_average")
    for metric in data.metric.unique():
        actual = set(map(tuple, data.loc[data.metric.eq(metric), ["model", "variant"]].to_numpy()))
        expected = {(model, variant) for model in models for variant, _ in variants}
        if actual != expected:
            raise ValueError("Each outcome requires both sources for every generator and their average.")
    data["coverage_complete"] = data.coverage_complete.map(_truth)
    numeric = ["mean_difference", "p_value_holm", "family_size", "n_generators_averaged"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    valid = data.loc[data.coverage_complete, "mean_difference"]
    if not np.isfinite(valid).all() or not valid.between(-1, 1).all():
        raise ValueError("Complete gains must be finite differences in [-1, 1].")
    if not data.p_value_holm.dropna().between(0, 1).all():
        raise ValueError("Adjusted gain p-values must lie in [0, 1].")
    family_size = 2 * len(models)
    if not data.family_size.eq(family_size).all():
        raise ValueError("Holm family must include both sources, all generators, and their average.")
    counts = np.where(data.model.eq("model_average"), len(models) - 1, 1)
    if not data.n_generators_averaged.eq(counts).all():
        raise ValueError("Average rows must include every generator with equal weights.")
    lines = [
        "% Generated from baseline-significance CSVs; regenerate rather than editing numerical cells.",
        "% Use table* instead of table to span both columns; panel widths follow \\linewidth.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{2pt}",
    ]
    for panel, (variant, label) in zip(("a", "b"), variants):
        if panel == "b":
            lines.append(r"\par\vspace{6pt}")
        lines += [
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lll@{}}", r"\toprule",
            rf"\multicolumn{{3}}{{@{{}}l}}{{\textbf{{({panel}) {label}}}}} \\[2pt]",
            r" & \multicolumn{2}{c@{}}{$\Delta$ vs. No retrieval} \\",
            r"\cmidrule(l){2-3}",
            r"Generator & Diagnosis & Treatment plan \\",
            r"\midrule",
        ]
        for model in models:
            average = model == "model_average"
            if average:
                lines.append(r"\addlinespace[3pt]")
            name = "Average across generators" if average else MODEL_LABELS.get(model, model)
            cells = [_tex(name)]
            for metric in METRICS:
                rows = data.loc[data.model.eq(model) & data.variant.eq(variant) & data.metric.eq(metric)]
                if rows.empty or not rows.iloc[0].coverage_complete:
                    cells.append("--")
                    continue
                row = rows.iloc[0]
                gain, pvalue = row.mean_difference, row.p_value_holm
                stars = ("**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "") if gain > 0 else ""
                number = f"{0.0 if abs(gain) < 0.005 else gain:+.2f}"
                suffix = "^{" + stars + "}" if stars else ""
                cells.append(f"${number}{suffix}$")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular*}"]
    caption = 'Mean changes in diagnosis and treatment-plan scores relative to "No retrieval." $^{*}p<0.05$, $^{**}p<0.01$: significant improvement in one-sided paired tests.'
    lines += [r"\captionsetup{justification=raggedright,singlelinecheck=false}",
              rf"\caption{{{caption}}}", r"\label{tab:retrieval-gains}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_group_retrieval_gains_table(scores):
    """Render group raw baseline scores and starred retrieval-minus-baseline deltas."""
    variants = ("patient_retrieval", "guidelines", "patient_retrieval_guidelines")
    required = {"model", "metric", "group", "variant", "mean_baseline", "mean_retrieval",
                "mean_difference", "p_value_holm", "coverage_complete"}
    if required - set(scores):
        raise ValueError(f"Missing group score columns: {sorted(required - set(scores))}")
    data = scores.copy()
    if data.empty or data.duplicated(["model", "metric", "group", "variant"]).any():
        raise ValueError("group identities must be present and unique.")
    if (not set(data["metric"]).issubset(METRICS) or not set(data["group"]).issubset(GROUPS)
            or not set(data["variant"]).issubset(variants)):
        raise ValueError("Unknown group metric, group, or configuration.")
    data["coverage_complete"] = data["coverage_complete"].map(_truth)
    columns = ["mean_baseline", "mean_retrieval", "mean_difference", "p_value_holm"]
    data[columns] = data[columns].apply(pd.to_numeric, errors="raise")
    for column in ("mean_baseline", "mean_retrieval", "p_value_holm"):
        if not data[column].dropna().between(0, 1).all():
            raise ValueError("group raw means and p-values must lie in [0, 1].")
    complete = data.loc[data["coverage_complete"]]
    if not np.isfinite(complete[["mean_baseline", "mean_retrieval", "mean_difference"]]).all().all():
        raise ValueError("Complete group comparisons require finite means and deltas.")
    if not np.allclose(complete["mean_retrieval"] - complete["mean_baseline"],
                       complete["mean_difference"], rtol=0, atol=1e-12):
        raise ValueError("group deltas must equal retrieval minus baseline means.")
    retrieval_headers = (r"{\footnotesize\shortstack[l]{Sim. patient\\retrieval}} & "
                         r"{\footnotesize\shortstack[l]{Guideline\\retrieval}} & {\footnotesize Both}")
    baseline_header = r"{\footnotesize\shortstack[l]{No retrieval\\(score)}}"
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l@{\hspace{2pt}}l@{\hspace{6pt}}l@{\hspace{8pt}}llll@{\hspace{8pt}}lll@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{Diagnoses} & \multicolumn{4}{c}{Treatment Plans} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r" & & & \multicolumn{3}{c}{\footnotesize $\Delta$ vs. No retrieval} & & \multicolumn{3}{c}{\footnotesize $\Delta$ vs. No retrieval} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){8-10}",
        r"\shortstack[l]{Generator\\Model} & Group & " + baseline_header + " & " + retrieval_headers + " & "
        + baseline_header + " & " + retrieval_headers + r" \\", r"\midrule",
    ]
    for model_index, model in enumerate(_models(data)):
        if model_index:
            lines.append(r"\midrule")
        for group_index, group in enumerate(GROUPS):
            cells = [_tex(MODEL_LABELS.get(model, model)) if group_index == 0 else "", group.title()]
            for metric in METRICS:
                selected = data.loc[data["model"].eq(model) & data["metric"].eq(metric) & data["group"].eq(group)]
                if len(selected) != 3 or set(selected["variant"]) != set(variants):
                    raise ValueError(f"Missing group configurations for {model}/{metric}/{group}.")
                selected = selected.set_index("variant").loc[list(variants)]
                baseline = selected["mean_baseline"]
                if baseline.nunique(dropna=False) != 1:
                    raise ValueError("group configurations must share the same baseline mean.")
                means = [baseline.iloc[0], *selected["mean_retrieval"].tolist()]
                best = max((value for value in means if pd.notna(value)), default=np.nan)
                for index, raw_mean in enumerate(means):
                    row = selected.iloc[index - 1] if index else None
                    value = (row["mean_difference"] if row["coverage_complete"] else np.nan) if index else raw_mean
                    if pd.isna(value):
                        cells.append(r"{\footnotesize --}")
                        continue
                    mean = (f"{0.0 if abs(value) < 0.005 else value:+.2f}" if index
                            else f"{value:.2f}")
                    if np.isclose(raw_mean, best, rtol=0, atol=1e-12):
                        mean = rf"\mathbf{{{mean}}}"
                    stars = ""
                    if index:
                        if row["coverage_complete"] and row["mean_difference"] > 0:
                            pvalue = row["p_value_holm"]
                            stars = "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
                    suffix = "^{" + stars + "}" if stars else ""
                    cells.append(r"{\footnotesize " + f"${mean}{suffix}$" + "}")
            lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular*}",
        r"\captionsetup{justification=raggedright,singlelinecheck=false}",
        '\\caption{Diagnosis and treatment-plan results by condition group. $^{*}p<0.05$, $^{**}p<0.01$: significant improvement over "No retrieval" in one-sided paired tests. Bold marks the best configuration within each group; raw scores range from 0 to 1.}',
        r"\label{tab:group-retrieval-gains}", r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"
