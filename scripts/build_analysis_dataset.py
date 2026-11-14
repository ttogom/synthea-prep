import json
from pathlib import Path
import pandas as pd

RESULTS = "experiments/output/2026-11-12_022701PST__test_so__main/results.jsonl"
LABELS = "experiments/input/test_labels.jsonl"
ROOT = Path(__file__).resolve().parents[1]

results_df = pd.read_json(ROOT / RESULTS, lines=True).set_index("id")
labels_df = pd.read_json(ROOT / LABELS, lines=True).set_index("id")
labels_df = labels_df[["bin", "condition", "plan"]]
labels_df = labels_df.rename(columns={"condition": "condition_label", "plan": "plan_label"})
labels_df["plan_label"] = labels_df["plan_label"].map(lambda actions: "; ".join(actions or []))

if not results_df.index.is_unique or not labels_df.index.is_unique:
    raise ValueError("Results and labels must each have unique patient IDs.")

labels_df = labels_df.loc[results_df.index]

rows = []
for patient_id, variants in results_df["variants"].items():
    for variant_name, variant in variants.items():
        for model_name, model in variant["assessment_plans"].items():
            row = {"id": patient_id, "model": model_name, "variant": variant_name}
            predictions = model.get("predictions") or []
            evaluations = {entry["rank"]: entry for entry in model.get("evaluation", [])}
            for rank in range(1, 4):
                prediction = predictions[rank - 1] if len(predictions) >= rank else {}
                evaluation = evaluations.get(rank, {})
                biolord = evaluation.get("diagnosis_correctness", {})
                bert = evaluation.get("condition", {})
                plan = evaluation.get("plan_correctness", {})
                row[f"diagnosis{rank}"] = prediction.get("condition") or ""
                biolord_score = biolord.get("score") if biolord.get("status") == "scored" else None
                row[f"biolord_score_diagnosis{rank}"] = biolord_score
                bert_f1 = bert.get("metrics", {}).get("BERTScore_F1") if bert.get("status") == "scored" else None
                row[f"bertf1_diagnosis{rank}"] = bert_f1
                row[f"plan{rank}"] = "; ".join(prediction.get("plan") or [])
                row[f"plan{rank}_score"] = plan.get("score") if plan.get("status") == "scored" else None
                credits = plan.get("credits")
                row[f"plan{rank}_reference_item_credits"] = json.dumps(credits) if credits is not None else None
                row[f"plan{rank}_unsupported_extra_count"] = plan.get("unsupported_extra_count")
                row[f"plan{rank}_status"] = plan.get("status")
                row[f"plan{rank}_reason"] = plan.get("reason")
            rows.append(row)

patient_df = pd.DataFrame(rows).set_index(["id", "model", "variant"])
prediction_columns = list(patient_df.columns)
patient_df = patient_df.join(labels_df, on="id", validate="many_to_one")
patient_df = patient_df[list(labels_df.columns) + prediction_columns]
for rank in range(1, 4):
    column = f"plan{rank}_unsupported_extra_count"
    patient_df[column] = patient_df[column].astype("Int64")
grouped = patient_df.groupby(["model", "variant", "bin", "condition_label"], sort=False)
summary_df = grouped.size().to_frame("n")
summary_df["mean_biolord_score"] = grouped["biolord_score_diagnosis1"].mean()
summary_df["mean_bert_f1"] = grouped["bertf1_diagnosis1"].mean()

if __name__ == "__main__":
    csv_path = (ROOT / RESULTS).parent / "experiment_results.csv"
    patient_df.to_csv(csv_path, na_rep="")
    print(f"Saved patient-level results to {csv_path}")
    print("Rank-1 diagnosis scores; n counts result patients; means exclude unavailable scores.")
    print(summary_df.to_string(float_format="{:.4f}".format))
