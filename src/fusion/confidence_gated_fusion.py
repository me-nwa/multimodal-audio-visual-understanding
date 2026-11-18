from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("results/final/predictions_all.csv")

OUTPUT_PREDICTIONS = Path(
    "results/final/predictions_reliability_fusion.csv"
)

OUTPUT_SUMMARY = Path(
    "results/final/reliability_fusion_results.csv"
)

OUTPUT_COMPARISON = Path(
    "results/final/reliability_fusion_vs_baseline.csv"
)


# ============================================================
# Load prediction data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = {
    "experiment",
    "condition",
    "model",
    "seed",
    "sample_id",
    "video_id",
    "true_class",
    "predicted_class",
    "confidence",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# Keep audio and visual predictions
# ============================================================

audio = df[df["model"] == "audio"].copy()
visual = df[df["model"] == "visual"].copy()

key_cols = [
    "experiment",
    "condition",
    "seed",
    "sample_id",
    "video_id",
    "true_class",
]

audio = audio[
    key_cols + ["predicted_class", "confidence"]
].rename(
    columns={
        "predicted_class": "audio_prediction",
        "confidence": "audio_confidence",
    }
)

visual = visual[
    key_cols + ["predicted_class", "confidence"]
].rename(
    columns={
        "predicted_class": "visual_prediction",
        "confidence": "visual_confidence",
    }
)


# ============================================================
# Pair audio and visual predictions
# ============================================================

paired = audio.merge(
    visual,
    on=key_cols,
    how="inner",
    validate="one_to_one",
)

expected_rows = (
    df[df["model"] == "audio"].shape[0]
)

if len(paired) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} paired rows, "
        f"but found {len(paired)}"
    )


# ============================================================
# Reliability estimation
# ============================================================

paired["audio_reliability"] = paired["audio_confidence"]
paired["visual_reliability"] = paired["visual_confidence"]


# Known missing-modality cases:
#
# E7 = audio unavailable
# E8 = video unavailable
#
# Because availability is directly observable, reliability of
# the unavailable stream is explicitly set to zero.

paired.loc[
    paired["experiment"] == "E7",
    "audio_reliability"
] = 0.0

paired.loc[
    paired["experiment"] == "E8",
    "visual_reliability"
] = 0.0


# ============================================================
# Confidence-gated reliability-aware fusion
# ============================================================

use_audio = (
    paired["audio_reliability"]
    >= paired["visual_reliability"]
)

paired["selected_modality"] = "visual"

paired.loc[
    use_audio,
    "selected_modality"
] = "audio"

paired["predicted_class"] = paired["visual_prediction"]

paired.loc[
    use_audio,
    "predicted_class"
] = paired.loc[
    use_audio,
    "audio_prediction"
]

paired["correct"] = (
    paired["predicted_class"]
    == paired["true_class"]
).astype(int)

paired["selected_reliability"] = paired[
    ["audio_reliability", "visual_reliability"]
].max(axis=1)


# ============================================================
# Save sample-level predictions
# ============================================================

OUTPUT_PREDICTIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

paired.to_csv(
    OUTPUT_PREDICTIONS,
    index=False,
)


# ============================================================
# Compute metrics for each experiment and seed
# ============================================================

seed_results = []

for (experiment, condition, seed), group in paired.groupby(
    ["experiment", "condition", "seed"],
    sort=True,
):

    y_true = group["true_class"]
    y_pred = group["predicted_class"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    audio_selection_rate = (
        group["selected_modality"]
        .eq("audio")
        .mean()
    )

    visual_selection_rate = (
        group["selected_modality"]
        .eq("visual")
        .mean()
    )

    seed_results.append(
        {
            "experiment": experiment,
            "condition": condition,
            "seed": seed,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "audio_selection_rate": audio_selection_rate,
            "visual_selection_rate": visual_selection_rate,
        }
    )

seed_df = pd.DataFrame(seed_results)


# ============================================================
# Mean ± standard deviation across seeds
# ============================================================

summary = (
    seed_df
    .groupby(
        ["experiment", "condition"],
        as_index=False,
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        audio_selection_rate=(
            "audio_selection_rate",
            "mean",
        ),
        visual_selection_rate=(
            "visual_selection_rate",
            "mean",
        ),
    )
)

summary.insert(
    2,
    "model",
    "reliability_fusion",
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# Compare against existing multimodal baseline
# ============================================================

baseline = (
    df[df["model"] == "multimodal"]
    .groupby(
        ["experiment", "condition", "seed"],
        as_index=False,
    )
    .agg(
        baseline_accuracy=("correct", "mean")
    )
)

# Macro-F1 must be recomputed from actual predictions
baseline_f1_rows = []

for (experiment, condition, seed), group in (
    df[df["model"] == "multimodal"]
    .groupby(
        ["experiment", "condition", "seed"],
        sort=True,
    )
):

    baseline_f1_rows.append(
        {
            "experiment": experiment,
            "condition": condition,
            "seed": seed,
            "baseline_macro_f1": f1_score(
                group["true_class"],
                group["predicted_class"],
                average="macro",
                zero_division=0,
            ),
        }
    )

baseline_f1 = pd.DataFrame(
    baseline_f1_rows
)

baseline = baseline.merge(
    baseline_f1,
    on=[
        "experiment",
        "condition",
        "seed",
    ],
    validate="one_to_one",
)

comparison_seed = seed_df.merge(
    baseline,
    on=[
        "experiment",
        "condition",
        "seed",
    ],
    validate="one_to_one",
)

comparison_seed[
    "accuracy_change"
] = (
    comparison_seed["accuracy"]
    - comparison_seed["baseline_accuracy"]
)

comparison_seed[
    "macro_f1_change"
] = (
    comparison_seed["macro_f1"]
    - comparison_seed["baseline_macro_f1"]
)

comparison = (
    comparison_seed
    .groupby(
        ["experiment", "condition"],
        as_index=False,
    )
    .agg(
        baseline_accuracy=(
            "baseline_accuracy",
            "mean",
        ),
        reliability_accuracy=(
            "accuracy",
            "mean",
        ),
        accuracy_change=(
            "accuracy_change",
            "mean",
        ),
        baseline_macro_f1=(
            "baseline_macro_f1",
            "mean",
        ),
        reliability_macro_f1=(
            "macro_f1",
            "mean",
        ),
        macro_f1_change=(
            "macro_f1_change",
            "mean",
        ),
        audio_selection_rate=(
            "audio_selection_rate",
            "mean",
        ),
        visual_selection_rate=(
            "visual_selection_rate",
            "mean",
        ),
    )
)

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False,
)


# ============================================================
# Display key results
# ============================================================

display = comparison.copy()

percentage_cols = [
    "baseline_accuracy",
    "reliability_accuracy",
    "accuracy_change",
    "baseline_macro_f1",
    "reliability_macro_f1",
    "macro_f1_change",
    "audio_selection_rate",
    "visual_selection_rate",
]

for col in percentage_cols:
    display[col] = display[col] * 100


print("\n" + "=" * 100)
print(
    "CONFIDENCE-GATED RELIABILITY-AWARE FUSION"
)
print("=" * 100)

print(
    display[
        [
            "experiment",
            "condition",
            "baseline_accuracy",
            "reliability_accuracy",
            "accuracy_change",
            "audio_selection_rate",
            "visual_selection_rate",
        ]
    ].to_string(
        index=False,
        formatters={
            "baseline_accuracy":
                lambda x: f"{x:6.2f}%",
            "reliability_accuracy":
                lambda x: f"{x:6.2f}%",
            "accuracy_change":
                lambda x: f"{x:+6.2f} pp",
            "audio_selection_rate":
                lambda x: f"{x:6.2f}%",
            "visual_selection_rate":
                lambda x: f"{x:6.2f}%",
        }
    )
)

print("\nIMPORTANT CONDITIONS")
print("-" * 100)

for exp in [
    "E0",
    "E3",
    "E6",
    "E7",
    "E8",
    "E9",
]:

    row = display[
        display["experiment"] == exp
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(
        f"{exp}: "
        f"B0={row['baseline_accuracy']:.2f}% | "
        f"B1={row['reliability_accuracy']:.2f}% | "
        f"change={row['accuracy_change']:+.2f} pp | "
        f"audio selected={row['audio_selection_rate']:.2f}% | "
        f"visual selected={row['visual_selection_rate']:.2f}%"
    )


print("\nSaved:")
print(f"  {OUTPUT_PREDICTIONS}")
print(f"  {OUTPUT_SUMMARY}")
print(f"  {OUTPUT_COMPARISON}")

print("\nMETHOD NOTE:")
print(
    "B1 uses maximum unimodal softmax confidence as a "
    "lightweight reliability estimate. E7 and E8 use "
    "explicit modality availability because the missing "
    "stream is directly observable."
)

print("=" * 100)
