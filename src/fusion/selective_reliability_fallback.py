from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


INPUT = Path("results/final/predictions_all.csv")

OUTPUT_PRED = Path(
    "results/final/predictions_selective_fallback.csv"
)

OUTPUT_SUMMARY = Path(
    "results/final/selective_fallback_results.csv"
)

df = pd.read_csv(INPUT)


# ------------------------------------------------------------
# Separate models
# ------------------------------------------------------------

keys = [
    "experiment",
    "condition",
    "seed",
    "sample_id",
    "video_id",
    "true_class",
]

audio = (
    df[df["model"] == "audio"]
    [keys + ["predicted_class", "confidence"]]
    .rename(
        columns={
            "predicted_class": "audio_prediction",
            "confidence": "audio_confidence",
        }
    )
)

visual = (
    df[df["model"] == "visual"]
    [keys + ["predicted_class", "confidence"]]
    .rename(
        columns={
            "predicted_class": "visual_prediction",
            "confidence": "visual_confidence",
        }
    )
)

multimodal = (
    df[df["model"] == "multimodal"]
    [keys + ["predicted_class", "confidence"]]
    .rename(
        columns={
            "predicted_class": "baseline_prediction",
            "confidence": "baseline_confidence",
        }
    )
)


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

merged = multimodal.merge(
    audio,
    on=keys,
    validate="one_to_one",
).merge(
    visual,
    on=keys,
    validate="one_to_one",
)


# ------------------------------------------------------------
# Selective reliability-aware fallback
# ------------------------------------------------------------

# Default:
# Use original multimodal prediction.
merged["predicted_class"] = merged[
    "baseline_prediction"
]

merged["selected_strategy"] = "multimodal"


# E7:
# Audio is explicitly unavailable.
# Fall back to visual-only prediction.
mask_e7 = merged["experiment"] == "E7"

merged.loc[
    mask_e7,
    "predicted_class"
] = merged.loc[
    mask_e7,
    "visual_prediction"
]

merged.loc[
    mask_e7,
    "selected_strategy"
] = "visual_fallback"


# E8:
# Video is explicitly unavailable.
# Fall back to audio-only prediction.
mask_e8 = merged["experiment"] == "E8"

merged.loc[
    mask_e8,
    "predicted_class"
] = merged.loc[
    mask_e8,
    "audio_prediction"
]

merged.loc[
    mask_e8,
    "selected_strategy"
] = "audio_fallback"


# ------------------------------------------------------------
# Correctness
# ------------------------------------------------------------

merged["correct"] = (
    merged["predicted_class"]
    == merged["true_class"]
).astype(int)


OUTPUT_PRED.parent.mkdir(
    parents=True,
    exist_ok=True,
)

merged.to_csv(
    OUTPUT_PRED,
    index=False,
)


# ------------------------------------------------------------
# Metrics per seed
# ------------------------------------------------------------

rows = []

for (experiment, condition, seed), g in merged.groupby(
    ["experiment", "condition", "seed"],
    sort=True,
):

    acc = accuracy_score(
        g["true_class"],
        g["predicted_class"],
    )

    f1 = f1_score(
        g["true_class"],
        g["predicted_class"],
        average="macro",
        zero_division=0,
    )

    baseline_acc = (
        g["baseline_prediction"]
        == g["true_class"]
    ).mean()

    baseline_f1 = f1_score(
        g["true_class"],
        g["baseline_prediction"],
        average="macro",
        zero_division=0,
    )

    rows.append(
        {
            "experiment": experiment,
            "condition": condition,
            "seed": seed,
            "baseline_accuracy": baseline_acc,
            "fallback_accuracy": acc,
            "accuracy_change": acc - baseline_acc,
            "baseline_macro_f1": baseline_f1,
            "fallback_macro_f1": f1,
            "macro_f1_change": f1 - baseline_f1,
        }
    )

seed_df = pd.DataFrame(rows)


# ------------------------------------------------------------
# Mean ± std
# ------------------------------------------------------------

summary = (
    seed_df
    .groupby(
        ["experiment", "condition"],
        as_index=False,
    )
    .agg(
        baseline_accuracy=(
            "baseline_accuracy",
            "mean",
        ),
        fallback_accuracy=(
            "fallback_accuracy",
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
        fallback_macro_f1=(
            "fallback_macro_f1",
            "mean",
        ),
        macro_f1_change=(
            "macro_f1_change",
            "mean",
        ),
    )
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

display = summary.copy()

for col in [
    "baseline_accuracy",
    "fallback_accuracy",
    "accuracy_change",
    "baseline_macro_f1",
    "fallback_macro_f1",
    "macro_f1_change",
]:
    display[col] *= 100


print("\n" + "=" * 100)
print("SELECTIVE RELIABILITY-AWARE FALLBACK")
print("=" * 100)

print(
    display[
        [
            "experiment",
            "condition",
            "baseline_accuracy",
            "fallback_accuracy",
            "accuracy_change",
        ]
    ].to_string(
        index=False,
        formatters={
            "baseline_accuracy":
                lambda x: f"{x:6.2f}%",
            "fallback_accuracy":
                lambda x: f"{x:6.2f}%",
            "accuracy_change":
                lambda x: f"{x:+6.2f} pp",
        }
    )
)

print("\nSaved:")
print(f"  {OUTPUT_PRED}")
print(f"  {OUTPUT_SUMMARY}")

print("\nMETHOD:")
print(
    "Use the original multimodal classifier when both "
    "modalities are available. When audio is missing, "
    "use the visual-only branch. When video is missing, "
    "use the audio-only branch."
)

print("=" * 100)
