from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import binomtest


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "results/final/predictions_all.csv"
)

THRESHOLD_FILE = Path(
    "results/final/hybrid_reliability_threshold_locked.json"
)

OUTPUT_PREDICTIONS = Path(
    "results/final/hybrid_reliability_test_predictions.csv"
)

OUTPUT_BY_SEED = Path(
    "results/final/hybrid_reliability_test_by_seed.csv"
)

OUTPUT_SUMMARY = Path(
    "results/final/hybrid_reliability_test_summary.csv"
)

OUTPUT_MCNEMAR = Path(
    "results/final/hybrid_reliability_mcnemar.csv"
)

OUTPUT_MCNEMAR_SUMMARY = Path(
    "results/final/hybrid_reliability_mcnemar_summary.csv"
)


# ============================================================
# LOAD LOCKED THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)

TAU = float(
    config["selected_threshold"]
)

print(
    f"Locked threshold tau = {TAU:.2f}"
)


# ============================================================
# LOAD EXISTING FINAL TEST PREDICTIONS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

required = {
    "experiment",
    "condition",
    "model",
    "seed",
    "sample_id",
    "video_id",
    "true_class",
    "predicted_class",
    "correct",
    "confidence",
}

missing = required - set(
    df.columns
)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# SPLIT THREE EXISTING MODELS
# ============================================================

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
    [
        keys
        + [
            "predicted_class",
            "confidence",
        ]
    ]
    .rename(
        columns={
            "predicted_class":
                "audio_prediction",
            "confidence":
                "audio_confidence",
        }
    )
)

visual = (
    df[df["model"] == "visual"]
    [
        keys
        + [
            "predicted_class",
            "confidence",
        ]
    ]
    .rename(
        columns={
            "predicted_class":
                "visual_prediction",
            "confidence":
                "visual_confidence",
        }
    )
)

multimodal = (
    df[df["model"] == "multimodal"]
    [
        keys
        + [
            "predicted_class",
            "confidence",
            "correct",
        ]
    ]
    .rename(
        columns={
            "predicted_class":
                "baseline_prediction",
            "confidence":
                "baseline_confidence",
            "correct":
                "baseline_correct",
        }
    )
)


# ============================================================
# MERGE
# ============================================================

merged = (
    multimodal
    .merge(
        audio,
        on=keys,
        validate="one_to_one",
    )
    .merge(
        visual,
        on=keys,
        validate="one_to_one",
    )
)


# ============================================================
# B3: HYBRID RELIABILITY-AWARE FUSION
# ============================================================

# Default:
# preserve original multimodal prediction.
merged["hybrid_prediction"] = (
    merged["baseline_prediction"]
)

merged["selected_strategy"] = (
    "multimodal"
)

merged["confidence_gap"] = (
    merged["audio_confidence"]
    - merged["visual_confidence"]
).abs()


# ------------------------------------------------------------
# E7: audio explicitly missing
# ------------------------------------------------------------

mask_e7 = (
    merged["experiment"]
    == "E7"
)

merged.loc[
    mask_e7,
    "hybrid_prediction"
] = merged.loc[
    mask_e7,
    "visual_prediction"
]

merged.loc[
    mask_e7,
    "selected_strategy"
] = "visual_fallback"


# ------------------------------------------------------------
# E8: video explicitly missing
# ------------------------------------------------------------

mask_e8 = (
    merged["experiment"]
    == "E8"
)

merged.loc[
    mask_e8,
    "hybrid_prediction"
] = merged.loc[
    mask_e8,
    "audio_prediction"
]

merged.loc[
    mask_e8,
    "selected_strategy"
] = "audio_fallback"


# ------------------------------------------------------------
# E0-E6 + E9:
# reliability-aware intervention only if
# confidence difference >= locked tau
# ------------------------------------------------------------

both_available = ~(
    mask_e7
    | mask_e8
)

switch = (
    both_available
    & (
        merged["confidence_gap"]
        >= TAU
    )
)

audio_more_reliable = (
    merged["audio_confidence"]
    >= merged["visual_confidence"]
)

use_audio = (
    switch
    & audio_more_reliable
)

use_visual = (
    switch
    & ~audio_more_reliable
)


merged.loc[
    use_audio,
    "hybrid_prediction"
] = merged.loc[
    use_audio,
    "audio_prediction"
]

merged.loc[
    use_audio,
    "selected_strategy"
] = "audio_reliability_switch"


merged.loc[
    use_visual,
    "hybrid_prediction"
] = merged.loc[
    use_visual,
    "visual_prediction"
]

merged.loc[
    use_visual,
    "selected_strategy"
] = "visual_reliability_switch"


# ============================================================
# CORRECTNESS
# ============================================================

merged["hybrid_correct"] = (
    merged["hybrid_prediction"]
    == merged["true_class"]
).astype(int)


# ============================================================
# SAVE SAMPLE-LEVEL OUTPUT
# ============================================================

merged.to_csv(
    OUTPUT_PREDICTIONS,
    index=False,
)


# ============================================================
# METRICS PER CONDITION / SEED
# ============================================================

seed_rows = []

for (
    experiment,
    condition,
    seed
), g in merged.groupby(
    [
        "experiment",
        "condition",
        "seed",
    ],
    sort=True,
):

    baseline_accuracy = (
        g["baseline_correct"]
        .mean()
    )

    hybrid_accuracy = (
        g["hybrid_correct"]
        .mean()
    )

    baseline_f1 = f1_score(
        g["true_class"],
        g["baseline_prediction"],
        average="macro",
        zero_division=0,
    )

    hybrid_f1 = f1_score(
        g["true_class"],
        g["hybrid_prediction"],
        average="macro",
        zero_division=0,
    )

    intervention_rate = (
        g["selected_strategy"]
        .ne("multimodal")
        .mean()
    )

    audio_switch_rate = (
        g["selected_strategy"]
        .eq(
            "audio_reliability_switch"
        )
        .mean()
    )

    visual_switch_rate = (
        g["selected_strategy"]
        .eq(
            "visual_reliability_switch"
        )
        .mean()
    )

    seed_rows.append(
        {
            "experiment":
                experiment,

            "condition":
                condition,

            "seed":
                seed,

            "baseline_accuracy":
                baseline_accuracy,

            "hybrid_accuracy":
                hybrid_accuracy,

            "accuracy_change":
                hybrid_accuracy
                - baseline_accuracy,

            "baseline_macro_f1":
                baseline_f1,

            "hybrid_macro_f1":
                hybrid_f1,

            "macro_f1_change":
                hybrid_f1
                - baseline_f1,

            "intervention_rate":
                intervention_rate,

            "audio_switch_rate":
                audio_switch_rate,

            "visual_switch_rate":
                visual_switch_rate,
        }
    )


seed_df = pd.DataFrame(
    seed_rows
)

seed_df.to_csv(
    OUTPUT_BY_SEED,
    index=False,
)


# ============================================================
# SUMMARY ACROSS SEEDS
# ============================================================

summary = (
    seed_df
    .groupby(
        [
            "experiment",
            "condition",
        ],
        as_index=False,
    )
    .agg(
        baseline_accuracy=(
            "baseline_accuracy",
            "mean",
        ),

        hybrid_accuracy=(
            "hybrid_accuracy",
            "mean",
        ),

        hybrid_accuracy_std=(
            "hybrid_accuracy",
            "std",
        ),

        accuracy_change=(
            "accuracy_change",
            "mean",
        ),

        baseline_macro_f1=(
            "baseline_macro_f1",
            "mean",
        ),

        hybrid_macro_f1=(
            "hybrid_macro_f1",
            "mean",
        ),

        hybrid_macro_f1_std=(
            "hybrid_macro_f1",
            "std",
        ),

        macro_f1_change=(
            "macro_f1_change",
            "mean",
        ),

        intervention_rate=(
            "intervention_rate",
            "mean",
        ),

        audio_switch_rate=(
            "audio_switch_rate",
            "mean",
        ),

        visual_switch_rate=(
            "visual_switch_rate",
            "mean",
        ),
    )
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# EXACT McNEMAR
# ============================================================

mcnemar_rows = []

for (
    experiment,
    condition,
    seed
), g in merged.groupby(
    [
        "experiment",
        "condition",
        "seed",
    ],
    sort=True,
):

    # Baseline correct, hybrid wrong
    b = (
        (g["baseline_correct"] == 1)
        & (g["hybrid_correct"] == 0)
    ).sum()

    # Baseline wrong, hybrid correct
    c = (
        (g["baseline_correct"] == 0)
        & (g["hybrid_correct"] == 1)
    ).sum()

    discordant = int(
        b + c
    )

    if discordant == 0:

        p_raw = 1.0

    else:

        p_raw = binomtest(
            k=min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue


    mcnemar_rows.append(
        {
            "experiment":
                experiment,

            "condition":
                condition,

            "seed":
                seed,

            "accuracy_change":
                (
                    g["hybrid_correct"].mean()
                    - g["baseline_correct"].mean()
                ),

            "b_baseline_correct_hybrid_wrong":
                int(b),

            "c_baseline_wrong_hybrid_correct":
                int(c),

            "discordant_pairs":
                discordant,

            "p_raw":
                p_raw,
        }
    )


mcnemar = pd.DataFrame(
    mcnemar_rows
)


# ============================================================
# HOLM CORRECTION WITHIN EACH SEED
# ============================================================

def holm_adjust(pvalues):

    n = len(pvalues)

    order = sorted(
        range(n),
        key=lambda i: pvalues[i],
    )

    adjusted = [
        None
    ] * n

    running_max = 0.0

    for rank, idx in enumerate(
        order
    ):

        factor = (
            n - rank
        )

        adjusted_p = min(
            1.0,
            pvalues[idx]
            * factor,
        )

        running_max = max(
            running_max,
            adjusted_p,
        )

        adjusted[idx] = (
            running_max
        )

    return adjusted


parts = []

for seed, g in mcnemar.groupby(
    "seed"
):

    g = (
        g.copy()
        .reset_index(drop=True)
    )

    g["p_holm"] = holm_adjust(
        g["p_raw"].tolist()
    )

    g["significant_0_05"] = (
        g["p_holm"]
        < 0.05
    )

    parts.append(
        g
    )


mcnemar = pd.concat(
    parts,
    ignore_index=True,
).sort_values(
    [
        "experiment",
        "seed",
    ]
)


mcnemar.to_csv(
    OUTPUT_MCNEMAR,
    index=False,
)


mcnemar_summary = (
    mcnemar
    .groupby(
        [
            "experiment",
            "condition",
        ],
        as_index=False,
    )
    .agg(
        mean_accuracy_change=(
            "accuracy_change",
            "mean",
        ),

        significant_seeds=(
            "significant_0_05",
            "sum",
        ),

        total_seeds=(
            "seed",
            "count",
        ),
    )
)


mcnemar_summary.to_csv(
    OUTPUT_MCNEMAR_SUMMARY,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

display = summary.copy()

for col in [
    "baseline_accuracy",
    "hybrid_accuracy",
    "accuracy_change",
    "baseline_macro_f1",
    "hybrid_macro_f1",
    "macro_f1_change",
    "intervention_rate",
]:
    display[col] *= 100


print()
print("=" * 115)

print(
    "FINAL B3 HYBRID RELIABILITY-AWARE FUSION — E0-E9 TEST"
)

print("=" * 115)

print(
    f"Locked validation threshold: tau={TAU:.2f}"
)

print()

print(
    display[
        [
            "experiment",
            "condition",
            "baseline_accuracy",
            "hybrid_accuracy",
            "accuracy_change",
            "intervention_rate",
        ]
    ].to_string(
        index=False,
        formatters={

            "baseline_accuracy":
                lambda x: f"{x:6.2f}%",

            "hybrid_accuracy":
                lambda x: f"{x:6.2f}%",

            "accuracy_change":
                lambda x: f"{x:+6.2f} pp",

            "intervention_rate":
                lambda x: f"{x:6.2f}%",
        },
    )
)


print()
print(
    "MACRO-F1"
)

print("-" * 115)

print(
    display[
        [
            "experiment",
            "baseline_macro_f1",
            "hybrid_macro_f1",
            "macro_f1_change",
        ]
    ].to_string(
        index=False,
        formatters={

            "baseline_macro_f1":
                lambda x: f"{x:6.2f}%",

            "hybrid_macro_f1":
                lambda x: f"{x:6.2f}%",

            "macro_f1_change":
                lambda x: f"{x:+6.2f} pp",
        },
    )
)


sig_display = (
    mcnemar_summary.copy()
)

sig_display[
    "mean_accuracy_change"
] *= 100


print()
print(
    "PAIRED SIGNIFICANCE — EXACT McNEMAR + HOLM"
)

print("-" * 115)

print(
    sig_display.to_string(
        index=False,
        formatters={
            "mean_accuracy_change":
                lambda x: f"{x:+.2f} pp"
        },
    )
)


print()
print("Saved:")
print(
    " ",
    OUTPUT_PREDICTIONS
)
print(
    " ",
    OUTPUT_BY_SEED
)
print(
    " ",
    OUTPUT_SUMMARY
)
print(
    " ",
    OUTPUT_MCNEMAR
)
print(
    " ",
    OUTPUT_MCNEMAR_SUMMARY
)


print()
print(
    "IMPORTANT: tau was locked from validation "
    "before this E0-E9 test evaluation."
)

print("=" * 115)
