from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support


# ==================================================
# PATHS
# ==================================================

PREDICTIONS = Path(
    "results/final/predictions_all.csv"
)

OUTPUT = Path(
    "results/final/statistics"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    PREDICTIONS
)

SEEDS = [42, 43, 44]

CLASSES = sorted(
    df["true_class"].unique()
)

print("Prediction rows:", len(df))
print("Classes:", len(CLASSES))


# ==================================================
# HOLM CORRECTION
# ==================================================

def holm_adjust(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    m = len(pvalues)

    order = np.argsort(
        pvalues
    )

    adjusted = np.empty(
        m,
        dtype=float
    )

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (
            (m - rank)
            * pvalues[index]
        )

        running_max = max(
            running_max,
            value
        )

        adjusted[index] = min(
            running_max,
            1.0
        )

    return adjusted


# ==================================================
# EXACT McNEMAR TEST
# ==================================================

def mcnemar_exact(
    left,
    right
):

    merged = left[
        [
            "sample_id",
            "correct"
        ]
    ].merge(

        right[
            [
                "sample_id",
                "correct"
            ]
        ],

        on="sample_id",

        suffixes=(
            "_left",
            "_right"
        )
    )


    # Left correct, right wrong
    b = int(
        (
            (merged["correct_left"] == 1)
            &
            (merged["correct_right"] == 0)
        ).sum()
    )


    # Left wrong, right correct
    c = int(
        (
            (merged["correct_left"] == 0)
            &
            (merged["correct_right"] == 1)
        ).sum()
    )


    discordant = b + c


    if discordant == 0:

        pvalue = 1.0

    else:

        pvalue = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue


    left_accuracy = (
        merged["correct_left"]
        .mean()
    )

    right_accuracy = (
        merged["correct_right"]
        .mean()
    )


    return {
        "n": len(merged),
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "difference": (
            left_accuracy
            -
            right_accuracy
        ),
        "left_correct_right_wrong": b,
        "left_wrong_right_correct": c,
        "discordant": discordant,
        "p_value": pvalue,
    }


# ==================================================
# PART 1
# CORRUPTED CONDITION VS CLEAN
# ==================================================

condition_tests = []


# Relevant conditions for each model
CONDITION_MAP = {

    "audio": [
        "E1",
        "E2",
        "E3",
        "E7",
        "E9",
    ],

    "visual": [
        "E4",
        "E5",
        "E6",
        "E8",
        "E9",
    ],

    "multimodal": [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
        "E9",
    ],
}


for model, experiments in (
    CONDITION_MAP.items()
):

    for experiment in experiments:

        for seed in SEEDS:

            clean = df[
                (df["experiment"] == "E0")
                &
                (df["model"] == model)
                &
                (df["seed"] == seed)
            ]


            corrupted = df[
                (df["experiment"] == experiment)
                &
                (df["model"] == model)
                &
                (df["seed"] == seed)
            ]


            result = mcnemar_exact(
                corrupted,
                clean
            )


            condition_name = (
                corrupted[
                    "condition"
                ].iloc[0]
            )


            condition_tests.append({

                "model":
                    model,

                "experiment":
                    experiment,

                "condition":
                    condition_name,

                "seed":
                    seed,

                "corrupted_accuracy":
                    result[
                        "left_accuracy"
                    ],

                "clean_accuracy":
                    result[
                        "right_accuracy"
                    ],

                "accuracy_change":
                    result[
                        "difference"
                    ],

                "corrupted_correct_clean_wrong":
                    result[
                        "left_correct_right_wrong"
                    ],

                "corrupted_wrong_clean_correct":
                    result[
                        "left_wrong_right_correct"
                    ],

                "discordant":
                    result[
                        "discordant"
                    ],

                "p_value":
                    result[
                        "p_value"
                    ],
            })


condition_df = pd.DataFrame(
    condition_tests
)


# Holm correction separately per model
condition_parts = []

for model in condition_df[
    "model"
].unique():

    part = condition_df[
        condition_df["model"] == model
    ].copy()


    part["p_holm"] = holm_adjust(
        part["p_value"].values
    )


    part["significant_0_05"] = (
        part["p_holm"]
        < 0.05
    )


    condition_parts.append(
        part
    )


condition_df = pd.concat(
    condition_parts,
    ignore_index=True
)


condition_df.to_csv(

    OUTPUT
    / "mcnemar_condition_vs_clean.csv",

    index=False
)


# ==================================================
# CONDITION SUMMARY ACROSS SEEDS
# ==================================================

condition_summary = (

    condition_df

    .groupby(
        [
            "model",
            "experiment",
            "condition"
        ]
    )

    .agg(

        mean_accuracy_change=(
            "accuracy_change",
            "mean"
        ),

        min_p_holm=(
            "p_holm",
            "min"
        ),

        max_p_holm=(
            "p_holm",
            "max"
        ),

        significant_seeds=(
            "significant_0_05",
            "sum"
        ),
    )

    .reset_index()
)


condition_summary[
    "all_3_seeds_significant"
] = (
    condition_summary[
        "significant_seeds"
    ] == 3
)


condition_summary.to_csv(

    OUTPUT
    / "mcnemar_condition_summary.csv",

    index=False
)


# ==================================================
# PART 2
# KEY MODEL COMPARISONS
# ==================================================

KEY_COMPARISONS = [

    (
        "E0",
        "multimodal",
        "audio",
        "Clean: multimodal vs audio"
    ),

    (
        "E3",
        "multimodal",
        "audio",
        "0 dB audio: multimodal vs audio"
    ),

    (
        "E6",
        "multimodal",
        "audio",
        "Sigma4 blur: multimodal vs audio"
    ),

    (
        "E7",
        "multimodal",
        "visual",
        "Missing audio: multimodal vs visual"
    ),

    (
        "E8",
        "multimodal",
        "audio",
        "Missing video: multimodal vs audio"
    ),

    (
        "E9",
        "multimodal",
        "audio",
        "Dual degradation: multimodal vs audio"
    ),
]


model_tests = []


for (
    experiment,
    left_model,
    right_model,
    description
) in KEY_COMPARISONS:

    for seed in SEEDS:

        left = df[
            (df["experiment"] == experiment)
            &
            (df["model"] == left_model)
            &
            (df["seed"] == seed)
        ]


        right = df[
            (df["experiment"] == experiment)
            &
            (df["model"] == right_model)
            &
            (df["seed"] == seed)
        ]


        result = mcnemar_exact(
            left,
            right
        )


        model_tests.append({

            "comparison":
                description,

            "experiment":
                experiment,

            "seed":
                seed,

            "left_model":
                left_model,

            "right_model":
                right_model,

            "left_accuracy":
                result[
                    "left_accuracy"
                ],

            "right_accuracy":
                result[
                    "right_accuracy"
                ],

            "accuracy_difference":
                result[
                    "difference"
                ],

            "left_correct_right_wrong":
                result[
                    "left_correct_right_wrong"
                ],

            "left_wrong_right_correct":
                result[
                    "left_wrong_right_correct"
                ],

            "discordant":
                result[
                    "discordant"
                ],

            "p_value":
                result[
                    "p_value"
                ],
        })


model_df = pd.DataFrame(
    model_tests
)


model_df["p_holm"] = holm_adjust(
    model_df["p_value"].values
)


model_df["significant_0_05"] = (
    model_df["p_holm"]
    < 0.05
)


model_df.to_csv(

    OUTPUT
    / "mcnemar_key_model_comparisons.csv",

    index=False
)


model_summary = (

    model_df

    .groupby(
        [
            "comparison",
            "experiment",
            "left_model",
            "right_model"
        ]
    )

    .agg(

        mean_accuracy_difference=(
            "accuracy_difference",
            "mean"
        ),

        significant_seeds=(
            "significant_0_05",
            "sum"
        ),

        min_p_holm=(
            "p_holm",
            "min"
        ),

        max_p_holm=(
            "p_holm",
            "max"
        ),
    )

    .reset_index()
)


model_summary[
    "all_3_seeds_significant"
] = (
    model_summary[
        "significant_seeds"
    ] == 3
)


model_summary.to_csv(

    OUTPUT
    / "mcnemar_key_model_summary.csv",

    index=False
)


# ==================================================
# PART 3
# PER-CLASS METRICS
# ==================================================

class_rows = []


for (
    experiment,
    model,
    seed
), group in df.groupby(
    [
        "experiment",
        "model",
        "seed"
    ]
):


    precision, recall, f1, support = (
        precision_recall_fscore_support(

            group["true_class"],

            group["predicted_class"],

            labels=CLASSES,

            zero_division=0
        )
    )


    condition = (
        group["condition"]
        .iloc[0]
    )


    for i, class_name in enumerate(
        CLASSES
    ):

        class_rows.append({

            "experiment":
                experiment,

            "condition":
                condition,

            "model":
                model,

            "seed":
                seed,

            "class_name":
                class_name,

            "precision":
                precision[i],

            "recall":
                recall[i],

            "f1":
                f1[i],

            "support":
                support[i],
        })


class_seed_df = pd.DataFrame(
    class_rows
)


class_seed_df.to_csv(

    OUTPUT
    / "class_metrics_by_seed.csv",

    index=False
)


# ==================================================
# AGGREGATE ACROSS SEEDS
# ==================================================

class_summary = (

    class_seed_df

    .groupby(
        [
            "experiment",
            "condition",
            "model",
            "class_name"
        ]
    )

    .agg(

        precision_mean=(
            "precision",
            "mean"
        ),

        precision_std=(
            "precision",
            "std"
        ),

        recall_mean=(
            "recall",
            "mean"
        ),

        recall_std=(
            "recall",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        support=(
            "support",
            "first"
        ),
    )

    .reset_index()
)


class_summary.to_csv(

    OUTPUT
    / "class_metrics.csv",

    index=False
)


# ==================================================
# PART 4
# MULTIMODAL CLASS CHANGES
# ==================================================

mm = class_summary[
    class_summary["model"]
    == "multimodal"
].copy()


clean_mm = (

    mm[
        mm["experiment"]
        == "E0"
    ][
        [
            "class_name",
            "f1_mean",
            "recall_mean"
        ]
    ]

    .rename(
        columns={

            "f1_mean":
                "clean_f1",

            "recall_mean":
                "clean_recall",
        }
    )
)


KEY_CONDITIONS = [
    "E3",
    "E6",
    "E7",
    "E8",
    "E9",
]


changes = (

    mm[
        mm["experiment"]
        .isin(
            KEY_CONDITIONS
        )
    ]

    .merge(
        clean_mm,
        on="class_name",
        how="left"
    )
)


changes[
    "f1_change"
] = (
    changes["f1_mean"]
    -
    changes["clean_f1"]
)


changes[
    "recall_change"
] = (
    changes["recall_mean"]
    -
    changes["clean_recall"]
)


changes.to_csv(

    OUTPUT
    / "multimodal_class_changes.csv",

    index=False
)


# ==================================================
# PRINT IMPORTANT RESULTS
# ==================================================

print()
print("=" * 80)
print("KEY MODEL COMPARISONS")
print("=" * 80)

print(

    model_summary[
        [
            "comparison",
            "mean_accuracy_difference",
            "significant_seeds",
            "all_3_seeds_significant",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 80)
print("MULTIMODAL: WORST CLASSES UNDER MISSING AUDIO")
print("=" * 80)


missing_audio = (

    changes[
        changes["experiment"]
        == "E7"
    ]

    .sort_values(
        "f1_change"
    )

    .head(10)
)


print(

    missing_audio[
        [
            "class_name",
            "clean_f1",
            "f1_mean",
            "f1_change",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 80)
print("MULTIMODAL: WORST CLASSES UNDER MISSING VIDEO")
print("=" * 80)


missing_video = (

    changes[
        changes["experiment"]
        == "E8"
    ]

    .sort_values(
        "f1_change"
    )

    .head(10)
)


print(

    missing_video[
        [
            "class_name",
            "clean_f1",
            "f1_mean",
            "f1_change",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved statistical analysis to:",
    OUTPUT
)
