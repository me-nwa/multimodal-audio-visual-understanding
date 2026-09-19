from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

PREDICTIONS = Path(
    "results/final/predictions_all.csv"
)

OUTPUT_DIR = Path(
    "results/final/source_disjoint"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD
# ==================================================

manifest = pd.read_csv(
    MANIFEST
)

predictions = pd.read_csv(
    PREDICTIONS
)


# ==================================================
# FIND SOURCE VIDEO OVERLAP
# ==================================================

train_val_ids = set(

    manifest[
        manifest["split"].isin(
            ["train", "val"]
        )
    ]["video_id"]
)


test_manifest = (

    manifest[
        manifest["split"] == "test"
    ]
    .copy()
)


test_manifest[
    "source_seen_during_training"
] = (

    test_manifest[
        "video_id"
    ].isin(
        train_val_ids
    )
)


overlapping = test_manifest[
    test_manifest[
        "source_seen_during_training"
    ]
]


clean_test = test_manifest[
    ~test_manifest[
        "source_seen_during_training"
    ]
]


print("=" * 70)
print("SOURCE-DISJOINT SENSITIVITY CHECK")
print("=" * 70)

print(
    "Original test samples:",
    len(test_manifest)
)

print(
    "Overlapping test samples removed:",
    len(overlapping)
)

print(
    "Remaining source-disjoint test samples:",
    len(clean_test)
)

print(
    "Unique overlapping video IDs:",
    overlapping["video_id"].nunique()
)


# ==================================================
# SAVE OVERLAP AUDIT
# ==================================================

overlapping.to_csv(

    OUTPUT_DIR
    / "excluded_test_samples.csv",

    index=False
)


# ==================================================
# FILTER PREDICTIONS
# ==================================================

valid_sample_ids = set(
    clean_test["sample_id"]
)


filtered = predictions[
    predictions[
        "sample_id"
    ].isin(
        valid_sample_ids
    )
].copy()


# ==================================================
# EVALUATE
# ==================================================

results = []


for (
    experiment,
    condition,
    model,
    seed
), group in filtered.groupby(

    [
        "experiment",
        "condition",
        "model",
        "seed",
    ]
):

    accuracy = accuracy_score(

        group["true_class"],
        group["predicted_class"],
    )


    macro_f1 = f1_score(

        group["true_class"],
        group["predicted_class"],

        average="macro",
        zero_division=0,
    )


    results.append({

        "experiment":
            experiment,

        "condition":
            condition,

        "model":
            model,

        "seed":
            seed,

        "n_samples":
            len(group),

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,
    })


results_df = pd.DataFrame(
    results
)


results_df.to_csv(

    OUTPUT_DIR
    / "metrics_by_seed.csv",

    index=False
)


# ==================================================
# SUMMARY
# ==================================================

summary = (

    results_df

    .groupby(
        [
            "experiment",
            "condition",
            "model",
        ]
    )

    .agg(

        n_samples=(
            "n_samples",
            "first"
        ),

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),

        macro_f1_std=(
            "macro_f1",
            "std"
        ),
    )

    .reset_index()
)


summary[
    "experiment_number"
] = (

    summary[
        "experiment"
    ]
    .str.replace(
        "E",
        "",
        regex=False
    )
    .astype(int)
)


model_order = {
    "audio": 0,
    "visual": 1,
    "multimodal": 2,
}


summary[
    "model_order"
] = (

    summary[
        "model"
    ].map(
        model_order
    )
)


summary = (

    summary

    .sort_values(
        [
            "experiment_number",
            "model_order",
        ]
    )

    .drop(
        columns=[
            "experiment_number",
            "model_order",
        ]
    )
)


summary.to_csv(

    OUTPUT_DIR
    / "summary.csv",

    index=False
)


# ==================================================
# IMPORTANT CONDITIONS
# ==================================================

important = summary[
    summary["experiment"].isin(
        [
            "E0",
            "E3",
            "E6",
            "E7",
            "E8",
            "E9",
        ]
    )
]


print()
print("=" * 70)
print("KEY SOURCE-DISJOINT RESULTS")
print("=" * 70)

print(

    important[
        [
            "experiment",
            "model",
            "n_samples",
            "accuracy_mean",
            "macro_f1_mean",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved to:",
    OUTPUT_DIR
)
