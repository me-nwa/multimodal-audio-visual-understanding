from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from torch import nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ==================================================
# PATHS
# ==================================================

MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

MISSING_AUDIO_DIR = Path(
    "data/embeddings/audio_corrupt/missing_audio"
)

VISUAL_DIR = Path(
    "data/embeddings/visual"
)

MODEL_DIR = Path(
    "models/baselines"
)

BASELINE_RESULTS = Path(
    "results/baseline_clean"
)

RESULT_DIR = Path(
    "results/robustness_missing_audio"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SEEDS = [
    42,
    43,
    44
]


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ==================================================
# CLASS MAPPING
# ==================================================

with open(
    BASELINE_RESULTS
    / "class_mapping.json",
    "r",
    encoding="utf-8",
) as f:

    class_to_idx = json.load(f)


# ==================================================
# TEST SET
# ==================================================

df = pd.read_csv(
    MANIFEST
)

test_df = (
    df[
        df["split"] == "test"
    ]
    .copy()
    .reset_index(drop=True)
)

test_df["label"] = (
    test_df["class_name"]
    .map(class_to_idx)
)


targets = (
    test_df["label"]
    .to_numpy()
)


print(
    "Test samples:",
    len(test_df)
)


# ==================================================
# MODEL
# ==================================================

class EventClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=256,
        dropout=0.30,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_classes
            ),
        )


    def forward(
        self,
        x
    ):

        return self.network(x)


# ==================================================
# FEATURES
# ==================================================

def load_features(
    modality
):

    features = []


    for _, row in (
        test_df.iterrows()
    ):

        sample_id = row[
            "sample_id"
        ]


        missing_audio = np.load(

            MISSING_AUDIO_DIR
            / f"{sample_id}.npy"

        ).astype(
            np.float32
        )


        visual = np.load(

            VISUAL_DIR
            / f"{sample_id}.npy"

        ).astype(
            np.float32
        )


        if modality == "audio":

            feature = (
                missing_audio
            )


        elif modality == "visual":

            feature = visual


        elif modality == "multimodal":

            feature = (
                np.concatenate(
                    [
                        missing_audio,
                        visual
                    ]
                )
                .astype(
                    np.float32
                )
            )


        else:

            raise ValueError(
                modality
            )


        features.append(
            feature
        )


    return np.stack(
        features
    )


# ==================================================
# EVALUATION
# ==================================================

def evaluate(
    modality,
    seed,
    features
):

    checkpoint = torch.load(

        MODEL_DIR
        / f"{modality}_seed{seed}.pt",

        map_location="cpu",
        weights_only=False,
    )


    model = EventClassifier(

        input_dim=checkpoint[
            "input_dim"
        ],

        num_classes=checkpoint[
            "num_classes"
        ],
    )


    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )


    model = model.to(device)
    model.eval()


    x = torch.tensor(
        features,
        dtype=torch.float32
    ).to(device)


    with torch.no_grad():

        logits = model(x)

        predictions = (
            logits.argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )


    accuracy = accuracy_score(
        targets,
        predictions
    )


    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )


    return (
        accuracy,
        macro_f1
    )


# ==================================================
# RUN
# ==================================================

metrics = []


for modality in [
    "audio",
    "visual",
    "multimodal",
]:

    features = load_features(
        modality
    )


    for seed in SEEDS:

        accuracy, macro_f1 = evaluate(
            modality,
            seed,
            features
        )


        print(

            f"{modality:10s} "
            f"seed={seed} "
            f"Acc={accuracy:.4f} "
            f"F1={macro_f1:.4f}"
        )


        metrics.append({

            "condition":
                "missing_audio",

            "model":
                modality,

            "seed":
                seed,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,
        })


# ==================================================
# SUMMARY
# ==================================================

metrics_df = pd.DataFrame(
    metrics
)


metrics_df.to_csv(

    RESULT_DIR
    / "metrics_by_seed.csv",

    index=False,
)


summary = (

    metrics_df

    .groupby(
        "model"
    )

    .agg(

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


# ==================================================
# ADD CLEAN BASELINES
# ==================================================

clean = pd.read_csv(
    BASELINE_RESULTS
    / "summary.csv"
)


clean = clean.rename(
    columns={

        "test_accuracy_mean":
            "clean_accuracy",

        "test_macro_f1_mean":
            "clean_macro_f1",
    }
)


summary = summary.merge(

    clean[
        [
            "model",
            "clean_accuracy",
            "clean_macro_f1",
        ]
    ],

    on="model",
    how="left",
)


# ==================================================
# ROBUSTNESS METRICS
# ==================================================

summary[
    "accuracy_drop"
] = (
    summary["clean_accuracy"]
    -
    summary["accuracy_mean"]
)


summary[
    "accuracy_retention"
] = (
    summary["accuracy_mean"]
    /
    summary["clean_accuracy"]
)


summary[
    "macro_f1_drop"
] = (
    summary["clean_macro_f1"]
    -
    summary["macro_f1_mean"]
)


summary[
    "macro_f1_retention"
] = (
    summary["macro_f1_mean"]
    /
    summary["clean_macro_f1"]
)


summary.to_csv(

    RESULT_DIR
    / "summary.csv",

    index=False,
)


print()
print("=" * 60)
print("MISSING AUDIO RESULTS")
print("=" * 60)

print(
    summary.to_string(
        index=False
    )
)
