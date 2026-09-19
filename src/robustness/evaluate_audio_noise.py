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

CLEAN_AUDIO = Path(
    "data/embeddings/audio"
)

NOISY_AUDIO_ROOT = Path(
    "data/embeddings/audio_corrupt"
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
    "results/robustness_audio_noise"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SEEDS = [
    42,
    43,
    44,
]


CONDITIONS = {
    "clean": CLEAN_AUDIO,
    "20db": (
        NOISY_AUDIO_ROOT
        / "snr_20db"
    ),
    "10db": (
        NOISY_AUDIO_ROOT
        / "snr_10db"
    ),
    "0db": (
        NOISY_AUDIO_ROOT
        / "snr_0db"
    ),
}


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


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


idx_to_class = {
    int(index):
        class_name

    for class_name, index
    in class_to_idx.items()
}


# ==================================================
# TEST MANIFEST
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


print(
    "Test samples:",
    len(test_df)
)


# ==================================================
# SAME CLASSIFIER USED DURING TRAINING
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
# FEATURE LOADING
# ==================================================

def load_features(
    modality,
    audio_dir
):

    features = []

    for _, row in (
        test_df.iterrows()
    ):

        sample_id = row[
            "sample_id"
        ]

        audio = np.load(
            audio_dir
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

            feature = audio


        elif modality == "visual":

            feature = visual


        elif modality == "multimodal":

            feature = (
                np.concatenate(
                    [
                        audio,
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

    model_path = (
        MODEL_DIR
        / f"{modality}_seed{seed}.pt"
    )


    checkpoint = torch.load(
        model_path,
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

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted = (
            probabilities.max(
                dim=1
            )
        )


    predicted = (
        predicted
        .cpu()
        .numpy()
    )

    confidence = (
        confidence
        .cpu()
        .numpy()
    )


    targets = (
        test_df[
            "label"
        ]
        .to_numpy()
    )


    accuracy = accuracy_score(
        targets,
        predicted
    )


    macro_f1 = f1_score(
        targets,
        predicted,
        average="macro",
        zero_division=0,
    )


    return (
        accuracy,
        macro_f1,
        predicted,
        confidence,
    )


# ==================================================
# RUN CONDITIONS
# ==================================================

metrics = []


for condition, audio_dir in (
    CONDITIONS.items()
):

    print()
    print("=" * 60)
    print(
        "CONDITION:",
        condition
    )
    print("=" * 60)


    for modality in [
        "audio",
        "visual",
        "multimodal",
    ]:


        features = load_features(
            modality,
            audio_dir
        )


        for seed in SEEDS:

            (
                accuracy,
                macro_f1,
                predicted,
                confidence,
            ) = evaluate(
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
                    condition,

                "model":
                    modality,

                "seed":
                    seed,

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,
            })


            prediction_df = pd.DataFrame({

                "sample_id":
                    test_df[
                        "sample_id"
                    ],

                "true_class":
                    test_df[
                        "class_name"
                    ],

                "predicted_class":
                    [
                        idx_to_class[
                            int(index)
                        ]
                        for index
                        in predicted
                    ],

                "confidence":
                    confidence,
            })


            prediction_df.to_csv(

                RESULT_DIR
                / (
                    f"predictions_"
                    f"{condition}_"
                    f"{modality}_"
                    f"seed{seed}.csv"
                ),

                index=False,
            )


# ==================================================
# SAVE PER-SEED METRICS
# ==================================================

metrics_df = pd.DataFrame(
    metrics
)

metrics_df.to_csv(
    RESULT_DIR
    / "metrics_by_seed.csv",
    index=False,
)


# ==================================================
# SUMMARY
# ==================================================

summary = (
    metrics_df
    .groupby(
        [
            "condition",
            "model"
        ]
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
# ROBUSTNESS DROP + RETENTION
# ==================================================

clean = (
    summary[
        summary[
            "condition"
        ] == "clean"
    ][
        [
            "model",
            "accuracy_mean",
            "macro_f1_mean",
        ]
    ]
    .rename(
        columns={
            "accuracy_mean":
                "clean_accuracy",

            "macro_f1_mean":
                "clean_macro_f1",
        }
    )
)


summary = summary.merge(
    clean,
    on="model",
    how="left",
)


summary[
    "accuracy_drop"
] = (
    summary[
        "clean_accuracy"
    ]
    -
    summary[
        "accuracy_mean"
    ]
)


summary[
    "accuracy_retention"
] = (
    summary[
        "accuracy_mean"
    ]
    /
    summary[
        "clean_accuracy"
    ]
)


summary[
    "macro_f1_drop"
] = (
    summary[
        "clean_macro_f1"
    ]
    -
    summary[
        "macro_f1_mean"
    ]
)


summary[
    "macro_f1_retention"
] = (
    summary[
        "macro_f1_mean"
    ]
    /
    summary[
        "clean_macro_f1"
    ]
)


# Make table easier to read
condition_order = {
    "clean": 0,
    "20db": 1,
    "10db": 2,
    "0db": 3,
}

model_order = {
    "audio": 0,
    "visual": 1,
    "multimodal": 2,
}


summary["_condition_order"] = (
    summary[
        "condition"
    ].map(condition_order)
)

summary["_model_order"] = (
    summary[
        "model"
    ].map(model_order)
)


summary = (
    summary
    .sort_values(
        [
            "_condition_order",
            "_model_order"
        ]
    )
    .drop(
        columns=[
            "_condition_order",
            "_model_order"
        ]
    )
)


summary.to_csv(
    RESULT_DIR
    / "summary.csv",
    index=False,
)


print()
print("=" * 60)
print("AUDIO ROBUSTNESS RESULTS")
print("=" * 60)

print(
    summary[
        [
            "condition",
            "model",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "accuracy_drop",
            "accuracy_retention",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Results saved to:",
    RESULT_DIR
)
