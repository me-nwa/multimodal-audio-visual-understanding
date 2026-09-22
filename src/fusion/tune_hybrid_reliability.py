from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

MODEL_DIR = Path(
    "models/baselines"
)

CLASS_MAPPING = Path(
    "results/baseline_clean/class_mapping.json"
)

OUTPUT_DIR = Path(
    "results/final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIO_DIR = Path(
    "data/embeddings/audio"
)

VISUAL_DIR = Path(
    "data/embeddings/visual"
)

SEEDS = [42, 43, 44]

HIDDEN_DIM = 256
DROPOUT = 0.30

# Maximum permitted clean-validation loss:
# 0.005 = 0.5 percentage points
CLEAN_TOLERANCE = 0.005

THRESHOLDS = np.round(
    np.arange(0.00, 1.001, 0.01),
    2
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CLASS MAPPING
# ============================================================

with open(
    CLASS_MAPPING,
    "r",
    encoding="utf-8",
) as f:
    class_to_idx = json.load(f)

idx_to_class = {
    int(v): k
    for k, v in class_to_idx.items()
}


# ============================================================
# VALIDATION SPLIT ONLY
# ============================================================

df = pd.read_csv(
    MANIFEST
)

val_df = (
    df[df["split"] == "val"]
    .copy()
    .reset_index(drop=True)
)

val_df["label"] = (
    val_df["class_name"]
    .map(class_to_idx)
)

targets = (
    val_df["label"]
    .to_numpy()
)

print(
    "Validation samples:",
    len(val_df)
)

assert len(val_df) == 402


# ============================================================
# MODEL
# ============================================================

class EventClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM,
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                num_classes,
            ),
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD CLEAN VALIDATION FEATURES
# ============================================================

audio_features = []
visual_features = []

for _, row in val_df.iterrows():

    sample_id = row["sample_id"]

    audio = np.load(
        AUDIO_DIR
        / f"{sample_id}.npy"
    ).astype(np.float32)

    visual = np.load(
        VISUAL_DIR
        / f"{sample_id}.npy"
    ).astype(np.float32)

    audio_features.append(
        audio
    )

    visual_features.append(
        visual
    )


audio_features = np.stack(
    audio_features
)

visual_features = np.stack(
    visual_features
)

multimodal_features = np.concatenate(
    [
        audio_features,
        visual_features,
    ],
    axis=1,
).astype(np.float32)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    modality,
    seed,
    features,
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

    model = model.to(
        device
    )

    model.eval()

    x = torch.tensor(
        features,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )

    return (
        prediction.cpu().numpy(),
        confidence.cpu().numpy(),
    )


# ============================================================
# GENERATE CLEAN VALIDATION PREDICTIONS
# ============================================================

rows = []

for seed in SEEDS:

    print()
    print(
        f"Generating clean validation predictions "
        f"for seed {seed}"
    )

    audio_pred, audio_conf = predict(
        "audio",
        seed,
        audio_features,
    )

    visual_pred, visual_conf = predict(
        "visual",
        seed,
        visual_features,
    )

    multimodal_pred, multimodal_conf = predict(
        "multimodal",
        seed,
        multimodal_features,
    )

    for i, row in val_df.iterrows():

        rows.append({

            "seed":
                seed,

            "sample_id":
                row["sample_id"],

            "video_id":
                row["video_id"],

            "true_class":
                row["class_name"],

            "true_idx":
                int(targets[i]),

            "audio_prediction_idx":
                int(audio_pred[i]),

            "audio_prediction":
                idx_to_class[
                    int(audio_pred[i])
                ],

            "audio_confidence":
                float(audio_conf[i]),

            "visual_prediction_idx":
                int(visual_pred[i]),

            "visual_prediction":
                idx_to_class[
                    int(visual_pred[i])
                ],

            "visual_confidence":
                float(visual_conf[i]),

            "multimodal_prediction_idx":
                int(multimodal_pred[i]),

            "multimodal_prediction":
                idx_to_class[
                    int(multimodal_pred[i])
                ],

            "multimodal_confidence":
                float(
                    multimodal_conf[i]
                ),
        })


pred_df = pd.DataFrame(
    rows
)

prediction_file = (
    OUTPUT_DIR
    / "clean_validation_reliability_predictions.csv"
)

pred_df.to_csv(
    prediction_file,
    index=False,
)


# ============================================================
# HYBRID RULE
# ============================================================

def hybrid_predict(
    frame,
    threshold,
):

    baseline = (
        frame[
            "multimodal_prediction_idx"
        ]
        .to_numpy()
        .copy()
    )

    audio_pred = (
        frame[
            "audio_prediction_idx"
        ]
        .to_numpy()
    )

    visual_pred = (
        frame[
            "visual_prediction_idx"
        ]
        .to_numpy()
    )

    audio_conf = (
        frame[
            "audio_confidence"
        ]
        .to_numpy()
    )

    visual_conf = (
        frame[
            "visual_confidence"
        ]
        .to_numpy()
    )

    confidence_gap = np.abs(
        audio_conf
        - visual_conf
    )

    switch = (
        confidence_gap
        >= threshold
    )

    audio_more_reliable = (
        audio_conf
        >= visual_conf
    )

    use_audio = (
        switch
        & audio_more_reliable
    )

    use_visual = (
        switch
        & ~audio_more_reliable
    )

    final_pred = baseline.copy()

    final_pred[
        use_audio
    ] = audio_pred[
        use_audio
    ]

    final_pred[
        use_visual
    ] = visual_pred[
        use_visual
    ]

    return (
        final_pred,
        switch,
    )


# ============================================================
# BASELINE CLEAN VALIDATION ACCURACY
# ============================================================

baseline_seed_rows = []

for seed, group in pred_df.groupby(
    "seed"
):

    true_idx = (
        group["true_idx"]
        .to_numpy()
    )

    baseline_pred = (
        group[
            "multimodal_prediction_idx"
        ]
        .to_numpy()
    )

    acc = (
        baseline_pred
        == true_idx
    ).mean()

    baseline_seed_rows.append(
        acc
    )


baseline_clean_accuracy = float(
    np.mean(
        baseline_seed_rows
    )
)


# ============================================================
# SEARCH THRESHOLDS
# ============================================================

grid_rows = []

for threshold in THRESHOLDS:

    seed_accuracies = []
    seed_switch_rates = []

    for seed, group in pred_df.groupby(
        "seed"
    ):

        true_idx = (
            group[
                "true_idx"
            ]
            .to_numpy()
        )

        final_pred, switch = (
            hybrid_predict(
                group,
                threshold,
            )
        )

        accuracy = (
            final_pred
            == true_idx
        ).mean()

        switch_rate = (
            switch.mean()
        )

        seed_accuracies.append(
            accuracy
        )

        seed_switch_rates.append(
            switch_rate
        )


    mean_accuracy = float(
        np.mean(
            seed_accuracies
        )
    )

    mean_switch_rate = float(
        np.mean(
            seed_switch_rates
        )
    )

    clean_change = (
        mean_accuracy
        - baseline_clean_accuracy
    )


    grid_rows.append({

        "threshold":
            threshold,

        "clean_accuracy":
            mean_accuracy,

        "baseline_clean_accuracy":
            baseline_clean_accuracy,

        "clean_change":
            clean_change,

        "switch_rate":
            mean_switch_rate,

        "passes_clean_guard":
            (
                clean_change
                >= -CLEAN_TOLERANCE
            ),
    })


grid_df = pd.DataFrame(
    grid_rows
)


# ============================================================
# SELECT THRESHOLD
# ============================================================

eligible = (
    grid_df[
        grid_df[
            "passes_clean_guard"
        ]
    ]
    .copy()
)


if eligible.empty:

    raise RuntimeError(
        "No threshold satisfies "
        "the clean-validation guard."
    )


# ------------------------------------------------------------
# Selection rule:
#
# Choose the SMALLEST threshold that keeps
# clean validation accuracy within 0.5 pp
# of the original multimodal baseline.
#
# This gives the reliability mechanism the
# greatest opportunity to intervene while
# constraining clean-data degradation.
# ------------------------------------------------------------

eligible = eligible.sort_values(
    by="threshold",
    ascending=True,
)

best = eligible.iloc[0]

best_threshold = float(
    best["threshold"]
)


# ============================================================
# SAVE
# ============================================================

grid_file = (
    OUTPUT_DIR
    / "hybrid_clean_validation_grid.csv"
)

grid_df.to_csv(
    grid_file,
    index=False,
)


config = {

    "selected_threshold":
        best_threshold,

    "selection_split":
        "validation",

    "validation_samples":
        int(len(val_df)),

    "selection_data":
        "clean_validation_only",

    "clean_tolerance":
        CLEAN_TOLERANCE,

    "selection_rule":
        (
            "Smallest confidence-gap threshold "
            "whose mean clean-validation accuracy "
            "is no more than 0.5 percentage points "
            "below the original multimodal baseline."
        ),

    "reliability_signal":
        (
            "Maximum unimodal softmax confidence"
        ),

    "seeds":
        SEEDS,
}


config_file = (
    OUTPUT_DIR
    / "hybrid_reliability_threshold.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 90)

print(
    "HYBRID RELIABILITY THRESHOLD "
    "— CLEAN VALIDATION ONLY"
)

print("=" * 90)

print(
    "Baseline clean validation accuracy: "
    f"{baseline_clean_accuracy * 100:.2f}%"
)

print(
    "Selected threshold tau: "
    f"{best_threshold:.2f}"
)

print(
    "Hybrid clean validation accuracy:   "
    f"{best['clean_accuracy'] * 100:.2f}%"
)

print(
    "Clean validation change:             "
    f"{best['clean_change'] * 100:+.2f} pp"
)

print(
    "Clean validation switch rate:        "
    f"{best['switch_rate'] * 100:.2f}%"
)


print()
print(
    "FIRST 10 ELIGIBLE THRESHOLDS"
)

print("-" * 90)

display = (
    eligible
    .head(10)
    .copy()
)

for col in [
    "clean_accuracy",
    "baseline_clean_accuracy",
    "clean_change",
    "switch_rate",
]:

    display[col] *= 100


print(
    display[
        [
            "threshold",
            "clean_accuracy",
            "clean_change",
            "switch_rate",
        ]
    ].to_string(
        index=False,
        formatters={

            "threshold":
                lambda x: f"{x:.2f}",

            "clean_accuracy":
                lambda x: f"{x:.2f}%",

            "clean_change":
                lambda x: f"{x:+.2f} pp",

            "switch_rate":
                lambda x: f"{x:.2f}%",
        }
    )
)


print()
print("Saved:")
print(
    " ",
    prediction_file
)
print(
    " ",
    grid_file
)
print(
    " ",
    config_file
)

print()
print(
    "IMPORTANT:"
)

print(
    "No test samples and no corrupted "
    "test conditions were used for "
    "threshold selection."
)

print("=" * 90)
