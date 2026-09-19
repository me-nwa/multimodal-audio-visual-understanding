from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from torch import nn


# ==================================================
# PATHS
# ==================================================

MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

MODEL_DIR = Path(
    "models/baselines"
)

BASELINE_RESULTS = Path(
    "results/baseline_clean"
)

OUTPUT_DIR = Path(
    "results/final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# FEATURE DIRECTORIES FOR E0-E9
# ==================================================

CONDITIONS = {

    "E0": {
        "name": "Clean",
        "audio": Path(
            "data/embeddings/audio"
        ),
        "visual": Path(
            "data/embeddings/visual"
        ),
    },

    "E1": {
        "name": "Audio noise 20 dB",
        "audio": Path(
            "data/embeddings/audio_corrupt/snr_20db"
        ),
        "visual": Path(
            "data/embeddings/visual"
        ),
    },

    "E2": {
        "name": "Audio noise 10 dB",
        "audio": Path(
            "data/embeddings/audio_corrupt/snr_10db"
        ),
        "visual": Path(
            "data/embeddings/visual"
        ),
    },

    "E3": {
        "name": "Audio noise 0 dB",
        "audio": Path(
            "data/embeddings/audio_corrupt/snr_0db"
        ),
        "visual": Path(
            "data/embeddings/visual"
        ),
    },

    "E4": {
        "name": "Video blur sigma=1",
        "audio": Path(
            "data/embeddings/audio"
        ),
        "visual": Path(
            "data/embeddings/visual_corrupt/blur_sigma_1"
        ),
    },

    "E5": {
        "name": "Video blur sigma=2",
        "audio": Path(
            "data/embeddings/audio"
        ),
        "visual": Path(
            "data/embeddings/visual_corrupt/blur_sigma_2"
        ),
    },

    "E6": {
        "name": "Video blur sigma=4",
        "audio": Path(
            "data/embeddings/audio"
        ),
        "visual": Path(
            "data/embeddings/visual_corrupt/blur_sigma_4"
        ),
    },

    "E7": {
        "name": "Missing audio",
        "audio": Path(
            "data/embeddings/audio_corrupt/missing_audio"
        ),
        "visual": Path(
            "data/embeddings/visual"
        ),
    },

    "E8": {
        "name": "Missing video",
        "audio": Path(
            "data/embeddings/audio"
        ),
        "visual": Path(
            "data/embeddings/visual_corrupt/missing_video"
        ),
    },

    "E9": {
        "name": "Audio 0 dB + Video sigma=4",
        "audio": Path(
            "data/embeddings/audio_corrupt/snr_0db"
        ),
        "visual": Path(
            "data/embeddings/visual_corrupt/blur_sigma_4"
        ),
    },
}


SEEDS = [
    42,
    43,
    44
]

MODALITIES = [
    "audio",
    "visual",
    "multimodal"
]


# ==================================================
# DEVICE
# ==================================================

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


idx_to_class = {

    int(index):
        class_name

    for class_name, index
    in class_to_idx.items()
}


# ==================================================
# TEST DATA
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
# CLASSIFIER
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
# LOAD FEATURES
# ==================================================

def load_features(
    modality,
    audio_dir,
    visual_dir
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

            visual_dir
            / f"{sample_id}.npy"

        ).astype(
            np.float32
        )


        if modality == "audio":

            feature = audio


        elif modality == "visual":

            feature = visual


        elif modality == "multimodal":

            feature = np.concatenate(
                [
                    audio,
                    visual
                ]
            ).astype(
                np.float32
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
# PREDICT
# ==================================================

def predict(
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

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(
                dim=1
            )
        )


    return (
        prediction
        .cpu()
        .numpy(),

        confidence
        .cpu()
        .numpy()
    )


# ==================================================
# EXPORT EVERYTHING
# ==================================================

output_rows = []


for experiment, config in (
    CONDITIONS.items()
):

    print()
    print(
        "=" * 60
    )

    print(
        experiment,
        config["name"]
    )

    print(
        "=" * 60
    )


    for modality in MODALITIES:

        print(
            "Loading:",
            modality
        )


        features = load_features(

            modality,

            config["audio"],

            config["visual"]
        )


        for seed in SEEDS:

            prediction, confidence = predict(

                modality,
                seed,
                features
            )


            accuracy = (
                prediction
                == targets
            ).mean()


            print(

                f"{modality:10s} "
                f"seed={seed} "
                f"accuracy={accuracy:.4f}"
            )


            for i, row in (
                test_df.iterrows()
            ):

                predicted_idx = int(
                    prediction[i]
                )

                true_idx = int(
                    targets[i]
                )


                output_rows.append({

                    "experiment":
                        experiment,

                    "condition":
                        config["name"],

                    "model":
                        modality,

                    "seed":
                        seed,

                    "sample_id":
                        row["sample_id"],

                    "video_id":
                        row["video_id"],

                    "true_class":
                        row["class_name"],

                    "predicted_class":
                        idx_to_class[
                            predicted_idx
                        ],

                    "correct":
                        int(
                            predicted_idx
                            == true_idx
                        ),

                    "confidence":
                        float(
                            confidence[i]
                        ),
                })


# ==================================================
# SAVE
# ==================================================

predictions = pd.DataFrame(
    output_rows
)


output_file = (
    OUTPUT_DIR
    / "predictions_all.csv"
)


predictions.to_csv(
    output_file,
    index=False
)


print()
print(
    "=" * 60
)

print(
    "EXPORT COMPLETE"
)

print(
    "=" * 60
)

print(
    "Rows:",
    len(predictions)
)

print(
    "Saved:",
    output_file
)
