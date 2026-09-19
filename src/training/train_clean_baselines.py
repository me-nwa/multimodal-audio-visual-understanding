from pathlib import Path
import csv
import json
import random
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score


# ==================================================
# CONFIGURATION
# ==================================================

MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

AUDIO_DIR = Path("data/embeddings/audio")
VISUAL_DIR = Path("data/embeddings/visual")

RESULT_DIR = Path("results/baseline_clean")
MODEL_DIR = Path("models/baselines")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
DROPOUT = 0.30


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ==================================================
# READ MANIFEST
# ==================================================

df = pd.read_csv(MANIFEST)

classes = sorted(df["class_name"].unique())

class_to_idx = {
    class_name: i
    for i, class_name in enumerate(classes)
}

idx_to_class = {
    i: class_name
    for class_name, i in class_to_idx.items()
}

df["label"] = df["class_name"].map(class_to_idx)

print("Samples:", len(df))
print("Classes:", len(classes))

print(
    "Train / Val / Test:",
    len(df[df["split"] == "train"]),
    len(df[df["split"] == "val"]),
    len(df[df["split"] == "test"]),
)

with open(
    RESULT_DIR / "class_mapping.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        class_to_idx,
        f,
        indent=2
    )


# ==================================================
# LOAD FEATURES
# ==================================================

def load_features(frame, modality):

    features = []
    labels = []
    sample_ids = []
    class_names = []

    for _, row in frame.iterrows():

        sample_id = row["sample_id"]

        audio = np.load(
            AUDIO_DIR / f"{sample_id}.npy"
        ).astype(np.float32)

        visual = np.load(
            VISUAL_DIR / f"{sample_id}.npy"
        ).astype(np.float32)

        if modality == "audio":

            feature = audio

        elif modality == "visual":

            feature = visual

        elif modality == "multimodal":

            feature = np.concatenate(
                [audio, visual]
            ).astype(np.float32)

        else:

            raise ValueError(
                f"Unknown modality: {modality}"
            )

        features.append(feature)
        labels.append(int(row["label"]))
        sample_ids.append(sample_id)
        class_names.append(row["class_name"])

    return (
        np.stack(features),
        np.array(labels),
        sample_ids,
        class_names,
    )


# ==================================================
# CLASSIFIER
# ==================================================

class EventClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                num_classes
            ),
        )

    def forward(self, x):

        return self.network(x)


# ==================================================
# DATA LOADER
# ==================================================

def create_loader(
    features,
    labels,
    shuffle=False
):

    dataset = TensorDataset(
        torch.tensor(
            features,
            dtype=torch.float32
        ),
        torch.tensor(
            labels,
            dtype=torch.long
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


# ==================================================
# EVALUATION
# ==================================================

def evaluate(
    model,
    loader
):

    model.eval()

    predictions = []
    targets = []
    confidences = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, predicted = (
                probabilities.max(dim=1)
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                y.numpy()
            )

            confidences.extend(
                confidence.cpu().numpy()
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

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": np.array(predictions),
        "targets": np.array(targets),
        "confidence": np.array(confidences),
    }


# ==================================================
# TRAIN ONE MODEL
# ==================================================

def train_model(
    modality,
    seed
):

    print()
    print("=" * 60)
    print(
        f"MODEL: {modality.upper()} | "
        f"SEED: {seed}"
    )
    print("=" * 60)

    set_seed(seed)

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()


    X_train, y_train, _, _ = (
        load_features(
            train_df,
            modality
        )
    )

    X_val, y_val, _, _ = (
        load_features(
            val_df,
            modality
        )
    )

    (
        X_test,
        y_test,
        test_sample_ids,
        test_class_names,
    ) = load_features(
        test_df,
        modality
    )


    print(
        "Feature dimension:",
        X_train.shape[1]
    )

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True
    )

    val_loader = create_loader(
        X_val,
        y_val
    )

    test_loader = create_loader(
        X_test,
        y_test
    )


    model = EventClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(classes)
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()


    best_val_f1 = -1
    best_epoch = 0
    best_state = None

    patience_counter = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        val_results = evaluate(
            model,
            val_loader
        )

        val_f1 = val_results[
            "macro_f1"
        ]

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {total_loss / len(train_loader):.4f} | "
            f"Val Acc {val_results['accuracy']:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )


        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:

            print(
                "Early stopping."
            )

            break


    # ----------------------------------------------
    # Restore best validation model
    # ----------------------------------------------

    model.load_state_dict(
        best_state
    )


    val_results = evaluate(
        model,
        val_loader
    )

    test_results = evaluate(
        model,
        test_loader
    )


    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{modality}_seed{seed}.pt"
    )

    torch.save(
        {
            "state_dict":
                model.state_dict(),

            "input_dim":
                X_train.shape[1],

            "num_classes":
                len(classes),

            "class_to_idx":
                class_to_idx,

            "seed":
                seed,

            "best_epoch":
                best_epoch,
        },
        model_path,
    )


    # ----------------------------------------------
    # Save test predictions
    # ----------------------------------------------

    prediction_file = (
        RESULT_DIR
        / f"predictions_{modality}_seed{seed}.csv"
    )


    prediction_rows = []

    for i in range(
        len(test_sample_ids)
    ):

        predicted_idx = int(
            test_results[
                "predictions"
            ][i]
        )

        prediction_rows.append({

            "sample_id":
                test_sample_ids[i],

            "true_class":
                test_class_names[i],

            "predicted_class":
                idx_to_class[
                    predicted_idx
                ],

            "confidence":
                float(
                    test_results[
                        "confidence"
                    ][i]
                ),
        })


    pd.DataFrame(
        prediction_rows
    ).to_csv(
        prediction_file,
        index=False
    )


    print()
    print("BEST EPOCH:", best_epoch)

    print(
        "TEST ACCURACY:",
        f"{test_results['accuracy']:.4f}"
    )

    print(
        "TEST MACRO-F1:",
        f"{test_results['macro_f1']:.4f}"
    )


    return {

        "model": modality,

        "seed": seed,

        "best_epoch":
            best_epoch,

        "val_accuracy":
            val_results["accuracy"],

        "val_macro_f1":
            val_results["macro_f1"],

        "test_accuracy":
            test_results["accuracy"],

        "test_macro_f1":
            test_results["macro_f1"],
    }


# ==================================================
# RUN ALL BASELINES
# ==================================================

all_results = []

for modality in [
    "audio",
    "visual",
    "multimodal"
]:

    for seed in SEEDS:

        result = train_model(
            modality,
            seed
        )

        all_results.append(
            result
        )


# ==================================================
# SAVE METRICS
# ==================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    RESULT_DIR / "metrics_by_seed.csv",
    index=False
)


# ==================================================
# SUMMARY
# ==================================================

summary = (
    results_df
    .groupby("model")
    .agg(
        test_accuracy_mean=(
            "test_accuracy",
            "mean"
        ),

        test_accuracy_std=(
            "test_accuracy",
            "std"
        ),

        test_macro_f1_mean=(
            "test_macro_f1",
            "mean"
        ),

        test_macro_f1_std=(
            "test_macro_f1",
            "std"
        ),
    )
    .reset_index()
)


summary.to_csv(
    RESULT_DIR / "summary.csv",
    index=False
)


print()
print("=" * 60)
print("FINAL CLEAN BASELINE RESULTS")
print("=" * 60)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "Results saved to:",
    RESULT_DIR
)
