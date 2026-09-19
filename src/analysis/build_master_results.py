from pathlib import Path
import pandas as pd


OUTPUT = Path("results/final")
OUTPUT.mkdir(parents=True, exist_ok=True)

rows = []


# ==================================================
# E0 CLEAN
# ==================================================

clean = pd.read_csv(
    "results/baseline_clean/summary.csv"
)

for _, row in clean.iterrows():

    rows.append({
        "experiment": "E0",
        "condition": "Clean",
        "model": row["model"],
        "accuracy_mean": row["test_accuracy_mean"],
        "accuracy_std": row["test_accuracy_std"],
        "macro_f1_mean": row["test_macro_f1_mean"],
        "macro_f1_std": row["test_macro_f1_std"],
        "accuracy_drop": 0.0,
        "accuracy_retention": 1.0,
        "macro_f1_drop": 0.0,
        "macro_f1_retention": 1.0,
    })


# ==================================================
# E1-E3 AUDIO NOISE
# ==================================================

audio_noise = pd.read_csv(
    "results/robustness_audio_noise/summary.csv"
)

audio_map = {
    "20db": ("E1", "Audio noise 20 dB"),
    "10db": ("E2", "Audio noise 10 dB"),
    "0db": ("E3", "Audio noise 0 dB"),
}

for _, row in audio_noise.iterrows():

    condition = row["condition"]

    if condition == "clean":
        continue

    experiment, label = audio_map[
        condition
    ]

    rows.append({
        "experiment": experiment,
        "condition": label,
        "model": row["model"],
        "accuracy_mean": row["accuracy_mean"],
        "accuracy_std": row["accuracy_std"],
        "macro_f1_mean": row["macro_f1_mean"],
        "macro_f1_std": row["macro_f1_std"],
        "accuracy_drop": row["accuracy_drop"],
        "accuracy_retention": row["accuracy_retention"],
        "macro_f1_drop": row["macro_f1_drop"],
        "macro_f1_retention": row["macro_f1_retention"],
    })


# ==================================================
# E4-E6 VIDEO BLUR
# ==================================================

video_blur = pd.read_csv(
    "results/robustness_video_blur/summary.csv"
)

video_map = {
    "sigma1": ("E4", "Video blur σ=1"),
    "sigma2": ("E5", "Video blur σ=2"),
    "sigma4": ("E6", "Video blur σ=4"),
}

for _, row in video_blur.iterrows():

    condition = row["condition"]

    if condition == "clean":
        continue

    experiment, label = video_map[
        condition
    ]

    rows.append({
        "experiment": experiment,
        "condition": label,
        "model": row["model"],
        "accuracy_mean": row["accuracy_mean"],
        "accuracy_std": row["accuracy_std"],
        "macro_f1_mean": row["macro_f1_mean"],
        "macro_f1_std": row["macro_f1_std"],
        "accuracy_drop": row["accuracy_drop"],
        "accuracy_retention": row["accuracy_retention"],
        "macro_f1_drop": row["macro_f1_drop"],
        "macro_f1_retention": row["macro_f1_retention"],
    })


# ==================================================
# HELPER FOR E7-E9
# ==================================================

def add_single_condition(
    csv_path,
    experiment,
    condition
):

    df = pd.read_csv(
        csv_path
    )

    for _, row in df.iterrows():

        rows.append({
            "experiment": experiment,
            "condition": condition,
            "model": row["model"],
            "accuracy_mean": row["accuracy_mean"],
            "accuracy_std": row["accuracy_std"],
            "macro_f1_mean": row["macro_f1_mean"],
            "macro_f1_std": row["macro_f1_std"],
            "accuracy_drop": row["accuracy_drop"],
            "accuracy_retention": row["accuracy_retention"],
            "macro_f1_drop": row["macro_f1_drop"],
            "macro_f1_retention": row["macro_f1_retention"],
        })


# E7
add_single_condition(
    "results/robustness_missing_audio/summary.csv",
    "E7",
    "Missing audio"
)

# E8
add_single_condition(
    "results/robustness_missing_video/summary.csv",
    "E8",
    "Missing video"
)

# E9
add_single_condition(
    "results/robustness_dual_degradation/summary.csv",
    "E9",
    "Audio 0 dB + Video σ=4"
)


# ==================================================
# FINAL TABLE
# ==================================================

final = pd.DataFrame(
    rows
)


experiment_order = {
    f"E{i}": i
    for i in range(10)
}

model_order = {
    "audio": 0,
    "visual": 1,
    "multimodal": 2,
}


final["_exp"] = (
    final["experiment"]
    .map(experiment_order)
)

final["_model"] = (
    final["model"]
    .map(model_order)
)


final = (
    final
    .sort_values(
        ["_exp", "_model"]
    )
    .drop(
        columns=[
            "_exp",
            "_model"
        ]
    )
)


final.to_csv(
    OUTPUT / "master_results.csv",
    index=False
)


# ==================================================
# PERCENTAGE VERSION
# ==================================================

display = final.copy()

for column in [
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "accuracy_drop",
    "macro_f1_drop",
]:

    display[column] = (
        display[column] * 100
    )


display[
    "accuracy_retention"
] = (
    display[
        "accuracy_retention"
    ] * 100
)

display[
    "macro_f1_retention"
] = (
    display[
        "macro_f1_retention"
    ] * 100
)


display.to_csv(
    OUTPUT / "master_results_percent.csv",
    index=False
)


print()
print("=" * 80)
print("FINAL E0-E9 RESULTS")
print("=" * 80)

print(
    display[
        [
            "experiment",
            "condition",
            "model",
            "accuracy_mean",
            "macro_f1_mean",
            "accuracy_retention",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUTPUT / "master_results.csv"
)

print(
    "Saved:",
    OUTPUT / "master_results_percent.csv"
)
