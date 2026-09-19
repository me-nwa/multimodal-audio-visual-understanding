from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==================================================
# CONFIG
# ==================================================

RESULTS = Path(
    "results/final/master_results_percent.csv"
)

FIGURES = Path(
    "figures"
)

FIGURES.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(
    RESULTS
)


MODEL_ORDER = [
    "audio",
    "visual",
    "multimodal",
]


MODEL_LABELS = {
    "audio": "Audio-only",
    "visual": "Visual-only",
    "multimodal": "Audio + Visual",
}


# ==================================================
# HELPER
# ==================================================

def save_figure(name):

    plt.tight_layout()

    plt.savefig(
        FIGURES / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        name
    )


# ==================================================
# FIGURE 1
# CLEAN BASELINE
# ==================================================

clean = df[
    df["experiment"] == "E0"
].set_index("model")


models = [
    MODEL_LABELS[m]
    for m in MODEL_ORDER
]

accuracy = [
    clean.loc[m, "accuracy_mean"]
    for m in MODEL_ORDER
]

errors = [
    clean.loc[m, "accuracy_std"]
    for m in MODEL_ORDER
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    models,
    accuracy,
    yerr=errors,
    capsize=5,
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Clean AVE Classification Performance"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.25
)

save_figure(
    "fig1_clean_baseline"
)


# ==================================================
# FIGURE 2
# AUDIO NOISE ROBUSTNESS
# ==================================================

audio_conditions = [
    "E0",
    "E1",
    "E2",
    "E3",
]

audio_labels = [
    "Clean",
    "20 dB",
    "10 dB",
    "0 dB",
]


plt.figure(
    figsize=(8, 5)
)


for model in MODEL_ORDER:

    values = []

    errors = []

    for experiment in audio_conditions:

        row = df[
            (df["experiment"] == experiment)
            &
            (df["model"] == model)
        ].iloc[0]

        values.append(
            row["accuracy_mean"]
        )

        errors.append(
            row["accuracy_std"]
        )


    plt.errorbar(
        audio_labels,
        values,
        yerr=errors,
        marker="o",
        capsize=4,
        label=MODEL_LABELS[model],
    )


plt.xlabel(
    "Audio condition"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Robustness to Audio Noise"
)

plt.ylim(
    0,
    100
)

plt.grid(
    alpha=0.25
)

plt.legend()

save_figure(
    "fig2_audio_noise"
)


# ==================================================
# FIGURE 3
# VIDEO BLUR ROBUSTNESS
# ==================================================

video_conditions = [
    "E0",
    "E4",
    "E5",
    "E6",
]

video_labels = [
    "Clean",
    "σ = 1",
    "σ = 2",
    "σ = 4",
]


plt.figure(
    figsize=(8, 5)
)


for model in MODEL_ORDER:

    values = []

    errors = []

    for experiment in video_conditions:

        row = df[
            (df["experiment"] == experiment)
            &
            (df["model"] == model)
        ].iloc[0]

        values.append(
            row["accuracy_mean"]
        )

        errors.append(
            row["accuracy_std"]
        )


    plt.errorbar(
        video_labels,
        values,
        yerr=errors,
        marker="o",
        capsize=4,
        label=MODEL_LABELS[model],
    )


plt.xlabel(
    "Visual condition"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Robustness to Visual Blur"
)

plt.ylim(
    0,
    100
)

plt.grid(
    alpha=0.25
)

plt.legend()

save_figure(
    "fig3_video_blur"
)


# ==================================================
# FIGURE 4
# MISSING MODALITY
# ==================================================

missing_conditions = [
    "E0",
    "E7",
    "E8",
]

missing_labels = [
    "Clean",
    "Missing Audio",
    "Missing Video",
]


plot_rows = []


for experiment, label in zip(
    missing_conditions,
    missing_labels
):

    for model in MODEL_ORDER:

        row = df[
            (df["experiment"] == experiment)
            &
            (df["model"] == model)
        ].iloc[0]

        plot_rows.append({
            "condition": label,
            "model": MODEL_LABELS[model],
            "accuracy": row["accuracy_mean"],
        })


plot_df = pd.DataFrame(
    plot_rows
)


pivot = plot_df.pivot(
    index="condition",
    columns="model",
    values="accuracy",
)


pivot = pivot[
    [
        "Audio-only",
        "Visual-only",
        "Audio + Visual",
    ]
]


pivot = pivot.reindex(
    missing_labels
)


plt.figure(
    figsize=(9, 5)
)

pivot.plot(
    kind="bar",
    ax=plt.gca()
)


plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    ""
)

plt.title(
    "Effect of Complete Modality Loss"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=0
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    title="Model"
)

save_figure(
    "fig4_missing_modalities"
)


# ==================================================
# FIGURE 5
# ALL EXPERIMENTS - MULTIMODAL ONLY
# ==================================================

multimodal = (
    df[
        df["model"]
        == "multimodal"
    ]
    .copy()
)


multimodal["number"] = (
    multimodal["experiment"]
    .str.replace(
        "E",
        "",
        regex=False
    )
    .astype(int)
)


multimodal = (
    multimodal
    .sort_values("number")
)


labels = [

    f"{row.experiment}\n{row.condition}"

    for row in multimodal.itertuples()
]


plt.figure(
    figsize=(13, 6)
)


plt.bar(
    labels,
    multimodal["accuracy_mean"],
    yerr=multimodal["accuracy_std"],
    capsize=4,
)


plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Experiment"
)

plt.title(
    "Multimodal Accuracy Across Robustness Conditions"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.25
)

save_figure(
    "fig5_multimodal_all_conditions"
)


print()
print(
    "All figures generated successfully."
)
