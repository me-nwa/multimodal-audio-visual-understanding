from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Locate the repository from this script's location in src/analysis/.
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/final/hybrid_reliability_test_predictions.csv"
OUTPUT = ROOT / "results/review_metrics"

CONDITIONS = [f"E{i}" for i in range(10)]
SEEDS = [42, 43, 44]

# The saved routing decision tells us which confidence HRAR actually used.
ROUTES = {
    "multimodal": "baseline",
    "audio_fallback": "audio",
    "visual_fallback": "visual",
    "audio_reliability_switch": "audio",
    "visual_reliability_switch": "visual",
}


def load_predictions():
    data = pd.read_csv(INPUT)
    keys = ["experiment", "seed", "sample_id"]

    if data.duplicated(keys).any():
        raise ValueError("Duplicate predictions found for a condition, seed, and segment.")

    expected = pd.MultiIndex.from_product(
        [CONDITIONS, SEEDS], names=["experiment", "seed"]
    )
    sizes = data.groupby(["experiment", "seed"]).size()
    if len(sizes) != 30 or not sizes.reindex(expected).eq(402).all():
        raise ValueError("Expected 402 segments for each of 10 conditions and 3 seeds.")

    if data.isna().any().any():
        raise ValueError("The predictions CSV contains missing values.")

    by_segment = data.groupby("sample_id")
    if (len(by_segment) != 402
            or by_segment.true_class.nunique().ne(1).any()
            or by_segment.video_id.nunique().ne(1).any()
            or data.true_class.nunique() != 28):
        raise ValueError("Segment IDs, source videos, or class labels are inconsistent.")

    unknown = set(data["selected_strategy"]) - set(ROUTES)
    if unknown:
        raise ValueError(f"Unknown HRAR routing strategies: {unknown}")

    for strategy, head in ROUTES.items():
        rows = data["selected_strategy"].eq(strategy)
        selected_predictions = data.loc[rows, f"{head}_prediction"]
        if not selected_predictions.eq(data.loc[rows, "hybrid_prediction"]).all():
            raise ValueError(f"HRAR predictions disagree with the selected head: {strategy}")
        data.loc[rows, "hybrid_confidence"] = data.loc[rows, f"{head}_confidence"]

    for head in ["baseline", "hybrid"]:
        correct = data[f"{head}_prediction"].eq(data["true_class"])
        if not correct.eq(data[f"{head}_correct"]).all():
            raise ValueError(f"Stored correctness flags disagree with {head} predictions.")
        if not data[f"{head}_confidence"].between(0, 1).all():
            raise ValueError(f"Invalid confidence values for {head}.")

    return data


def calibration_and_curves(data):
    """Calculate top-label ECE and risk–coverage separately for each seed."""
    ece_rows = []
    curve_rows = []

    for (condition, seed), group in data.groupby(["experiment", "seed"]):
        for head in ["baseline", "hybrid"]:
            confidence = group[f"{head}_confidence"].to_numpy()
            correct = group[f"{head}_correct"].to_numpy()

            # 15 equal-width bins. Confidence 1.0 belongs in the final bin.
            bins = np.minimum((confidence * 15).astype(int), 14)
            bin_errors = np.bincount(bins, weights=correct - confidence, minlength=15)
            ece_pct = 100 * np.abs(bin_errors).sum() / len(group)
            ece_rows.append((condition, seed, head, ece_pct))

            # Keep the most confident segments first. Include tied scores together.
            order = np.argsort(-confidence)
            sorted_confidence = confidence[order]
            end_of_tie = np.r_[sorted_confidence[:-1] != sorted_confidence[1:], True]
            retained = np.arange(1, len(group) + 1)[end_of_tie]
            coverage = retained / len(group)
            risk = 1 - np.cumsum(correct[order])[end_of_tie] / retained

            for c, r in zip(coverage, risk):
                curve_rows.append((condition, seed, head, c, r))

    ece = pd.DataFrame(ece_rows, columns=["experiment", "seed", "head", "ece_pct"])
    ece.to_csv(OUTPUT / "ece_by_seed.csv", index=False)
    mean_ece = ece.pivot_table(index="experiment", columns="head", values="ece_pct")
    mean_ece.to_csv(OUTPUT / "ece_mean_pct.csv")

    curves = pd.DataFrame(
        curve_rows, columns=["experiment", "seed", "head", "coverage", "risk"]
    )
    curves.to_csv(OUTPUT / "risk_coverage.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    colors = {"baseline": "tab:blue", "hybrid": "tab:orange"}
    names = {"baseline": "B0", "hybrid": "HRAR"}

    for condition, ax in zip(CONDITIONS, axes.flat):
        subset = curves[curves["experiment"].eq(condition)]
        for (head, seed), line in subset.groupby(["head", "seed"]):
            ax.step(
                line["coverage"], line["risk"], where="post",
                color=colors[head], alpha=0.65,
                label=names[head] if seed == 42 else None,
            )
        ax.set(title=condition, xlim=(0, 1), ylim=(0, 1))

    axes[0, 0].legend()
    fig.supxlabel("Coverage (fraction of test segments retained)")
    fig.supylabel("Risk (error rate on retained segments)")
    fig.suptitle("Risk–coverage across three training seeds")
    fig.tight_layout()
    fig.savefig(OUTPUT / "risk_coverage.png", dpi=180)
    plt.close(fig)


def paired_accuracy_intervals(data):
    """Bootstrap source videos while keeping paired predictions and seeds together."""
    rows = []

    for condition, group in data.groupby("experiment"):
        difference = group["hybrid_correct"] - group["baseline_correct"]
        videos = difference.groupby(group["video_id"]).agg(["sum", "size"]).to_numpy()

        rng = np.random.default_rng(42)
        draws = rng.integers(len(videos), size=(10_000, len(videos)))
        totals = videos[draws].sum(axis=1)
        bootstrap_gains = 100 * totals[:, 0] / totals[:, 1]
        low, high = np.percentile(bootstrap_gains, [2.5, 97.5])

        rows.append((
            condition,
            100 * group["baseline_correct"].mean(),
            100 * group["hybrid_correct"].mean(),
            100 * difference.mean(),
            low,
            high,
        ))

    results = pd.DataFrame(rows, columns=[
        "experiment", "b0_acc_pct", "hrar_acc_pct",
        "gain_pp", "ci_low_pp", "ci_high_pp",
    ])
    results.to_csv(OUTPUT / "paired_accuracy_ci.csv", index=False)
    print(results.round(2).to_string(index=False))


def per_class_results(data):
    """Calculate clean unimodal recall and the fallback gap for each AVE class."""
    clean = data[data["experiment"].eq("E0")]
    table = clean.groupby("true_class")["sample_id"].nunique().to_frame("test_segments")

    for head in ["audio", "visual"]:
        correct = clean[f"{head}_prediction"].eq(clean["true_class"])
        table[f"{head}_clean_recall_pct"] = 100 * correct.groupby(clean["true_class"]).mean()

    for condition, surviving_head in [("E7", "visual"), ("E8", "audio")]:
        group = data[data["experiment"].eq(condition)]
        expert_correct = group[f"{surviving_head}_prediction"].eq(group["true_class"])
        expert_recall = 100 * expert_correct.groupby(group["true_class"]).mean()
        b0_recall = 100 * group.groupby("true_class")["baseline_correct"].mean()

        table[f"{condition}_{surviving_head}_recall_pct"] = expert_recall
        table[f"{condition}_b0_recall_pct"] = b0_recall
        table[f"{condition}_fallback_gap_pp"] = expert_recall - b0_recall

    table.to_csv(OUTPUT / "per_class_fallback.csv")


if __name__ == "__main__":
    predictions = load_predictions()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    calibration_and_curves(predictions)
    paired_accuracy_intervals(predictions)
    per_class_results(predictions)

    print(f"\nAnalyzed {len(predictions):,} predictions.")
    print(f"Results saved in {OUTPUT}")