from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

BASELINE_FILE = Path("results/final/predictions_all.csv")
FUSION_FILE = Path("results/final/predictions_reliability_fusion.csv")
OUTPUT_FILE = Path("results/final/reliability_fusion_mcnemar.csv")
SUMMARY_FILE = Path("results/final/reliability_fusion_mcnemar_summary.csv")

baseline = pd.read_csv(BASELINE_FILE)
fusion = pd.read_csv(FUSION_FILE)

baseline = baseline[baseline["model"] == "multimodal"].copy()

keys = [
    "experiment",
    "condition",
    "seed",
    "sample_id",
    "video_id",
    "true_class",
]

baseline = baseline[
    keys + ["predicted_class", "correct"]
].rename(
    columns={
        "predicted_class": "baseline_prediction",
        "correct": "baseline_correct",
    }
)

fusion = fusion[
    keys + ["predicted_class", "correct"]
].rename(
    columns={
        "predicted_class": "fusion_prediction",
        "correct": "fusion_correct",
    }
)

paired = baseline.merge(
    fusion,
    on=keys,
    how="inner",
    validate="one_to_one",
)

rows = []

for (experiment, condition, seed), g in paired.groupby(
    ["experiment", "condition", "seed"]
):

    # B0 correct, B1 wrong
    b = (
        (g["baseline_correct"] == 1)
        & (g["fusion_correct"] == 0)
    ).sum()

    # B0 wrong, B1 correct
    c = (
        (g["baseline_correct"] == 0)
        & (g["fusion_correct"] == 1)
    ).sum()

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    baseline_acc = g["baseline_correct"].mean()
    fusion_acc = g["fusion_correct"].mean()

    rows.append(
        {
            "experiment": experiment,
            "condition": condition,
            "seed": seed,
            "baseline_accuracy": baseline_acc,
            "fusion_accuracy": fusion_acc,
            "accuracy_change": fusion_acc - baseline_acc,
            "b_baseline_correct_fusion_wrong": int(b),
            "c_baseline_wrong_fusion_correct": int(c),
            "discordant_pairs": int(discordant),
            "p_raw": p_value,
        }
    )

res = pd.DataFrame(rows)


# ------------------------------------------------------------
# Holm correction within each seed across E0-E9
# ------------------------------------------------------------

def holm_adjust(pvals):
    n = len(pvals)
    order = sorted(
        range(n),
        key=lambda i: pvals[i]
    )

    adjusted = [None] * n
    running_max = 0.0

    for rank, idx in enumerate(order):
        factor = n - rank
        adj = min(1.0, pvals[idx] * factor)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    return adjusted


corrected_parts = []

for seed, g in res.groupby("seed"):
    g = g.copy().reset_index(drop=True)

    g["p_holm"] = holm_adjust(
        g["p_raw"].tolist()
    )

    g["significant_0_05"] = (
        g["p_holm"] < 0.05
    )

    corrected_parts.append(g)

res = pd.concat(
    corrected_parts,
    ignore_index=True
)

res = res.sort_values(
    ["experiment", "seed"]
)

res.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Summary across seeds
# ------------------------------------------------------------

summary = (
    res
    .groupby(
        ["experiment", "condition"],
        as_index=False,
    )
    .agg(
        mean_accuracy_change=(
            "accuracy_change",
            "mean",
        ),
        significant_seeds=(
            "significant_0_05",
            "sum",
        ),
        total_seeds=(
            "seed",
            "count",
        ),
    )
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

summary_display = summary.copy()
summary_display["mean_accuracy_change"] *= 100

print("\n" + "=" * 90)
print("B0 VS B1 — EXACT McNEMAR TEST WITH HOLM CORRECTION")
print("=" * 90)

print(
    summary_display.to_string(
        index=False,
        formatters={
            "mean_accuracy_change":
                lambda x: f"{x:+.2f} pp"
        }
    )
)

print("\nDetailed seed-level results:")
print("-" * 90)

for _, row in res.iterrows():
    print(
        f"{row['experiment']} | "
        f"seed {row['seed']} | "
        f"change={row['accuracy_change']*100:+.2f} pp | "
        f"b={row['b_baseline_correct_fusion_wrong']} | "
        f"c={row['c_baseline_wrong_fusion_correct']} | "
        f"p_raw={row['p_raw']:.6f} | "
        f"p_holm={row['p_holm']:.6f} | "
        f"sig={row['significant_0_05']}"
    )

print("\nSaved:")
print(f"  {OUTPUT_FILE}")
print(f"  {SUMMARY_FILE}")

print("=" * 90)
