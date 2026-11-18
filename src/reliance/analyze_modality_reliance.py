from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = Path("results/final/master_results.csv")
OUTPUT_FILE = Path("results/final/modality_reliance_analysis.csv")

MODEL = "multimodal"


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "experiment",
    "condition",
    "model",
    "accuracy_mean",
    "macro_f1_mean",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def get_row(experiment, model):
    rows = df[
        (df["experiment"] == experiment)
        & (df["model"] == model)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for "
            f"experiment={experiment}, model={model}, "
            f"but found {len(rows)}"
        )

    return rows.iloc[0]


# ------------------------------------------------------------
# Retrieve required conditions
# ------------------------------------------------------------

# E0 = Clean
e0_mm = get_row("E0", "multimodal")

# E7 = Missing audio
e7_mm = get_row("E7", "multimodal")
e7_visual = get_row("E7", "visual")

# E8 = Missing video
e8_mm = get_row("E8", "multimodal")
e8_audio = get_row("E8", "audio")


# ------------------------------------------------------------
# ACCURACY-BASED MODALITY RELIANCE
# ------------------------------------------------------------

clean_acc = float(e0_mm["accuracy_mean"])

missing_audio_acc = float(e7_mm["accuracy_mean"])
missing_video_acc = float(e8_mm["accuracy_mean"])

# Performance loss when each modality disappears
audio_removal_drop = clean_acc - missing_audio_acc
video_removal_drop = clean_acc - missing_video_acc

total_removal_effect = audio_removal_drop + video_removal_drop

if total_removal_effect > 0:
    audio_reliance_ratio = (
        audio_removal_drop / total_removal_effect
    )
    visual_reliance_ratio = (
        video_removal_drop / total_removal_effect
    )
else:
    audio_reliance_ratio = float("nan")
    visual_reliance_ratio = float("nan")


# ------------------------------------------------------------
# FALLBACK ANALYSIS
# ------------------------------------------------------------

# If audio is absent, the multimodal system should ideally
# approach the surviving visual-only model.
visual_e7_acc = float(e7_visual["accuracy_mean"])
missing_audio_fallback_gap = (
    missing_audio_acc - visual_e7_acc
)

# If video is absent, the multimodal system should ideally
# approach the surviving audio-only model.
audio_e8_acc = float(e8_audio["accuracy_mean"])
missing_video_fallback_gap = (
    missing_video_acc - audio_e8_acc
)


# ------------------------------------------------------------
# MACRO-F1 SUPPORTING ANALYSIS
# ------------------------------------------------------------

clean_f1 = float(e0_mm["macro_f1_mean"])

missing_audio_f1 = float(e7_mm["macro_f1_mean"])
missing_video_f1 = float(e8_mm["macro_f1_mean"])

audio_removal_f1_drop = clean_f1 - missing_audio_f1
video_removal_f1_drop = clean_f1 - missing_video_f1

total_f1_effect = (
    audio_removal_f1_drop
    + video_removal_f1_drop
)

if total_f1_effect > 0:
    audio_reliance_ratio_f1 = (
        audio_removal_f1_drop / total_f1_effect
    )
else:
    audio_reliance_ratio_f1 = float("nan")


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results = pd.DataFrame(
    [
        {
            "metric": "clean_multimodal_accuracy",
            "value": clean_acc,
        },
        {
            "metric": "missing_audio_multimodal_accuracy",
            "value": missing_audio_acc,
        },
        {
            "metric": "missing_video_multimodal_accuracy",
            "value": missing_video_acc,
        },
        {
            "metric": "audio_removal_accuracy_drop",
            "value": audio_removal_drop,
        },
        {
            "metric": "video_removal_accuracy_drop",
            "value": video_removal_drop,
        },
        {
            "metric": "audio_reliance_ratio",
            "value": audio_reliance_ratio,
        },
        {
            "metric": "visual_reliance_ratio",
            "value": visual_reliance_ratio,
        },
        {
            "metric": "missing_audio_fallback_gap",
            "value": missing_audio_fallback_gap,
        },
        {
            "metric": "missing_video_fallback_gap",
            "value": missing_video_fallback_gap,
        },
        {
            "metric": "audio_removal_macro_f1_drop",
            "value": audio_removal_f1_drop,
        },
        {
            "metric": "video_removal_macro_f1_drop",
            "value": video_removal_f1_drop,
        },
        {
            "metric": "audio_reliance_ratio_macro_f1",
            "value": audio_reliance_ratio_f1,
        },
    ]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)


# ------------------------------------------------------------
# Display research summary
# ------------------------------------------------------------

print("\n" + "=" * 68)
print("MODALITY-RELIANCE ANALYSIS — CONCATENATION BASELINE")
print("=" * 68)

print("\n1. CLEAN BASELINE")
print(
    f"   Multimodal accuracy: "
    f"{clean_acc * 100:.2f}%"
)

print("\n2. MODALITY REMOVAL")
print(
    f"   Missing audio accuracy: "
    f"{missing_audio_acc * 100:.2f}%"
)
print(
    f"   Missing video accuracy: "
    f"{missing_video_acc * 100:.2f}%"
)

print("\n3. PERFORMANCE DROP")
print(
    f"   Audio removal drop: "
    f"{audio_removal_drop * 100:.2f} percentage points"
)
print(
    f"   Video removal drop: "
    f"{video_removal_drop * 100:.2f} percentage points"
)

print("\n4. DESCRIPTIVE MODALITY RELIANCE")
print(
    f"   Audio Reliance Ratio (ARR): "
    f"{audio_reliance_ratio:.4f} "
    f"({audio_reliance_ratio * 100:.2f}%)"
)
print(
    f"   Visual Reliance Ratio: "
    f"{visual_reliance_ratio:.4f} "
    f"({visual_reliance_ratio * 100:.2f}%)"
)

print("\n5. FALLBACK BEHAVIOR")

print(
    f"   E7 visual-only accuracy: "
    f"{visual_e7_acc * 100:.2f}%"
)
print(
    f"   E7 multimodal accuracy: "
    f"{missing_audio_acc * 100:.2f}%"
)
print(
    f"   Missing-audio fallback gap: "
    f"{missing_audio_fallback_gap * 100:.2f} percentage points"
)

print()

print(
    f"   E8 audio-only accuracy: "
    f"{audio_e8_acc * 100:.2f}%"
)
print(
    f"   E8 multimodal accuracy: "
    f"{missing_video_acc * 100:.2f}%"
)
print(
    f"   Missing-video fallback gap: "
    f"{missing_video_fallback_gap * 100:.2f} percentage points"
)

print("\n6. MACRO-F1 CHECK")
print(
    f"   Audio-removal Macro-F1 drop: "
    f"{audio_removal_f1_drop * 100:.2f} pp"
)
print(
    f"   Video-removal Macro-F1 drop: "
    f"{video_removal_f1_drop * 100:.2f} pp"
)
print(
    f"   Audio Reliance Ratio (Macro-F1): "
    f"{audio_reliance_ratio_f1:.4f}"
)

print("\nSaved:")
print(f"   {OUTPUT_FILE}")

print("\nNOTE:")
print(
    "   ARR is a study-specific descriptive diagnostic. "
    "It is not presented as an established standardized metric."
)

print("=" * 68)
