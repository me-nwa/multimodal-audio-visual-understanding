from pathlib import Path
import csv
import subprocess
import tempfile

import av
import librosa
import numpy as np
import torch

from transformers import (
    ClapModel,
    ClapProcessor,
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
)

MANIFEST = Path("data/metadata/ave_manifest_validated.csv")
OUTPUT_DIR = Path("results/smoke_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLAP_MODEL = "laion/clap-htsat-unfused"
VIDEO_MODEL = "MCG-NJU/videomae-base"

SAMPLE_RATE = 48000
NUM_FRAMES = 16
NUM_SAMPLES = 5

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Select 5 training samples from different classes
# --------------------------------------------------

selected = []
seen_classes = set()

with MANIFEST.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        if row["split"] != "train":
            continue

        if row["class_name"] in seen_classes:
            continue

        selected.append(row)
        seen_classes.add(row["class_name"])

        if len(selected) == NUM_SAMPLES:
            break


print("\nSelected samples:")

for row in selected:
    print(
        row["sample_id"],
        "|",
        row["class_name"],
        "|",
        row["video_id"]
    )


# --------------------------------------------------
# AUDIO FEATURES
# --------------------------------------------------

print("\n==============================")
print("CLAP AUDIO EXTRACTION")
print("==============================")

clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
clap_model = ClapModel.from_pretrained(CLAP_MODEL)

clap_model = clap_model.to(device)
clap_model.eval()

audio_embeddings = {}

for row in selected:

    video_path = row["video_path"]

    print("\nAudio:", row["class_name"])

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                tmp_audio.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        audio, _ = librosa.load(
            tmp_audio.name,
            sr=SAMPLE_RATE,
            mono=True,
            duration=10.0,
        )

    inputs = clap_processor(
        audios=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if torch.is_tensor(value)
    }

    with torch.no_grad():
        embedding = clap_model.get_audio_features(**inputs)

    embedding = embedding.cpu()

    audio_embeddings[row["sample_id"]] = embedding

    print("  Shape:", embedding.shape)


del clap_model

if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# --------------------------------------------------
# VIDEO FEATURES
# --------------------------------------------------

print("\n==============================")
print("VIDEOMAE EXTRACTION")
print("==============================")

video_processor = VideoMAEImageProcessor.from_pretrained(
    VIDEO_MODEL
)

video_model = VideoMAEForPreTraining.from_pretrained(
    VIDEO_MODEL
)

video_model = video_model.to(device)
video_model.eval()

video_embeddings = {}

for row in selected:

    video_path = row["video_path"]

    print("\nVideo:", row["class_name"])

    container = av.open(video_path)

    frames = [
        frame.to_ndarray(format="rgb24")
        for frame in container.decode(video=0)
    ]

    container.close()

    indices = np.linspace(
        0,
        len(frames) - 1,
        NUM_FRAMES
    ).astype(int)

    sampled_frames = [
        frames[index]
        for index in indices
    ]

    inputs = video_processor(
        sampled_frames,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = video_model.videomae(
            pixel_values=pixel_values
        )

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .cpu()
    )

    video_embeddings[row["sample_id"]] = embedding

    print("  Shape:", embedding.shape)


# --------------------------------------------------
# MULTIMODAL FUSION
# --------------------------------------------------

print("\n==============================")
print("MULTIMODAL CHECK")
print("==============================")

summary_rows = []

for row in selected:

    sample_id = row["sample_id"]

    audio_embedding = torch.nn.functional.normalize(
        audio_embeddings[sample_id],
        p=2,
        dim=1
    )

    video_embedding = torch.nn.functional.normalize(
        video_embeddings[sample_id],
        p=2,
        dim=1
    )

    multimodal_embedding = torch.cat(
        [audio_embedding, video_embedding],
        dim=1
    )

    output_file = OUTPUT_DIR / f"{sample_id}.pt"

    torch.save(
        {
            "sample_id": sample_id,
            "video_id": row["video_id"],
            "class_name": row["class_name"],
            "audio_embedding": audio_embedding,
            "video_embedding": video_embedding,
            "multimodal_embedding": multimodal_embedding,
        },
        output_file,
    )

    print()
    print(sample_id, "-", row["class_name"])
    print("  Audio:", audio_embedding.shape)
    print("  Video:", video_embedding.shape)
    print("  AV:", multimodal_embedding.shape)

    summary_rows.append({
        "sample_id": sample_id,
        "video_id": row["video_id"],
        "class_name": row["class_name"],
        "audio_dim": audio_embedding.shape[1],
        "video_dim": video_embedding.shape[1],
        "multimodal_dim": multimodal_embedding.shape[1],
    })


summary_file = OUTPUT_DIR / "summary.csv"

with summary_file.open("w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary_rows)


print()
print("==============================")
print("SMOKE TEST COMPLETE")
print("==============================")

print("Processed:", len(selected), "real AVE samples")
print("Summary:", summary_file)
print("Embeddings:", OUTPUT_DIR)
