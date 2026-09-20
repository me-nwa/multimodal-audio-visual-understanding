from pathlib import Path
import csv
import subprocess
import tempfile
import time

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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MANIFEST = Path("data/metadata/ave_manifest_validated.csv")

AUDIO_DIR = Path("data/embeddings/audio")
VIDEO_DIR = Path("data/embeddings/visual")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

CLAP_MODEL = "laion/clap-htsat-unfused"
VIDEO_MODEL = "MCG-NJU/videomae-base"

SAMPLE_RATE = 48000
NUM_FRAMES = 16

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Read manifest
# --------------------------------------------------

with MANIFEST.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("Samples:", len(rows))


# --------------------------------------------------
# Load CLAP
# --------------------------------------------------

print("\nLoading CLAP...")

clap_processor = ClapProcessor.from_pretrained(
    CLAP_MODEL
)

clap_model = ClapModel.from_pretrained(
    CLAP_MODEL
)

clap_model = clap_model.to(device)
clap_model.eval()

print("CLAP ready.")


# --------------------------------------------------
# AUDIO EXTRACTION
# --------------------------------------------------

print("\n==============================")
print("AUDIO EMBEDDING EXTRACTION")
print("==============================")

audio_failed = []

for index, row in enumerate(rows, start=1):

    sample_id = row["sample_id"]
    output_file = AUDIO_DIR / f"{sample_id}.npy"

    # Resume support
    if output_file.exists():
        continue

    video_path = row["video_path"]
    start = float(row["start_time"])
    end = float(row["end_time"])

    duration = max(end - start, 0.1)

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as tmp_audio:

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    video_path,
                    "-t",
                    str(duration),
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
            )

        if len(audio) == 0:
            raise RuntimeError("Empty audio")

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
            embedding = clap_model.get_audio_features(
                **inputs
            )

        embedding = (
            torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )
            .cpu()
            .numpy()
            .squeeze(0)
        )

        np.save(output_file, embedding)

    except Exception as exc:

        print(
            f"\nAUDIO FAILED: {sample_id} "
            f"{row['video_id']} - {exc}"
        )

        audio_failed.append(sample_id)


    if index % 50 == 0:
        print(
            f"Audio progress: "
            f"{index}/{len(rows)}"
        )


# Free CLAP memory
del clap_model

if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# --------------------------------------------------
# Load VideoMAE
# --------------------------------------------------

print("\nLoading VideoMAE...")

video_processor = (
    VideoMAEImageProcessor.from_pretrained(
        VIDEO_MODEL
    )
)

video_model = (
    VideoMAEForPreTraining.from_pretrained(
        VIDEO_MODEL
    )
)

video_model = video_model.to(device)
video_model.eval()

print("VideoMAE ready.")


# --------------------------------------------------
# VIDEO EXTRACTION
# --------------------------------------------------

print("\n==============================")
print("VIDEO EMBEDDING EXTRACTION")
print("==============================")

video_failed = []

for index, row in enumerate(rows, start=1):

    sample_id = row["sample_id"]
    output_file = VIDEO_DIR / f"{sample_id}.npy"

    # Resume support
    if output_file.exists():
        continue

    video_path = row["video_path"]

    start = float(row["start_time"])
    end = float(row["end_time"])

    try:

        container = av.open(video_path)

        segment_frames = []

        for frame in container.decode(video=0):

            frame_time = frame.time

            if frame_time is None:
                continue

            if start <= frame_time <= end:

                segment_frames.append(
                    frame.to_ndarray(
                        format="rgb24"
                    )
                )

        container.close()

        if len(segment_frames) == 0:
            raise RuntimeError(
                "No video frames in segment"
            )

        indices = np.linspace(
            0,
            len(segment_frames) - 1,
            NUM_FRAMES
        ).astype(int)

        sampled_frames = [
            segment_frames[i]
            for i in indices
        ]

        inputs = video_processor(
            sampled_frames,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(device)
        )

        with torch.no_grad():

            outputs = video_model.videomae(
                pixel_values=pixel_values
            )

        embedding = (
            outputs.last_hidden_state
            .mean(dim=1)
        )

        embedding = (
            torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )
            .cpu()
            .numpy()
            .squeeze(0)
        )

        np.save(output_file, embedding)

    except Exception as exc:

        print(
            f"\nVIDEO FAILED: {sample_id} "
            f"{row['video_id']} - {exc}"
        )

        video_failed.append(sample_id)


    if index % 50 == 0:
        print(
            f"Video progress: "
            f"{index}/{len(rows)}"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

audio_files = list(
    AUDIO_DIR.glob("*.npy")
)

video_files = list(
    VIDEO_DIR.glob("*.npy")
)

print()
print("==============================")
print("EXTRACTION COMPLETE")
print("==============================")

print("Expected samples:", len(rows))

print(
    "Audio embeddings:",
    len(audio_files)
)

print(
    "Video embeddings:",
    len(video_files)
)

print(
    "Audio failures:",
    len(audio_failed)
)

print(
    "Video failures:",
    len(video_failed)
)

if audio_failed:
    print(
        "Audio failed IDs:",
        audio_failed[:20]
    )

if video_failed:
    print(
        "Video failed IDs:",
        video_failed[:20]
    )
