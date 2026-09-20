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
# Select samples from different classes
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


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("\nLoading CLAP...")

clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
clap_model = ClapModel.from_pretrained(CLAP_MODEL)

clap_model = clap_model.to(device)
clap_model.eval()


print("Loading VideoMAE...")

video_processor = VideoMAEImageProcessor.from_pretrained(
    VIDEO_MODEL
)

video_model = VideoMAEForPreTraining.from_pretrained(
    VIDEO_MODEL
)

video_model = video_model.to(device)
video_model.eval()


# --------------------------------------------------
# Process each annotated segment
# --------------------------------------------------

for row in selected:

    sample_id = row["sample_id"]
    video_path = row["video_path"]
    class_name = row["class_name"]

    start = float(row["start_time"])
    end = float(row["end_time"])

    duration = end - start

    print("\n================================")
    print("Sample:", sample_id)
    print("Class:", class_name)
    print("Video:", row["video_id"])
    print("Segment:", start, "->", end)
    print("Duration:", duration)
    print("================================")


    # ----------------------------------------------
    # AUDIO SEGMENT
    # ----------------------------------------------

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

    print("Audio duration:", len(audio) / SAMPLE_RATE)

    audio_inputs = clap_processor(
        audios=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    audio_inputs = {
        key: value.to(device)
        for key, value in audio_inputs.items()
        if torch.is_tensor(value)
    }

    with torch.no_grad():

        audio_embedding = clap_model.get_audio_features(
            **audio_inputs
        )

    audio_embedding = audio_embedding.cpu()


    # ----------------------------------------------
    # VIDEO SEGMENT
    # ----------------------------------------------

    container = av.open(video_path)

    segment_frames = []

    for frame in container.decode(video=0):

        frame_time = frame.time

        if frame_time is None:
            continue

        if start <= frame_time <= end:

            segment_frames.append(
                frame.to_ndarray(format="rgb24")
            )

    container.close()

    print(
        "Frames inside annotated segment:",
        len(segment_frames)
    )

    if len(segment_frames) == 0:
        raise RuntimeError(
            f"No frames found for {sample_id}"
        )

    # Uniformly select exactly 16 frames.
    # np.linspace automatically repeats indices
    # if a very short segment has few frames.
    indices = np.linspace(
        0,
        len(segment_frames) - 1,
        NUM_FRAMES
    ).astype(int)

    sampled_frames = [
        segment_frames[i]
        for i in indices
    ]

    video_inputs = video_processor(
        sampled_frames,
        return_tensors="pt"
    )

    pixel_values = (
        video_inputs["pixel_values"]
        .to(device)
    )

    with torch.no_grad():

        outputs = video_model.videomae(
            pixel_values=pixel_values
        )

    video_embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .cpu()
    )


    # ----------------------------------------------
    # NORMALIZE + FUSE
    # ----------------------------------------------

    audio_embedding = torch.nn.functional.normalize(
        audio_embedding,
        p=2,
        dim=1
    )

    video_embedding = torch.nn.functional.normalize(
        video_embedding,
        p=2,
        dim=1
    )

    multimodal_embedding = torch.cat(
        [
            audio_embedding,
            video_embedding
        ],
        dim=1
    )

    print("Audio embedding:", audio_embedding.shape)
    print("Video embedding:", video_embedding.shape)
    print("AV embedding:", multimodal_embedding.shape)


print("\n================================")
print("SEGMENT SMOKE TEST SUCCESSFUL")
print("================================")
