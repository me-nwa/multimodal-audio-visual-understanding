from pathlib import Path
import csv

import numpy as np
import torch

from transformers import (
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
)


# ==================================================
# CONFIG
# ==================================================

MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

OUTPUT_DIR = Path(
    "data/embeddings/visual_corrupt/missing_video"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIDEO_MODEL = "MCG-NJU/videomae-base"

NUM_FRAMES = 16
FRAME_SIZE = 224


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# TEST SET
# ==================================================

with MANIFEST.open(
    "r",
    encoding="utf-8"
) as f:

    rows = [
        row
        for row in csv.DictReader(f)
        if row["split"] == "test"
    ]

print("Test samples:", len(rows))


# ==================================================
# LOAD VIDEOMAE
# ==================================================

print("\nLoading VideoMAE...")

processor = (
    VideoMAEImageProcessor.from_pretrained(
        VIDEO_MODEL
    )
)

model = (
    VideoMAEForPreTraining.from_pretrained(
        VIDEO_MODEL
    )
)

model = model.to(device)
model.eval()

print("VideoMAE ready.")


# ==================================================
# CREATE 16 BLANK FRAMES
# ==================================================

blank_frames = [

    np.zeros(
        (
            FRAME_SIZE,
            FRAME_SIZE,
            3
        ),
        dtype=np.uint8
    )

    for _ in range(NUM_FRAMES)
]


inputs = processor(
    blank_frames,
    return_tensors="pt"
)

pixel_values = (
    inputs["pixel_values"]
    .to(device)
)


with torch.no_grad():

    outputs = model.videomae(
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


print(
    "Missing-video embedding:",
    embedding.shape
)


# ==================================================
# SAME BLANK REPRESENTATION FOR EVERY TEST SAMPLE
# ==================================================

for index, row in enumerate(
    rows,
    start=1
):

    sample_id = row["sample_id"]

    output_file = (
        OUTPUT_DIR
        / f"{sample_id}.npy"
    )

    np.save(
        output_file,
        embedding
    )

    if index % 50 == 0:

        print(
            f"Progress: {index}/{len(rows)}"
        )


count = len(
    list(
        OUTPUT_DIR.glob("*.npy")
    )
)


print()
print("=" * 60)
print("MISSING VIDEO EXTRACTION COMPLETE")
print("=" * 60)

print(
    "Embeddings:",
    count
)

