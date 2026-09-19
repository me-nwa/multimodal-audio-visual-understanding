from pathlib import Path
import csv

import av
import numpy as np
import torch

from PIL import Image, ImageFilter

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

OUTPUT_ROOT = Path(
    "data/embeddings/visual_corrupt"
)

VIDEO_MODEL = "MCG-NJU/videomae-base"

NUM_FRAMES = 16

BLUR_LEVELS = [
    1,
    2,
    4,
]


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# READ TEST SET
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


print(
    "Test samples:",
    len(rows)
)


# ==================================================
# OUTPUT FOLDERS
# ==================================================

for sigma in BLUR_LEVELS:

    (
        OUTPUT_ROOT
        / f"blur_sigma_{sigma}"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ==================================================
# LOAD VIDEOMAE
# ==================================================

print("\nLoading VideoMAE...")

processor = (
    VideoMAEImageProcessor
    .from_pretrained(
        VIDEO_MODEL
    )
)

model = (
    VideoMAEForPreTraining
    .from_pretrained(
        VIDEO_MODEL
    )
)

model = model.to(device)
model.eval()

print("VideoMAE ready.")


# ==================================================
# PROCESS TEST SET
# ==================================================

failed = []


for index, row in enumerate(
    rows,
    start=1
):

    sample_id = row["sample_id"]

    expected_files = [

        OUTPUT_ROOT
        / f"blur_sigma_{sigma}"
        / f"{sample_id}.npy"

        for sigma in BLUR_LEVELS
    ]


    # Resume support
    if all(
        path.exists()
        for path in expected_files
    ):
        continue


    video_path = row["video_path"]

    start = float(
        row["start_time"]
    )

    end = float(
        row["end_time"]
    )


    try:

        # ------------------------------------------
        # Decode annotated segment
        # ------------------------------------------

        container = av.open(
            video_path
        )

        segment_frames = []


        for frame in container.decode(
            video=0
        ):

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
                "No frames found in segment"
            )


        # ------------------------------------------
        # Use the same 16-frame sampling strategy
        # as the clean baseline
        # ------------------------------------------

        indices = np.linspace(
            0,
            len(segment_frames) - 1,
            NUM_FRAMES
        ).astype(int)


        sampled_frames = [

            segment_frames[i]

            for i in indices
        ]


        # ------------------------------------------
        # Create each blur condition
        # ------------------------------------------

        for sigma in BLUR_LEVELS:

            output_file = (

                OUTPUT_ROOT
                / f"blur_sigma_{sigma}"
                / f"{sample_id}.npy"
            )


            if output_file.exists():
                continue


            blurred_frames = []


            for frame in sampled_frames:

                image = Image.fromarray(
                    frame
                )

                image = image.filter(
                    ImageFilter.GaussianBlur(
                        radius=sigma
                    )
                )

                blurred_frames.append(
                    np.array(image)
                )


            # --------------------------------------
            # VideoMAE
            # --------------------------------------

            inputs = processor(
                blurred_frames,
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


            np.save(
                output_file,
                embedding
            )


    except Exception as exc:

        print(
            "\nFAILED:",
            sample_id,
            exc
        )

        failed.append(
            sample_id
        )


    if index % 25 == 0:

        print(
            f"Progress: "
            f"{index}/{len(rows)}"
        )


# ==================================================
# SUMMARY
# ==================================================

print()
print("=" * 60)
print("BLURRED VIDEO EXTRACTION COMPLETE")
print("=" * 60)


for sigma in BLUR_LEVELS:

    count = len(
        list(
            (
                OUTPUT_ROOT
                / f"blur_sigma_{sigma}"
            )
            .glob("*.npy")
        )
    )

    print(
        f"Sigma {sigma}:",
        count,
        "embeddings"
    )


print(
    "Failures:",
    len(failed)
)


if failed:

    print(
        "Failed IDs:",
        failed[:20]
    )
