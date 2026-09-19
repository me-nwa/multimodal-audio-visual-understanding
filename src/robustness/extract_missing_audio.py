from pathlib import Path
import csv

import numpy as np
import torch

from transformers import (
    ClapModel,
    ClapProcessor,
)


# ==================================================
# CONFIG
# ==================================================

MANIFEST = Path(
    "data/metadata/ave_manifest_validated.csv"
)

OUTPUT_DIR = Path(
    "data/embeddings/audio_corrupt/missing_audio"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLAP_MODEL = "laion/clap-htsat-unfused"

SAMPLE_RATE = 48000


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

print(
    "Test samples:",
    len(rows)
)


# ==================================================
# LOAD CLAP
# ==================================================

print("\nLoading CLAP...")

processor = ClapProcessor.from_pretrained(
    CLAP_MODEL
)

model = ClapModel.from_pretrained(
    CLAP_MODEL
)

model = model.to(device)
model.eval()

print("CLAP ready.")


# ==================================================
# CREATE SILENCE EMBEDDINGS
# ==================================================

failed = []


for index, row in enumerate(
    rows,
    start=1
):

    sample_id = row["sample_id"]

    output_file = (
        OUTPUT_DIR
        / f"{sample_id}.npy"
    )


    # Resume support
    if output_file.exists():
        continue


    try:

        start = float(
            row["start_time"]
        )

        end = float(
            row["end_time"]
        )

        duration = max(
            end - start,
            0.1
        )


        # ------------------------------------------
        # Completely silent waveform
        # ------------------------------------------

        num_samples = max(
            int(
                duration
                * SAMPLE_RATE
            ),
            1
        )


        silent_audio = np.zeros(
            num_samples,
            dtype=np.float32
        )


        inputs = processor(
            audios=silent_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )


        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
            if torch.is_tensor(value)
        }


        with torch.no_grad():

            embedding = (
                model.get_audio_features(
                    **inputs
                )
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

count = len(
    list(
        OUTPUT_DIR.glob(
            "*.npy"
        )
    )
)


print()
print("=" * 60)
print("MISSING AUDIO EXTRACTION COMPLETE")
print("=" * 60)

print(
    "Embeddings:",
    count
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
