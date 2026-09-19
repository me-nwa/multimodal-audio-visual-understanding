from pathlib import Path
import csv
import hashlib
import subprocess
import tempfile

import librosa
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

OUTPUT_ROOT = Path(
    "data/embeddings/audio_corrupt"
)

CLAP_MODEL = "laion/clap-htsat-unfused"

SAMPLE_RATE = 48000

SNR_LEVELS = [
    20,
    10,
    0,
]

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# REPRODUCIBLE NOISE
# ==================================================

def stable_seed(sample_id, snr_db):

    text = f"{sample_id}_{snr_db}"

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        "little"
    )


def add_noise_at_snr(
    audio,
    snr_db,
    seed
):

    audio = audio.astype(np.float32)

    signal_rms = np.sqrt(
        np.mean(audio ** 2) + 1e-12
    )

    # Avoid division problems for extremely quiet clips
    signal_rms = max(
        signal_rms,
        1e-6
    )

    desired_noise_rms = (
        signal_rms
        / (10 ** (snr_db / 20.0))
    )

    rng = np.random.default_rng(seed)

    noise = rng.standard_normal(
        len(audio)
    ).astype(np.float32)

    noise_rms = np.sqrt(
        np.mean(noise ** 2) + 1e-12
    )

    noise = (
        noise
        / noise_rms
        * desired_noise_rms
    )

    corrupted = audio + noise

    # Keep waveform in a safe range.
    # Scaling the whole mixture preserves SNR.
    peak = np.max(
        np.abs(corrupted)
    )

    if peak > 1.0:
        corrupted = corrupted / peak

    return corrupted.astype(
        np.float32
    )


# ==================================================
# READ TEST SET ONLY
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
# OUTPUT DIRECTORIES
# ==================================================

for snr in SNR_LEVELS:

    (
        OUTPUT_ROOT
        / f"snr_{snr}db"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ==================================================
# LOAD CLAP
# ==================================================

print("\nLoading CLAP...")

processor = (
    ClapProcessor.from_pretrained(
        CLAP_MODEL
    )
)

model = (
    ClapModel.from_pretrained(
        CLAP_MODEL
    )
)

model = model.to(device)
model.eval()

print("CLAP ready.")


# ==================================================
# PROCESS TEST SAMPLES
# ==================================================

failed = []

for index, row in enumerate(
    rows,
    start=1
):

    sample_id = row["sample_id"]

    # Check whether this sample already exists
    # for every SNR condition.
    expected_files = [
        OUTPUT_ROOT
        / f"snr_{snr}db"
        / f"{sample_id}.npy"

        for snr in SNR_LEVELS
    ]

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

    duration = max(
        end - start,
        0.1
    )


    try:

        # ------------------------------------------
        # Extract clean annotated audio once
        # ------------------------------------------

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
            raise RuntimeError(
                "Empty audio segment"
            )


        # ------------------------------------------
        # Generate each noise condition
        # ------------------------------------------

        for snr in SNR_LEVELS:

            output_file = (
                OUTPUT_ROOT
                / f"snr_{snr}db"
                / f"{sample_id}.npy"
            )

            if output_file.exists():
                continue


            noisy_audio = add_noise_at_snr(
                audio,
                snr_db=snr,
                seed=stable_seed(
                    sample_id,
                    snr
                ),
            )


            inputs = processor(
                audios=noisy_audio,
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

print()
print("=" * 60)
print("NOISY AUDIO EXTRACTION COMPLETE")
print("=" * 60)

for snr in SNR_LEVELS:

    count = len(
        list(
            (
                OUTPUT_ROOT
                / f"snr_{snr}db"
            ).glob("*.npy")
        )
    )

    print(
        f"{snr:>2} dB:",
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
