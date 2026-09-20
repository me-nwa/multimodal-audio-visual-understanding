from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class ConditionSpec:
    experiment: str
    label: str
    audio_mode: str = "clean"
    video_mode: str = "clean"
    snr_db: int | None = None
    blur_sigma: int | None = None


CONDITIONS: dict[str, ConditionSpec] = {
    "E0": ConditionSpec("E0", "Clean"),
    "E1": ConditionSpec(
        "E1", "Audio noise 20 dB",
        audio_mode="noise", snr_db=20
    ),
    "E2": ConditionSpec(
        "E2", "Audio noise 10 dB",
        audio_mode="noise", snr_db=10
    ),
    "E3": ConditionSpec(
        "E3", "Audio noise 0 dB",
        audio_mode="noise", snr_db=0
    ),
    "E4": ConditionSpec(
        "E4", "Video blur sigma=1",
        video_mode="blur", blur_sigma=1
    ),
    "E5": ConditionSpec(
        "E5", "Video blur sigma=2",
        video_mode="blur", blur_sigma=2
    ),
    "E6": ConditionSpec(
        "E6", "Video blur sigma=4",
        video_mode="blur", blur_sigma=4
    ),
    "E7": ConditionSpec(
        "E7", "Missing audio",
        audio_mode="missing"
    ),
    "E8": ConditionSpec(
        "E8", "Missing video",
        video_mode="missing"
    ),
    "E9": ConditionSpec(
        "E9",
        "Audio 0 dB + Video blur sigma=4",
        audio_mode="noise",
        video_mode="blur",
        snr_db=0,
        blur_sigma=4,
    ),
}


def get_condition(experiment: str) -> ConditionSpec:
    key = experiment.upper().strip()
    if key not in CONDITIONS:
        valid = ", ".join(CONDITIONS)
        raise ValueError(
            f"Unknown condition '{experiment}'. Valid conditions: {valid}"
        )
    return CONDITIONS[key]


# Exact research implementation from:
# src/robustness/extract_noisy_audio.py
def stable_seed(sample_id: str, snr_db: int) -> int:
    text = f"{sample_id}_{snr_db}"

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        "little",
    )


def add_noise_at_snr(
    audio: np.ndarray,
    snr_db: int,
    seed: int,
) -> np.ndarray:

    audio = audio.astype(np.float32)

    signal_rms = np.sqrt(
        np.mean(audio ** 2) + 1e-12
    )

    # Same quiet-signal guard used by the research script.
    signal_rms = max(
        signal_rms,
        1e-6,
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

    # Exact peak protection from the research extraction.
    # Scaling the whole mixture preserves SNR.
    peak = np.max(
        np.abs(corrupted)
    )

    if peak > 1.0:
        corrupted = corrupted / peak

    return corrupted.astype(
        np.float32
    )


def missing_audio(
    audio: np.ndarray,
) -> np.ndarray:
    return np.zeros_like(
        audio,
        dtype=np.float32,
    )


# Exact blur family used by:
# src/robustness/extract_blurred_video.py
def blur_frames(
    frames: list[np.ndarray],
    sigma: int,
) -> list[np.ndarray]:

    blurred_frames: list[np.ndarray] = []

    for frame in frames:
        image = Image.fromarray(frame)

        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=sigma
            )
        )

        blurred_frames.append(
            np.array(image)
        )

    return blurred_frames


def missing_video(
    frames: list[np.ndarray],
) -> list[np.ndarray]:

    return [
        np.zeros_like(
            frame,
            dtype=np.uint8,
        )
        for frame in frames
    ]


def apply_condition(
    audio: np.ndarray,
    frames: list[np.ndarray],
    experiment: str,
    sample_key: str,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    ConditionSpec,
    int | None,
]:

    spec = get_condition(experiment)

    audio_out = audio
    frames_out = frames
    noise_seed: int | None = None

    if spec.audio_mode == "noise":
        assert spec.snr_db is not None

        noise_seed = stable_seed(
            sample_key,
            spec.snr_db,
        )

        audio_out = add_noise_at_snr(
            audio,
            snr_db=spec.snr_db,
            seed=noise_seed,
        )

    elif spec.audio_mode == "missing":
        audio_out = missing_audio(
            audio
        )

    if spec.video_mode == "blur":
        assert spec.blur_sigma is not None

        frames_out = blur_frames(
            frames,
            sigma=spec.blur_sigma,
        )

    elif spec.video_mode == "missing":
        frames_out = missing_video(
            frames
        )

    return (
        audio_out,
        frames_out,
        spec,
        noise_seed,
    )
