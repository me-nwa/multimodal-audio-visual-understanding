from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import av
import librosa
import numpy as np
import torch
from torch import nn
from transformers import (
    ClapModel,
    ClapProcessor,
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
)

from .corruptions import (
    apply_condition,
    get_condition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_MODEL_ID = "laion/clap-htsat-unfused"
VIDEO_MODEL_ID = "MCG-NJU/videomae-base"

CLASS_MAPPING = (
    PROJECT_ROOT
    / "results/baseline_clean/class_mapping.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models/baselines"
)

SAMPLE_RATE = 48_000
NUM_FRAMES = 16
MAX_SECONDS = 10.0


class EventClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.network(x)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResearchInference:
    def __init__(
        self,
        seed: int = 42,
    ):
        self.seed = seed
        self.device = _device()

        if not CLASS_MAPPING.exists():
            raise FileNotFoundError(
                f"Missing class mapping: {CLASS_MAPPING}"
            )

        with CLASS_MAPPING.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.class_to_idx = json.load(f)

        self.idx_to_class = {
            int(idx): name
            for name, idx
            in self.class_to_idx.items()
        }

        print(
            f"[app] device={self.device}"
        )

        print("[app] loading CLAP...")

        self.clap_processor = (
            ClapProcessor.from_pretrained(
                AUDIO_MODEL_ID
            )
        )

        self.clap_model = (
            ClapModel.from_pretrained(
                AUDIO_MODEL_ID
            )
            .to(self.device)
            .eval()
        )

        print("[app] loading VideoMAE...")

        self.video_processor = (
            VideoMAEImageProcessor
            .from_pretrained(
                VIDEO_MODEL_ID
            )
        )

        self.video_model = (
            VideoMAEForPreTraining
            .from_pretrained(
                VIDEO_MODEL_ID
            )
            .to(self.device)
            .eval()
        )

        print(
            "[app] loading trained heads..."
        )

        self.heads = {
            modality: self._load_head(
                modality
            )
            for modality in (
                "audio",
                "visual",
                "multimodal",
            )
        }

        print("[app] ready")

    def _load_head(
        self,
        modality: str,
    ) -> EventClassifier:

        checkpoint_path = (
            MODEL_DIR
            / f"{modality}_seed{self.seed}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing trained classifier: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model = EventClassifier(
            input_dim=int(
                checkpoint["input_dim"]
            ),
            num_classes=int(
                checkpoint["num_classes"]
            ),
        )

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        return (
            model
            .to(self.device)
            .eval()
        )

    def _extract_audio(
        self,
        video_path: Path,
    ) -> np.ndarray:

        with tempfile.TemporaryDirectory(
            prefix="av_audio_"
        ) as tmp:

            wav_path = (
                Path(tmp)
                / "audio.wav"
            )

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                "0",
                "-i",
                str(video_path),
                "-t",
                str(MAX_SECONDS),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(wav_path),
            ]

            subprocess.run(
                command,
                check=True,
            )

            audio, _ = librosa.load(
                wav_path,
                sr=SAMPLE_RATE,
                mono=True,
            )

        if audio.size == 0:
            raise RuntimeError(
                "No audio stream could be "
                "extracted from the video."
            )

        return audio.astype(
            np.float32
        )

    def _extract_frames(
        self,
        video_path: Path,
    ) -> list[np.ndarray]:

        container = av.open(
            str(video_path)
        )

        segment_frames: list[
            np.ndarray
        ] = []

        for frame in container.decode(
            video=0
        ):
            frame_time = frame.time

            if frame_time is None:
                continue

            if (
                0.0
                <= frame_time
                <= MAX_SECONDS
            ):
                segment_frames.append(
                    frame.to_ndarray(
                        format="rgb24"
                    )
                )

        container.close()

        if not segment_frames:
            raise RuntimeError(
                "No video frames could be decoded."
            )

        indices = np.linspace(
            0,
            len(segment_frames) - 1,
            NUM_FRAMES,
        ).astype(int)

        return [
            segment_frames[i]
            for i in indices
        ]

    @torch.inference_mode()
    def _audio_embedding(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        inputs = self.clap_processor(
            audio=audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
            if isinstance(
                value,
                torch.Tensor,
            )
        }

        feature = (
            self.clap_model
            .get_audio_features(
                **inputs
            )
        )

        feature = (
            torch.nn.functional
            .normalize(
                feature,
                p=2,
                dim=1,
            )
        )

        feature = (
            feature
            .squeeze(0)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        if feature.shape != (512,):
            raise RuntimeError(
                "Unexpected CLAP feature "
                f"shape: {feature.shape}; "
                "expected (512,)"
            )

        return feature.astype(
            np.float32
        )

    @torch.inference_mode()
    def _visual_embedding(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray:

        inputs = self.video_processor(
            frames,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        output = (
            self.video_model
            .videomae(
                pixel_values=pixel_values
            )
        )

        feature = (
            output.last_hidden_state
            .mean(dim=1)
        )

        feature = (
            torch.nn.functional
            .normalize(
                feature,
                p=2,
                dim=1,
            )
        )

        feature = (
            feature
            .squeeze(0)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        if feature.shape != (768,):
            raise RuntimeError(
                "Unexpected VideoMAE feature "
                f"shape: {feature.shape}; "
                "expected (768,)"
            )

        return feature.astype(
            np.float32
        )

    @torch.inference_mode()
    def _predict_feature(
        self,
        modality: str,
        feature: np.ndarray,
    ) -> dict:

        x = (
            torch.from_numpy(
                feature
            )
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        logits = (
            self.heads[modality](x)
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[0]

        values, indices = torch.topk(
            probs,
            k=min(
                3,
                len(self.idx_to_class),
            ),
        )

        top3 = [
            {
                "class_name":
                    self.idx_to_class[
                        int(idx.item())
                    ],
                "confidence":
                    float(prob.item()),
            }
            for prob, idx
            in zip(
                values,
                indices,
            )
        ]

        return {
            "class_name":
                top3[0]["class_name"],
            "confidence":
                top3[0]["confidence"],
            "top3": top3,
        }

    def _prediction_bundle(
        self,
        audio_feature: np.ndarray,
        visual_feature: np.ndarray,
    ) -> dict:

        # Both embeddings are already L2-normalized,
        # exactly as in the stored research features.
        multimodal_feature = (
            np.concatenate(
                [
                    audio_feature,
                    visual_feature,
                ]
            )
            .astype(np.float32)
        )

        return {
            "audio":
                self._predict_feature(
                    "audio",
                    audio_feature,
                ),
            "visual":
                self._predict_feature(
                    "visual",
                    visual_feature,
                ),
            "multimodal":
                self._predict_feature(
                    "multimodal",
                    multimodal_feature,
                ),
        }

    def predict_video(
        self,
        video_path: Path,
        condition: str = "E0",
        sample_key: str = "upload",
    ) -> dict:

        spec = get_condition(
            condition
        )

        clean_audio = (
            self._extract_audio(
                video_path
            )
        )

        clean_frames = (
            self._extract_frames(
                video_path
            )
        )

        clean_audio_feature = (
            self._audio_embedding(
                clean_audio
            )
        )

        clean_visual_feature = (
            self._visual_embedding(
                clean_frames
            )
        )

        clean_predictions = (
            self._prediction_bundle(
                clean_audio_feature,
                clean_visual_feature,
            )
        )

        noise_seed = None

        if spec.experiment == "E0":
            degraded_predictions = (
                clean_predictions
            )

        else:
            (
                corrupted_audio,
                corrupted_frames,
                _,
                noise_seed,
            ) = apply_condition(
                clean_audio,
                clean_frames,
                experiment=(
                    spec.experiment
                ),
                sample_key=sample_key,
            )

            if (
                spec.audio_mode
                == "clean"
            ):
                degraded_audio_feature = (
                    clean_audio_feature
                )
            else:
                degraded_audio_feature = (
                    self._audio_embedding(
                        corrupted_audio
                    )
                )

            if (
                spec.video_mode
                == "clean"
            ):
                degraded_visual_feature = (
                    clean_visual_feature
                )
            else:
                degraded_visual_feature = (
                    self._visual_embedding(
                        corrupted_frames
                    )
                )

            degraded_predictions = (
                self._prediction_bundle(
                    degraded_audio_feature,
                    degraded_visual_feature,
                )
            )

        return {
            "seed": self.seed,
            "sample_key": sample_key,
            "noise_seed": noise_seed,
            "clip_seconds":
                MAX_SECONDS,
            "condition": {
                "experiment":
                    spec.experiment,
                "label":
                    spec.label,
                "audio_mode":
                    spec.audio_mode,
                "video_mode":
                    spec.video_mode,
                "snr_db":
                    spec.snr_db,
                "blur_sigma":
                    spec.blur_sigma,
            },
            "clean":
                clean_predictions,
            "degraded":
                degraded_predictions,
        }
