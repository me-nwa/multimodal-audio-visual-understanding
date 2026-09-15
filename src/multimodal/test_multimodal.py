import subprocess
import tempfile

import av
import librosa
import numpy as np
import torch

from huggingface_hub import hf_hub_download
from transformers import (
    ClapModel,
    ClapProcessor,
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
)

CLAP_MODEL = "laion/clap-htsat-unfused"
VIDEO_MODEL = "MCG-NJU/videomae-base"

SAMPLE_RATE = 48000
NUM_FRAMES = 16


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Download test video
# --------------------------------------------------

print("\nGetting test video...")

video_path = hf_hub_download(
    repo_id="nielsr/video-demo",
    filename="eating_spaghetti.mp4",
    repo_type="dataset",
)

print("Video:", video_path)


# --------------------------------------------------
# AUDIO PIPELINE
# --------------------------------------------------

print("\n==============================")
print("AUDIO PIPELINE - CLAP")
print("==============================")

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

print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / SAMPLE_RATE, "seconds")

print("Loading CLAP...")

clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)

clap_model = ClapModel.from_pretrained(CLAP_MODEL)
clap_model = clap_model.to(device)
clap_model.eval()

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

print("Audio embedding shape:", audio_embedding.shape)


# Free CLAP before loading VideoMAE
del clap_model

if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# --------------------------------------------------
# VIDEO PIPELINE
# --------------------------------------------------

print("\n==============================")
print("VIDEO PIPELINE - VideoMAE")
print("==============================")

container = av.open(video_path)

frames = []

for frame in container.decode(video=0):
    frames.append(
        frame.to_ndarray(format="rgb24")
    )

container.close()

print("Total decoded frames:", len(frames))

indices = np.linspace(
    0,
    len(frames) - 1,
    NUM_FRAMES
).astype(int)

sampled_frames = [
    frames[index]
    for index in indices
]

print("Sampled frames:", len(sampled_frames))

print("Loading VideoMAE...")

video_processor = VideoMAEImageProcessor.from_pretrained(
    VIDEO_MODEL
)

video_model = VideoMAEForPreTraining.from_pretrained(
    VIDEO_MODEL
)

video_model = video_model.to(device)
video_model.eval()

video_inputs = video_processor(
    sampled_frames,
    return_tensors="pt"
)

pixel_values = video_inputs["pixel_values"].to(device)

print("Video input shape:", pixel_values.shape)

with torch.no_grad():

    outputs = video_model.videomae(
        pixel_values=pixel_values
    )

video_embedding = (
    outputs.last_hidden_state
    .mean(dim=1)
    .cpu()
)

print("Video embedding shape:", video_embedding.shape)


# --------------------------------------------------
# MULTIMODAL FUSION
# --------------------------------------------------

print("\n==============================")
print("MULTIMODAL REPRESENTATION")
print("==============================")

# L2-normalize each modality before fusion

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
        video_embedding,
    ],
    dim=1,
)

print("Audio representation:", audio_embedding.shape)
print("Visual representation:", video_embedding.shape)
print("Multimodal representation:", multimodal_embedding.shape)

print("\nSUCCESS: Audio and video were processed from the same video.")
