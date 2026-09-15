import av
import numpy as np
import torch

from huggingface_hub import hf_hub_download
from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining

MODEL_ID = "MCG-NJU/videomae-base"
NUM_FRAMES = 16


def load_video_frames(video_path, num_frames=16):
    container = av.open(video_path)

    frames = []

    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))

    container.close()

    print("Total decoded frames:", len(frames))

    # Uniformly select exactly 16 frames
    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames
    ).astype(int)

    sampled_frames = [frames[i] for i in indices]

    print("Sampled frames:", len(sampled_frames))

    return sampled_frames


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

print("Getting test video...")

video_path = hf_hub_download(
    repo_id="nielsr/video-demo",
    filename="eating_spaghetti.mp4",
    repo_type="dataset",
)

print("Video:", video_path)

print("Decoding and sampling video...")

frames = load_video_frames(
    video_path,
    num_frames=NUM_FRAMES
)

print("Loading VideoMAE processor...")

processor = VideoMAEImageProcessor.from_pretrained(
    MODEL_ID
)

print("Loading VideoMAE model...")

model = VideoMAEForPreTraining.from_pretrained(
    MODEL_ID
)

model = model.to(device)
model.eval()

inputs = processor(
    frames,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"].to(device)

print("Input shape:", pixel_values.shape)

with torch.no_grad():

    # Use only the pretrained VideoMAE encoder
    outputs = model.videomae(
        pixel_values=pixel_values
    )

hidden_states = outputs.last_hidden_state

# Convert all visual tokens into one video representation
video_embedding = hidden_states.mean(dim=1)

print()
print("VideoMAE loaded successfully.")
print("Hidden-state shape:", hidden_states.shape)
print("Video embedding shape:", video_embedding.shape)
print("Embedding device:", video_embedding.device)

