import torch
import numpy as np

from transformers import AutoProcessor, AutoModel

MODEL_ID = "laion/clap-htsat-unfused"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)
print("Loading CLAP...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

model = model.to(device)
model.eval()

# Create a 10-second silent test waveform at 48 kHz
sample_rate = 48000
duration = 10

audio = np.zeros(sample_rate * duration, dtype=np.float32)

inputs = processor(
    audios=audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
    if torch.is_tensor(value)
}

with torch.no_grad():
    audio_features = model.get_audio_features(**inputs)

print("CLAP loaded successfully.")
print("Embedding shape:", audio_features.shape)
print("Embedding device:", audio_features.device)
