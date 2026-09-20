from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "UnFaZeD07/AVE-Dataset"
OUTPUT_DIR = Path("data/raw/ave")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading AVE videos.zip ...")

zip_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename="videos.zip",
    local_dir=OUTPUT_DIR,
)

print("Downloaded to:")
print(zip_path)
