from huggingface_hub import snapshot_download

REPO_ID = "Helios1208/AVE_Dataset"

print("Downloading AVE metadata...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        "Annotations.txt",
        "trainSet.txt",
        "valSet.txt",
        "testSet.txt",
    ],
    local_dir="data/metadata",
)

print("AVE metadata downloaded successfully.")
