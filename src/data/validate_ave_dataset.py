from pathlib import Path
import csv
from collections import Counter

MANIFEST = Path("data/metadata/ave_manifest.csv")
VIDEO_DIR = Path("data/raw/ave/videos")
OUTPUT = Path("data/metadata/ave_manifest_validated.csv")

rows = []

missing = []
found = []

with MANIFEST.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        video_id = row["video_id"]

        video_path = VIDEO_DIR / f"{video_id}.mp4"

        row["video_path"] = str(video_path)

        if video_path.exists():
            row["video_exists"] = "yes"
            found.append(video_id)
        else:
            row["video_exists"] = "no"
            missing.append(video_id)

        rows.append(row)


# Save validated manifest
fieldnames = list(rows[0].keys())

with OUTPUT.open("w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# Summary
unique_manifest_ids = {
    row["video_id"]
    for row in rows
}

physical_videos = {
    path.stem
    for path in VIDEO_DIR.glob("*.mp4")
}

split_counts = Counter(
    row["split"]
    for row in rows
)

print()
print("==============================")
print("AVE DATASET VALIDATION")
print("==============================")

print("Manifest samples:", len(rows))
print("Unique manifest video IDs:", len(unique_manifest_ids))

print()
print("Physical MP4 files:", len(physical_videos))

print()
print("Train samples:", split_counts["train"])
print("Validation samples:", split_counts["val"])
print("Test samples:", split_counts["test"])

print()
print("Missing sample references:", len(missing))

missing_unique = sorted(set(missing))

print("Unique missing videos:", len(missing_unique))

if missing_unique:
    print()
    print("Missing video IDs:")
    for video_id in missing_unique[:20]:
        print(" -", video_id)

extra_videos = sorted(
    physical_videos - unique_manifest_ids
)

print()
print("Extra physical videos:", len(extra_videos))

if extra_videos:
    print()
    print("Example extra videos:")
    for video_id in extra_videos[:20]:
        print(" -", video_id)

print()
print("Validated manifest saved to:")
print(OUTPUT)

if len(missing_unique) == 0:
    print()
    print("SUCCESS: All manifest videos were found.")
