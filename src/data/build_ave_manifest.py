from pathlib import Path
import csv
from collections import Counter

META_DIR = Path("data/metadata")
OUTPUT_FILE = META_DIR / "ave_manifest.csv"

SPLIT_FILES = {
    "train": META_DIR / "trainSet.txt",
    "val": META_DIR / "valSet.txt",
    "test": META_DIR / "testSet.txt",
}


def clean_time(value):
    """
    Convert values such as '10', '8.' or '8.0' safely to int.
    """
    value = value.strip().rstrip(".")
    return int(float(value))


def parse_line(line):
    line = line.strip()

    if not line:
        return None

    parts = [part.strip() for part in line.split("&")]

    # Skip header
    if parts[0].lower() == "category":
        return None

    if len(parts) < 5:
        raise ValueError(
            f"Expected 5 fields but found {len(parts)}:\n{line}"
        )

    return {
        "class_name": parts[0],
        "video_id": parts[1],
        "quality": parts[2],
        "start_time": clean_time(parts[3]),
        "end_time": clean_time(parts[4]),
    }


all_rows = []

for split_name, split_file in SPLIT_FILES.items():

    print(f"Reading {split_name}: {split_file}")

    with split_file.open("r", encoding="utf-8") as file:
        split_rows = []

        for line_number, line in enumerate(file, start=1):

            try:
                record = parse_line(line)
            except Exception as exc:
                raise RuntimeError(
                    f"Error in {split_file}, line {line_number}"
                ) from exc

            if record is None:
                continue

            record["split"] = split_name
            split_rows.append(record)

        print(f"  Samples: {len(split_rows)}")

        all_rows.extend(split_rows)


# Give every annotated sample a stable ID
for index, row in enumerate(all_rows):
    row["sample_id"] = f"ave_{index:05d}"


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

fieldnames = [
    "sample_id",
    "split",
    "class_name",
    "video_id",
    "quality",
    "start_time",
    "end_time",
]

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in all_rows:
        writer.writerow(row)


# --------------------------------------------------
# Validation
# --------------------------------------------------

split_counts = Counter(
    row["split"]
    for row in all_rows
)

classes = sorted(
    set(row["class_name"] for row in all_rows)
)

unique_videos = set(
    row["video_id"]
    for row in all_rows
)

print()
print("==============================")
print("AVE MANIFEST SUMMARY")
print("==============================")

print("Total samples:", len(all_rows))
print("Train:", split_counts["train"])
print("Validation:", split_counts["val"])
print("Test:", split_counts["test"])

print()
print("Number of classes:", len(classes))
print("Unique video IDs:", len(unique_videos))

print()
print("Classes:")
for index, class_name in enumerate(classes):
    print(f"{index:02d}: {class_name}")


# --------------------------------------------------
# Check source-video overlap between splits
# --------------------------------------------------

videos_by_split = {
    split: {
        row["video_id"]
        for row in all_rows
        if row["split"] == split
    }
    for split in ["train", "val", "test"]
}

train_val_overlap = (
    videos_by_split["train"]
    & videos_by_split["val"]
)

train_test_overlap = (
    videos_by_split["train"]
    & videos_by_split["test"]
)

val_test_overlap = (
    videos_by_split["val"]
    & videos_by_split["test"]
)

print()
print("==============================")
print("SPLIT VIDEO OVERLAP CHECK")
print("==============================")

print(
    "Train ↔ Validation:",
    len(train_val_overlap)
)

print(
    "Train ↔ Test:",
    len(train_test_overlap)
)

print(
    "Validation ↔ Test:",
    len(val_test_overlap)
)


print()
print("Manifest saved to:")
print(OUTPUT_FILE)

print()
print("SUCCESS.")
