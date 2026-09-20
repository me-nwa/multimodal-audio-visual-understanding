import csv
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("data/metadata/ave_manifest_validated.csv")

video_records = defaultdict(list)

with MANIFEST.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        video_records[row["video_id"]].append(row)

duplicate_videos = {
    video_id: rows
    for video_id, rows in video_records.items()
    if len(rows) > 1
}

same_class = []
different_class = []

for video_id, rows in duplicate_videos.items():

    classes = {
        row["class_name"]
        for row in rows
    }

    if len(classes) == 1:
        same_class.append(video_id)
    else:
        different_class.append((video_id, rows))


print("==============================")
print("AVE DUPLICATE VIDEO AUDIT")
print("==============================")

print("Total samples:", sum(len(v) for v in video_records.values()))
print("Unique videos:", len(video_records))
print("Repeated video IDs:", len(duplicate_videos))

print()
print("Repeated videos with SAME class:", len(same_class))
print("Repeated videos with DIFFERENT classes:", len(different_class))

if different_class:
    print()
    print("Videos with multiple class labels:")

    for video_id, rows in different_class:

        print()
        print("Video:", video_id)

        for row in rows:
            print(
                " ",
                row["split"],
                "|",
                row["class_name"],
                "|",
                row["start_time"],
                "-",
                row["end_time"],
            )
