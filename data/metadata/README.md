# AVE Metadata

This project uses the Audio-Visual Event (AVE) dataset.

Dataset files are not included in this repository.

Expected local metadata files:

- `Annotations.txt`
- `trainSet.txt`
- `valSet.txt`
- `testSet.txt`
- `ave_manifest.csv`
- `ave_manifest_validated.csv`

The AVE dataset and official annotations should be obtained from the original
dataset source according to its distribution terms.

The manifest files are generated locally during dataset preparation and are
used by the training, robustness, and evaluation pipelines.

## Local structure

```text
data/
└── metadata/
    ├── Annotations.txt
    ├── trainSet.txt
    ├── valSet.txt
    ├── testSet.txt
    ├── ave_manifest.csv
    └── ave_manifest_validated.csv
