from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from .corruptions import (
    CONDITIONS,
    get_condition,
)
from .inference import (
    ResearchInference,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MANIFEST = (
    PROJECT_ROOT
    / "data/metadata/ave_manifest_validated.csv"
)


app = FastAPI(
    title=(
        "Multimodal Audio-Visual "
        "Understanding API"
    ),
    version="0.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: (
    ResearchInference
    | None
) = None


def get_engine() -> ResearchInference:
    global _engine

    if _engine is None:
        _engine = (
            ResearchInference(
                seed=42
            )
        )

    return _engine


def resolve_sample_key(
    filename: str,
    explicit_sample_id: str | None,
) -> tuple[str, str]:

    # Explicit sample ID always wins.
    if explicit_sample_id:
        return (
            explicit_sample_id.strip(),
            "explicit_sample_id",
        )

    video_stem = (
        Path(filename)
        .stem
    )

    # For AVE clips, automatically recover the
    # research sample_id when the uploaded file
    # matches a unique 0-10 second manifest row.
    if MANIFEST.exists():
        matches: list[str] = []

        with MANIFEST.open(
            "r",
            encoding="utf-8",
        ) as f:

            for row in csv.DictReader(f):

                try:
                    row_video_stem = (
                        Path(
                            row["video_path"]
                        )
                        .stem
                    )

                    start = float(
                        row["start_time"]
                    )
                    end = float(
                        row["end_time"]
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    row_video_stem
                    == video_stem
                    and abs(start - 0.0)
                    < 1e-9
                    and abs(end - 10.0)
                    < 1e-9
                ):
                    matches.append(
                        row["sample_id"]
                    )

        unique_matches = sorted(
            set(matches)
        )

        if len(unique_matches) == 1:
            return (
                unique_matches[0],
                "ave_manifest",
            )

    # Generic uploads remain deterministic.
    return (
        video_stem,
        "filename_stem",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": (
            "multimodal-audio-visual-"
            "understanding"
        ),
        "version": "0.2.1",
    }


@app.get("/conditions")
def conditions():
    return [
        {
            "experiment":
                key,
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
        }
        for key, spec
        in CONDITIONS.items()
    ]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    condition: str = Form("E0"),
    sample_id: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "No file name supplied."
            ),
        )

    suffix = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if suffix not in {
        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".webm",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a video file: "
                "mp4, mov, m4v, avi, "
                "or webm."
            ),
        )

    try:
        get_condition(
            condition
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    sample_key, sample_key_source = (
        resolve_sample_key(
            file.filename,
            sample_id,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="av_demo_"
    ) as tmp:

        input_path = (
            Path(tmp)
            / f"input{suffix}"
        )

        with input_path.open(
            "wb"
        ) as f:
            shutil.copyfileobj(
                file.file,
                f,
            )

        try:
            result = (
                get_engine()
                .predict_video(
                    input_path,
                    condition=condition,
                    sample_key=sample_key,
                )
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    return {
        "filename":
            file.filename,
        "sample_key_source":
            sample_key_source,
        **result,
    }
