"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";

type Prediction = {
  class_name: string;
  confidence: number;
  top3: Array<{
    class_name: string;
    confidence: number;
  }>;
};

type PredictionBundle = {
  audio: Prediction;
  visual: Prediction;
  multimodal: Prediction;
};

type Condition = {
  experiment: string;
  label: string;
  audio_mode: string;
  video_mode: string;
  snr_db: number | null;
  blur_sigma: number | null;
};

type ApiResult = {
  filename: string;
  seed: number;
  corruption_seed: number;
  clip_seconds: number;
  condition: Condition;
  clean: PredictionBundle;
  degraded: PredictionBundle;
};

type StudyAccuracy = {
  audio: number;
  visual: number;
  multimodal: number;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const CONDITIONS = [
  { id: "E0", label: "E0 · Clean", detail: "No degradation. This is the baseline comparison." },
  { id: "E1", label: "E1 · Audio noise · 20 dB", detail: "Mild additive noise is applied to the audio track." },
  { id: "E2", label: "E2 · Audio noise · 10 dB", detail: "Moderate additive noise is applied to the audio track." },
  { id: "E3", label: "E3 · Audio noise · 0 dB", detail: "Severe additive noise is applied to the audio track." },
  { id: "E4", label: "E4 · Video blur · σ = 1", detail: "A light Gaussian blur is applied to the video frames." },
  { id: "E5", label: "E5 · Video blur · σ = 2", detail: "A moderate Gaussian blur is applied to the video frames." },
  { id: "E6", label: "E6 · Video blur · σ = 4", detail: "A strong Gaussian blur is applied to the video frames." },
  { id: "E7", label: "E7 · Missing audio", detail: "The audio stream is removed so only visual evidence remains." },
  { id: "E8", label: "E8 · Missing video", detail: "The video stream is removed so only audio evidence remains." },
  { id: "E9", label: "E9 · Audio 0 dB + Video blur σ = 4", detail: "Severe audio noise is combined with strong visual blur." },
];

const STUDY_ACCURACY: Record<string, StudyAccuracy> = {
  E0: { audio: 0.8615257048, visual: 0.5033167496, multimodal: 0.8673300166 },
  E1: { audio: 0.818408, visual: 0.503317, multimodal: 0.840796 },
  E2: { audio: 0.752902, visual: 0.503317, multimodal: 0.781095 },
  E3: { audio: 0.578773, visual: 0.503317, multimodal: 0.619403 },
  E4: { audio: 0.861526, visual: 0.501658, multimodal: 0.870647 },
  E5: { audio: 0.861526, visual: 0.415423, multimodal: 0.869818 },
  E6: { audio: 0.861526, visual: 0.290216, multimodal: 0.868159 },
  E7: { audio: 0.044776, visual: 0.503317, multimodal: 0.109453 },
  E8: { audio: 0.861526, visual: 0.024876, multimodal: 0.832504 },
  E9: { audio: 0.578773, visual: 0.290216, multimodal: 0.591211 },
};

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function isVideoFile(item: File) {
  return item.type.startsWith("video/") || /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(item.name);
}

function ResultCard({
  title,
  prediction,
}: {
  title: string;
  prediction: Prediction;
}) {
  return (
    <div className="resultCard">
      <div className="eyebrow">{title}</div>
      <h2>{prediction.class_name}</h2>
      <div className="confidenceHead">
        <span>Confidence</span>
        <strong>{pct(prediction.confidence)}</strong>
      </div>
      <div className="meter" aria-hidden="true">
        <span style={{ width: `${prediction.confidence * 100}%` }} />
      </div>

      <div className="top3">
        {prediction.top3.map((item, index) => (
          <div className="topRow" key={`${item.class_name}-${index}`}>
            <span>
              {index + 1}. {item.class_name}
            </span>
            <strong>{pct(item.confidence)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function StudyMetric({
  label,
  clean,
  selected,
}: {
  label: string;
  clean: number;
  selected: number;
}) {
  const drop = clean - selected;

  const dropPp =
  Math.abs((clean - selected) * 100) < 0.05
    ? 0
    : (clean - selected) * 100;

  return (
    <div className="studyMetric">
      <div>
        <div className="studyLabel">{label}</div>
        <div className="studyNumbers">
          {pct(clean)} <span>→</span> {pct(selected)}
        </div>
      </div>

      <div
        className={`studyDrop ${
          drop > 0.001 ? "negative" : drop < -0.001 ? "positive" : ""
        }`}
      >
        Δ {drop.toFixed(1)} pp
      </div>
    </div>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [condition, setCondition] = useState("E0");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const selectedStudy = useMemo(
    () => STUDY_ACCURACY[condition],
    [condition]
  );

  const selectedCondition = useMemo(
    () => CONDITIONS.find((item) => item.id === condition) ?? CONDITIONS[0],
    [condition]
  );

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : ""),
    [file]
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(nextFile: File | null) {
    if (nextFile && !isVideoFile(nextFile)) {
      setError("Please drop a video file.");
      return;
    }

    setFile(nextFile);
    setResult(null);
    setError("");
  }

  function fileFromDataTransfer(dataTransfer: DataTransfer | null) {
    const items = Array.from(dataTransfer?.files ?? []);
    return items.find(isVideoFile) ?? items[0] ?? null;
  }

  async function analyze() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("condition", condition);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const body = await response.json();

      if (!response.ok) {
        throw new Error(body.detail ?? "Prediction failed.");
      }

      setResult(body);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Prediction failed."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="appShell">
      <header className="siteHeader">
        <div className="headerInner">
          <div className="brand">
            <Image
              className="brandLogo"
              src="/logo.png"
              alt="Abu Dhabi University"
              width={1120}
              height={336}
              priority
            />
          </div>
          <div className="headerMeta">
            <strong>Research Demonstrator</strong>
            <span>Done by student: 1109499</span>
          </div>
        </div>
      </header>

      <main className="page">
        <section className="hero">
          <div className="badge">Research Demonstrator · Version 2</div>
          <h1>Multimodal Audio-Visual Understanding</h1>
          <p>
            Compare audio-only, visual-only, and audio-visual predictions
            under the same clean and robustness conditions evaluated in the
            research study.
          </p>
        </section>

        <section className="controlPanel">
          <div className="formSection">
            <div className="formHeading">
              <span className="step">01</span>
              <div>
                <h2>Upload video</h2>
                <p>Drop a short clip or browse locally. The model uses up to the first 10 seconds.</p>
              </div>
            </div>

            <label
              className={`fileBox${isDragging ? " isDragging" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                event.dataTransfer.dropEffect = "copy";
                setIsDragging(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                if (!event.currentTarget.contains(event.relatedTarget as Node)) {
                  setIsDragging(false);
                }
              }}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                selectFile(fileFromDataTransfer(event.dataTransfer));
              }}
            >
              <strong>
                {file
                  ? file.name
                  : isDragging
                    ? "Drop the video to upload"
                    : "Choose a video clip"}
              </strong>
              <span>
                {file
                  ? `${formatBytes(file.size)} · ${file.type || "video"}`
                  : "MP4, MOV, MKV, WebM, or other common video formats"}
              </span>
              <input
                type="file"
                accept="video/*"
                onChange={(event) => {
                  selectFile(event.target.files?.[0] ?? null);
                }}
              />
            </label>

            {previewUrl && (
              <div className="previewCard">
                <video src={previewUrl} controls playsInline />
              </div>
            )}
          </div>

          <div className="formSection">
            <div className="formHeading">
              <span className="step">02</span>
              <div>
                <h2>Evaluation condition</h2>
              </div>
            </div>

            <select
              id="condition"
              aria-label="Evaluation condition"
              className="conditionSelect"
              value={condition}
              onChange={(event) => {
                setCondition(event.target.value);
                setResult(null);
                setError("");
              }}
            >
              {CONDITIONS.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
            <p className="fieldNote">{selectedCondition.detail}</p>
          </div>

          <div className="formActions">
            <p className="hint">
              Classifier seed 42 · robustness conditions E0–E9.
            </p>
            <button
              className="analyzeButton"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading ? "Analyzing…" : "Analyze video"}
            </button>
          </div>
        </section>

        {error && <div className="error">{error}</div>}

        {result && (
          <>
            <section className="summary">
              <span>File: {result.filename}</span>
              <span>Seed: {result.seed}</span>
              <span>Condition: {result.condition.experiment}</span>
              <span>{result.condition.label}</span>
              <span>Window: up to {result.clip_seconds}s</span>
            </section>

            <section className="resultsSection">
              <div className="sectionHeading">
                <div>
                  <div className="eyebrow">Single-video inference</div>
                  <h2>Clean prediction</h2>
                </div>
                <p>
                  Prediction confidence is for this uploaded video, not
                  dataset-level accuracy.
                </p>
              </div>

              <div className="grid">
                <ResultCard title="Audio only" prediction={result.clean.audio} />
                <ResultCard title="Visual only" prediction={result.clean.visual} />
                <ResultCard
                  title="Audio + Visual"
                  prediction={result.clean.multimodal}
                />
              </div>
            </section>

            {result.condition.experiment !== "E0" && (
              <section className="resultsSection degradedSection">
                <div className="sectionHeading">
                  <div>
                    <div className="eyebrow">
                      {result.condition.experiment} · Robustness test
                    </div>
                    <h2>{result.condition.label}</h2>
                  </div>
                </div>

                <div className="grid">
                  <ResultCard
                    title="Audio only"
                    prediction={result.degraded.audio}
                  />
                  <ResultCard
                    title="Visual only"
                    prediction={result.degraded.visual}
                  />
                  <ResultCard
                    title="Audio + Visual"
                    prediction={result.degraded.multimodal}
                  />
                </div>
              </section>
            )}

            <section className="studyPanel">
              <div className="sectionHeading">
                <div>
                  <div className="eyebrow">Dataset-level research context</div>
                  <h2>
                    AVE test-set mean accuracy · E0 vs {condition}
                  </h2>
                </div>
                <p>
                  These are study-level mean accuracies across the three
                  experimental seeds, not confidence scores for this video.
                </p>
              </div>

              <div className="studyGrid">
                <StudyMetric
                  label="Audio only"
                  clean={STUDY_ACCURACY.E0.audio}
                  selected={selectedStudy.audio}
                />
                <StudyMetric
                  label="Visual only"
                  clean={STUDY_ACCURACY.E0.visual}
                  selected={selectedStudy.visual}
                />
                <StudyMetric
                  label="Audio + Visual"
                  clean={STUDY_ACCURACY.E0.multimodal}
                  selected={selectedStudy.multimodal}
                />
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="siteFooter">
        <div className="footerInner">
          <span>Abu Dhabi University · Research Demonstrator</span>
          <strong>Done by student: 1109499</strong>
        </div>
      </footer>
    </div>
  );
}
