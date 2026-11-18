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

type HrarPrediction = Prediction & {
  tau: number;
  routed_expert: "audio" | "visual" | "fusion";
  selected_strategy: string;
  reason: string;
  audio_available: boolean;
  video_available: boolean;
  audio_confidence: number;
  visual_confidence: number;
  confidence_gap: number;
};

type PredictionBundle = {
  audio: Prediction;
  visual: Prediction;
  multimodal: Prediction;
  hrar: HrarPrediction;
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
  clip_seconds: number;
  true_class?: string | null;
  sample_key?: string;
  condition: Condition;
  clean: PredictionBundle;
  degraded: PredictionBundle;
};

type StudyAccuracy = {
  audio: number;
  visual: number;
  fusion: number;
  hrar: number;
};

type MatrixRow = {
  experiment: string;
  label: string;
  audio: Prediction;
  visual: Prediction;
  fusion: Prediction;
  hrar: {
    class_name: string;
    confidence: number;
    routed_expert: string;
    reason: string;
  };
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "/api";

const GITHUB_URL =
  "https://github.com/me-nwa/multimodal-audio-visual-understanding";

const CONDITIONS = [
  { id: "E0", label: "E0 · Clean", detail: "No degradation. This is the baseline comparison." },
  { id: "E1", label: "E1 · Audio noise · 20 dB", detail: "Mild additive noise is applied to the audio track." },
  { id: "E2", label: "E2 · Audio noise · 10 dB", detail: "Moderate additive noise is applied to the audio track." },
  { id: "E3", label: "E3 · Audio noise · 0 dB", detail: "Severe additive noise is applied to the audio track." },
  { id: "E4", label: "E4 · Video blur · σ = 1", detail: "A light Gaussian blur is applied to the video frames." },
  { id: "E5", label: "E5 · Video blur · σ = 2", detail: "A moderate Gaussian blur is applied to the video frames." },
  { id: "E6", label: "E6 · Video blur · σ = 4", detail: "A strong Gaussian blur is applied to the video frames." },
  { id: "E7", label: "E7 · Missing audio", detail: "The audio waveform is replaced with silence while the visual stream remains unchanged." },
  { id: "E8", label: "E8 · Missing video", detail: "The video stream is removed so only audio evidence remains." },
  { id: "E9", label: "E9 · Audio 0 dB + Video blur σ = 4", detail: "Severe audio noise is combined with strong visual blur." },
];

const STUDY_ACCURACY: Record<string, StudyAccuracy> = {
  E0: { audio: 86.15, visual: 50.33, fusion: 86.73, hrar: 86.82 },
  E1: { audio: 81.84, visual: 50.33, fusion: 84.08, hrar: 84.16 },
  E2: { audio: 75.29, visual: 50.33, fusion: 78.11, hrar: 78.28 },
  E3: { audio: 57.88, visual: 50.33, fusion: 61.94, hrar: 62.77 },
  E4: { audio: 86.15, visual: 50.17, fusion: 87.06, hrar: 87.06 },
  E5: { audio: 86.15, visual: 41.54, fusion: 86.98, hrar: 86.98 },
  E6: { audio: 86.15, visual: 29.02, fusion: 86.82, hrar: 86.82 },
  E7: { audio: 4.48, visual: 50.33, fusion: 10.95, hrar: 50.33 },
  E8: { audio: 86.15, visual: 2.49, fusion: 83.25, hrar: 86.15 },
  E9: { audio: 57.88, visual: 29.02, fusion: 59.12, hrar: 59.70 },
};

const EXAMPLES = [
  {
    id: "church_bell_e7",
    label: "Church bell · E7",
    condition: "E7",
    sample_id: "ave_03741",
    filename: "EsfJpLbBeb4.mp4",
  },
  {
    id: "church_bell_e3",
    label: "Church bell · E3",
    condition: "E3",
    sample_id: "ave_03741",
    filename: "EsfJpLbBeb4.mp4",
  },
];

function pct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function pctPoints(value: number) {
  return `${value.toFixed(2)}%`;
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function isVideoFile(item: File) {
  return item.type.startsWith("video/") || /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(item.name);
}

function apiError(body: { detail?: unknown }) {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail
      .map((item: { msg?: string }) => item.msg)
      .filter(Boolean)
      .join(" ");
  }
  return "";
}

function ResultCard({
  title,
  prediction,
  highlight = false,
  note,
}: {
  title: string;
  prediction: Prediction;
  highlight?: boolean;
  note?: string;
}) {
  return (
    <div className={`resultCard${highlight ? " hrarCard" : ""}`}>
      <div className="eyebrow">{title}</div>
      <h2>{prediction.class_name}</h2>
      <div className="confidenceHead">
        <span>Confidence</span>
        <strong>{pct(prediction.confidence)}</strong>
      </div>
      <div className="meter" aria-hidden="true">
        <span style={{ width: `${prediction.confidence * 100}%` }} />
      </div>
      {note && <div className="routeNote">{note}</div>}
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

function RoutingSignals({ hrar }: { hrar: HrarPrediction }) {
  return (
    <div className="signalRow">
      <span className={`signalChip${hrar.audio_available ? "" : " on"}`}>
        Audio {hrar.audio_available ? "present" : "missing"}
      </span>
      <span className={`signalChip${hrar.video_available ? "" : " on"}`}>
        Video {hrar.video_available ? "present" : "missing"}
      </span>
      <span className={`signalChip${hrar.confidence_gap >= hrar.tau ? " on" : ""}`}>
        Gap {hrar.confidence_gap.toFixed(2)} {hrar.confidence_gap >= hrar.tau ? "≥" : "<"} τ {hrar.tau.toFixed(2)}
      </span>
      <span className="signalChip">
        Routed to {hrar.routed_expert}
      </span>
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
  const dropPpRaw = clean - selected;
  const dropPp = Math.abs(dropPpRaw) < 0.005 ? 0 : dropPpRaw;

  return (
    <div className="studyMetric">
      <div>
        <div className="studyLabel">{label}</div>
        <div className="studyNumbers">
          {pctPoints(clean)} <span>→</span> {pctPoints(selected)}
        </div>
      </div>
      <div
        className={`studyDrop ${
          dropPp > 0.05 ? "negative" : dropPp < -0.05 ? "positive" : ""
        }`}
      >
        Δ {dropPp.toFixed(2)} pp
      </div>
    </div>
  );
}

function fallbackGap(result: ApiResult) {
  const experiment = result.condition.experiment;
  if (experiment !== "E7" && experiment !== "E8") return null;

  const fusion = result.degraded.multimodal;
  const expert =
    experiment === "E7" ? result.degraded.visual : result.degraded.audio;
  const expertName =
    experiment === "E7"
      ? "surviving visual expert"
      : "surviving audio expert";
  const truth = result.true_class;

  const tag = (prediction: Prediction) => {
    if (!truth) return prediction.class_name;
    return prediction.class_name === truth ? "correct" : "wrong class";
  };

  return `Fallback gap: fusion ${pct(fusion.confidence)} (${tag(fusion)}) vs. ${expertName} ${pct(expert.confidence)} (${tag(expert)})`;
}

function PredictionGrid({ bundle }: { bundle: PredictionBundle }) {
  return (
    <>
      <div className="grid">
        <ResultCard title="Audio only" prediction={bundle.audio} />
        <ResultCard title="Visual only" prediction={bundle.visual} />
        <ResultCard title="Fusion (B0)" prediction={bundle.multimodal} />
        <ResultCard
          title="HRAR (τ = 0.55)"
          prediction={bundle.hrar}
          highlight
          note={bundle.hrar.reason}
        />
      </div>
      <RoutingSignals hrar={bundle.hrar} />
    </>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [condition, setCondition] = useState("E0");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [matrix, setMatrix] = useState<MatrixRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [matrixLoading, setMatrixLoading] = useState(false);
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

  function selectFile(nextFile: File | null, nextSampleId: string | null = null) {
    if (nextFile && !isVideoFile(nextFile)) {
      setError("Please drop a video file.");
      return;
    }

    setFile(nextFile);
    setSampleId(nextSampleId);
    setResult(null);
    setMatrix(null);
    setError("");
  }

  function fileFromDataTransfer(dataTransfer: DataTransfer | null) {
    const items = Array.from(dataTransfer?.files ?? []);
    return items.find(isVideoFile) ?? items[0] ?? null;
  }

  async function analyze(
    nextFile = file,
    nextCondition = condition,
    nextSampleId = sampleId
  ) {
    if (!nextFile) return;

    setLoading(true);
    setError("");
    setResult(null);
    setMatrix(null);

    try {
      const formData = new FormData();
      formData.append("file", nextFile);
      formData.append("condition", nextCondition);
      if (nextSampleId) formData.append("sample_id", nextSampleId);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const body = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(apiError(body) || "Prediction failed.");
      }

      setResult(body);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Prediction failed.";
      setError(
        message === "Failed to fetch"
          ? "Cannot reach the prediction API. Start the FastAPI backend on port 8000, then open the local app instead of the Vercel site."
          : message
      );
    } finally {
      setLoading(false);
    }
  }

  async function loadExample(exampleId: string) {
    const example = EXAMPLES.find((item) => item.id === exampleId);
    if (!example) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/examples/${exampleId}/video`);
      if (!response.ok) {
        throw new Error(
          "Example clip is not available. Keep the local AVE videos under data/raw/ave/videos/."
        );
      }

      const blob = await response.blob();
      const nextFile = new File([blob], example.filename, {
        type: "video/mp4",
      });

      setCondition(example.condition);
      setFile(nextFile);
      setSampleId(example.sample_id);
      await analyze(nextFile, example.condition, example.sample_id);
    } catch (err) {
      setLoading(false);
      setError(err instanceof Error ? err.message : "Could not load example.");
    }
  }

  async function analyzeMatrix() {
    if (!file) return;

    setMatrixLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      if (sampleId) formData.append("sample_id", sampleId);

      const response = await fetch(`${API_URL}/predict_matrix`, {
        method: "POST",
        body: formData,
      });

      const body = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(apiError(body) || "Matrix evaluation failed.");
      }

      setMatrix(body.conditions);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Matrix evaluation failed."
      );
    } finally {
      setMatrixLoading(false);
    }
  }

  const gap = result ? fallbackGap(result) : null;

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
            <div className="headerLinks">
              <a href={GITHUB_URL} target="_blank" rel="noreferrer">
                GitHub
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="page">
        <section className="hero">
          <div className="badge">Research Demonstrator · Version 2</div>
          <h1>Multimodal Audio-Visual Understanding</h1>
          <p>
            Compare audio-only, visual-only, Fusion (B0), and HRAR (τ = 0.55)
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
                <p>Drop a short clip, browse locally, or load a paper example.</p>
              </div>
            </div>

            <div className="exampleRow">
              {EXAMPLES.map((example) => (
                <button
                  key={example.id}
                  className="exampleButton"
                  disabled={loading}
                  onClick={() => loadExample(example.id)}
                >
                  {example.label}
                </button>
              ))}
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
                setMatrix(null);
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
              Classifier seed 42 · robustness conditions E0–E9 · HRAR τ = 0.55.
            </p>
            <button
              className="analyzeButton"
              onClick={() => analyze()}
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
              {result.true_class && <span>True class: {result.true_class}</span>}
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
              <PredictionGrid bundle={result.clean} />
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
                {gap && <p className="fallbackGap"><strong>{gap}</strong></p>}
                <PredictionGrid bundle={result.degraded} />
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
                  These are study-level mean accuracies from Tables 9–11, not
                  confidence scores for this video.
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
                  label="Fusion (B0)"
                  clean={STUDY_ACCURACY.E0.fusion}
                  selected={selectedStudy.fusion}
                />
                <StudyMetric
                  label="HRAR (τ = 0.55)"
                  clean={STUDY_ACCURACY.E0.hrar}
                  selected={selectedStudy.hrar}
                />
              </div>
            </section>

            <section className="studyPanel">
              <div className="sectionHeading">
                <div>
                  <div className="eyebrow">Evaluation matrix</div>
                  <h2>All conditions · this clip</h2>
                </div>
                <button
                  className="secondaryButton"
                  onClick={analyzeMatrix}
                  disabled={matrixLoading || loading}
                >
                  {matrixLoading ? "Evaluating E0–E9…" : "Run E0–E9 on this clip"}
                </button>
              </div>

              {matrix && (
                <table className="matrixTable">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Audio only</th>
                      <th>Visual only</th>
                      <th>Fusion (B0)</th>
                      <th>HRAR (τ = 0.55)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {matrix.map((row) => (
                      <tr key={row.experiment}>
                        <td>
                          {row.experiment}
                          <div className="fieldNote">{row.label}</div>
                        </td>
                        <td>
                          {row.audio.class_name}
                          <div className="fieldNote">{pct(row.audio.confidence)}</div>
                        </td>
                        <td>
                          {row.visual.class_name}
                          <div className="fieldNote">{pct(row.visual.confidence)}</div>
                        </td>
                        <td>
                          {row.fusion.class_name}
                          <div className="fieldNote">{pct(row.fusion.confidence)}</div>
                        </td>
                        <td className="hrarCell">
                          {row.hrar.class_name}
                          <div className="fieldNote">
                            {pct(row.hrar.confidence)} · {row.hrar.reason}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          </>
        )}
      </main>

      <footer className="siteFooter">
        <div className="footerInner">
          <span>Abu Dhabi University · Research Demonstrator</span>
          <div className="footerLinks">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <strong>Done by student: 1109499</strong>
          </div>
        </div>
      </footer>
    </div>
  );
}
