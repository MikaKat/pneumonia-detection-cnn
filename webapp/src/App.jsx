import { useEffect, useRef, useState } from "react";
import {
  getHealth,
  getLimits,
  getSamples,
  analyzeFile,
  analyzeSample,
  pollJob,
} from "./api.js";
import { DisclaimerGate, DisclaimerBanner } from "./components/Disclaimer.jsx";
import { RateLimitBar } from "./components/RateLimitBar.jsx";
import { ImageSource } from "./components/ImageSource.jsx";
import { QueueStatus } from "./components/QueueStatus.jsx";
import { ResultView } from "./components/ResultView.jsx";

export default function App() {
  // phase: "idle" | "running" | "done" | "error"
  const [phase, setPhase] = useState("idle");
  const [selection, setSelection] = useState(null);

  const [limits, setLimits] = useState(null);
  const [samples, setSamples] = useState([]);
  const [samplesError, setSamplesError] = useState(null);
  const [backendDown, setBackendDown] = useState(false);

  const [jobStatus, setJobStatus] = useState(null); // submitting|queued|processing
  const [jobPosition, setJobPosition] = useState(null);
  const [result, setResult] = useState(null);
  const [errorInfo, setErrorInfo] = useState(null); // { message, code, reset_at }

  const abortRef = useRef(null);

  async function refreshLimits() {
    try {
      setLimits(await getLimits());
    } catch {
      /* leave stale limits */
    }
  }

  useEffect(() => {
    (async () => {
      try {
        await getHealth();
        setBackendDown(false);
      } catch {
        setBackendDown(true);
      }
      refreshLimits();
      try {
        setSamples(await getSamples());
      } catch (e) {
        setSamplesError(e.message);
      }
    })();
  }, []);

  const outOfQuota = limits && limits.remaining <= 0;
  const canAnalyze = selection && phase !== "running" && !outOfQuota;

  async function handleAnalyze() {
    if (!canAnalyze) return;
    setErrorInfo(null);
    setResult(null);
    setPhase("running");
    setJobStatus("submitting");
    setJobPosition(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const submit =
        selection.kind === "file"
          ? await analyzeFile(selection.file)
          : await analyzeSample(selection.sample.id);

      setJobStatus(submit.status || "queued");
      setJobPosition(submit.position ?? null);

      const finished = await pollJob(
        submit.job_id,
        ({ status, position }) => {
          setJobStatus(status);
          setJobPosition(position ?? null);
        },
        { signal: controller.signal }
      );

      setResult(finished.result);
      setPhase("done");
      refreshLimits();
    } catch (e) {
      if (e.name === "AbortError") {
        setPhase("idle");
        setJobStatus(null);
        return;
      }
      setErrorInfo({ message: e.message, code: e.code, reset_at: e.reset_at });
      setPhase("error");
      refreshLimits();
    } finally {
      abortRef.current = null;
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
  }

  function reset() {
    // Free any object URL created for an uploaded file preview.
    if (selection?.kind === "file" && selection.previewUrl) {
      URL.revokeObjectURL(selection.previewUrl);
    }
    setSelection(null);
    setResult(null);
    setErrorInfo(null);
    setJobStatus(null);
    setJobPosition(null);
    setPhase("idle");
  }

  function handleSelect(next) {
    // revoke previous upload preview if replacing it
    if (selection?.kind === "file" && selection.previewUrl && selection.previewUrl !== next.previewUrl) {
      URL.revokeObjectURL(selection.previewUrl);
    }
    setSelection(next);
  }

  return (
    <DisclaimerGate>
      <div className="app">
        <DisclaimerBanner />

        <header className="app-header">
          <h1>Chest X-ray Pneumonia Detection</h1>
          <p className="subtitle">
            Educational demo of a ResNet-18 classifier with Grad-CAM explanations.
          </p>
        </header>

        {backendDown && (
          <div className="card banner-error">
            The analysis service is currently unreachable. You can still browse the page;
            please try again shortly.
          </div>
        )}

        <RateLimitBar limits={limits} />

        {phase !== "done" && (
          <ImageSource
            samples={samples}
            samplesError={samplesError}
            selection={selection}
            onSelect={handleSelect}
            disabled={phase === "running"}
          />
        )}

        {(phase === "idle" || phase === "error") && (
          <>
            {selection && (
              <section className="card preview-card">
                <img className="preview-img" src={selection.previewUrl} alt="Selected X-ray" />
              </section>
            )}

            {errorInfo && <ErrorNotice info={errorInfo} />}

            <div className="actions">
              <button className="btn btn-primary btn-lg" disabled={!canAnalyze} onClick={handleAnalyze}>
                {outOfQuota ? "Analysis limit reached" : "Run analysis"}
              </button>
              {selection && (
                <button className="btn btn-ghost" onClick={reset} disabled={phase === "running"}>
                  Clear
                </button>
              )}
            </div>
            {outOfQuota && (
              <p className="muted center">
                You've used all your analyses for now. The counter above shows when it resets.
              </p>
            )}
          </>
        )}

        {phase === "running" && (
          <QueueStatus status={jobStatus} position={jobPosition} onCancel={handleCancel} />
        )}

        {phase === "done" && result && (
          <ResultView result={result} originalUrl={selection?.previewUrl} onReset={reset} />
        )}

        <footer className="app-footer">
          <p>
            Portfolio project · not a medical device · no medical decisions should be based on
            this tool. Please do not upload identifiable patient data.
          </p>
        </footer>
      </div>
    </DisclaimerGate>
  );
}

function ErrorNotice({ info }) {
  let text = info.message || "Something went wrong.";
  if (info.code === "rate_limited") {
    text = "You've reached the analysis limit. Please wait for the counter above to reset.";
  } else if (info.code === "queue_full") {
    text = "The server is busy right now. Please try again in a moment.";
  } else if (info.code === "file_too_large") {
    text = "That image is too large. Please use a file under 10 MB.";
  } else if (info.code === "unsupported_type") {
    text = "Unsupported file type. Please use PNG or JPEG.";
  }
  return <div className="card banner-error">{text}</div>;
}
