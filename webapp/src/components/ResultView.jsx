import { useState } from "react";

// Renders the finished analysis: verdict, probability, and the Grad-CAM heatmap
// blended over the original image with an opacity slider.
export function ResultView({ result, originalUrl, onReset }) {
  const [opacity, setOpacity] = useState(0.6);

  const isPneumonia = result.prediction === "PNEUMONIA";
  const pct = Math.round((result.probability ?? 0) * 100);
  const heatmapSrc = result.heatmap_png_base64
    ? `data:image/png;base64,${result.heatmap_png_base64}`
    : null;

  return (
    <section className="card result">
      <div className={"verdict " + (isPneumonia ? "verdict--pos" : "verdict--neg")}>
        <span className="verdict-icon" aria-hidden="true">
          {isPneumonia ? "⚠️" : "✓"}
        </span>
        <div>
          <p className="verdict-label">
            {isPneumonia ? "Signs of pneumonia detected" : "No signs of pneumonia"}
          </p>
          <p className="verdict-sub">
            Model probability of pneumonia: <strong>{pct}%</strong>
            {typeof result.threshold === "number" && (
              <span className="muted"> · decision threshold {Math.round(result.threshold * 100)}%</span>
            )}
          </p>
        </div>
      </div>

      <div className="prob-track" aria-hidden="true">
        <div
          className={"prob-fill " + (isPneumonia ? "prob-fill--pos" : "prob-fill--neg")}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="heatmap-area">
        <div className="heatmap-stack">
          {originalUrl && <img className="heatmap-base" src={originalUrl} alt="Original X-ray" />}
          {heatmapSrc && (
            <img
              className="heatmap-overlay"
              src={heatmapSrc}
              alt="Grad-CAM heatmap"
              style={{ opacity }}
            />
          )}
          {!heatmapSrc && <p className="muted">No heatmap returned for this image.</p>}
        </div>

        {heatmapSrc && (
          <label className="slider">
            <span>Heatmap intensity</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
            />
          </label>
        )}
        <p className="muted heatmap-caption">
          Warmer areas show where the model looked most when making its prediction
          (Grad-CAM). This is an explanation aid, not a diagnosis.
        </p>
      </div>

      <div className="result-footer">
        {typeof result.inference_ms === "number" && (
          <span className="muted">Inference: {result.inference_ms} ms</span>
        )}
        <button className="btn btn-primary" onClick={onReset}>
          Analyze another image
        </button>
      </div>
    </section>
  );
}
