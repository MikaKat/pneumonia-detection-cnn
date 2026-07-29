import { useState } from "react";

// Renders the finished analysis.
//
// Deliberately NO verdict. The backend stopped sending a `prediction` field and
// this view stopped drawing one: in this project's external validation the
// operating threshold did not transfer, and half of the images the model called
// negative did have pneumonia (NPV 0.500, README section 6). A green "no signs
// of pneumonia" box would be the one statement the measurements contradict.
//
// What is shown instead: the probability, the range it moves through when the
// framing is nudged by a couple of percent, and what that does and does not mean.
export function ResultView({ result, stages = [], originalUrl, onReset }) {
  const [opacity, setOpacity] = useState(0.6);
  const [showViews, setShowViews] = useState(false);

  const u = result.uncertainty;
  const base = result.probability ?? 0;
  const median = u?.median ?? base;
  const lo = u?.min ?? base;
  const hi = u?.max ?? base;
  const spread = u?.spread ?? 0;

  const pct = (x) => Math.round(x * 100);
  // Where the score sits matters less than how far it travels. Ten points of
  // movement from a 2 % change in framing means the digits are not meaningful.
  const unstable = spread >= 0.1;

  const heatmapSrc = result.heatmap_png_base64
    ? `data:image/png;base64,${result.heatmap_png_base64}`
    : null;

  // The heatmap is rendered in the model's 224x224 geometry, where the image is
  // squeezed to a square. The layer underneath has to be the same picture in the
  // same geometry, otherwise the warm areas sit next to the anatomy they belong
  // to. The upload stage image is used for it (full resolution, same framing);
  // the thumbnail from the picker is only the fallback.
  const uploadStage = stages.find((s) => s.key === "upload" && s.image_png_base64);
  const baseSrc = uploadStage
    ? `data:image/png;base64,${uploadStage.image_png_base64}`
    : originalUrl;

  return (
    <section className="card result">
      <div className="score">
        <div className="score-head">
          <span className="score-label">Model score for pneumonia</span>
          <span className="score-value">{pct(median)}%</span>
        </div>

        {/* The band is the point of this display: a single tick would claim a
            precision the model does not have. */}
        <div className="score-track" aria-hidden="true">
          <div
            className="score-band"
            style={{ left: `${pct(lo)}%`, width: `${Math.max(pct(hi) - pct(lo), 1)}%` }}
          />
          <div className="score-mark" style={{ left: `${pct(median)}%` }} />
        </div>
        <div className="score-scale" aria-hidden="true">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>

        <p className="score-range">
          {u ? (
            <>
              Range{" "}
              <strong>
                {pct(lo)}% to {pct(hi)}%
              </strong>{" "}
              across {u.views.length} slightly different framings of the same image,{" "}
              {pct(spread)} points apart.{" "}
              {unstable
                ? "The score travels that far when the frame moves by two percent, so the individual digits carry little information here."
                : "The score barely moves under those changes, which says the number is stable, not that it is right."}
            </>
          ) : (
            <>Single pass, no stability estimate available.</>
          )}
        </p>

        {u && (
          <>
            <button className="btn btn-ghost btn-sm" onClick={() => setShowViews((v) => !v)}>
              {showViews ? "Hide the individual runs" : "Show the individual runs"}
            </button>
            {showViews && (
              <ul className="view-list">
                {u.views.map((v) => (
                  <li key={v.view}>
                    <span className="muted">{v.view}</span>
                    <span>{pct(v.probability)}%</span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>

      <div className="heatmap-area">
        <div className="heatmap-stack">
          {baseSrc && <img className="heatmap-base" src={baseSrc} alt="Original X-ray" />}
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
          Warmer areas show where the model looked most (Grad-CAM). Expect a diffuse blob
          rather than a sharp finding: measurements in this project showed that the evidence
          the model uses is spread out and partly outside the lungs. Read it as a
          plausibility check on the model, not as a marked lesion.
        </p>
      </div>

      {/* Last block in the card, under the score and the heatmap, so it is the
          note the reader leaves with. */}
      <div className="honesty">
        <p>
          <strong>No verdict is shown, on purpose.</strong> This is a research
          demonstrator, not a diagnostic tool, and it reports a probability rather than a
          label.
        </p>
        <p>
          A low score does not rule out pneumonia. When this project carried its decision
          threshold from one dataset to another,{" "}
          <strong>half of the images the model called negative did have pneumonia</strong>{" "}
          (NPV 0.500). The ranking transferred, the calibration did not, and turning this
          score into a yes or no would rest on exactly the part that failed.
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
