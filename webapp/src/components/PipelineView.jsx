import { useEffect, useState } from "react";

// The preprocessing chain: one tile per step, joined by arrows, filling up from
// left to right while the backend works. Tiles are not animated on a timer -
// each one appears when its stage actually arrives from GET /api/jobs (see
// api.js `pollJob`), so the chain is a progress display, not a decoration.
//
// `plan`   static description of all steps (GET /api/pipeline), may be null
// `stages` the steps that have arrived so far, in order
// `running` whether the job is still being processed
export function PipelineView({ plan, stages, running }) {
  const [zoom, setZoom] = useState(null); // the tile shown enlarged

  // Fall back to the arrived stages if the static plan could not be loaded, so
  // the chain still renders (just without placeholders for what is still to come).
  const steps = plan?.length ? plan : stages;
  const byKey = new Map(stages.map((s) => [s.key, s]));
  const doneCount = stages.length;

  // The first step that has no stage yet is the one currently being computed.
  const activeIndex = running ? doneCount : -1;

  const tiles = steps.map((step, i) => {
    const arrived = byKey.get(step.key);
    let state = "pending";
    if (arrived?.skipped) state = "skipped";
    else if (arrived) state = "done";
    else if (i === activeIndex) state = "running";
    return { ...step, ...(arrived || {}), state };
  });

  return (
    <section className="card pipeline">
      <div className="pipeline-head">
        <h2 className="pipeline-title">Processing chain</h2>
        <p className="muted pipeline-sub">
          Every picture below is computed on your image, in this order. Steps marked{" "}
          <span className="chip chip--explored">explored</span> were built and measured
          during development but are <strong>not</strong> part of the scored path. The
          classifier runs on the full image, exactly as it was trained.
        </p>
      </div>

      <ol className="chain">
        {tiles.map((tile, i) => (
          <li key={tile.key} className="chain-item">
            {i > 0 && (
              <span
                className={"arrow" + (tile.state === "pending" ? " arrow--pending" : "")}
                aria-hidden="true"
              />
            )}
            <PipelineTile tile={tile} onZoom={() => tile.image_png_base64 && setZoom(tile)} />
          </li>
        ))}
      </ol>

      {zoom && <ZoomOverlay tile={zoom} onClose={() => setZoom(null)} />}
    </section>
  );
}

function PipelineTile({ tile, onZoom }) {
  const src = tile.image_png_base64 ? `data:image/png;base64,${tile.image_png_base64}` : null;
  const explored = tile.status === "explored";

  return (
    <figure
      className={
        "tile" +
        ` tile--${tile.state}` +
        (explored ? " tile--explored" : "") +
        (src ? " tile--clickable" : "")
      }
      onClick={src ? onZoom : undefined}
      role={src ? "button" : undefined}
      tabIndex={src ? 0 : undefined}
      onKeyDown={(e) => src && (e.key === "Enter" || e.key === " ") && onZoom()}
      title={src ? "Click to enlarge" : undefined}
    >
      <div className="tile-frame">
        {src && <img src={src} alt={tile.title} />}
        {!src && tile.state === "running" && <div className="spinner spinner--sm" />}
        {!src && tile.state === "skipped" && <span className="tile-skip">skipped</span>}
        {!src && tile.state === "pending" && <span className="tile-wait" aria-hidden="true" />}
      </div>
      <figcaption className="tile-caption">
        <span className="tile-title">{tile.title}</span>
        {explored && <span className="chip chip--explored">explored</span>}
        {typeof tile.ms === "number" && <span className="tile-ms">{tile.ms} ms</span>}
        {tile.state === "skipped" && tile.reason && (
          <span className="tile-reason">{tile.reason}</span>
        )}
      </figcaption>
    </figure>
  );
}

function ZoomOverlay({ tile, onClose }) {
  // Escape closes, like the disclaimer modal.
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal modal--zoom" onClick={(e) => e.stopPropagation()}>
        <h2>{tile.title}</h2>
        <img
          className="zoom-img"
          src={`data:image/png;base64,${tile.image_png_base64}`}
          alt={tile.title}
        />
        <p className="muted">{tile.caption}</p>
        {tile.note && <p className="tile-note">{tile.note}</p>}
        <div className="actions">
          <button className="btn btn-primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
