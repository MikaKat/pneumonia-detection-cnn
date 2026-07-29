import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

// The preprocessing chain, drawn as a fork rather than a line.
//
// The upload branches into two rows. The top row is the lung finder, which is a
// dead end: it is computed on your image and then nothing reads it. The bottom
// row is the scored path, where each tile really is the input to the next.
//
// It used to be one row with an "explored" badge on the stages that are not
// scored. That failed in the most direct way possible: the author of the
// project read his own interface and concluded the crop was being classified. A
// row of identical arrows is a diagram, a badge is a label, and the diagram
// wins. So the geometry now carries the claim, and the badge only repeats it.
//
// `plan`     static description of the scored path (GET /api/pipeline .stages)
// `asides`   static description of the side results (.asides), may be empty
// `stages`   the steps that have arrived so far, in the order they arrived
// `running`  whether the job is still being processed
export function PipelineView({ plan, asides, stages, running }) {
  const [zoom, setZoom] = useState(null);

  const isAside = (s) => s.group === "aside";
  const byKey = new Map(stages.map((s) => [s.key, s]));

  // Progress is counted over the scored path alone. The lung mask arrives
  // between the upload and the model input, and counting it would advance the
  // "computing now" marker onto a step that has not run yet.
  const chainArrived = stages.filter((s) => !isAside(s));
  const chainSteps = (plan?.length ? plan : chainArrived).filter((s) => !isAside(s));
  const activeIndex = running ? chainArrived.length : -1;

  const build = (step, state) => ({ ...step, ...(byKey.get(step.key) || {}), state });

  const chainTiles = chainSteps.map((step, i) => {
    const arrived = byKey.get(step.key);
    let state = "pending";
    if (arrived?.skipped) state = "skipped";
    else if (arrived) state = "done";
    else if (i === activeIndex) state = "running";
    return build(step, state);
  });

  // The side branch hangs off the upload, so it is being computed once the
  // upload is through and it has not arrived yet.
  const asideSteps = (asides?.length
    ? asides
    : stages.filter(isAside)
  );
  const asideTiles = asideSteps.map((step) => {
    const arrived = byKey.get(step.key);
    let state = "pending";
    if (arrived?.skipped) state = "skipped";
    else if (arrived) state = "done";
    else if (running && chainArrived.length >= 1) state = "running";
    return build(step, state);
  });

  const [root, ...scoredRest] = chainTiles;
  const forked = Boolean(root) && asideTiles.length > 0 && scoredRest.length > 0;

  const openZoom = (tile) => tile.image_png_base64 && setZoom(tile);

  return (
    <section className="card pipeline">
      <div className="pipeline-head">
        <h2 className="pipeline-title">Processing chain</h2>
        <p className="muted pipeline-sub">
          {forked ? (
            <>
              The lower row is the scored path: each picture is the input to the next,
              and the classifier runs on the full image, exactly as it was trained. The
              upper row is computed on the same image and then{" "}
              <strong>goes nowhere</strong>.
            </>
          ) : (
            <>
              Every picture below is computed on your image, in this order, and each one
              is the input to the next. The classifier runs on the full image, exactly as
              it was trained: no cropping, no masking.
            </>
          )}
        </p>
      </div>

      {forked ? (
        <div className="fork">
          <div className="fork-root">
            <PipelineTile tile={root} onZoom={() => openZoom(root)} />
          </div>
          <div className="fork-rows">
            <Row tiles={asideTiles} onZoom={openZoom} variant="aside" first />
            <Row tiles={scoredRest} onZoom={openZoom} />
          </div>
        </div>
      ) : (
        <Row tiles={chainTiles} onZoom={openZoom} lead={false} />
      )}

      {zoom && <ZoomOverlay tile={zoom} onClose={() => setZoom(null)} />}
    </section>
  );
}

// One horizontal run of tiles. `lead` draws an arrow before the first tile,
// which is what turns a row into a branch of the fork rather than a start.
function Row({ tiles, onZoom, variant, first, lead = true }) {
  return (
    <ol
      className={
        "chain fork-row" +
        (variant === "aside" ? " fork-row--aside" : "") +
        (first ? " fork-row--first" : "")
      }
    >
      {tiles.map((tile, i) => (
        <li key={tile.key} className="chain-item">
          {(i > 0 || lead) && (
            <span
              className={"arrow" + (tile.state === "pending" ? " arrow--pending" : "")}
              aria-hidden="true"
            />
          )}
          <PipelineTile tile={tile} onZoom={() => onZoom(tile)} />
        </li>
      ))}
    </ol>
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
      onKeyDown={(e) => {
        if (!src || (e.key !== "Enter" && e.key !== " ")) return;
        // Space on a non-button element scrolls the page; the tile has to claim it.
        e.preventDefault();
        onZoom();
      }}
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
        {explored && <span className="chip chip--explored">not scored</span>}
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

  // Rendered into document.body, not into the pipeline card. A `position:
  // fixed` overlay is only fixed to the viewport while no ancestor has a
  // transform, filter or containment; any one of those turns the ancestor into
  // the containing block and lets ordinary page content paint over the dialog.
  // The card above it is animated, so this is not hypothetical.
  return createPortal(
    <div
      className="modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="zoom-title"
    >
      {/* Header and footer do not scroll. A stage image can be taller than the
          screen, and when it is, the only Close button used to end up below the
          fold inside the scrolled body with no way out in view. */}
      <div className="modal modal--zoom" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2 id="zoom-title">{tile.title}</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
        <div className="modal-scroll">
          <img
            className="zoom-img"
            src={`data:image/png;base64,${tile.image_png_base64}`}
            alt={tile.title}
          />
          <p className="muted">{tile.caption}</p>
          {tile.note && <p className="tile-note">{tile.note}</p>}
        </div>
        <div className="modal-foot">
          <button className="btn btn-primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
