// Shows live progress while a job is queued / processing.
export function QueueStatus({ status, position, onCancel }) {
  let label;
  if (status === "submitting") label = "Submitting…";
  else if (status === "queued")
    label =
      position && position > 0
        ? `Waiting in queue — position ${position}`
        : "Waiting in queue…";
  else if (status === "processing") label = "Analyzing the image…";
  else label = "Working…";

  return (
    <section className="card queue">
      <div className="spinner" aria-hidden="true" />
      <div className="queue-text">
        <p className="queue-label">{label}</p>
        <p className="muted">
          Only one image is processed at a time, so this can take a moment under load.
        </p>
      </div>
      {onCancel && (
        <button className="btn btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      )}
    </section>
  );
}
