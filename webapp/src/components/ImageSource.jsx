import { useRef, useState } from "react";
import { apiUrl } from "../api.js";

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB (matches recommended server limit)
const ACCEPT = ["image/png", "image/jpeg"];

// Lets the user either upload their own X-ray or pick a server sample.
// Calls onSelect({ kind: "file", file, previewUrl } | { kind: "sample", sample, previewUrl }).
export function ImageSource({ samples, samplesError, selection, onSelect, disabled }) {
  const [tab, setTab] = useState("upload");
  const [localError, setLocalError] = useState(null);
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  function handleFile(file) {
    setLocalError(null);
    if (!file) return;
    if (!ACCEPT.includes(file.type)) {
      setLocalError("Please choose a PNG or JPEG image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setLocalError("That file is larger than 10 MB.");
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    onSelect({ kind: "file", file, previewUrl });
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    handleFile(e.dataTransfer.files?.[0]);
  }

  return (
    <section className="card">
      <div className="tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === "upload"}
          className={"tab" + (tab === "upload" ? " tab--active" : "")}
          onClick={() => setTab("upload")}
        >
          Upload image
        </button>
        <button
          role="tab"
          aria-selected={tab === "samples"}
          className={"tab" + (tab === "samples" ? " tab--active" : "")}
          onClick={() => setTab("samples")}
        >
          Sample images
        </button>
      </div>

      {tab === "upload" && (
        <div className="tabpanel">
          <div
            className={"dropzone" + (dragOver ? " dropzone--over" : "") + (disabled ? " dropzone--disabled" : "")}
            onDragOver={(e) => {
              e.preventDefault();
              if (!disabled) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !disabled && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (disabled || (e.key !== "Enter" && e.key !== " ")) return;
              // Space on a div scrolls the page; the dropzone has to claim it.
              e.preventDefault();
              inputRef.current?.click();
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg"
              hidden
              disabled={disabled}
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <p className="dropzone-title">Drop a chest X-ray here, or click to browse</p>
            <p className="muted">PNG or JPEG, up to 10 MB. No patient-identifying data.</p>
          </div>
          {selection?.kind === "file" && (
            <p className="selected-note">
              Selected: <strong>{selection.file.name}</strong>
            </p>
          )}
          {localError && <p className="error-text">{localError}</p>}
        </div>
      )}

      {tab === "samples" && (
        <div className="tabpanel">
          {samplesError && <p className="error-text">Could not load samples: {samplesError}</p>}
          {!samplesError && samples.length === 0 && <p className="muted">No sample images available.</p>}
          <div className="sample-grid">
            {samples.map((s) => {
              const active = selection?.kind === "sample" && selection.sample.id === s.id;
              return (
                <button
                  key={s.id}
                  className={"sample" + (active ? " sample--active" : "")}
                  disabled={disabled}
                  onClick={() =>
                    onSelect({ kind: "sample", sample: s, previewUrl: apiUrl(s.thumbnail_url) })
                  }
                  title={s.label}
                >
                  {s.thumbnail_url ? (
                    <img src={apiUrl(s.thumbnail_url)} alt={s.label} loading="lazy" />
                  ) : (
                    <div className="sample-placeholder">{s.label}</div>
                  )}
                  {s.category && <span className={`tag tag--${s.category.toLowerCase()}`}>{s.category}</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
