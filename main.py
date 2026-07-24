"""FastAPI-Backend für die Pneumonie-Web-App.

Setzt den Vertrag aus webapp/API_CONTRACT.md um:
  GET  /api/health
  GET  /api/limits
  GET  /api/samples          (+ /api/samples/{id}/thumb)
  POST /api/analyze          (Datei-Upload ODER {"sample_id": ...})
  GET  /api/jobs/{job_id}

Kernideen:
  * EIN einziger Worker-Thread verarbeitet immer nur EIN Bild gleichzeitig.
    Alle Anfragen landen in einer Warteschlange (queue.Queue) und werden der
    Reihe nach abgearbeitet. So kann kein Ansturm mehrerer Nutzer den Server
    überlasten - selbst bei 1 GB RAM läuft nie mehr als eine Inferenz parallel.
  * Rate-Limiting im Arbeitsspeicher (pro X-Client-Id, festes Zeitfenster).
  * Das Modell wird EINMAL beim Start geladen, nicht pro Anfrage.
"""

import base64
import io
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Wiederverwendung deiner bestehenden Module
from data import transform
from model import build_model

# --------------------------------------------------------------------------
# Konfiguration (per Umgebungsvariable überschreibbar)
# --------------------------------------------------------------------------
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "best_model.pth")
TEST_DIR = os.getenv("TEST_DIR", "data/chest_xray/test")
CLASSES = ["NORMAL", "PNEUMONIA"]
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))       # Entscheidungsschwelle für PNEUMONIA
SAMPLES_PER_CLASS = int(os.getenv("SAMPLES_PER_CLASS", "4"))

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))         # max. Analysen pro Fenster
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "3600"))     # Fensterlänge in Sekunden
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
JOB_TTL = int(os.getenv("JOB_TTL", "300"))              # fertige Jobs nach 5 min vergessen
ALLOWED_TYPES = {"image/png", "image/jpeg"}

torch.set_num_threads(1)  # begrenzt CPU/RAM auf schwacher Hardware

# --------------------------------------------------------------------------
# Modell + Grad-CAM einmalig laden
# --------------------------------------------------------------------------
print("Lade Modell ...")
model = build_model(pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
print("Modell geladen.")


def _now_iso(ts: float) -> str:
    """Unix-Zeit -> ISO-8601 in UTC, z.B. 2026-07-23T18:30:00Z."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _png_base64(rgb_uint8: np.ndarray) -> str:
    """Ein RGB-uint8-Array als base64-kodiertes PNG (ohne data:-Präfix)."""
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_inference(pil_img: Image.Image) -> dict:
    """Klassifikation + Grad-CAM für ein Bild. Läuft NUR im Worker-Thread."""
    t0 = time.time()
    input_tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    prob_pneu = float(probs[1])                      # Index 1 = PNEUMONIA (wie in evaluate.py)
    prediction = "PNEUMONIA" if prob_pneu >= THRESHOLD else "NORMAL"

    # Grad-CAM (braucht Gradienten, daher kein no_grad)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # uint8 RGB

    return {
        "prediction": prediction,
        "probability": round(prob_pneu, 4),
        "threshold": THRESHOLD,
        "heatmap_png_base64": _png_base64(overlay),
        "inference_ms": int((time.time() - t0) * 1000),
    }


# --------------------------------------------------------------------------
# Beispielbilder registrieren (feste Auswahl aus dem Test-Ordner)
# --------------------------------------------------------------------------
def _build_sample_registry() -> dict:
    registry = {}
    for cls in CLASSES:
        folder = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for i, fname in enumerate(files[:SAMPLES_PER_CLASS], start=1):
            sid = f"{cls.lower()}_{i:02d}"
            path = os.path.join(folder, fname)
            # Thumbnail einmalig erzeugen und als PNG-Bytes vorhalten (nur wenige Bilder)
            img = Image.open(path).convert("RGB")
            img.thumbnail((160, 160))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            registry[sid] = {
                "id": sid,
                "label": f"{cls.title()} X-ray #{i}",
                "category": cls,
                "path": path,
                "thumb_png": buf.getvalue(),
            }
    return registry


SAMPLES = _build_sample_registry()
print(f"{len(SAMPLES)} Beispielbilder registriert.")

# --------------------------------------------------------------------------
# Job-Verwaltung + Warteschlange
# --------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work_queue: "queue.Queue[str]" = queue.Queue()
_seq_counter = 0


def _next_seq() -> int:
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


def _position_of(job: dict) -> int:
    """Wie viele Jobs müssen vor diesem noch laufen? (1 = als nächstes dran)."""
    ahead = 0
    for j in jobs.values():
        if j["status"] == "processing":
            ahead += 1
        elif j["status"] == "queued" and j["seq"] < job["seq"]:
            ahead += 1
    return ahead + 1


def _purge_old_jobs() -> None:
    """Fertige/fehlerhafte Jobs nach JOB_TTL entfernen (spart RAM)."""
    now = time.time()
    dead = [
        jid for jid, j in jobs.items()
        if j["status"] in ("done", "error") and now - j.get("finished_at", now) > JOB_TTL
    ]
    for jid in dead:
        jobs.pop(jid, None)


def _worker() -> None:
    """Endlosschleife: nimmt einen Job aus der Queue und verarbeitet ihn."""
    while True:
        job_id = work_queue.get()
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:            # inzwischen entfernt
                work_queue.task_done()
                continue
            job["status"] = "processing"
            img = job.pop("image")     # Bild aus dem Job herausnehmen und verarbeiten
        try:
            result = run_inference(img)
            with jobs_lock:
                job["status"] = "done"
                job["result"] = result
                job["finished_at"] = time.time()
        except Exception as exc:  # noqa: BLE001
            with jobs_lock:
                job["status"] = "error"
                job["error"] = "inference_failed"
                job["message"] = str(exc)
                job["finished_at"] = time.time()
        finally:
            work_queue.task_done()


# Genau EIN Worker-Thread => immer nur eine Inferenz gleichzeitig
threading.Thread(target=_worker, daemon=True).start()

# --------------------------------------------------------------------------
# Rate-Limiting (fixes Zeitfenster pro Client)
# --------------------------------------------------------------------------
_rl: dict[str, dict] = {}
_rl_lock = threading.Lock()


def _rl_entry(client_id: str, now: float) -> dict:
    e = _rl.get(client_id)
    if e is None or now - e["start"] >= RATE_WINDOW:
        e = {"count": 0, "start": now}
        _rl[client_id] = e
    return e


def limits_for(client_id: str) -> dict:
    now = time.time()
    with _rl_lock:
        e = _rl_entry(client_id, now)
        remaining = max(0, RATE_LIMIT - e["count"])
        reset_at = e["start"] + RATE_WINDOW
    return {
        "limit": RATE_LIMIT,
        "remaining": remaining,
        "window_seconds": RATE_WINDOW,
        "reset_at": _now_iso(reset_at),
    }


def consume_quota(client_id: str):
    """Verbraucht 1 Analyse. Gibt (ok, reset_at_iso) zurück."""
    now = time.time()
    with _rl_lock:
        e = _rl_entry(client_id, now)
        if e["count"] >= RATE_LIMIT:
            return False, _now_iso(e["start"] + RATE_WINDOW)
        e["count"] += 1
        return True, _now_iso(e["start"] + RATE_WINDOW)


# --------------------------------------------------------------------------
# FastAPI-App
# --------------------------------------------------------------------------
app = FastAPI(title="Pneumonia Detection API")

# Im Dev-Betrieb läuft die UI auf :5173 (andere Origin) -> CORS erlauben.
# In Produktion liegt alles hinter nginx (gleiche Origin), dann ist das egal.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_id(x_client_id: str | None) -> str:
    return x_client_id or "anonymous"


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/api/limits")
def get_limits(x_client_id: str | None = Header(default=None)):
    return limits_for(_client_id(x_client_id))


@app.get("/api/samples")
def get_samples():
    return {
        "samples": [
            {
                "id": s["id"],
                "label": s["label"],
                "category": s["category"],
                "thumbnail_url": f"/api/samples/{s['id']}/thumb",
            }
            for s in SAMPLES.values()
        ]
    }


@app.get("/api/samples/{sample_id}/thumb")
def sample_thumb(sample_id: str):
    s = SAMPLES.get(sample_id)
    if s is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return Response(content=s["thumb_png"], media_type="image/png")


def _enqueue(image: Image.Image) -> dict:
    job_id = uuid.uuid4().hex
    with jobs_lock:
        _purge_old_jobs()
        job = {"job_id": job_id, "status": "queued", "seq": _next_seq(), "image": image}
        jobs[job_id] = job
        position = _position_of(job)
    work_queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "position": position}


@app.post("/api/analyze")
async def analyze(request: Request, x_client_id: str | None = Header(default=None)):
    client = _client_id(x_client_id)

    # 1) Rate-Limit prüfen und verbrauchen
    ok, reset_at = consume_quota(client)
    if not ok:
        return JSONResponse(
            {"error": "rate_limited", "message": "Analysis limit reached.", "reset_at": reset_at},
            status_code=429,
        )

    content_type = request.headers.get("content-type", "")

    # 2a) Datei-Upload (multipart/form-data)
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            return JSONResponse({"error": "invalid_request", "message": "No file field."}, status_code=400)
        if upload.content_type not in ALLOWED_TYPES:
            return JSONResponse({"error": "unsupported_type", "message": "Use PNG or JPEG."}, status_code=415)
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "file_too_large", "message": "Max 10 MB."}, status_code=413)
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid_request", "message": "Not a valid image."}, status_code=400)
        return JSONResponse(_enqueue(image), status_code=202)

    # 2b) Server-Beispielbild ({"sample_id": "..."})
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid_request", "message": "Expected JSON or file."}, status_code=400)
    sample_id = (body or {}).get("sample_id")
    s = SAMPLES.get(sample_id)
    if s is None:
        return JSONResponse({"error": "invalid_request", "message": "Unknown sample_id."}, status_code=400)
    image = Image.open(s["path"]).convert("RGB")
    return JSONResponse(_enqueue(image), status_code=202)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "job_not_found"}, status_code=404)

        status = job["status"]
        if status == "queued":
            return {"job_id": job_id, "status": "queued", "position": _position_of(job)}
        if status == "processing":
            return {"job_id": job_id, "status": "processing"}
        if status == "done":
            return {"job_id": job_id, "status": "done", "result": job["result"]}
        # error
        return {
            "job_id": job_id,
            "status": "error",
            "error": job.get("error", "inference_failed"),
            "message": job.get("message", "Analysis failed."),
        }


if __name__ == "__main__":
    # Lokaler Start:  python main.py   (oder: uvicorn main:app --reload)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
