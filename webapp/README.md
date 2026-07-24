# Pneumonia Detection — Web UI

Front end for the chest X-ray pneumonia classifier. React + Vite, served as static
files by nginx which also reverse-proxies `/api` to your Python backend. Built to be
tiny (the production bundle is ~50 KB gzipped and nginx serving static files uses only
a few MB of RAM — your **1 GB budget is effectively all for the Python/model service**).

## What the UI does

- Upload your own chest X-ray (PNG/JPEG) **or** pick one of the server's sample images.
- Runs the model via the backend and shows the verdict (**pneumonia / no pneumonia**) with
  the probability and the decision threshold.
- Shows the **Grad-CAM heatmap** blended over the original, with an intensity slider.
- Reflects the backend's **rate limit** (a quota bar) and the **single-worker queue**
  (live "position in queue" while waiting) — see below.
- Shows the required **legal disclaimer** as a first-visit modal and a persistent banner.

The UI never blocks the browser on inference: it submits a job and **polls**, which is
exactly what lets your backend process **one image at a time** and queue the rest.

## The division of labour

You build (to learn): the Docker Compose stack, the Python server, and the Python↔web
communication. This folder is only the **web UI** and its own nginx container.

The UI talks to your backend through the endpoints in **[`API_CONTRACT.md`](./API_CONTRACT.md)**.
Implement those and the UI works unchanged. The important behaviors the UI relies on:

- **Rate limiting** is enforced by *your* server (keyed on the `X-Client-Id` header
  and/or IP). The UI reads `GET /api/limits` and reacts to HTTP `429`; it does not set policy.
- **Single-image processing + queue**: your server runs one inference at a time and
  returns a `job_id` + queue `position`. The UI polls `GET /api/jobs/{id}` and shows the
  position until the result is ready.
- **Heatmap** comes back base64-inline in the job result, so you don't have to persist
  or clean up result files (good for low RAM).

## Local development

```bash
cd webapp
npm install
npm run dev          # UI on http://localhost:5173
```

`npm run dev` proxies `/api` to `http://localhost:8000` by default. If your Python
backend listens elsewhere:

```bash
VITE_DEV_API_TARGET=http://localhost:9000 npm run dev
```

Copy `.env.example` to `.env` if you want to change `VITE_API_BASE`.

## Production build

```bash
npm run build        # outputs static files to dist/
npm run preview      # optional: preview the built bundle locally
```

## Docker (this UI container)

Multi-stage build → nginx serving `dist/` and proxying `/api` to your backend.

```bash
docker build -t pneumo-web .
docker run -p 8080:80 -e API_UPSTREAM=http://host.docker.internal:8000 pneumo-web
```

`API_UPSTREAM` is where nginx forwards `/api`. In a Compose stack it's your backend
service name, e.g. `http://backend:8000`.

### Fitting it into your Compose stack

Rough shape (yours will differ — you're building this part):

```yaml
services:
  backend:                     # your Python server + model
    build: .                   # repo root
    # no published port needed; only the web container talks to it
    mem_limit: 900m            # keep the model under your RAM budget

  web:                         # this folder
    build: ./webapp
    environment:
      API_UPSTREAM: http://backend:8000
    ports:
      - "80:80"
    depends_on:
      - backend
```

## Deploying with Coolify (deploy branch + git webhook)

1. **Push a `deploy` branch** to your git host:
   ```bash
   git checkout -b deploy && git push -u origin deploy
   ```
2. In Coolify, create a resource from your repository and set the **branch to `deploy`**.
   - If deploying this UI on its own: set **Base directory** to `/webapp` and use the
     **Dockerfile** build pack (Coolify reads `webapp/Dockerfile`).
   - If deploying the whole stack: point Coolify at your **docker-compose** file instead.
3. Set the env vars in Coolify: `API_UPSTREAM` for the web service (and whatever your
   backend needs). Optionally set the build arg `VITE_API_BASE` (default `/api` is fine).
4. Add a **health check** hitting `/` (static) or `/api/health` (through the proxy).
5. Enable the **git webhook**: copy the webhook URL Coolify generates into your git
   host's repo settings (GitHub: *Settings → Webhooks*; GitLab: *Settings → Webhooks*).
   Now every push to `deploy` triggers a rebuild + redeploy.

Typical workflow: develop on `main`, and when you want to ship, merge/fast-forward into
`deploy` and push — the webhook does the rest.

## Wiring notes for the backend (matches your existing code)

- Model: ResNet-18, classes `["NORMAL", "PNEUMONIA"]`, **PNEUMONIA = index 1** (as in
  `evaluate.py`). Report `probability` = softmax prob of index 1.
- Grad-CAM: target layer `model.layer4[-1]` (as in `gradcam.py`); return the
  `show_cam_on_image` overlay as base64 PNG in `heatmap_png_base64`.
- Preprocess uploads with the same `transform` from `data.py` (resize 224×224 + ImageNet
  normalize) before inference.
- Load the model **once at startup** (not per request) to stay within the RAM budget and
  keep the single-worker queue fast.
