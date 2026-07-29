// Thin client for the backend defined in API_CONTRACT.md.
// Every call carries a stable X-Client-Id so the server can rate-limit / queue per client.

const BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");

// --- stable per-browser client id ------------------------------------------
function getClientId() {
  const KEY = "pneumo_client_id";
  let id = null;
  try {
    id = localStorage.getItem(KEY);
    if (!id) {
      id =
        (crypto.randomUUID && crypto.randomUUID()) ||
        `c_${Date.now()}_${Math.random().toString(16).slice(2)}`;
      localStorage.setItem(KEY, id);
    }
  } catch {
    // localStorage blocked (private mode) — fall back to a per-session id.
    id = `c_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }
  return id;
}

const CLIENT_ID = getClientId();

function headers(extra = {}) {
  return { "X-Client-Id": CLIENT_ID, ...extra };
}

// Turns a non-OK response into a thrown Error carrying { status, code, reset_at }.
async function toError(res) {
  let body = {};
  try {
    body = await res.json();
  } catch {
    /* non-JSON error body */
  }
  const err = new Error(body.message || body.error || `HTTP ${res.status}`);
  err.status = res.status;
  err.code = body.error || null;
  err.reset_at = body.reset_at || null;
  return err;
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`, { headers: headers() });
  if (!res.ok) throw await toError(res);
  return res.json();
}

export async function getLimits() {
  const res = await fetch(`${BASE}/limits`, { headers: headers() });
  if (!res.ok) throw await toError(res);
  return res.json();
}

// Static description of the preprocessing chain: titles, captions, and which
// stages are production and which were explored during development and dropped.
// Fetched once at startup so the empty chain can be drawn before any upload.
export async function getPipeline() {
  const res = await fetch(`${BASE}/pipeline`, { headers: headers() });
  if (!res.ok) throw await toError(res);
  return res.json();
}

export async function getSamples() {
  const res = await fetch(`${BASE}/samples`, { headers: headers() });
  if (!res.ok) throw await toError(res);
  const data = await res.json();
  return data.samples || [];
}

// Absolute-ise a possibly-relative URL coming from the API (e.g. thumbnail_url).
export function apiUrl(maybeRelative) {
  if (!maybeRelative) return maybeRelative;
  if (/^https?:\/\//i.test(maybeRelative)) return maybeRelative;
  if (maybeRelative.startsWith("/api")) return maybeRelative; // already rooted at proxy
  if (maybeRelative.startsWith("/")) return maybeRelative;
  return `${BASE}/${maybeRelative}`;
}

// Submit a File upload. Returns { job_id, status, position }.
export async function analyzeFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: headers(), // don't set Content-Type — browser sets multipart boundary
    body: fd,
  });
  if (!res.ok) throw await toError(res);
  return res.json();
}

// Submit a server-side sample by id.
export async function analyzeSample(sampleId) {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ sample_id: sampleId }),
  });
  if (!res.ok) throw await toError(res);
  return res.json();
}

// `since` tells the server how many pipeline stages we already hold, so it only
// sends the new ones. Without it every poll would re-send the whole image chain.
export async function getJob(jobId, since = 0) {
  const res = await fetch(
    `${BASE}/jobs/${encodeURIComponent(jobId)}?since=${since}`,
    { headers: headers() }
  );
  if (res.status === 404) {
    const err = new Error("Result expired or job not found.");
    err.status = 404;
    err.code = "job_not_found";
    throw err;
  }
  if (!res.ok) throw await toError(res);
  return res.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Polls getJob until status is "done" or "error".
// onUpdate({status, position, stages}) fires on every poll; `stages` is the full
// list accumulated so far, so the UI can grow the chain while the job runs.
// Returns the final job object (with `stages` attached). Throws on error/timeout/abort.
export async function pollJob(jobId, onUpdate, { signal } = {}) {
  const startedAt = Date.now();
  const MAX_MS = 5 * 60 * 1000; // 5 min hard stop
  const stages = [];
  // Two cadences: while the job is still waiting its turn there is nothing to
  // see, so poll lazily. Once it is being processed the stages arrive one by
  // one and a slow poll would make the chain appear all at once at the end.
  const QUEUED_START = 1200;
  const PROCESSING = 400;
  let delay = QUEUED_START;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    if (Date.now() - startedAt > MAX_MS) {
      const err = new Error("Timed out waiting for the result.");
      err.code = "timeout";
      throw err;
    }

    const job = await getJob(jobId, stages.length);
    if (Array.isArray(job.stages) && job.stages.length) stages.push(...job.stages);
    onUpdate?.({ status: job.status, position: job.position, stages: [...stages] });

    if (job.status === "done") return { ...job, stages: [...stages] };
    if (job.status === "error") {
      const err = new Error(job.message || "Analysis failed.");
      err.code = job.error || "inference_failed";
      err.stages = [...stages];
      throw err;
    }

    await sleep(delay);
    if (job.status === "processing") {
      delay = PROCESSING;
    } else {
      // Gentle backoff while queued so we don't hammer the server; cap at 3s.
      delay = Math.min(delay + 300, 3000);
    }
  }
}
