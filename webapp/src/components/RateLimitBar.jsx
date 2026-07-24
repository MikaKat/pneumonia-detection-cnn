// Shows how many analyses the client has left in the current window.
export function RateLimitBar({ limits }) {
  if (!limits) return null;
  const { limit = 0, remaining = 0, reset_at } = limits;
  const used = Math.max(0, limit - remaining);
  const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
  const low = remaining <= 1;

  return (
    <div className="ratelimit">
      <div className="ratelimit-head">
        <span>
          Analyses left: <strong>{remaining}</strong> / {limit}
        </span>
        {reset_at && <span className="muted">resets {formatReset(reset_at)}</span>}
      </div>
      <div className="ratelimit-track">
        <div
          className={"ratelimit-fill" + (low ? " ratelimit-fill--low" : "")}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function formatReset(iso) {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const diffMs = t - Date.now();
  if (diffMs <= 0) return "shortly";
  const mins = Math.round(diffMs / 60000);
  if (mins < 60) return `in ${mins} min`;
  const hrs = Math.round(mins / 60);
  return `in ${hrs} h`;
}
