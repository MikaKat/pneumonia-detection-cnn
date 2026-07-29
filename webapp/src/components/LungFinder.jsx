// The U-Net segmenter, on its own card.
//
// It runs on every uploaded image and it does its job well, which is the whole
// reason it is still on the page. What it does NOT do is touch the score:
// feeding the classifier only the lungs was built, measured and rejected, so
// the mask is a side result and nothing downstream reads it.
//
// It used to sit inside the processing chain with an "explored" badge. That did
// not work. A row of identical arrows is a diagram and the badge is a label,
// and the diagram wins: readers concluded the mask and the crop fed the
// classifier. Hence a separate card rather than a better badge.
export function LungFinder({ plan, stages }) {
  const arrived = stages.find((s) => s.key === "mask");
  const src = arrived?.image_png_base64
    ? `data:image/png;base64,${arrived.image_png_base64}`
    : null;

  // No segmenter checkpoint, no card. A box saying "unavailable" would spend a
  // card's worth of attention on something that was never part of the answer.
  if (!src) return null;

  const meta = plan?.find((s) => s.key === "mask") || {};
  const title = arrived.title || meta.title || "Lung finder";
  const caption = arrived.caption || meta.caption;
  const note = arrived.note || meta.note;

  return (
    <section className="card aside-card">
      <div className="aside-head">
        <h2 className="aside-title">{title}</h2>
        <span className="chip chip--explored">not used for the score</span>
      </div>

      {caption && <p className="muted aside-sub">{caption}</p>}

      <figure className="aside-figure">
        <img className="aside-img" src={src} alt="Lung mask found by the U-Net" />
        {typeof arrived.ms === "number" && (
          <figcaption className="muted aside-ms">{arrived.ms} ms</figcaption>
        )}
      </figure>

      {note && <p className="tile-note aside-note">{note}</p>}
    </section>
  );
}
