// The localisation head, on its own card.
//
// This is the second output of the very network that produced the score, so it
// is not rejected work the way the lung finder is. It still gets its own card
// rather than a place in the chain, for one reason: it does not feed the score.
// The chain is the scored path and nothing else.
//
// Everything about how this is drawn follows from one measurement. The level of
// the field is not calibrated: on images without pneumonia it lights up in 62 %
// of cases. So there is no box, no outline and no cut-off at 0.5, because each
// of those would be a claim about exactly the quantity that was measured and
// found wanting. A gradient is the strongest statement the measurement carries.
//
// The server renders the overlay, because the colouring is part of that claim:
// the values are used as they are, with no per-image stretch, so a field that
// says nothing looks like nothing.
export function HeadField({ plan, stages, result }) {
  const arrived = stages.find((s) => s.key === "head_field");
  const src = arrived?.image_png_base64
    ? `data:image/png;base64,${arrived.image_png_base64}`
    : result?.head_field_png_base64
    ? `data:image/png;base64,${result.head_field_png_base64}`
    : null;

  if (!src) return null;

  const meta = plan?.find((s) => s.key === "head_field") || {};
  const title = arrived?.title || meta.title || "Where the model points";
  const caption = arrived?.caption || meta.caption;
  const note = arrived?.note || meta.note;
  const peak = result?.head_field?.max;

  return (
    <section className="card aside-card">
      <div className="aside-head">
        <h2 className="aside-title">{title}</h2>
        <span className="chip chip--explored">not used for the score</span>
      </div>

      {caption && <p className="muted aside-sub">{caption}</p>}

      <figure className="aside-figure">
        <img className="aside-img" src={src} alt="Localisation field of the ensemble" />
        <figcaption className="muted aside-ms">
          {typeof peak === "number" && <>Strongest tile: {Math.round(peak * 100)}%. </>}
          {typeof arrived?.ms === "number" && <>{arrived.ms} ms</>}
        </figcaption>
      </figure>

      {note && <p className="tile-note aside-note">{note}</p>}
    </section>
  );
}
