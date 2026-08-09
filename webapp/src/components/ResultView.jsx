import { useState } from "react";

// Renders the finished analysis.
//
// Deliberately NO verdict. The backend stopped sending a `prediction` field and
// this view stopped drawing one: in this project's external validation the
// operating threshold did not transfer, and half of the images the model called
// negative did have pneumonia (NPV 0.500, README section 6). A green "no signs
// of pneumonia" box would be the one statement the measurements contradict.
//
// What is shown instead: the probability, where it sits on a scale whose
// meaning is stated, the range it moves through when the framing is nudged by a
// couple of percent, and what all of that does and does not mean.
//
// Since phase 10 the number is the mean of five separately calibrated models,
// so there is a second spread to report: how far those five disagree with each
// other on this image. It is a different question from the framing spread and
// it is shown next to it rather than folded into it.
//
// TWO CHANGES OF 09.08.2026, both from reading the finished page:
//
// 1. THE SCALE HAS MARKS NOW. A bare 0-to-100 bar is read against the felt
//    middle at 50 %, and that middle is wrong here. The operating point is
//    0.2003, and on 3812 held-out images the highest value ever produced was
//    0.8927 with only 9 % above 0.60. So 45 % is well ABOVE the point where
//    this model would call an image positive, and it used to look like "rather
//    not". Two marks fix that: the threshold as a line in the track, and one
//    sentence placing the value among the 22872 development images.
//
// 2. THE LOCALISATION HEAD IS THE FIRST MAP, NOT THE THIRD. Measured against
//    the radiologist boxes the head field reaches 0.9123, Grad-CAM 0.7312, and
//    a fixed template that never looks at the image 0.7520 (phase 5). The
//    weaker of the two maps had the big frame and the slider; the stronger one
//    sat below as a still picture. Now they share one frame with a switch, and
//    the head is preselected. Grad-CAM stays reachable because it is the only
//    map that comes from the number above.
//
// The numbers quoted in the honesty block are from the held-out set of 3812
// images, read once: `predictions_holdout/holdout.csv`, evaluated by
// `rsna/befunde/rsna_phase10_auswertung.py`. The development figures next to
// them are in `serving/model/kalibrierung_p10.json`.
export function ResultView({ result, stages = [], originalUrl, onReset }) {
  const [opacity, setOpacity] = useState(0.6);
  const [showViews, setShowViews] = useState(false);
  // "head" or "cam". The head is preselected: it is the better pointer by a
  // wide margin, and preselection is the part of a switch that most users
  // never change.
  const [map, setMap] = useState("head");

  const u = result.uncertainty;
  const ens = result.ensemble?.per_fold?.length ? result.ensemble : null;
  const base = result.probability ?? 0;
  const median = u?.median ?? base;
  const lo = u?.min ?? base;
  const hi = u?.max ?? base;
  const spread = u?.spread ?? 0;
  const threshold = result.threshold;
  const reference = result.reference;

  const pct = (x) => Math.round(x * 100);
  // Where the score sits matters less than how far it travels. Ten points of
  // movement from a 2 % change in framing means the digits are not meaningful.
  const unstable = spread >= 0.1;

  const camSrc = result.heatmap_png_base64
    ? `data:image/png;base64,${result.heatmap_png_base64}`
    : null;
  // The transparent layer, not the version already composited onto the X-ray:
  // fading a composite over the same picture would be a second pass of the same
  // blend. Where the head says nothing this layer is transparent, so the X-ray
  // underneath stays untouched.
  const headSrc = result.head_field_layer_png_base64
    ? `data:image/png;base64,${result.head_field_layer_png_base64}`
    : null;

  // Both maps are rendered in the model's 224x224 geometry, where the image is
  // squeezed to a square. The layer underneath has to be the same picture in the
  // same geometry, otherwise the warm areas sit next to the anatomy they belong
  // to. The upload stage image is used for it (full resolution, same framing);
  // the thumbnail from the picker is only the fallback.
  const uploadStage = stages.find((s) => s.key === "upload" && s.image_png_base64);
  const baseSrc = uploadStage
    ? `data:image/png;base64,${uploadStage.image_png_base64}`
    : originalUrl;

  const overlaySrc = map === "head" ? headSrc : camSrc;
  const bothMaps = Boolean(headSrc && camSrc);

  return (
    <section className="card result">
      <div className="score">
        <div className="score-head">
          <span className="score-label">Model score for pneumonia</span>
          <span className="score-value">{pct(median)}%</span>
        </div>

        {/* The band is the point of this display: a single tick would claim a
            precision the model does not have. The dashed line is the operating
            point, which is what makes the rest of the bar readable. */}
        <div className="score-track" aria-hidden="true">
          <div
            className="score-band"
            style={{ left: `${pct(lo)}%`, width: `${Math.max(pct(hi) - pct(lo), 1)}%` }}
          />
          <div className="score-mark" style={{ left: `${pct(median)}%` }} />
          {typeof threshold === "number" && (
            <div className="score-threshold" style={{ left: `${pct(threshold)}%` }} />
          )}
        </div>
        <div className="score-scale" aria-hidden="true">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>

        {typeof threshold === "number" && (
          <p className="score-anchor">
            The dashed line sits at <strong>{pct(threshold)}%</strong>, this model's
            operating point. It was fixed on 22872 development images before the
            held-out set was ever touched.{" "}
            {reference?.percentile != null && (
              <>
                This score is higher than{" "}
                <strong>{reference.percentile}%</strong> of those{" "}
                {reference.n.toLocaleString("en-US").replace(/,/g, " ")} images.{" "}
              </>
            )}
            The probabilities are calibrated to a population where about 22 in 100
            images show pneumonia, so they run low by design: on 3812 held-out images
            the highest score ever produced was 89%, and only 9% of images scored
            above 60%. <strong>The line is a mark on a scale, not a verdict</strong> —
            see below for why none is shown.
          </p>
        )}

        <p className="score-range">
          {u ? (
            <>
              Range{" "}
              <strong>
                {pct(lo)}% to {pct(hi)}%
              </strong>{" "}
              across {u.views.length} slightly different framings of the same image,{" "}
              {pct(spread)} points apart.{" "}
              {unstable
                ? "The score travels that far when the frame moves by two percent, so the individual digits carry little information here."
                : "The score barely moves under those changes, which says the number is stable, not that it is right."}
            </>
          ) : (
            <>Single pass, no stability estimate available.</>
          )}
        </p>

        {ens && (
          <p className="score-range">
            The number above is the mean of <strong>{ens.folds} separately trained
            models</strong>, each calibrated on its own data before averaging. On this
            image they range from {pct(Math.min(...ens.per_fold))}% to{" "}
            {pct(Math.max(...ens.per_fold))}%. That is disagreement between models,
            which is a different thing from the framing spread above: five models can
            agree with each other and still be wrong together. Averaging also pulls the
            extremes in, which is part of why no score here reaches 95%.
          </p>
        )}

        {u && (
          <>
            <button className="btn btn-ghost btn-sm" onClick={() => setShowViews((v) => !v)}>
              {showViews ? "Hide the individual runs" : "Show the individual runs"}
            </button>
            {showViews && (
              <ul className="view-list">
                {u.views.map((v) => (
                  <li key={v.view}>
                    <span className="muted">{v.view}</span>
                    <span>{pct(v.probability)}%</span>
                  </li>
                ))}
                {ens?.per_fold?.map((p, i) => (
                  <li key={`fold-${i}`}>
                    <span className="muted">model {i + 1} of {ens.folds}</span>
                    <span>{pct(p)}%</span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>

      <div className="heatmap-area">
        {bothMaps && (
          <div className="map-switch" role="tablist" aria-label="Which map to show">
            <button
              role="tab"
              aria-selected={map === "head"}
              className={"map-tab" + (map === "head" ? " map-tab--active" : "")}
              onClick={() => setMap("head")}
            >
              Where the model points
              <span className="map-tab-sub">localisation head &middot; 0.91</span>
            </button>
            <button
              role="tab"
              aria-selected={map === "cam"}
              className={"map-tab" + (map === "cam" ? " map-tab--active" : "")}
              onClick={() => setMap("cam")}
            >
              What the score came from
              <span className="map-tab-sub">Grad-CAM &middot; 0.73</span>
            </button>
          </div>
        )}

        <div className="heatmap-stack">
          {baseSrc && <img className="heatmap-base" src={baseSrc} alt="Original X-ray" />}
          {overlaySrc && (
            <img
              className="heatmap-overlay"
              src={overlaySrc}
              alt={map === "head" ? "Localisation field of the ensemble" : "Grad-CAM heatmap"}
              style={{ opacity }}
            />
          )}
          {!overlaySrc && <p className="muted">No map returned for this image.</p>}
        </div>

        {overlaySrc && (
          <label className="slider">
            <span>Overlay intensity</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
            />
          </label>
        )}

        {/* One caption per map, not one caption for both. The two maps are
            measured against the same yardstick and land far apart, and a shared
            caption would have to blur that to stay true of either. */}
        {map === "head" ? (
          <p className="muted heatmap-caption">
            The <strong>second output of the same network</strong>, a 14x14 field trained
            against radiologist boxes and averaged over the five models. Against those
            boxes it reaches <strong>0.91</strong>, where Grad-CAM reaches 0.73 and a
            fixed template that never looks at the image reaches 0.75. It is the better
            pointer, and it does <strong>not</strong> feed the score.
            <br />
            Drawn with no box and no cut-off on purpose: the level of this field is not
            calibrated, and on images without pneumonia it still lights up in 62% of
            cases. Where it says nothing it is transparent and the X-ray shows through,
            which is the difference between "no statement" and "statement: no". Read it
            as a hint about the region, never as a finding.
          </p>
        ) : (
          <p className="muted heatmap-caption">
            Where the last convolutional block contributed most to the score, averaged
            over the five models. This is the <strong>only map tied to the number
            above</strong>, which is why it is kept.
            <br />
            As a pointer it is weak: <strong>0.73</strong> against the radiologist boxes,
            below the 0.75 of a fixed template that ignores the image entirely. Expect a
            diffuse blob rather than a sharp finding — measurements in this project
            showed the evidence the model uses is spread out and partly outside the
            lungs. Read it as a plausibility check on the model, not as a marked lesion.
          </p>
        )}
      </div>

      {/* Last block in the card, under the score and the maps, so it is the
          note the reader leaves with. */}
      <div className="honesty">
        <p>
          <strong>No verdict is shown, on purpose.</strong> This is a research
          demonstrator, not a diagnostic tool, and it reports a probability rather than a
          label.
        </p>
        <p>
          A low score does not rule out pneumonia. When this project carried its decision
          threshold from one dataset to another,{" "}
          <strong>half of the images the model called negative did have pneumonia</strong>{" "}
          (NPV 0.500). The ranking transferred, the calibration did not, and turning this
          score into a yes or no would rest on exactly the part that failed.
        </p>
        <p>
          <strong>The score depends on how the film was taken, and this page cannot
          tell.</strong> On the held-out set the model found 88% of pneumonias in AP
          films and 81% in PA films, a gap of 7 points, and on the development data the
          gap was 14. Separate thresholds per projection would close it, but nothing in
          an uploaded image says which projection it is, so one threshold has to serve
          both. The same asymmetry runs deeper than the threshold: the model can still
          tell AP from PA at 0.75, and nine attempts to take that ability away did not
          work. It is shipped with the model rather than hidden.
        </p>
      </div>

      <div className="result-footer">
        {typeof result.inference_ms === "number" && (
          <span className="muted">Inference: {result.inference_ms} ms</span>
        )}
        <button className="btn btn-primary" onClick={onReset}>
          Analyze another image
        </button>
      </div>
    </section>
  );
}
