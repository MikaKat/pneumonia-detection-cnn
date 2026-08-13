import { useState } from "react";

import {
  PRAEVALENZ_VORGABEN,
  REFERENZ_PRAEVALENZ,
  likelihoodRatio,
  priorShift,
} from "../prior";
import "../prevalence.css";
import { GEMESSEN, SENS_ZUSAGE, STUFE, TEXTE, T_HIGH, T_LOW, stufeVon } from "../stufen";

// Renders the finished analysis.
//
// NO VERDICT WAS SHOWN UNTIL 13.08.2026, AND THE REASON HAS BEEN REFINED
// RATHER THAN DROPPED.
//
// The original reasoning: in this project's external validation the operating
// threshold did not transfer, and half of the images the model called negative
// did have pneumonia (NPV 0.500). A green "no signs of pneumonia" box would be
// the one statement the measurements contradict.
//
// That was right about the danger and wrong about the cause. The problem was
// never the word, it was a word WITHOUT A NUMBER. "Unremarkable" on its own is
// a promise nobody measured. "Unremarkable, and 4 to 18 pneumonias in 100 land
// here, measured on three datasets" is not a promise at all, it is a finding.
//
// So the tier is shown, and it always carries its own price. The three
// thresholds and every figure behind them live in `../stufen`, chosen on the
// development data alone under the FDA's clinical-action-point logic rather
// than by Youden. The tier is computed from the REFERENCE probability, so the
// prevalence control below cannot move it.
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
// THE CHANGE OF 13.08.2026: THE ASSUMED PREVALENCE IS VISIBLE AND ADJUSTABLE.
//
// The external validation on Kermany put a number on the last unstated
// assumption in this page. Every percentage shown here is conditional on a
// population where 22.5 in 100 images have pneumonia, because that is what the
// Platt curves were fitted on. On Kermany, where 73 in 100 do, the calibration
// error went from 0.0094 to 0.4783 — and two thirds of that was the prior
// alone, recoverable by adding one constant to the logits.
//
// So the page now (a) says which population the number assumes, (b) lets that
// population be changed, and (c) shows next to it the one quantity that does
// NOT depend on the population: the likelihood ratio. Move the control and the
// percentage moves; the likelihood ratio does not. That is exactly the split
// the external validation measured — the ordering travels, the probability
// does not — and it is worth more on the screen than in a limitations section.
//
// The decision is untouched by all of this. The threshold is shifted by the
// same constant as the score, so `score >= threshold` holds or fails exactly as
// before, and sensitivity and specificity are preserved to the last bit.
//
// The numbers quoted in the honesty block are from the held-out set of 3812
// images, read once: `predictions_holdout/holdout.csv`, evaluated by
// `rsna/befunde/rsna_phase10_auswertung.py`. The development figures next to
// them are in `serving/model/kalibrierung_p10.json`.
export function ResultView({ result, stages = [], originalUrl, onReset }) {
  const [opacity, setOpacity] = useState(0.6);
  const [showViews, setShowViews] = useState(false);
  // Defaults to the reference, where the correction is the identity and the
  // page reads exactly as it did before.
  const [praevalenz, setPraevalenz] = useState(REFERENZ_PRAEVALENZ);
  // "head" or "cam". The head is preselected: it is the better pointer by a
  // wide margin, and preselection is the part of a switch that most users
  // never change.
  const [map, setMap] = useState("head");

  const u = result.uncertainty;
  const ens = result.ensemble?.per_fold?.length ? result.ensemble : null;
  const base = result.probability ?? 0;
  // The reference scale: what the backend computed, calibrated at 0.2253.
  // Everything the model actually knows lives here.
  const medianRef = u?.median ?? base;
  const loRef = u?.min ?? base;
  const hiRef = u?.max ?? base;
  const spread = u?.spread ?? 0;
  const thresholdRef = result.threshold;
  const reference = result.reference;

  // The display scale: the same evidence, re-expressed for the chosen
  // population. One additive constant in logit space, applied to the score AND
  // to the threshold, so their relation is invariant.
  const angepasst = Math.abs(praevalenz - REFERENZ_PRAEVALENZ) > 1e-12;
  const shift = (x) => priorShift(x, REFERENZ_PRAEVALENZ, praevalenz);
  const median = shift(medianRef);
  const lo = shift(loRef);
  const hi = shift(hiRef);
  const threshold =
    typeof thresholdRef === "number" ? shift(thresholdRef) : thresholdRef;

  // Prevalence-free. Computed from the REFERENCE score on purpose: it is a
  // property of the model, and it must not move when the control does.
  const lr = likelihoodRatio(medianRef);
  const lrText = lr >= 10 ? lr.toFixed(0) : lr.toFixed(2);
  const gewaehlt =
    PRAEVALENZ_VORGABEN.find((v) => Math.abs(v.wert - praevalenz) < 1e-12) ??
    null;

  // The tier comes from the REFERENCE probability on purpose. The two
  // thresholds were fixed on the development data, on that scale; reading them
  // against a prior-shifted number would compare against a moving ruler.
  const stufe = stufeVon(medianRef);
  const stufeText = TEXTE[stufe];
  // Die zwei Schwellen wandern mit der Anzeige mit, damit die Marken weiter
  // dort sitzen, wo die Stufe wechselt. Die STUFE selbst wird trotzdem auf der
  // Referenzskala bestimmt, siehe oben: sonst haenge die Einordnung am Regler.
  const tLow = shift(T_LOW);
  const tHigh = shift(T_HIGH);

  const pct = (x) => Math.round(x * 100);
  // Where the score sits matters less than how far it travels. Ten points of
  // movement from a 2 % change in framing means the digits are not meaningful.
  const unstable = spread >= 0.1;
  // The band straddling a threshold is worth saying out loud: it means the tier
  // itself would flip if the image had been framed two percent differently.
  const stufeWackelt = stufeVon(loRef) !== stufeVon(hiRef);

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

        {/* THE TIER, AND ITS PRICE IN THE SAME BREATH.
            The asterisk is not a disclaimer. There are already three of those
            on this page and a fourth would add nothing. It is a measured
            number, and it is what turns the word from a claim into a finding. */}
        <div className={`tier tier--${stufeText.ton}`}>
          <div className="tier-head">
            <span className="tier-label">{stufeText.label}<sup>*</sup></span>
            <span className="tier-sub">{stufeText.kurz}</span>
          </div>
          <p className="tier-star">
            <span className="tier-star-mark">*</span>
            {stufeText.stern()}
          </p>
          {stufeWackelt && (
            <p className="tier-star tier-star--warn">
              <span className="tier-star-mark">!</span>
              Across the {u ? u.views.length : 0} framings of this image the tier
              itself changes. The wording above is therefore not stable for this
              upload, and the number above it is the more honest output.
            </p>
          )}
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
          {/* ZWEI Marken seit 13.08.2026, nicht mehr eine.
              Der alte Youden-Punkt 0.2003 ist aus der Skala verschwunden. Er
              war das Maximum von Sensitivitaet plus Spezifitaet, also eine
              Antwort ohne Frage. An seiner Stelle stehen die beiden Schwellen,
              die einen Zweck haben: ausschliessen und einschliessen. Drei
              Striche waeren einer zu viel gewesen, und der ohne Begruendung
              musste gehen. `result.threshold` kommt weiter vom Server und wird
              hier bewusst nicht mehr gezeichnet. */}
          <div className="score-threshold score-threshold--low"
               style={{ left: `${pct(tLow)}%` }} />
          <div className="score-threshold score-threshold--high"
               style={{ left: `${pct(tHigh)}%` }} />
        </div>
        <div className="score-scale" aria-hidden="true">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>

        <p className="score-anchor">
          The two dashed lines sit at <strong>{pct(tLow)}%</strong> and{" "}
          <strong>{pct(tHigh)}%</strong>. They are not one operating point but two
          purposes: the lower was set for 90% sensitivity to rule out, the upper for
          95% specificity to rule in, both on 22 872 development images and on
          nothing else.{" "}
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
          above 60%. <strong>The lines are marks on a scale</strong>, and what
          standing on either side of them has cost is stated above, measured rather
          than promised.
        </p>

        {/* THE PREVALENCE CONTROL.
            A percentage is meaningless without the population it refers to, and
            until now this page never named one. It does now, and it lets it be
            changed, because the assumption is wrong nearly everywhere the app
            would be used. */}
        <div className="prevalence">
          <label className="prevalence-head" htmlFor="prevalence-select">
            <span className="prevalence-label">Population this number assumes</span>
            <select
              id="prevalence-select"
              className="prevalence-select"
              value={gewaehlt ? gewaehlt.id : "eigen"}
              onChange={(e) => {
                const v = PRAEVALENZ_VORGABEN.find((x) => x.id === e.target.value);
                if (v) setPraevalenz(v.wert);
              }}
            >
              {PRAEVALENZ_VORGABEN.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label} &middot; {(v.wert * 100).toFixed(v.wert < 0.1 ? 0 : 1)}%
                </option>
              ))}
              {!gewaehlt && <option value="eigen">custom</option>}
            </select>
          </label>

          <input
            className="prevalence-range"
            type="range"
            min="0.005"
            max="0.80"
            step="0.005"
            value={praevalenz}
            aria-label="Assumed prevalence of pneumonia in the population"
            onChange={(e) => setPraevalenz(parseFloat(e.target.value))}
          />

          <p className="prevalence-note">
            {angepasst ? (
              <>
                Re-expressed for a population where{" "}
                <strong>{(praevalenz * 100).toFixed(1)} in 100</strong> images show
                pneumonia. The evidence in the image has not changed — only one
                constant was added to the logits, which moves the score and the
                dashed line by the same amount.{" "}
                <strong>The decision is identical either way.</strong>
              </>
            ) : (
              <>
                No correction applied: this is the population the model was
                calibrated on. Change it and the percentage moves, because a
                probability is always a statement about a population and never
                about an image alone.
              </>
            )}
            {gewaehlt?.hinweis && <> {gewaehlt.hinweis}</>}
          </p>

          {/* The number that does NOT move. This is the point of the whole
              block: it makes visible which half of the output is a property of
              the model and which half is borrowed from the population. */}
          <p className="prevalence-lr">
            <span className="prevalence-lr-value">&times;{lrText}</span>
            <span>
              <strong>Likelihood ratio.</strong> This image multiplies the odds of
              pneumonia by {lrText}, whatever the population.{" "}
              <em>This number does not move when the control above does</em> — it
              is a property of the model, not of the setting. On external data the
              ranking transferred and the probability did not; this is the part
              that transferred.
              {lr < 1 && (
                <> Below 1, so this image argues mildly <em>against</em> pneumonia.</>
              )}
            </span>
          </p>
        </div>

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
          <strong>The tier above always carries its own error rate, on purpose.</strong>{" "}
          This is a research demonstrator and not a diagnostic tool. For a long time it
          showed no tier at all, because a bare word like "unremarkable" is a promise
          nobody measured. It is shown now because the promise has been measured, three
          times, and the number travels with the word.
        </p>
        <p>
          <strong>Where the two thresholds come from.</strong> Not from Youden, which
          maximises a statistic and answers no clinical question, but from a purpose:{" "}
          <strong>{pct(T_LOW)}%</strong> was fixed for 90% sensitivity (ruling out) and{" "}
          <strong>{pct(T_HIGH)}%</strong> for 95% specificity (ruling in), both on the
          22 872 development images and on nothing else. {SENS_ZUSAGE.satz}
        </p>
        <table className="tier-table">
          <caption>
            The same two thresholds on every dataset this model has been measured on.
            The tier is one thing; what it is worth depends on who sends the images.
          </caption>
          <thead>
            <tr>
              <th>Dataset</th><th>prevalence</th>
              <th>unremarkable</th><th>unclear</th><th>suspicious</th>
            </tr>
          </thead>
          <tbody>
            {GEMESSEN.map((g) => (
              <tr key={g.id} className={g.heimat ? "tier-table--home" : undefined}>
                <td>
                  {g.name}
                  <span className="muted"> &middot; {g.was}</span>
                </td>
                <td>{pct(g.praevalenz)}%</td>
                {[STUFE.UNAUFFAELLIG, STUFE.UNKLAR, STUFE.AUFFAELLIG].map((s) => (
                  <td key={s}>
                    {pct(g[s].anteil)}% of images
                    <br />
                    <strong>{pct(g[s].krank)}% had it</strong>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
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
