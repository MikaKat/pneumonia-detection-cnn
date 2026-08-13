import { useState } from "react";

import {
  PRAEVALENZ_VORGABEN,
  REFERENZ_PRAEVALENZ,
  likelihoodRatio,
  priorShift,
} from "../prior";
import "../prevalence.css";
import { GEMESSEN, SENS_ZUSAGE, STUFE, TEXTE, T_HIGH, T_LOW, stufeVon } from "../stufen";

// The finished analysis.
//
// LAYOUT
// ------
// Two columns from 1000px up: the numbers on the left, the maps on the right,
// the measured limits underneath across both. Below that width the same blocks
// stack in the same order. Long explanations sit in <details>, so the page
// carries one sentence per point and the derivation is one click away.
//
// THE PAGE SEPARATES TWO KINDS OF OUTPUT, AND THAT SEPARATION IS THE DESIGN
// ------------------------------------------------------------------------
// Population-dependent is the percentage. It is p(pneumonia | image) under the
// one prevalence the Platt curves were fitted at, 0.2253, and that assumption
// is wrong nearly everywhere the app would be used. The control below it
// re-expresses the number for another population by adding a single constant
// in logit space.
//
// Fixed are the tier and the likelihood ratio. Both are computed from the
// REFERENCE score, so the control cannot move them. The two thresholds were
// chosen on the development data on that scale, and reading them against a
// shifted number would measure against a moving ruler.
//
// The labels say which is which, because a control that moves one number and
// leaves the one next to it alone is otherwise just confusing. The split is
// also the main finding of the external validation: on Kermany the calibration
// error went from 0.0094 to 0.4783 while the ranking held. The ordering
// travels between populations, the probability does not.
//
// WHERE THE NUMBERS COME FROM
// ---------------------------
// Holdout figures from predictions_holdout/holdout.csv, development figures
// from serving/model/kalibrierung_p10.json, tier figures and both thresholds
// from ../stufen.
export function ResultView({ result, stages = [], originalUrl, onReset }) {
  const [opacity, setOpacity] = useState(0.6);
  const [showViews, setShowViews] = useState(false);
  // Defaults to the reference, where the correction is the identity.
  const [praevalenz, setPraevalenz] = useState(REFERENZ_PRAEVALENZ);
  // "head" or "cam". The head is preselected: it is the better pointer by a
  // wide margin, and preselection is the part of a switch most users never
  // touch.
  const [map, setMap] = useState("head");

  const u = result.uncertainty;
  const ens = result.ensemble?.per_fold?.length ? result.ensemble : null;
  const base = result.probability ?? 0;
  // The reference scale: what the backend computed, calibrated at 0.2253.
  const medianRef = u?.median ?? base;
  const loRef = u?.min ?? base;
  const hiRef = u?.max ?? base;
  const spread = u?.spread ?? 0;
  const reference = result.reference;

  // The display scale: the same evidence for the chosen population. One
  // additive constant in logit space, applied to the score and to both
  // thresholds, so their relation is invariant.
  const angepasst = Math.abs(praevalenz - REFERENZ_PRAEVALENZ) > 1e-12;
  const shift = (x) => priorShift(x, REFERENZ_PRAEVALENZ, praevalenz);
  const median = shift(medianRef);
  const lo = shift(loRef);
  const hi = shift(hiRef);

  // Prevalence-free, and computed from the REFERENCE score on purpose: it is a
  // property of the model and must not move when the control does.
  const lr = likelihoodRatio(medianRef);
  const lrText = lr >= 10 ? lr.toFixed(0) : lr.toFixed(2);
  const gewaehlt =
    PRAEVALENZ_VORGABEN.find((v) => Math.abs(v.wert - praevalenz) < 1e-12) ??
    null;

  const stufe = stufeVon(medianRef);
  const stufeText = TEXTE[stufe];
  // The marks travel with the display so they keep sitting where the tier
  // changes. The tier itself is still decided on the reference scale.
  const tLow = shift(T_LOW);
  const tHigh = shift(T_HIGH);

  const pct = (x) => Math.round(x * 100);
  const unstable = spread >= 0.1;
  // A band straddling a threshold is worth saying out loud: the tier itself
  // would flip if the image had been framed two percent differently.
  const stufeWackelt = stufeVon(loRef) !== stufeVon(hiRef);

  const camSrc = result.heatmap_png_base64
    ? `data:image/png;base64,${result.heatmap_png_base64}`
    : null;
  // The transparent layer, not the version already composited onto the X-ray:
  // fading a composite over the same picture would be a second pass of the
  // same blend. Where the head says nothing this layer is transparent.
  const headSrc = result.head_field_layer_png_base64
    ? `data:image/png;base64,${result.head_field_layer_png_base64}`
    : null;

  // Both maps live in the model's 224x224 geometry, where the image is squeezed
  // to a square. The layer underneath has to be the same picture in the same
  // geometry, otherwise warm areas sit next to the anatomy they belong to. The
  // upload stage image is used for it; the picker thumbnail is the fallback.
  const uploadStage = stages.find((s) => s.key === "upload" && s.image_png_base64);
  const baseSrc = uploadStage
    ? `data:image/png;base64,${uploadStage.image_png_base64}`
    : originalUrl;

  const overlaySrc = map === "head" ? headSrc : camSrc;
  const bothMaps = Boolean(headSrc && camSrc);

  return (
    <section className="card result">
      <div className="result-grid">
        <div className="result-col">
          {/* THE TIER, WITH ITS PRICE IN THE SAME BOX.
              The asterisk is not a disclaimer. It is the measured number that
              turns the word from a claim into a finding. The header says the
              tier is fixed, because the control further down moves the
              percentage and leaves this alone. */}
          <div className={`tier tier--${stufeText.ton}`}>
            <p className="block-eyebrow">
              Assessment
              <span className="block-eyebrow-note">
                set on the reference scale, so the control below cannot move it
              </span>
            </p>
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
                Across the {u ? u.views.length : 0} framings of this image the
                tier itself changes, so the number below is the more honest
                output here.
              </p>
            )}
          </div>

          <div className="score">
            {/* The two numbers side by side. One moves with the control below,
                the other does not, and putting them next to each other is what
                makes that readable without a paragraph about it. */}
            <div className="score-duo">
              <div className="score-cell">
                <span className="score-cell-label">Probability</span>
                <span className="score-value">{pct(median)}%</span>
                <span className="score-cell-foot">for the population set below</span>
              </div>
              <div className="score-cell score-cell--fixed">
                <span className="score-cell-label">Likelihood ratio</span>
                <span className="score-value">&times;{lrText}</span>
                <span className="score-cell-foot">the same in every population</span>
              </div>
            </div>

            {/* The band is the point of this display: a single tick would claim
                a precision the model does not have. The two dashed lines are
                the rule-out and the rule-in threshold. */}
            <div className="score-track" aria-hidden="true">
              <div
                className="score-band"
                style={{ left: `${pct(lo)}%`, width: `${Math.max(pct(hi) - pct(lo), 1)}%` }}
              />
              <div className="score-mark" style={{ left: `${pct(median)}%` }} />
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
              Dashed lines: rule out below <strong>{pct(tLow)}%</strong>, rule in
              above <strong>{pct(tHigh)}%</strong>.
              {reference?.percentile != null && (
                <>
                  {" "}This score is higher than{" "}
                  <strong>{reference.percentile}%</strong> of{" "}
                  {reference.n.toLocaleString("en-US").replace(/,/g, " ")}{" "}
                  development images.
                </>
              )}
            </p>

            <details className="more">
              <summary>Why the percentages look low</summary>
              <p>
                They are calibrated for a population with about 23 pneumonias in
                100. On 3812 held-out images the highest score ever produced was
                89 %, and only 9 % of images scored above 60 %. A value of 45 %
                therefore sits well above the point where this model calls an
                image positive.
              </p>
            </details>

            <p className="score-range">
              {u ? (
                <>
                  Range <strong>{pct(lo)}% to {pct(hi)}%</strong> across{" "}
                  {u.views.length} slightly different framings of the same image.
                  {ens && (
                    <>
                      {" "}The {ens.folds} models range{" "}
                      {pct(Math.min(...ens.per_fold))}% to{" "}
                      {pct(Math.max(...ens.per_fold))}%.
                    </>
                  )}
                  {unstable &&
                    " That is a wide swing for a two percent change of frame, so the individual digits carry little information here."}
                </>
              ) : (
                <>Single pass, no stability estimate available.</>
              )}
            </p>

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

            {/* THE PREVALENCE CONTROL.
                A percentage means nothing without the population it refers to.
                This names the population, lets it be changed, and says in one
                line what the change touches and what it leaves alone. */}
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
                    Set to <strong>{(praevalenz * 100).toFixed(1)} in 100</strong>.
                    This moves the probability and both dashed lines by the same
                    amount. The assessment and the likelihood ratio stay where
                    they are.
                  </>
                ) : (
                  <>
                    The population the model was calibrated on, so nothing is
                    corrected. Move the slider and only the probability changes.
                  </>
                )}
              </p>

              <details className="more">
                <summary>What the slider actually does</summary>
                <p>
                  It adds one constant in logit space to the score, the textbook
                  correction for a known change of base rate. The ranking of
                  images is untouched, which is why the assessment and the
                  likelihood ratio do not move with it: this image multiplies
                  the odds of pneumonia by {lrText} in any population.
                  {lr < 1 && " Below 1, so it argues mildly against pneumonia."}{" "}
                  On external data the ranking transferred and the probability
                  did not, so the fixed half is the half that travels.
                </p>
                {gewaehlt?.hinweis && <p>{gewaehlt.hinweis}</p>}
              </details>
            </div>
          </div>
        </div>

        <div className="result-col heatmap-area">
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

          {/* One caption per map. The two are measured against the same
              yardstick and land far apart, and a shared caption would have to
              blur that to stay true of either. */}
          {map === "head" ? (
            <div className="heatmap-caption">
              <p className="muted">
                The second output of the same network, a 14x14 field trained
                against radiologist boxes. Against those boxes it reaches{" "}
                <strong>0.91</strong>, where Grad-CAM reaches 0.73 and a fixed
                template that never looks at the image reaches 0.75.
              </p>
              <details className="more">
                <summary>How to read it</summary>
                <p>
                  Drawn with no box and no cut-off, because the level of this
                  field is not calibrated: on images without pneumonia it still
                  lights up in 62 % of cases. Where it says nothing the layer is
                  transparent and the X-ray shows through, which is the
                  difference between no statement and a statement of no. Treat
                  it as a hint about the region, never as a finding. It does not
                  feed the score.
                </p>
              </details>
            </div>
          ) : (
            <div className="heatmap-caption">
              <p className="muted">
                Where the last convolutional block pushed the score, averaged
                over the five models. This is the only map tied to the number on
                the left, which is why it is kept.
              </p>
              <details className="more">
                <summary>How to read it</summary>
                <p>
                  As a pointer it is weak: <strong>0.73</strong> against the
                  radiologist boxes, below the 0.75 of a fixed template that
                  ignores the image entirely. Expect a diffuse blob rather than
                  a sharp finding. Measurements in this project showed the
                  evidence the model uses is spread out and partly outside the
                  lungs, so read it as a plausibility check on the model rather
                  than as a marked lesion.
                </p>
              </details>
            </div>
          )}
        </div>
      </div>

      {/* Across both columns, under the score and the maps, so it is the note
          the reader leaves with. */}
      <div className="honesty">
        <p className="honesty-lead">
          Research demonstrator, not a diagnostic tool. Every tier is shown with
          the miss rate measured for it, and a low score does not rule out
          pneumonia.
        </p>
        <table className="tier-table">
          <caption>
            The same two thresholds on every dataset this model has been
            measured on. What a tier is worth depends on who sends the images.
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

        <div className="honesty-more">
          <details className="more">
            <summary>Where the two thresholds come from</summary>
            <p>
              Not from Youden, which maximises a statistic and answers no
              clinical question. {pct(T_LOW)}% was fixed for 90 % sensitivity to
              rule out and {pct(T_HIGH)}% for 95 % specificity to rule in, both
              on the 22 872 development images and on nothing else.{" "}
              {SENS_ZUSAGE.satz}
            </p>
          </details>
          <details className="more">
            <summary>What did not transfer</summary>
            <p>
              When this project carried its decision threshold from one dataset
              to another, half of the images the model called negative did have
              pneumonia (NPV 0.500). The ranking transferred, the calibration
              did not.
            </p>
            <p>
              The score also depends on how the film was taken, and this page
              cannot tell which it was. On the held-out set the model found 88 %
              of pneumonias in AP films and 81 % in PA films; on the development
              data that gap was 14 points. Separate thresholds per projection
              would close it, but nothing in an uploaded image says which
              projection it is. The model can still tell AP from PA at 0.75, and
              nine attempts to take that ability away failed.
            </p>
          </details>
        </div>
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
