// Prior correction: turning the score into a probability for a DIFFERENT
// population than the one it was calibrated on.
//
// WHY THIS FILE EXISTS
// --------------------
// The number the backend returns is p(pneumonia | image) under exactly one
// assumed prevalence: the 0.2253 of the 22872 development images the Platt
// curves were fitted on. That assumption is invisible in the output and wrong
// almost everywhere the app would actually be used. Screening sees far less
// pneumonia than RSNA; a ward round sees far more.
//
// The external validation on Kermany measured the damage: prevalence jumped
// from 0.2253 to 0.7297 and the calibration error went from 0.0094 to 0.4783.
// Shifting the logits by the difference of the two KNOWN frequencies, without
// fitting anything, brought it back to 0.1641. Two thirds of the error was the
// prior alone.
//
// THE WHOLE CORRECTION IS ONE ADDITIVE CONSTANT IN LOGIT SPACE:
//
//     logit(p_new) = logit(p_old) + ln(pi_new / (1 - pi_new))
//                                 - ln(pi_old / (1 - pi_old))
//
// It is monotone, so the RANKING and therefore the AUC are untouched to the
// last bit. That is the formal version of the finding of 13.08.2026: the
// ordering survives a change of population, the probability does not.
//
// WHAT THIS CANNOT DO
// -------------------
// Prevalence is a property of the population, not of the image. A single
// upload carries evidence, never a base rate, so the number below has to come
// from the operator or from a running estimate over many requests. There is no
// third option and no clever trick that finds one.
//
// It is also only a FIRST ORDER fix. On Kermany the residual after the true
// prior correction was 0.1641, and it was structured rather than noisy: the
// low bins were over-predicted and the high bins under-predicted, which means
// the slope moved too. That part is covariate shift (different anatomy,
// different machines) and no prior in the world repairs it.

// The prevalence the shipped Platt curves were fitted at.
// serving/model/kalibrierung_p10.json -> dev.praevalenz
export const REFERENZ_PRAEVALENZ = 0.22534102833158448;

const EPS = 1e-7;

export const logit = (p) => {
  const q = Math.min(Math.max(p, EPS), 1 - EPS);
  return Math.log(q / (1 - q));
};

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export const odds = (p) => {
  const q = Math.min(Math.max(p, EPS), 1 - EPS);
  return q / (1 - q);
};

/** The logit offset between two prevalences. */
export function priorDelta(von, nach) {
  return Math.log(nach / (1 - nach)) - Math.log(von / (1 - von));
}

/** Re-express a probability calibrated at `von` for a population at `nach`. */
export function priorShift(p, von = REFERENZ_PRAEVALENZ, nach = REFERENZ_PRAEVALENZ) {
  if (!Number.isFinite(p)) return p;
  if (Math.abs(von - nach) < 1e-12) return p;
  return sigmoid(logit(p) + priorDelta(von, nach));
}

/**
 * The likelihood ratio of this score: how much the image alone multiplies the
 * odds of pneumonia.
 *
 *     LR(s) = odds(p_ref) / odds(pi_ref)
 *
 * This is the part of the output that is a property of the MODEL rather than
 * of the population, and it is invariant under every prior shift. Move the
 * prevalence control and the percentage moves; this number does not. That is
 * the honest split, and it is what a reader should carry away.
 *
 * Reference values on the delivered ensemble, at pi_ref = 0.22534:
 *   0.0792 (median of images without pneumonia)  ->  0.2957
 *   0.2003 (the operating point)                 ->  0.8611   <- BELOW 1
 *   0.4823 (median of true pneumonias)           ->  3.2026
 *   0.8927 (highest score ever produced)         -> 28.6007
 *
 * The operating point sitting below 1 is not a bug: the app deliberately
 * raises a flag even on images that argue slightly AGAINST pneumonia, because
 * the cost of the two mistakes is not the same.
 */
export function likelihoodRatio(p, referenz = REFERENZ_PRAEVALENZ) {
  return odds(p) / odds(referenz);
}

/**
 * Presets. These are ORDERS OF MAGNITUDE from the clinical literature for the
 * setting, not measurements of this model, and they are labelled that way in
 * the interface. A number the operator types beats every one of them.
 */
export const PRAEVALENZ_VORGABEN = [
  {
    id: "screening",
    label: "Screening / outpatient",
    wert: 0.02,
    hinweis: "Asymptomatic or unselected chest films. Pneumonia is rare here.",
  },
  {
    id: "hausarzt",
    label: "Primary care, cough",
    wert: 0.05,
    hinweis: "Patients presenting with a respiratory complaint in general practice.",
  },
  {
    id: "referenz",
    label: "RSNA development set",
    wert: REFERENZ_PRAEVALENZ,
    hinweis:
      "What this model is calibrated to. The default, and the only one where " +
      "the displayed number needs no correction at all.",
  },
  {
    id: "notaufnahme",
    label: "Emergency department",
    wert: 0.15,
    hinweis: "Chest film ordered for a respiratory presentation in an ED.",
  },
  {
    id: "station",
    label: "Inpatient / portable film",
    wert: 0.45,
    hinweis:
      "Bedside films on a ward or in intensive care. Note that these are also " +
      "AP films, where this model behaves differently for reasons that have " +
      "nothing to do with prevalence.",
  },
];
