// The three tiers, and the measured price of each of them.
//
// WHY THIS FILE EXISTS
// --------------------
// Until now the app showed a probability and refused to name a tier, on the
// grounds that "a green 'no signs of pneumonia' box would be the one statement
// the measurements contradict". That reasoning was right and it was answering
// the wrong objection: the problem was never the label, it was a label WITHOUT
// A NUMBER. A tier that carries its own measured miss rate is not a claim the
// measurements contradict, it is one they support.
//
// So the tiers come back, each with the price of standing in it.
//
// WHERE THE TWO THRESHOLDS COME FROM
// ----------------------------------
// The shipped operating point 0.2003 came from Youden, which is a statistical
// criterion: it maximises sensitivity plus specificity and answers no clinical
// question. Regulators do not work that way. The FDA guidance calls the
// threshold a "clinical action point" and expects it to be pre-specified for an
// intended use, then locked, then demonstrated on independent data.
//
// So these two follow a purpose rather than a formula, and both were chosen on
// the 22872 out of fold DEVELOPMENT predictions alone:
//
//   t_low   sensitivity 90 %   ruling out    -> below it: "unremarkable"
//   t_high  specificity 95 %   ruling in     -> above it: "suspicious"
//   between                                  -> "unclear", no statement
//
// The spent holdout was not touched. Kermany and VinDr never entered the
// choice either, which is what makes the external figures below unbiased for
// exactly these thresholds rather than a second look at a used set.
//
// THE MISS RATES ARE MEASURED, NOT COMPUTED, AND THAT IS THE POINT
// ----------------------------------------------------------------
// The obvious move is to derive the miss rate from sensitivity, specificity
// and the prevalence the user selects. It was tried and it is wrong here: for
// Kermany the formula predicts 30 % and the truth is 18 %. It rests on
// sensitivity and specificity travelling between populations, and the external
// validation of 13.08.2026 disproved exactly that: at the rule-out threshold
// sensitivity was 0.90 by construction on RSNA, 0.95 on Kermany and 0.81 on
// VinDr. Three measured numbers beat one formula whose assumption is known to
// be false.

export const T_LOW = 0.1368481908525916;
export const T_HIGH = 0.559134948529309;

export const STUFE = { UNAUFFAELLIG: "low", UNKLAR: "mid", AUFFAELLIG: "high" };

export function stufeVon(p) {
  if (!Number.isFinite(p)) return STUFE.UNKLAR;
  if (p < T_LOW) return STUFE.UNAUFFAELLIG;
  if (p >= T_HIGH) return STUFE.AUFFAELLIG;
  return STUFE.UNKLAR;
}

/**
 * What each tier cost on every dataset this model has ever been measured on.
 *
 * `anteil` is how many images landed in the tier, `krank` how many of those
 * actually had the finding. For the bottom tier `krank` is the miss rate, and
 * it is the number the asterisk carries.
 *
 * Note what the three rows are for. They are not decoration and not a
 * robustness flourish: they are the same tier meaning three different things
 * depending on who sends the images. On VinDr "unremarkable" carries a 1 %
 * miss, on RSNA 4 %, on Kermany 18 %. Same model, same threshold, twelve times
 * the risk between the ends.
 */
export const GEMESSEN = [
  {
    id: "rsna",
    name: "RSNA development set",
    was: "adults and children, US emergency departments",
    n: 22872,
    praevalenz: 0.2253,
    low: { anteil: 0.5164, krank: 0.0437 },
    mid: { anteil: 0.3569, krank: 0.3217 },
    high: { anteil: 0.1267, krank: 0.6943 },
    sens: 0.8999,
    heimat: true,
  },
  {
    // MAJORITY LABEL, changed 13.08.2026. VinDr gives three independent readers
    // for its training split and prescribes no way to combine them; the row
    // used to count an image positive as soon as ONE of the three drew a box,
    // which is the file's shape rather than a decision. A finding needs a
    // majority. Under the old rule this row read 4 % missed in the bottom tier
    // instead of 1 %, and the run that was pre-registered on it is reported
    // unchanged in erklaerungen/40_. See 43_ for why the switch is labelled as
    // what it is: made after the result was known.
    id: "vindr",
    name: "VinDr-CXR",
    was: "adults, Vietnam, two of three radiologists",
    n: 15000,
    praevalenz: 0.0559,
    low: { anteil: 0.7421, krank: 0.0146 },
    mid: { anteil: 0.2428, krank: 0.1425 },
    high: { anteil: 0.0151, krank: 0.6947 },
    sens: 0.8067,
    heimat: false,
  },
  {
    id: "kermany",
    name: "Kermany",
    was: "children aged one to five, Guangzhou",
    n: 5856,
    praevalenz: 0.7297,
    low: { anteil: 0.2220, krank: 0.1785 },
    mid: { anteil: 0.7488, krank: 0.8826 },
    high: { anteil: 0.0292, krank: 1.0 },
    sens: 0.9457,
    heimat: false,
  },
];

const spanne = (schluessel) => {
  const w = GEMESSEN.map((g) => g[schluessel].krank);
  return { min: Math.min(...w), max: Math.max(...w) };
};

/**
 * The wording of each tier, and the sentence that keeps it honest.
 *
 * The bottom tier deliberately does NOT say "no pneumonia". It says what the
 * model did, which is that the score fell below the rule-out threshold, and
 * then it says immediately what that has cost. The difference between those two
 * sentences is the whole reason this file has a licence to exist.
 */
export const TEXTE = {
  [STUFE.UNAUFFAELLIG]: {
    label: "Unremarkable",
    kurz: "below the rule-out threshold",
    ton: "low",
    stern: () => {
      const s = spanne("low");
      return `Between ${Math.round(s.min * 100)} and ${Math.round(s.max * 100)} ` +
        `pneumonias in 100 land here. Measured on three datasets, not estimated. ` +
        `Which end of that range applies depends on who sends the images, and ` +
        `this page cannot tell.`;
    },
  },
  [STUFE.UNKLAR]: {
    label: "Unclear",
    kurz: "between the two thresholds",
    ton: "mid",
    stern: () => {
      const s = spanne("mid");
      return `This tier makes no statement, and it is not small: it held ` +
        `between ${Math.round(GEMESSEN[1].mid.anteil * 100)} and ` +
        `${Math.round(GEMESSEN[2].mid.anteil * 100)} percent of all images, ` +
        `of which ${Math.round(s.min * 100)} to ${Math.round(s.max * 100)} ` +
        `percent had the finding.`;
    },
  },
  [STUFE.AUFFAELLIG]: {
    label: "Suspicious",
    kurz: "above the rule-in threshold",
    ton: "high",
    stern: () => {
      const s = spanne("high");
      return `The one tier that held up everywhere: ` +
        `${Math.round(s.min * 100)} to ${Math.round(s.max * 100)} of 100 images ` +
        `here had the finding, across three populations. It is also rare, ` +
        `reaching only ${Math.round(GEMESSEN[1].high.anteil * 100)} to ` +
        `${Math.round(GEMESSEN[0].high.anteil * 100)} percent of images.`;
    },
  },
};

/** The sensitivity claim, and every population it was checked against. */
export const SENS_ZUSAGE = {
  ziel: 0.9,
  satz:
    "The lower threshold was set for 90 % sensitivity on the development data. " +
    "It reached 95 % on Kermany and 81 % on VinDr. A sensitivity figure is a " +
    "statement about a population, not about a model, which is why the three " +
    "are named rather than averaged.",
};
