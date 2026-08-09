"""
Phase 2: the last measurement of the old regime, with the instrument from
phase 1.

WHAT IT ANSWERS
---------------
Does the heat map point at the infiltrate, and does it do so better than a map
that knows nothing but where infiltrates usually sit. Measured on EVERY
positive validation image instead of a sample of 300, from the existing
checkpoints, without retraining anything.

FOUR NUMBERS, ALWAYS SIDE BY SIDE
---------------------------------
Model, location prior, lung map, chance. Only the four together say anything.
That is the whole lesson of phase 1: the old chance value, the share of the
image the boxes cover, is the chance of a pointer that lands anywhere at all,
including the image border and the air beside the patient. Nobody guesses like
that. The location prior is the opponent that has to be beaten, and on the old
pointing game it already came out ahead of the model.

PRIMARY IS THE POINT AUC INSIDE THE LUNG MASK, from `rsna_lokalisation.py`. Its
chance value is exactly 0.5 whatever the boxes cover and whatever the crop did.
Hit rate and mass are carried along as secondary numbers so the figures
reported so far stay connected.

THREE ARMS
----------
base    the baseline, whole images
bal10   full decoupling, balance-view at strength 1.0, whole images
crop    the adaptive crop model, four folds. Fold 0 is lost, its checkpoint was
        overwritten. Scored on CROPPED images, because that is what it was
        trained on, and its maps are then projected back into original
        coordinates so that both arms see the same boxes with the same area
        share.

Every arm is checked against the predictions it reported before it is used. A
checkpoint file says nothing about which run wrote it, and four of the five
baseline checkpoints once turned out to hold the crop models.

RESUMABLE
---------
One CSV per fold and arm. A combination that is already on disk is not computed
again, so the run can be stopped and continued. `--force` recomputes.

CLI, from the repository root:
  python rsna\\befunde\\rsna_cam_power.py                       (everything, 1 to 2 h)
  python rsna\\befunde\\rsna_cam_power.py --folds 0 --n 20      (smoke test, a minute)
  python rsna\\befunde\\rsna_cam_power.py --arms base bal10     (without the crop arm)
  python rsna\\befunde\\rsna_cam_power.py --report-only         (report from what is there)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

from rsna_lokalisation import (REF_SIZE, box_mask, evaluate_map, load_boxes,
                               load_lung, to_reference, uncrop)

# torch is imported inside the functions that need it, not here. That keeps
# `--report-only` runnable anywhere the CSVs are, without a torch install and
# without loading a deep learning stack to print a table.

# Two sided five percent bounds of the t distribution, by degrees of freedom.
# Four folds are not five, and using the five fold bound for a four fold
# comparison would be a quiet gift to whichever arm happens to be ahead.
T_LIMIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
# Die beiden Nulllinien aus Phase 1. Sie stehen NICHT in ARMS, weil sie nichts
# rechnen: sie kommen fertig aus baselines_f*.csv.
NULL_ARME = ("Lagepriore", "Lungenkarte")
# Die Paare, die fuer Phase 2 vor dem Rechnen festgelegt wurden. Sie bleiben
# als Liste stehen, damit ihre Reihenfolge erhalten bleibt; alles Weitere baut
# report() aus den Daten.
KANON_PAARE = [("Lagepriore", "base"), ("Lagepriore", "bal10"),
               ("Lagepriore", "crop"), ("base", "bal10"), ("base", "crop"),
               ("Lungenkarte", "base")]
CHI2_LIMIT = 3.841       # five percent, one degree of freedom
MIN_FOLDS_FOR_VERDICT = 4  # below this the closing sentence stays unwritten

ARMS = {
    "base": dict(tag="_base", name="Basislinie",
                 pred_dir=Path("predictions_rsna_base"),
                 images=Path("data/rsna/png512"), crop=False,
                 folds=[0, 1, 2, 3, 4]),
    "bal10": dict(tag="_bal10", name="volle Entkopplung",
                  pred_dir=Path("predictions_rsna_bal10"),
                  images=Path("data/rsna/png512"), crop=False,
                  folds=[0, 1, 2, 3, 4]),
    "crop": dict(tag="", name="Zuschnitt",
                 pred_dir=Path("predictions_rsna_crop"),
                 images=Path("data/rsna/crop512"), crop=True,
                 folds=[1, 2, 3, 4]),
    # The phase 5 arms. `head` says whether the checkpoint carries a second
    # output; the measurement itself is the same for all of them, and that is
    # the point. Comparing the single headed Grad-CAM with the two headed HEAD
    # OUTPUT would compare two instruments and call it two models.
    "p5ref": dict(tag="_p5ref", name="Phase 5 ohne Kopf",
                  pred_dir=Path("predictions_p5_ref"),
                  images=Path("data/rsna/png512"), crop=False,
                  folds=[0, 1, 2, 3, 4], head=False),
    "p5head_ex": dict(tag="_p5head_ex", name="Phase 5 Kopf exclude",
                      pred_dir=Path("predictions_final_model"),
                      images=Path("data/rsna/png512"), crop=False,
                      folds=[0, 1, 2, 3, 4], head=True),
    "p5head_em": dict(tag="_p5head_em", name="Phase 5 Kopf empty",
                      pred_dir=Path("predictions_p5_head_em"),
                      images=Path("data/rsna/png512"), crop=False,
                      folds=[0, 1, 2, 3, 4], head=True),
}


# --------------------------------------------------------------------------
# Weights, and whether they belong to the numbers
# --------------------------------------------------------------------------

def load_model(ckpt: Path, head: bool = False):
    """Weights to the CPU, always.

    `torch.load(..., map_location=<DirectML device>)` dies with a TypeError
    that looks like a broken checkpoint and is none. Grad-CAM runs on the CPU
    anyway: it needs a backward pass through hooks, which is neither fast nor
    reliable under DirectML.

    With `head=True` the checkpoint carries the second output of phase 5. The
    model is then wrapped in `ClassifierView`, which hands out the
    classification logit and forwards `layer4`. Grad-CAM therefore sees a plain
    classifier and this file measures the two headed arm with exactly the same
    instrument as the single headed one. The head OUTPUT is a different map and
    is measured elsewhere; mixing the two here would compare two instruments
    and call it two models.

    The head grid does not matter for this file. `ClassifierView` drops the
    field, and the weight of the head is a 1x1 convolution whose shape does not
    depend on the grid, so the state dict loads either way.
    """
    import torch
    from rsna_train import HEAD_GRID, ClassifierView, TwoHeadNet, make_model

    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")

    if head:
        if "loc.weight" not in state:
            raise SystemExit(
                f"ABORT: {ckpt} is declared as a two headed arm but carries no "
                f"head weights. Either the arm table is wrong or the file was "
                f"overwritten.")
        net = TwoHeadNet(HEAD_GRID, pretrained=False)
        net.load_state_dict(state)
        model = ClassifierView(net)
    else:
        if "loc.weight" in state:
            raise SystemExit(
                f"ABORT: {ckpt} carries head weights but the arm table says it "
                f"has none. Reading it as a single headed model would silently "
                f"drop the second output.")
        model = make_model(torch.device("cpu"))
        model.load_state_dict(state)
    model.eval()
    return model


def fingerprint(model) -> float:
    """One number per weight set, so two arms cannot silently be one file."""
    import torch

    with torch.no_grad():
        return float(sum(t.double().abs().sum() for t in model.state_dict().values()))


def check_provenance(model, root: Path, ids: list, labels: dict, size: int,
                     pred_csv: Path) -> tuple:
    """Does this checkpoint still produce the predictions that were reported?

    A checkpoint file says nothing about which run wrote it. Four of the five
    baseline checkpoints turned out to hold the crop models and the fifth the
    reweighted model, because the runs before the `--tag` switch all wrote to
    `rsna_f{fold}_s{seed}.pth`. Nothing in the file names showed it.

    The prediction CSVs are the ground truth here. Scoring a handful of those
    images again and correlating with the stored `p_clean` settles in seconds
    whether the weights belong to the numbers. A checkpoint that belongs to the
    run reproduces it exactly, so r is 1.0 and the difference is around 1e-7.
    """
    import torch
    from torch.utils.data import DataLoader
    from rsna_train import RsnaDataset, build_transforms, predict

    stored = pd.read_csv(pred_csv).set_index("patientId")["p_clean"]
    common = [i for i in ids if i in stored.index]
    if not common:
        return float("nan"), float("nan")
    ds = RsnaDataset(root, common, labels, build_transforms(size, False))
    p_now, _ = predict(model, DataLoader(ds, batch_size=32, num_workers=0),
                       torch.device("cpu"))
    p_old = stored.loc[common].to_numpy()
    return float(np.corrcoef(p_now, p_old)[0, 1]), float(np.abs(p_now - p_old).max())


# --------------------------------------------------------------------------
# One arm, one fold
# --------------------------------------------------------------------------

def run_arm(key: str, fold: int, args, boxes: dict, labels: dict,
            val_ids: list, crop_params, t0: float) -> pd.DataFrame:
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    from rsna_train import build_transforms

    spec = ARMS[key]
    ckpt = args.ckpt_dir / f"rsna_f{fold}_s{args.seed}{spec['tag']}.pth"
    if not ckpt.exists():
        print(f"  MISSING {ckpt}, {spec['name']} fold {fold} skipped")
        return pd.DataFrame()

    print(f"\nFold {fold}, {spec['name']}: {ckpt.name}")
    model = load_model(ckpt, spec.get("head", False))
    fp = fingerprint(model)

    pos = [i for i in val_ids if i in boxes]
    pred_csv = spec["pred_dir"] / f"rsna_f{fold}_s{args.seed}.csv"
    if pred_csv.exists():
        corr, dmax = check_provenance(model, spec["images"], pos[:args.probe],
                                      labels, args.size, pred_csv)
        print(f"  provenance against {pred_csv}: r = {corr:.6f}, "
              f"largest difference {dmax:.2e}")
        if not (corr >= args.min_corr):
            print("  ABORT: this checkpoint does not reproduce the predictions "
                  "that were reported for it. It belongs to a different run. "
                  "Fold skipped.")
            return pd.DataFrame()
    else:
        print(f"  no {pred_csv}, provenance NOT checked")

    if args.n:
        pos = pos[:args.n]

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    tf = build_transforms(args.size, False)
    rows, no_mask, no_crop = [], 0, 0

    for j, pid in enumerate(pos, 1):
        lung = load_lung(args.masks, pid)
        if lung is None:
            no_mask += 1
            continue
        if spec["crop"]:
            if crop_params is None or pid not in crop_params.index:
                no_crop += 1
                continue
            cp = crop_params.loc[pid]
            if not bool(cp["ok"]):
                no_crop += 1
                continue

        img = Image.open(spec["images"] / f"{pid}.png").convert("L")
        x = tf(img).unsqueeze(0)
        heat = np.clip(cam(input_tensor=x,
                           targets=[BinaryClassifierOutputTarget(1)])[0], 0, None)
        heat = to_reference(heat, REF_SIZE)
        if spec["crop"]:
            heat = uncrop(heat, float(cp["top"]), float(cp["left"]),
                          float(cp["side"]), REF_SIZE)

        # Boxes and lung stay in ORIGINAL coordinates for every arm. That is
        # the point of the back projection: the same boxes with the same area
        # share on both sides of the comparison.
        r = evaluate_map(heat, box_mask(boxes[pid], REF_SIZE), lung)
        r.update({"fold": fold, "patientId": pid, "arm": key,
                  "n_boxes": len(boxes[pid]), "fingerprint": fp})
        rows.append(r)
        if j % 200 == 0:
            print(f"    {j}/{len(pos)}   [{(time.time() - t0) / 60:.1f} min]",
                  flush=True)

    if no_mask:
        print(f"  {no_mask} images without a lung mask skipped")
    if no_crop:
        print(f"  {no_crop} images without usable crop parameters skipped")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fold_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["arm", "fold"])
    return pd.DataFrame({
        "n": g.size(),
        "point_auc": g["point_auc"].mean(),
        "point_auc_lung": g["point_auc_lung"].mean(),
        "hit": g["hit"].mean(),
        "mass": g["mass"].mean(),
        "area": g["area"].mean(),
        "degenerate": g["degenerate"].sum(),
    }).reset_index()


def paired(d: pd.DataFrame, a: str, b: str, col: str) -> dict:
    """b minus a, per fold, over the folds both arms have.

    Paired means the same fold compared with itself, which is the only
    comparison that is not swamped by the spread between folds. Only images
    both arms actually scored enter it.
    """
    da = d[d["arm"] == a].set_index(["fold", "patientId"])[col]
    db = d[d["arm"] == b].set_index(["fold", "patientId"])[col]
    common = da.index.intersection(db.index)
    if len(common) == 0:
        return {}
    m = pd.DataFrame({"a": da.loc[common], "b": db.loc[common]}).reset_index()
    per = m.groupby("fold")[["a", "b"]].mean()
    v = (per["b"] - per["a"]).to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"n_folds": int(v.size), "mean": float(v.mean()) if v.size else float("nan"),
                "sd": float("nan"), "t": float("nan"), "limit": float("nan"),
                "secured": False, "n_images": len(m)}
    sd = float(v.std(ddof=1))
    t = float(v.mean() / (sd / np.sqrt(v.size))) if sd > 0 else float("nan")
    limit = T_LIMIT.get(v.size - 1, 2.0)
    return {"n_folds": int(v.size), "mean": float(v.mean()), "sd": sd, "t": t,
            "limit": limit, "secured": bool(np.isfinite(t) and abs(t) > limit),
            "n_images": len(m)}


def mcnemar(d: pd.DataFrame, a: str, b: str) -> dict:
    da = d[d["arm"] == a].set_index(["fold", "patientId"])["hit"]
    db = d[d["arm"] == b].set_index(["fold", "patientId"])["hit"]
    common = da.index.intersection(db.index)
    if len(common) == 0:
        return {}
    x = da.loc[common].to_numpy(dtype=bool)
    y = db.loc[common].to_numpy(dtype=bool)
    only_a = int((x & ~y).sum())
    only_b = int((~x & y).sum())
    n = only_a + only_b
    chi2 = ((abs(only_a - only_b) - 1) ** 2) / n if n else float("nan")
    return {"only_a": only_a, "only_b": only_b, "chi2": float(chi2)}


def report(d: pd.DataFrame, names: dict) -> None:
    tab = fold_table(d)
    print("\n" + "=" * 88)
    print("PER FOLD")
    print("=" * 88)
    print(tab.round(4).to_string(index=False))

    print("\n" + "=" * 88)
    print("FOUR NUMBERS SIDE BY SIDE, mean over folds")
    print("=" * 88)
    print(f"  {'':<22}{'Punkt-AUC Lunge':>18}{'Punkt-AUC Bild':>18}"
          f"{'Trefferquote':>14}{'Masse':>10}")
    # Die Reihenfolge kommt aus den DATEN, nicht aus einer Liste im Quelltext.
    # Vorher stand hier ("base", "bal10", "crop", ...) fest verdrahtet. Am
    # 06.08.2026 lief dieses Skript mit den drei Phase-5-Armen durch, rechnete
    # 19 Minuten, schrieb alle Dateien, druckte die Tabelle, und in der Tabelle
    # standen NUR die beiden Nulllinien. Kein Fehler, keine Warnung. Ein Filter
    # gegen eine feste Liste laesst alles Neue lautlos fallen.
    vorhanden = set(tab["arm"])
    modelle = [k for k in ARMS if k in vorhanden]
    nullen = [k for k in NULL_ARME if k in vorhanden]
    order = modelle + nullen
    # Der Waechter dahinter. Wenn ein Arm in den Daten steht und nicht in der
    # Tabelle landet, ist die Tabelle falsch, und zwar unsichtbar falsch.
    vergessen = sorted(vorhanden - set(order))
    if vergessen:
        print("\n  ABBRUCH: diese Arme stehen in den Daten und nicht in der")
        print(f"  Tabelle: {', '.join(vergessen)}")
        print("  Ein Arm, der in ARMS oder NULL_ARME fehlt, wuerde sonst still")
        print("  verschwinden, und genau das ist am 06.08. passiert.")
        raise SystemExit(1)
    for k in order:
        g = tab[tab["arm"] == k]
        print(f"  {names.get(k, k):<22}{g['point_auc_lung'].mean():>18.4f}"
              f"{g['point_auc'].mean():>18.4f}{g['hit'].mean():>14.4f}"
              f"{g['mass'].mean():>10.4f}")
    chance = tab["area"].mean()
    print(f"  {'Zufall':<22}{0.5:>18.4f}{0.5:>18.4f}{chance:>14.4f}{chance:>10.4f}")
    print("\n  The point AUC has a chance value of exactly 0.5, whatever the boxes")
    print("  cover and whatever the crop did. The hit rate does not, its chance")
    print("  value is the box area and it moves with the variant.")

    print("\n" + "=" * 88)
    print("PAIRED, primary is the point AUC inside the lung")
    print("=" * 88)
    have = set(d["arm"])
    # Zuerst die Paare, die fuer Phase 2 VOR dem Rechnen festgelegt wurden, in
    # ihrer damaligen Reihenfolge. Eine vorfestgelegte Ausgabe umzusortieren,
    # weil der Quelltext huebscher wird, ist genau die Art Aenderung, gegen die
    # eine Vorfestlegung existiert.
    pairs = [p for p in KANON_PAARE if p[0] in have and p[1] in have]
    # Danach alles Weitere, was in den Daten steht. Beschreibend, nicht
    # bestaetigend: die vorfestgelegten Endpunkte der Phase 5 liegen in
    # rsna_phase5_auswertung.py und werden hier nicht neu verhandelt.
    gesehen = {frozenset(p) for p in pairs}
    weitere = ([("Lagepriore", m) for m in modelle]
               + [(a, b) for i, a in enumerate(modelle) for b in modelle[i + 1:]]
               + [("Lungenkarte", m) for m in modelle])
    for p in weitere:
        if p[0] in have and p[1] in have and frozenset(p) not in gesehen:
            gesehen.add(frozenset(p))
            pairs.append(p)
    for a, b in pairs:
        if a not in have or b not in have:
            continue
        r = paired(d, a, b, "point_auc_lung")
        if not r:
            continue
        print(f"\n  {names.get(b, b)} minus {names.get(a, a)}"
              f"   ({r['n_folds']} folds, {r['n_images']} images)")
        for label, rr in (("point AUC in lung", r),
                          ("hit rate", paired(d, a, b, "hit"))):
            if not rr:
                continue
            if rr["n_folds"] < 2 or not np.isfinite(rr["t"]):
                # One fold has no spread, so there is nothing to test. Printing
                # "not secured" here would be true by construction and would
                # read like a result.
                print(f"    {label:<20}{rr['mean']:+.4f}   NO VERDICT, "
                      f"a spread needs at least two folds")
                continue
            v = "SECURED" if rr["secured"] else "not secured"
            print(f"    {label:<20}{rr['mean']:+.4f} +- {rr['sd']:.4f}   "
                  f"t = {rr['t']:+6.2f}   limit {rr['limit']:.3f}   {v}")
        mc = mcnemar(d, a, b)
        if mc:
            print(f"    McNemar over all images: {mc['only_a']} only "
                  f"{names.get(a, a)}, {mc['only_b']} only {names.get(b, b)}, "
                  f"chi2 = {mc['chi2']:.1f} against {CHI2_LIMIT}")

    print("\n  The fold level t is the verdict. McNemar has more power but treats")
    print("  images inside a fold as independent, which they are not.")
    kanon = {frozenset(q) for q in KANON_PAARE}
    extra = [p for p in pairs if frozenset(p) not in kanon]
    if extra:
        wieviel = ("Everything above is" if len(extra) == len(pairs)
                   else "Everything beyond the phase 2 pairs is")
        print(f"  {wieviel} DESCRIPTIVE: only the phase 2 pairs")
        print("  were pre registered in this script. The confirmatory endpoints")
        print("  of phase 5 live in rsna_phase5_auswertung.py and are not")
        print("  renegotiated here.")

    key = paired(d, "Lagepriore", "base", "point_auc_lung")
    if not key:
        # Der Satz unten wurde fuer Phase 2 vorfestgelegt und gehoert dem
        # Basislinien-Arm. Ohne ihn bleibt er ungeschrieben, und das muss
        # dastehen: eine Ueberschrift ohne Text liest sich wie ein Ergebnis.
        print("\n" + "=" * 88)
        print("NO CLOSING SENTENCE FOR THIS RUN")
        print("=" * 88)
        print("  The sentence below the paired block was pre registered for")
        print("  phase 2 and belongs to the arm 'base', which this run does not")
        print("  contain. It stays unwritten rather than being reused for a")
        print("  different arm.")
    if key:
        print("\n" + "=" * 88)
        print("THE SENTENCE THIS RUN DECIDES")
        print("=" * 88)
        if key["n_folds"] < MIN_FOLDS_FOR_VERDICT or not np.isfinite(key["t"]):
            # A verdict off two folds and a handful of images is worth less
            # than no verdict, because it looks like one. The smoke run used
            # to print the "no secured difference" branch here, which is true
            # by construction when there is no spread to test.
            print(f"  NO VERDICT. This rests on {key['n_folds']} fold(s) and "
                  f"{key['n_images']} images.")
            print(f"  A sentence needs at least {MIN_FOLDS_FOR_VERDICT} folds. "
                  f"This is the shape of a smoke test,")
            print("  the full run covers five.")
        elif key["secured"] and key["mean"] > 0:
            print("  The baseline sits above the location prior, secured over the")
            print("  folds. The model has learned something about WHERE that goes")
            print("  beyond the usual position of opacities, without ever having")
            print("  been trained on a box.")
        elif key["secured"] and key["mean"] < 0:
            print("  The baseline sits BELOW the location prior, secured. The heat")
            print("  map is worse at pointing than a fixed anatomical map. Emergent")
            print("  localisation is not there, and the README has to say so.")
        else:
            print("  No secured difference to the location prior. What the heat map")
            print("  shows is consistent with the usual position of opacities, and")
            print("  the claim 'the model points at the focus 4.6 times more often")
            print("  than chance' has to go, because its denominator is the wrong")
            print("  one. Phase 5, the second head, becomes the way to get")
            print("  localisation rather than a nice addition.")

        # The three sentences above were written before the first number
        # existed and they stay as they are, because rewriting a pre
        # registered sentence after seeing the data is the thing pre
        # registration exists to prevent. The run of 02.08. showed that all
        # three are too SHORT: an average over every image hides that the
        # model and the prior are strong on different images. That is a
        # second reading, not a correction, and it lives in its own script.
        print("\n  This average is not the whole reading. Whether the map is")
        print("  anatomy or about this image is a different question, and it is")
        print("  answered by:  python rsna\\befunde\\rsna_lokalisation_lesart.py")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--masks", type=Path, default=Path("data/rsna/masks224_dev"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--crop-params", type=Path,
                   default=Path("predictions_rsna/crop_params.csv"))
    p.add_argument("--baselines", type=Path, default=Path("predictions_lokalisation"),
                   help="where phase 1 put prior_f*.npy and baselines_f*.csv")
    p.add_argument("--arms", nargs="+", default=["base", "bal10", "crop"],
                   choices=list(ARMS))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=REF_SIZE)
    p.add_argument("--n", type=int, default=0,
                   help="images per fold and arm, 0 = every positive validation "
                        "image, which is the point of this script")
    p.add_argument("--min-corr", type=float, default=0.999)
    p.add_argument("--probe", type=int, default=120,
                   help="images used for the provenance check per fold and arm; "
                        "lower it for a smoke run, it costs a forward pass each")
    p.add_argument("--force", action="store_true",
                   help="recompute combinations that are already on disk")
    p.add_argument("--report-only", action="store_true",
                   help="no Grad-CAM, report from the CSVs that are there")
    p.add_argument("--out-dir", type=Path, default=Path("predictions_cam_full"))
    p.add_argument("--prefix", default="phase2",
                   help="name of the two summary files. A phase 5 run belongs "
                        "in its own folder with its own prefix; writing "
                        "phase2_summary.csv into a phase 5 folder would be a "
                        "file name that lies about its contents")
    args = p.parse_args()

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    boxes = load_boxes(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_params = None
    if "crop" in args.arms and args.crop_params.exists():
        crop_params = pd.read_csv(args.crop_params)
        crop_params["patientId"] = crop_params["patientId"].astype(str)
        crop_params = crop_params.set_index("patientId")

    t0 = time.time()
    if not args.report_only:
        for fold in args.folds:
            val_ids = sp["folds"][fold]["val"]
            for key in args.arms:
                if fold not in ARMS[key]["folds"]:
                    continue
                out = args.out_dir / f"campow_f{fold}_{key}.csv"
                if out.exists() and not args.force:
                    print(f"Fold {fold}, {ARMS[key]['name']}: {out.name} is "
                          f"already there, skipped")
                    continue
                df = run_arm(key, fold, args, boxes, labels, val_ids,
                             crop_params, t0)
                if df.empty:
                    continue
                df.sort_values("patientId").to_csv(out, index=False)
                print(f"  n {len(df)}   point AUC in lung "
                      f"{df['point_auc_lung'].mean():.4f}   hit "
                      f"{df['hit'].mean():.4f}   [{(time.time() - t0) / 60:.1f} min]")

    # ---- gather everything, models and null lines ------------------------
    have = sorted(args.out_dir.glob("campow_f*_*.csv"))
    if not have:
        print("\nnothing measured yet")
        return
    d = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)

    base_files = sorted(Path(args.baselines).glob("baselines_f*.csv"))
    if base_files:
        nulls = pd.concat([pd.read_csv(f) for f in base_files], ignore_index=True)
        nulls = nulls.rename(columns={"map": "arm"})
        # Only the images the models were actually scored on, so that every
        # row of the comparison rests on the same set.
        keep = set(zip(d["fold"], d["patientId"]))
        nulls = nulls[[t in keep for t in zip(nulls["fold"], nulls["patientId"])]]
        d = pd.concat([d, nulls], ignore_index=True)
    else:
        print(f"\nWARNING: no baselines_f*.csv in {args.baselines}. Run "
              f"rsna_lokalisation.py tor first, otherwise the null lines are "
              f"missing and the numbers cannot be read.")

    # Deliberately NOT summary.csv and paired.csv: those names belong to the
    # single fold run of 01.08. in the old format, and overwriting a record is
    # not the same as replacing it.
    d.to_csv(args.out_dir / f"{args.prefix}_per_image.csv", index=False)
    fold_table(d).to_csv(args.out_dir / f"{args.prefix}_summary.csv", index=False)
    names = {k: v["name"] for k, v in ARMS.items()}
    names.update({"Lagepriore": "Lagepriore", "Lungenkarte": "Lungenkarte"})
    report(d, names)
    # args.prefix, nicht "phase2". Der Lauf vom 06.08. schrieb korrekt
    # phase5_summary.csv und MELDETE phase2_summary.csv. Eine Ausgabezeile, die
    # einen anderen Dateinamen nennt als den geschriebenen, schickt den Leser
    # zur falschen Datei oder laesst ihn glauben, --prefix habe nicht gewirkt.
    print(f"\nsaved: {args.out_dir}/{args.prefix}_per_image.csv, "
          f"{args.out_dir}/{args.prefix}_summary.csv, campow_f*_*.csv   "
          f"[{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
