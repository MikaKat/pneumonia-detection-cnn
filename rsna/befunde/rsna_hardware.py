"""Phase 4: which chip is actually computing, is the other one faster, and does
swapping it change the arithmetic.

WHY THIS EXISTS
---------------
`torch_directml.device()` without an index returns adapter 0. On this machine
adapter 0 is the integrated graphics of the CPU and adapter 1 is the RX 5500 XT
in the slot. Every training run of this project up to 02.08.2026 went to the
integrated chip, and nothing in any log said so: `privateuseone:0` names the
interface, not the chip. The card was idle for weeks.

That costs time, not validity. Every comparison this project made was paired
per fold and ran on the same hardware, so the conclusions stand. What is
unknown is how much time was left on the table, and whether the card can even
hold the batch. This script turns both into numbers before a single long run is
started on new hardware.

WHAT IS MEASURED, PRE-REGISTERED BEFORE THE FIRST RUN
-----------------------------------------------------
E1  SPEED, primary. Seconds per training step (forward, backward, optimiser
    step) at 224 px and batch 16, measured in two loops that differ only in
    where the images come from:

      compute  one synthetic batch that already sits on the chip, reused every
               step. No PNG decode, no transforms, no transfer. This is the
               chip alone.
      voll     the real DataLoader over the fit split of fold 0, that is PNG
               decode plus transforms on the CPU plus the transfer. This is
               what an epoch really does.

    Reported per arm: seconds per step in both loops and the epoch time
    extrapolated from `voll` times the real number of steps per epoch.

    The two loops are not decoration. If `compute` gets much faster on the card
    while `voll` does not, then the bottleneck is the CPU-side image pipeline
    and a faster chip buys nothing. That is a different repair (workers,
    pre-decoded tensors) and it would be invisible with only one loop.

E2  EQUALITY. The same weights and the same 64 images through both adapters and
    through the CPU, forward pass only, in eval mode. Reported: the largest
    absolute difference in the logit, the largest in the probability, and the
    rank correlation of the scores.

    What E2 can and cannot say, because this is easy to over-read: it compares
    ONE forward pass. It answers "is this the same arithmetic, up to float32
    rounding". It does NOT license comparing an old APU run against a new card
    run. In training, a rounding difference in step one changes the weights of
    step two, and after eight epochs the two runs are different models, not the
    same model measured twice. The roadmap rule stays: both arms of a
    comparison on the same adapter.

E3  MEMORY. 224, 320, 448 and 512 px, each at batch 16, twenty steps each on
    one adapter. The APU borrows main memory, which is why 320 px next to
    --batch 16 was never a problem. The card has its VRAM and nothing else.

DECISION RULE, fixed here before any number exists
--------------------------------------------------
All new runs move to adapter 1 if and only if

  (a) E1 `voll` seconds per step on adapter 1 is at least 10 percent below
      adapter 0, in every repeat, and
  (b) E2 passes: max |delta logit| between the adapters below 1e-3 and rank
      correlation above 0.9999, and
  (c) E3 shows 224 px at batch 16 completing on adapter 1.

If (a) fails while `compute` is clearly faster, the finding is "the data
pipeline is the ceiling" and the next step is the pipeline, not the chip. If
(b) fails, DirectML computes something materially different on the two
adapters, which is a bug report, not a hardware upgrade. If (c) fails, the card
is unusable for this project at the current batch size and the batch size may
NOT simply be lowered: ResNet18 contains batch normalisation, whose mean and
variance are computed per physical batch, so a batch of 8 is a different model
and not a cheaper way to get the same one. Gradient accumulation does not
repair that either.

MEASUREMENT DESIGN
------------------
Crossover, like everywhere else in this project. Each repeat runs the adapters
in alternating order (0,1 then 1,0 then 0,1) so that a machine which warms up,
throttles or is disturbed by a background process cannot dress that up as a
difference between the adapters. Warm-up steps are timed and thrown away: the
first DirectML call of a process builds shaders and takes seconds.

Timing waits for the chip on purpose. DirectML queues work and returns
immediately, so a naive `time.perf_counter()` around the loop would measure how
fast Python can enqueue. Each block therefore ends by pulling one number of a
MODEL PARAMETER back to the CPU, which cannot happen before the last optimiser
step has finished. Pulling the loss back would not be enough: the loss does not
depend on the optimiser step.

Nothing here writes a checkpoint and nothing here is a training run. Runtime is
a few minutes.

CLI, from the repository root:
  python rsna\\befunde\\rsna_hardware.py liste
  python rsna\\befunde\\rsna_hardware.py messen --adapters 0 1
  python rsna\\befunde\\rsna_hardware.py gleich --adapters 0 1
  python rsna\\befunde\\rsna_hardware.py speicher --adapter 1
  python rsna\\befunde\\rsna_hardware.py bericht
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (puts rsna/pipeline on sys.path)

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rsna_train import (RsnaDataset, build_transforms, dml_adapters,  # noqa: E402
                        inner_split, make_model, pick_device)

OUT_DIR = Path("predictions_hardware")

# Pre-registration, read by `bericht`. Changing a number here after the fact is
# the one thing that would make this whole file worthless.
E1_MIN_GAIN = 0.10          # adapter 1 must be at least 10 % faster per step
E1_MIN_REPEATS = 3          # and it must have been asked at least three times
E2_MAX_DLOGIT = 1e-3        # largest tolerated logit difference between adapters
E2_MIN_RANKCORR = 0.9999    # and the scores must still order the images alike
E3_SIZES = (224, 320, 448, 512)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def resolve(index: int):
    """(device, label) for a DirectML adapter index, -1 meaning the CPU."""
    if index < 0:
        return torch.device("cpu"), "cpu"
    device, _pin, label = pick_device("directml", index)
    return device, label


def sync(model: nn.Module) -> float:
    """Waits for the chip. Returns a number only so nothing optimises it away.

    Reading a PARAMETER, not the loss. The loss is finished one operation
    before the optimiser step, so syncing on it would stop the clock too early
    and make every arm look faster than it is, by an amount that need not be
    the same on the two adapters.
    """
    p = next(model.parameters())
    return float(p.detach().reshape(-1)[:1].cpu())


def split_ids(splits: Path, fold: int, seed: int, inner_splits: int):
    """The same fit/sel/val division the training script would use.

    Taken from the real splits file rather than invented, so that "steps per
    epoch" below is the true number and the extrapolated epoch time can be held
    against the 540 to 590 s the logs of the APU runs actually show.
    """
    sp = json.loads(Path(splits).read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    f = sp["folds"][fold]
    fit_ids, sel_ids = inner_split(f["train"], labels, sp["viewpos"],
                                   seed, inner_splits)
    return labels, fit_ids, sel_ids, f["val"]


def fresh_model(device, seed: int):
    """A new, identically initialised model on `device`.

    Seeded, because two arms whose heads start from different random numbers
    would differ in the loss curve. For E1 that is irrelevant, for E2 it is the
    entire point, and one function for both is one thing less to get wrong.
    """
    torch.manual_seed(seed)
    return make_model(device)


# --------------------------------------------------------------------------
# E1  speed
# --------------------------------------------------------------------------

def block_compute(device, size: int, batch: int, steps: int, warmup: int,
                  seed: int) -> float:
    """Seconds per step with the data already on the chip."""
    model = fresh_model(device, seed)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 3, size, size, generator=g).to(device)
    t = (torch.rand(batch, generator=g) < 0.225).float().to(device)

    def run(n: int) -> float:
        model.train()
        t0 = time.perf_counter()
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x).squeeze(1), t)
            loss.backward()
            opt.step()
        sync(model)
        return time.perf_counter() - t0

    run(warmup)                      # shader build, thrown away
    return run(steps) / steps


def block_voll(device, images: Path, ids, labels, size: int, batch: int,
               steps: int, warmup: int, seed: int, workers: int) -> float:
    """Seconds per step through the real DataLoader, PNG decode included."""
    model = fresh_model(device, seed)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    torch.manual_seed(seed)
    loader = DataLoader(RsnaDataset(images, list(ids), labels,
                                    build_transforms(size, True)),
                        batch_size=batch, num_workers=workers,
                        shuffle=True, drop_last=True)

    def run(n: int) -> float:
        model.train()
        it = iter(loader)
        # The iterator is built BEFORE the clock starts. On Windows with
        # workers=0 that is cheap, with workers>0 it forks processes, and that
        # cost belongs to neither adapter.
        t0 = time.perf_counter()
        done = 0
        while done < n:
            try:
                x, t = next(it)
            except StopIteration:
                it = iter(loader)
                continue
            x, t = x.to(device), t.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x).squeeze(1), t)
            loss.backward()
            opt.step()
            done += 1
        sync(model)
        return time.perf_counter() - t0

    run(warmup)
    return run(steps) / steps


def cmd_messen(a) -> None:
    labels, fit_ids, _sel, _val = split_ids(a.splits, a.fold, a.seed,
                                            a.inner_splits)
    steps_per_epoch = len(fit_ids) // a.batch
    print(f"fit split of fold {a.fold}: {len(fit_ids)} images, "
          f"{steps_per_epoch} steps per epoch at batch {a.batch}")
    print(f"{a.repeats} repeats, {a.steps} timed steps per block, "
          f"{a.warmup} warm-up steps, adapters in alternating order\n")

    rows = []
    for r in range(a.repeats):
        # Crossover: reverse the order in every second repeat.
        order = list(a.adapters) if r % 2 == 0 else list(reversed(a.adapters))
        for idx in order:
            device, label = resolve(idx)
            for loop in ("compute", "voll"):
                t0 = time.perf_counter()
                if loop == "compute":
                    per = block_compute(device, a.size, a.batch, a.steps,
                                        a.warmup, a.seed)
                else:
                    per = block_voll(device, a.images, fit_ids, labels, a.size,
                                     a.batch, a.steps, a.warmup, a.seed,
                                     a.workers)
                rows.append({"repeat": r, "order_pos": order.index(idx),
                             "adapter": idx, "device_name": label,
                             "loop": loop, "size": a.size, "batch": a.batch,
                             "steps": a.steps, "warmup": a.warmup,
                             "sec_per_step": per,
                             "steps_per_epoch": steps_per_epoch,
                             "epoch_sec_extrapolated": per * steps_per_epoch,
                             "block_wall_sec": time.perf_counter() - t0})
                print(f"  repeat {r}  adapter {idx} ({label})  {loop:<8} "
                      f"{per:.4f} s/step   -> epoch "
                      f"{per * steps_per_epoch / 60:.1f} min")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "bench.csv", index=False)
    print(f"\nwritten: {OUT_DIR / 'bench.csv'}")
    report_speed(df)


# --------------------------------------------------------------------------
# E2  equality
# --------------------------------------------------------------------------

def logits_on(device, state, ids, images: Path, labels, size: int,
              batch: int, seed: int) -> np.ndarray:
    model = fresh_model(device, seed)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    loader = DataLoader(RsnaDataset(images, list(ids), labels,
                                    build_transforms(size, False)),
                        batch_size=batch, num_workers=0, shuffle=False)
    out = []
    with torch.no_grad():
        for x, _t in loader:
            out.append(model(x.to(device)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman without scipy: Pearson on the ranks."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_gleich(a) -> None:
    labels, _fit, _sel, val_ids = split_ids(a.splits, a.fold, a.seed,
                                            a.inner_splits)
    ids = sorted(val_ids)[:a.n]          # deterministic, not a random draw
    state = None
    if a.ckpt and Path(a.ckpt).exists():
        # map_location="cpu" is mandatory. torch.load straight onto a DirectML
        # device dies with a TypeError; see the module rules.
        state = torch.load(a.ckpt, map_location="cpu")
        print(f"weights: {a.ckpt}")
    else:
        print(f"weights: ImageNet plus a head seeded with {a.seed} "
              f"(no checkpoint given or found)")
    print(f"{len(ids)} fixed val images of fold {a.fold}\n")

    scores = {}
    # The CPU is always the third arm: it is the arbiter when the two adapters
    # disagree. dict.fromkeys de-duplicates and keeps the order, so calling
    # this with `--adapters -1` does not silently compare the CPU with itself
    # and hand back an empty table.
    for idx in dict.fromkeys([int(i) for i in a.adapters] + [-1]):
        device, label = resolve(idx)
        scores[idx] = logits_on(device, state, ids, a.images, labels, a.size,
                                a.batch, a.seed)
        print(f"  adapter {idx:>2} ({label}) done")

    rows = []
    keys = list(scores)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            la, lb = scores[ka], scores[kb]
            pa, pb = 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lb))
            rows.append({"a": ka, "b": kb, "n": len(la),
                         "max_dlogit": float(np.abs(la - lb).max()),
                         "mean_dlogit": float(np.abs(la - lb).mean()),
                         "max_dprob": float(np.abs(pa - pb).max()),
                         "rank_corr": rank_corr(la, lb)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gleich.csv", index=False)
    print(f"\nwritten: {OUT_DIR / 'gleich.csv'}")
    report_equal(df)


# --------------------------------------------------------------------------
# E3  memory
# --------------------------------------------------------------------------

def cmd_speicher(a) -> None:
    device, label = resolve(a.adapter)
    print(f"adapter {a.adapter} ({label}), batch {a.batch}, "
          f"{a.steps} steps per size\n")
    rows = []
    for size in a.sizes:
        t0 = time.perf_counter()
        try:
            per = block_compute(device, size, a.batch, a.steps, 2, a.seed)
            ok, err = True, ""
            print(f"  {size:>4} px  ok    {per:.4f} s/step")
        except Exception as exc:                       # noqa: BLE001
            per, ok, err = float("nan"), False, f"{type(exc).__name__}: {exc}"
            print(f"  {size:>4} px  FAIL  {err[:120]}")
        rows.append({"adapter": a.adapter, "device_name": label, "size": size,
                     "batch": a.batch, "ok": ok, "sec_per_step": per,
                     "wall_sec": time.perf_counter() - t0, "error": err})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "speicher.csv", index=False)
    print(f"\nwritten: {OUT_DIR / 'speicher.csv'}")
    print("Reminder: a size that fails may NOT be rescued by halving the "
          "batch. Batch normalisation makes a batch of 8 a different model, "
          "so the reference arm would have to be repeated at 8 as well.")


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def read_or_empty(path: Path):
    """A CSV with a header but no rows is a real state here, not an accident.

    `gleich` writes an empty table when only one arm was runnable. Letting
    pandas raise EmptyDataError in the report would hide the two reports that
    DO have data behind a stack trace.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def report_speed(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E1  SPEED")
    print("=" * 70)
    piv = (df.groupby(["adapter", "device_name", "loop"])["sec_per_step"]
             .agg(["median", "min", "max", "count"]).reset_index())
    for _, r in piv.iterrows():
        print(f"  adapter {r['adapter']} {r['device_name'][:34]:<34} "
              f"{r['loop']:<8} median {r['median']:.4f} s/step  "
              f"(range {r['min']:.4f} to {r['max']:.4f}, n={int(r['count'])})")

    for loop in ("compute", "voll"):
        d = df[df.loop == loop]
        if not {0, 1} <= set(int(x) for x in d.adapter):
            continue
        m0 = d[d.adapter == 0].sec_per_step.median()
        m1 = d[d.adapter == 1].sec_per_step.median()
        gain = 1 - m1 / m0
        word = "faster" if gain > 0 else "SLOWER"
        print(f"\n  {loop}: adapter 1 needs {m1 / m0:.2f} times the step time "
              f"of adapter 0, that is {abs(gain) * 100:.1f} percent {word}")
        if loop == "voll":
            # Every repeat separately, because the pre-registration says "in
            # every repeat" and a median can hide one reversed repeat.
            per_rep = []
            for r in sorted(d.repeat.unique()):
                dd = d[d.repeat == r]
                if {0, 1} <= set(dd.adapter):
                    per_rep.append(1 - (dd[dd.adapter == 1].sec_per_step.iloc[0] /
                                        dd[dd.adapter == 0].sec_per_step.iloc[0]))
            print("  gain per repeat (positive = adapter 1 faster): " +
                  ", ".join(f"{g * 100:+.1f} %" for g in per_rep))
            passed = bool(per_rep) and all(g >= E1_MIN_GAIN for g in per_rep)
            # NO verdict from a smoke run. The first quick pass on 02.08.
            # printed "E1 PASSED" off a single repeat, which is the same
            # mistake as the one-fold smoke test that printed a conclusion:
            # a single number has no spread, and a rule that reads "in EVERY
            # repeat" cannot be checked against one of them.
            if len(per_rep) < E1_MIN_REPEATS:
                print(f"  E1 NO VERDICT: {len(per_rep)} repeat(s), the "
                      f"pre-registration needs {E1_MIN_REPEATS}. A smoke run "
                      f"shows THAT the card computes, not HOW MUCH faster.")
                print(f"  (it would read {'PASSED' if passed else 'FAILED'} on "
                      f"this evidence, which is precisely why it is not "
                      f"printed as the verdict)")
                continue
            print(f"  E1 {'PASSED' if passed else 'FAILED'} "
                  f"(pre-registered: at least {E1_MIN_GAIN * 100:.0f} percent "
                  f"faster in EVERY repeat, over at least "
                  f"{E1_MIN_REPEATS} repeats)")
            if not passed:
                c = df[df.loop == "compute"]
                if {0, 1} <= set(c.adapter):
                    cg = 1 - (c[c.adapter == 1].sec_per_step.median() /
                              c[c.adapter == 0].sec_per_step.median())
                    if cg >= E1_MIN_GAIN:
                        print("  But `compute` IS faster on adapter 1. The "
                              "reading fixed in advance for exactly this case: "
                              "the CPU image pipeline is the ceiling, and the "
                              "next repair is the pipeline, not the chip.")

    d = df[df.loop == "voll"]
    for adapter in sorted(d.adapter.unique()):
        e = d[d.adapter == adapter].epoch_sec_extrapolated.median()
        print(f"  extrapolated epoch, adapter {adapter}: {e / 60:.1f} min "
              f"({e:.0f} s). The APU logs of the 224 px runs show 540 to 590 s.")


def report_equal(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E2  EQUALITY (one forward pass, identical weights)")
    print("=" * 70)
    if df.empty:
        print("  no pair to compare: fewer than two distinct arms were run.")
        return

    def nm(i) -> str:
        return "cpu" if int(i) < 0 else f"adapter {int(i)}"

    for _, r in df.iterrows():
        print(f"  {nm(r['a'])} vs {nm(r['b'])}: "
              f"max dlogit {r['max_dlogit']:.2e}, mean {r['mean_dlogit']:.2e}, "
              f"max dprob {r['max_dprob']:.2e}, rank corr {r['rank_corr']:.6f}")
    pair = df[(df.a == 0) & (df.b == 1)]
    if not pair.empty:
        r = pair.iloc[0]
        ok = (r["max_dlogit"] < E2_MAX_DLOGIT
              and r["rank_corr"] > E2_MIN_RANKCORR)
        print(f"\n  E2 {'PASSED' if ok else 'FAILED'} (pre-registered: "
              f"max dlogit < {E2_MAX_DLOGIT:g} and rank corr > "
              f"{E2_MIN_RANKCORR})")
    print("  What this does NOT say: that an old APU run may be compared "
          "against a new card run. One forward pass agreeing to float32 "
          "rounding says nothing about eight epochs, where the rounding of "
          "step one moves the weights of step two. Both arms of a comparison "
          "stay on one adapter.")


def cmd_bericht(a) -> None:
    print(f"host {platform.node()}, {platform.platform()}")
    print(f"torch {torch.__version__}")
    names = dml_adapters()
    for i, n in enumerate(names):
        print(f"  adapter {i}: {n}")
    for f, fn in (("bench.csv", report_speed), ("gleich.csv", report_equal)):
        p = OUT_DIR / f
        d = read_or_empty(p)
        if d is None:
            print(f"\n{p} is missing or empty, run `messen` / `gleich` first.")
        else:
            fn(d)
    p = OUT_DIR / "speicher.csv"
    print("\n" + "=" * 70)
    print("E3  MEMORY")
    print("=" * 70)
    d = read_or_empty(p)
    if d is not None:
        for _, r in d.iterrows():
            print(f"  {r['size']:>4} px batch {r['batch']}: "
                  f"{'ok' if r['ok'] else 'FAIL ' + str(r['error'])[:100]}")
        big = d[d.ok]
        if not big.empty:
            print(f"  largest size that completes: {int(big['size'].max())} px")
    else:
        print(f"  {p} is missing, run `speicher` first.")


def cmd_liste(a) -> None:
    names = dml_adapters()
    if not names:
        print("torch-directml reports no adapter.")
        return
    print(f"{len(names)} DirectML adapters:")
    for i, n in enumerate(names):
        note = ("  <- default of torch_directml.device(), the integrated "
                "graphics on this machine" if i == 0 else "")
        print(f"  {i}  {n}{note}")
    print("\nWhat trained on what: results_rsna.csv now carries device_name "
          "and dml_index. Rows without those two columns are older than "
          "02.08.2026 and ran on adapter 0.")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
        q.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
        q.add_argument("--fold", type=int, default=0)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--inner-splits", type=int, default=6)
        q.add_argument("--size", type=int, default=224)
        q.add_argument("--batch", type=int, default=16)
        return q

    sub.add_parser("liste", help="list the DirectML adapters").set_defaults(
        func=cmd_liste)

    q = common(sub.add_parser("messen", help="E1, seconds per training step"))
    q.add_argument("--adapters", type=int, nargs="+", default=[0, 1])
    q.add_argument("--steps", type=int, default=30)
    q.add_argument("--warmup", type=int, default=5)
    q.add_argument("--repeats", type=int, default=3)
    q.add_argument("--workers", type=int, default=0)
    q.set_defaults(func=cmd_messen)

    q = common(sub.add_parser("gleich", help="E2, same weights on both chips"))
    q.add_argument("--adapters", type=int, nargs="+", default=[0, 1])
    q.add_argument("--n", type=int, default=64)
    q.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/rsna_f0_s0_base.pth"))
    q.set_defaults(func=cmd_gleich)

    q = common(sub.add_parser("speicher", help="E3, which sizes still fit"))
    q.add_argument("--adapter", type=int, default=1)
    q.add_argument("--sizes", type=int, nargs="+", default=list(E3_SIZES))
    q.add_argument("--steps", type=int, default=20)
    q.set_defaults(func=cmd_speicher)

    sub.add_parser("bericht", help="print the verdict from the CSVs"
                   ).set_defaults(func=cmd_bericht)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
