"""Phase 4 audit: re-derive every number of the hardware measurement, then ask
the questions the measurement itself could not ask.

WHY THIS EXISTS
---------------
The working rule of this project: before a new phase starts, the conclusions of
the previous one are recomputed by a SECOND script from the raw output. It has
paid for itself once already (phase 3, five corrections in an hour, no compute).

Phase 4 has one soft spot that no amount of re-adding can reach, and it is the
whole foundation: **is the stopwatch measuring computation or is it measuring
how fast Python can hand work to a queue?** DirectML accepts work and returns
immediately. A benchmark that gets this wrong reports beautiful numbers that
mean nothing, and it fails in the flattering direction, which is the dangerous
one.

Three independent checks answer it, and none of them trusts the benchmark's own
arithmetic:

  1  ARITHMETIC. Medians, per-repeat gains and the verdict, recomputed from
     bench.csv. Must match the printed report to the digit.
  2  THE CROSSOVER ACTUALLY HAPPENED. The design claims alternating order; the
     column `order_pos` has to show it. A crossover that was only described in
     a docstring controls nothing.
  3  THE CLOCK IS HONEST. Two of these:
     3a  Accounting: (warm-up + timed steps) * seconds per step must account
         for most of the wall-clock time of the block. If the timer only
         measured enqueueing, the timed portion would be a small fraction of
         the wall clock and the rest would sit in an unexplained remainder.
     3b  Reality: the epoch time extrapolated for adapter 0 is held against the
         epoch times ACTUALLY LOGGED by the eight-epoch runs on that very
         adapter, from predictions_rsna_base/history_f*_s0.csv. This is the
         strongest available test. A stopwatch that measured queueing would
         land an order of magnitude low.

         Note the direction, it matters: the logged `sec` covers the training
         loop AND the prediction pass over the selection split that follows it
         every epoch. The extrapolation covers the training loop alone. So the
         extrapolated number must come out BELOW the logged one, and the gap is
         the selection pass. Extrapolation above the logged time would mean the
         benchmark is systematically slower than the real thing, which needs an
         explanation.

  4  E2 and E3, recomputed against the pre-registered thresholds, plus the one
     thing E2 must never be read as saying.

Nothing here computes on a GPU, nothing trains, runtime is a second.

CLI, from the repository root:
  python rsna\\befunde\\rsna_phase4_pruefung.py
  python rsna\\befunde\\rsna_phase4_pruefung.py --history-dir predictions_rsna_bal10
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Copied deliberately, not imported, so that a changed threshold in
# rsna_hardware.py shows up here as a disagreement instead of moving silently
# along with it. Same reasoning as in the phase 3 audit.
E1_MIN_GAIN = 0.10
E1_MIN_REPEATS = 3
E2_MAX_DLOGIT = 1e-3
E2_MIN_RANKCORR = 0.9999

FINDINGS: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FINDINGS.append(name)
    return cond


def note(text: str) -> None:
    print(f"        {text}")


# --------------------------------------------------------------------------

def check_arithmetic(b: pd.DataFrame) -> None:
    print("\n1  ARITHMETIC, recomputed from bench.csv")
    for loop in ("compute", "voll"):
        d = b[b.loop == loop]
        if not {0, 1} <= set(d.adapter):
            print(f"  skip {loop}: fewer than two adapters in the file")
            continue
        m0 = float(np.median(d[d.adapter == 0].sec_per_step))
        m1 = float(np.median(d[d.adapter == 1].sec_per_step))
        print(f"  {loop:<8} adapter 0 {m0:.4f} s/step, adapter 1 {m1:.4f} "
              f"s/step, gain {(1 - m1 / m0) * 100:+.1f} %")

    d = b[b.loop == "voll"]
    gains = []
    for r in sorted(d.repeat.unique()):
        dd = d[d.repeat == r]
        if not {0, 1} <= set(dd.adapter):
            continue
        gains.append(1 - (float(dd[dd.adapter == 1].sec_per_step.iloc[0]) /
                          float(dd[dd.adapter == 0].sec_per_step.iloc[0])))
    print("  per repeat: " + ", ".join(f"{g * 100:+.1f} %" for g in gains))

    enough = check(f"at least {E1_MIN_REPEATS} repeats", len(gains) >= E1_MIN_REPEATS,
                   f"{len(gains)} found")
    if not enough:
        note("Without them there is no verdict, only a smoke test. The quick")
        note("run of 02.08. 23:54 printed E1 PASSED off ONE repeat; that line")
        note("was the same error as the one-fold smoke test which printed a")
        note("conclusion, and rsna_hardware.py now refuses to print it.")
        return
    check("every repeat clears the pre-registered margin",
          all(g >= E1_MIN_GAIN for g in gains),
          f"smallest {min(gains) * 100:+.1f} %")
    spread = max(gains) - min(gains)
    check("the repeats agree with each other", spread < 0.15,
          f"spread {spread * 100:.1f} percentage points")
    if spread >= 0.15:
        note("Wide spread does NOT mean 'no difference', it means 'measured")
        note("too imprecisely'. The answer is more repeats, not a shrug.")


def check_crossover(b: pd.DataFrame) -> None:
    print("\n2  DID THE CROSSOVER ACTUALLY HAPPEN")
    d = b[b.loop == "voll"]
    if d.empty:
        check("bench.csv contains the full loop", False)
        return
    positions = d.groupby("adapter")["order_pos"].apply(lambda s: sorted(set(s)))
    for adapter, pos in positions.items():
        print(f"  adapter {adapter} ran at order position(s) {pos}")
    n_rep = d.repeat.nunique()
    if n_rep < 2:
        check("crossover is verifiable", False,
              "one repeat cannot alternate anything")
        note("Not a defect of the design, a consequence of the smoke run.")
        return
    check("every adapter ran first at least once and second at least once",
          all(len(p) >= 2 for p in positions), str(dict(positions)))


def check_clock(b: pd.DataFrame, history_dir: Path) -> None:
    print("\n3a  IS THE CLOCK MEASURING COMPUTATION (accounting)")
    if "warmup" not in b.columns:
        check("bench.csv records the warm-up steps", False,
              "column missing, file predates the audit")
        note("Rerun `messen` with the current rsna_hardware.py, then this")
        note("check works. Without the warm-up count the remainder cannot")
        note("be attributed.")
    else:
        worst = 0.0
        for _, r in b.iterrows():
            accounted = (r["steps"] + r["warmup"]) * r["sec_per_step"]
            frac = accounted / r["block_wall_sec"]
            worst = max(worst, abs(1 - frac))
            print(f"  adapter {r['adapter']} {r['loop']:<8} repeat "
                  f"{r['repeat']}: timed work {accounted:.2f} s of "
                  f"{r['block_wall_sec']:.2f} s wall  ({frac * 100:.0f} %)")
        check("the timed work explains most of the wall clock", worst < 0.5,
              f"largest deviation {worst * 100:.0f} %")
        note("A timer that only measured enqueueing would leave the bulk of")
        note("the wall clock unexplained, because the computation would still")
        note("have to happen somewhere.")

    print("\n3b  IS THE CLOCK MEASURING COMPUTATION (against the real logs)")
    hist = sorted(Path(history_dir).glob("history_f*_s*.csv"))
    if not hist:
        check(f"epoch logs found in {history_dir}", False)
        return
    secs = np.concatenate([pd.read_csv(h)["sec"].to_numpy() for h in hist])
    logged = float(np.median(secs))
    print(f"  {len(secs)} logged epochs in {len(hist)} files, median "
          f"{logged:.0f} s, range {secs.min():.0f} to {secs.max():.0f} s")

    d = b[(b.loop == "voll") & (b.adapter == 0)]
    if d.empty:
        check("adapter 0 was measured", False, "no reference to compare to")
        return
    extra = float(np.median(d.epoch_sec_extrapolated))
    print(f"  extrapolated training loop on adapter 0: {extra:.0f} s")

    # The order-of-magnitude test. This is the one that would catch a stopwatch
    # measuring the queue rather than the chip.
    check("the extrapolation is the same order of magnitude as reality",
          0.5 < extra / logged < 2.0, f"ratio {extra / logged:.2f}")

    # The direction test. Logged seconds = training loop + selection pass;
    # extrapolated = training loop only. So extrapolated must be smaller.
    below = check("and it lies BELOW the logged epoch, as it must",
                  extra < logged,
                  f"{extra:.0f} s against {logged:.0f} s, "
                  f"gap {logged - extra:.0f} s")
    note("The gap is the prediction pass over the selection split, which runs")
    note("after every epoch and is not part of the benchmark loop.")
    # How large should the gap be? Order of magnitude, from the benchmark's own
    # numbers rather than from a guess: the selection split is about one sixth
    # of the training part, and a forward pass costs roughly a third of a step
    # that also does backward and optimiser.
    per_step = float(np.median(d.sec_per_step))
    n_fit = int(np.median(d.steps_per_epoch)) * int(np.median(d.batch))
    expected = (n_fit / 5) * (per_step / int(np.median(d.batch))) / 3
    note(f"Order of magnitude expected for that pass: roughly "
         f"{expected:.0f} s ({n_fit // 5} images, forward only, at about a "
         f"third of the cost of a full step).")
    if below and (logged - extra) < expected / 3:
        FINDINGS.append("gap to the logged epoch is implausibly small")
        print("  FAIL  the gap is too small to be the selection pass")
        note(f"{logged - extra:.0f} s against the {expected:.0f} s expected. "
             f"The likely cause is too few timed")
        note("steps, so residual warm-up is still inside the average and the")
        note("seconds per step come out high. More steps per block fix it,")
        note("and the error is in the harmless direction: it understates the")
        note("gain of the card rather than inventing one.")


def check_equality(g: pd.DataFrame) -> None:
    print("\n4a  E2, EQUALITY")
    pair = g[(g.a == 0) & (g.b == 1)]
    if pair.empty:
        check("the two adapters were compared", False)
        return
    r = pair.iloc[0]
    print(f"  adapter 0 vs 1: max dlogit {r['max_dlogit']:.2e}, "
          f"rank corr {r['rank_corr']:.6f}, n={int(r['n'])}")
    check("max logit difference under the pre-registered bound",
          r["max_dlogit"] < E2_MAX_DLOGIT, f"{r['max_dlogit']:.2e}")
    check("the ranking is untouched", r["rank_corr"] > E2_MIN_RANKCORR)

    # The CPU is the arbiter: if the two adapters agree with each other much
    # better than either agrees with the CPU, they share a DirectML-specific
    # deviation, and "they agree" would be the wrong reading.
    cpu = g[(g.b == -1)]
    if len(cpu) >= 2:
        d_cpu = float(cpu["max_dlogit"].max())
        print(f"  largest difference against the CPU: {d_cpu:.2e}")
        check("adapter-to-adapter difference is not larger than "
              "adapter-to-CPU", r["max_dlogit"] <= d_cpu * 1.5,
              f"{r['max_dlogit']:.2e} against {d_cpu:.2e}")
        note("All three at the level of float32 rounding means the DirectML")
        note("path itself is not introducing anything the CPU does not.")

    print("  READING, and this is the one that gets over-read:")
    note("E2 compares ONE forward pass. It does NOT permit comparing an old")
    note("APU run against a new card run. In training the rounding of step one")
    note("moves the weights of step two, so after eight epochs there are two")
    note("different models. Both arms of a comparison stay on one adapter, and")
    note("the 224 px reference arm has to be repeated on the card.")


def check_memory(s: pd.DataFrame) -> None:
    print("\n4b  E3, MEMORY")
    for _, r in s.iterrows():
        print(f"  {int(r['size']):>4} px batch {int(r['batch'])}: "
              f"{'ok' if r['ok'] else 'FAIL'}")
    ok224 = s[(s['size'] == 224) & s.ok]
    check("224 px at batch 16 runs on the card", not ok224.empty)
    ok512 = s[(s['size'] == 512) & s.ok]
    if not ok512.empty:
        note("512 px also fits, so phase 8 does not need a smaller batch and")
        note("the doubled cost feared in the roadmap does not arise.")
        # Cost of phase 8, computed rather than guessed.
        d = s.set_index("size")["sec_per_step"]
        if 224 in d.index and 512 in d.index:
            factor = float(d[512]) / float(d[224])
            note(f"512 px costs {factor:.1f} times the step time of 224 px on "
                 f"the card, so a five-fold run scales accordingly.")
    else:
        note("512 px does not fit. The batch may NOT simply be halved: ResNet18")
        note("normalises per physical batch, so batch 8 is a different model.")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, default=Path("predictions_hardware"))
    p.add_argument("--history-dir", type=Path,
                   default=Path("predictions_rsna_base"),
                   help="the eight-epoch runs whose logged epoch times are the "
                        "ground truth for check 3b. They must come from the "
                        "SAME adapter the benchmark used as reference.")
    a = p.parse_args()

    print("=" * 70)
    print("PHASE 4 AUDIT")
    print("=" * 70)

    bench = a.dir / "bench.csv"
    if bench.exists():
        b = pd.read_csv(bench)
        b["device_name"] = b["device_name"].astype(str).str.strip()
        check_arithmetic(b)
        check_crossover(b)
        check_clock(b, a.history_dir)
    else:
        check(f"{bench} exists", False)

    gl = a.dir / "gleich.csv"
    if gl.exists() and gl.stat().st_size:
        check_equality(pd.read_csv(gl))
    else:
        check(f"{gl} exists", False)

    sp = a.dir / "speicher.csv"
    if sp.exists():
        check_memory(pd.read_csv(sp))
    else:
        check(f"{sp} exists", False)

    print("\n" + "=" * 70)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for f in FINDINGS:
            print(f"  - {f}")
        print("\nA finding is not automatically a defect. Read each one and")
        print("decide; what may not happen is carrying it into phase 5 unread.")
    else:
        print("No finding. Phase 4 carries.")
    sys.exit(1 if FINDINGS else 0)


if __name__ == "__main__":
    main()
