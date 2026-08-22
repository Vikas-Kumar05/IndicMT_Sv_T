"""
run_seed_sweep.py — Multi-Seed Significance Sweep
===================================================================
For a chosen set of experiments and seeds, fine-tunes + evaluates each
(experiment, seed) pair as a SEPARATE process (via sweep_runner.py), so
each run gets fresh, correctly-seed-suffixed paths from config.py.

PREREQUISITE:
  Run the normal default-seed pipeline first so the augmented datasets
  and default-seed (16) checkpoints already exist:
      python main.py --lang mni --all-aug

WHAT THIS ADDS:
  Seeds 42 and 123 (in addition to the default 16, already produced by the
  run above) for whichever experiments you list. After this finishes, run
  step9_significance.py to aggregate into a mean +/- std table and to run
  paired bootstrap significance tests between your top-2 systems.

Usage:
    python run_seed_sweep.py --lang mni --experiments exp4 exp6
    python run_seed_sweep.py --lang mni --experiments exp4 exp6 --seeds 42 123
    python run_seed_sweep.py --lang as  --experiments exp1 exp2 exp3 exp4 exp5 exp6
"""

import argparse
import os
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["mni", "as", "brx"])
    p.add_argument("--experiments", nargs="+", required=True,
                   choices=["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123],
                   help="Additional seeds beyond the default (16), which is "
                        "assumed already done by the normal pipeline run.")
    return p.parse_args()


def main():
    args = parse_args()
    total = len(args.experiments) * len(args.seeds)
    print(f"\n{'='*65}")
    print(f"  SEED SWEEP  |  lang={args.lang}")
    print(f"  Experiments : {args.experiments}")
    print(f"  Extra seeds : {args.seeds}  (default seed 16 assumed already run)")
    print(f"  Total runs  : {total}")
    print(f"{'='*65}")

    t0 = time.time()
    done = 0
    for seed in args.seeds:
        for exp in args.experiments:
            done += 1
            print(f"\n[{done}/{total}]  seed={seed}  exp={exp}  "
                  f"({time.time()-t0:.0f}s elapsed)")
            env = os.environ.copy()
            env["MT_LANG"] = args.lang
            env["MT_SEED"] = str(seed)
            result = subprocess.run(
                [sys.executable, "sweep_runner.py", "--exp", exp],
                env=env,
            )
            if result.returncode != 0:
                print(f"  ✗ FAILED: seed={seed} exp={exp} "
                      f"(exit code {result.returncode}). Continuing with "
                      f"remaining runs.")

    print(f"\n{'='*65}")
    print(f"  SEED SWEEP COMPLETE  |  {total} runs in {(time.time()-t0)/60:.1f} min")
    print(f"  Next: python step9_significance.py --lang {args.lang} "
          f"--experiments {' '.join(args.experiments)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
