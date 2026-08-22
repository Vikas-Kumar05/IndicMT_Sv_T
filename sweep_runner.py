"""
sweep_runner.py — Single (experiment, seed) worker for the significance sweep
===============================================================================
Called by run_seed_sweep.py as a SEPARATE PROCESS per (experiment, seed) pair,
with MT_LANG and MT_SEED already set as environment variables. Running as a
subprocess (rather than looping in-process) is deliberate: config.py computes
all paths once at import time based on MT_SEED, so each seed needs a fresh
Python process / fresh config import to get correctly seed-suffixed adapter
and prediction paths without any risk of stale module state.

PREREQUISITE: the augmented dataset for the requested experiment must already
exist (i.e. you already ran `python main.py --lang <lang> --steps 5a 5b 5c 5d
5e 5f` once at the default seed). Data-generation (BT/FT/Noise/... sampling)
is intentionally NOT repeated per seed -- it is controlled by config.DATA_SEED
and stays fixed across the sweep, so only fine-tuning stochasticity varies
between seeds. This also saves a large amount of compute (no repeated LaBSE
filtering / R1 inference passes for synthetic data).

Usage (normally invoked by run_seed_sweep.py, not by hand):
    MT_LANG=mni MT_SEED=42 python sweep_runner.py --exp exp4
"""

import argparse
import os
import sys

from config import (
    MT_LANG, MODEL_SEED,
    EXP1_DATASET, EXP1_ADAPTER, EXP1_PREDS_FILE,
    EXP2_DATASET, EXP2_ADAPTER, EXP2_PREDS_FILE,
    EXP3_DATASET, EXP3_ADAPTER, EXP3_PREDS_FILE,
    EXP4_DATASET, EXP4_ADAPTER, EXP4_PREDS_FILE,
    EXP5_DATASET, EXP5_ADAPTER, EXP5_PREDS_FILE,
    EXP6_DATASET, EXP6_ADAPTER, EXP6_PREDS_FILE,
)
from aug_finetune_eval import run_finetune_and_eval

# exp_key -> (dataset_file, adapter_out, pred_file, score_key, label)
EXPERIMENTS = {
    "exp1": (EXP1_DATASET, EXP1_ADAPTER, EXP1_PREDS_FILE, "exp1", "Exp-1 (BT)"),
    "exp2": (EXP2_DATASET, EXP2_ADAPTER, EXP2_PREDS_FILE, "exp2", "Exp-2 (BT+FT)"),
    "exp3": (EXP3_DATASET, EXP3_ADAPTER, EXP3_PREDS_FILE, "exp3", "Exp-3 (BT+FT+Iter)"),
    # "exp4": (EXP4_DATASET, EXP4_ADAPTER, EXP4_PREDS_FILE, "exp4", "Exp-4 (BT+FT+Noise)"),
    "exp5": (EXP5_DATASET, EXP5_ADAPTER, EXP5_PREDS_FILE, "exp5", "Exp-5 (BT+FT+SubwordReg)"),
    # "exp6": (EXP6_DATASET, EXP6_ADAPTER, EXP6_PREDS_FILE, "exp6", "Exp-6 (BT+FT+EngPara)"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()))
    args = p.parse_args()

    dataset_file, adapter_out, pred_file, score_key, label = EXPERIMENTS[args.exp]

    if not os.path.exists(dataset_file):
        print(f"\n  ERROR: {dataset_file} does not exist yet.")
        print(f"  Run the default-seed pipeline first, e.g.:")
        print(f"    python main.py --lang {MT_LANG} --all-aug")
        print(f"  This builds all Exp-1..6 datasets once (at the default seed),")
        print(f"  which the seed sweep then reuses for every seed.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  SWEEP WORKER  |  lang={MT_LANG}  seed={MODEL_SEED}  exp={args.exp}")
    print(f"  Adapter out : {adapter_out}")
    print(f"  Pred file   : {pred_file}")
    print(f"{'='*65}")

    metrics = run_finetune_and_eval(
        score_key    = f"{score_key}_s{MODEL_SEED}",
        dataset_file = dataset_file,
        adapter_out  = adapter_out,
        pred_file    = pred_file,
        label        = f"{label} [seed={MODEL_SEED}]",
        seed         = MODEL_SEED,
    )
    print(f"\n  Seed {MODEL_SEED} / {args.exp} primary (eng2tgt) metrics: {metrics}")


if __name__ == "__main__":
    main()
