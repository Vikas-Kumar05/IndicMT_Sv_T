"""
main.py — Master Orchestrator | Eng<->Indic Bidirectional MT Pipeline
=============================================================
Pipeline steps:
  1   — Dataset Analysis, Cleaning & Bidirectional Prep (pre-split train/dev/test)
  2   — Zero-Shot Baseline Evaluation (both directions)
  3   — R1 LoRA Fine-Tuning (bidirectional)
  4   — Evaluate R1 + Compare with Baseline (both directions)
  5a  — Exp-1 : BT augmentation -> fine-tune -> eval
  5b  — Exp-2 : BT + FT augmentation -> fine-tune -> eval
  5c  — Exp-3 : BT + FT + Iterative Retraining -> fine-tune -> eval

  5e  — Exp-5 : BT + FT + Subword Regularization -> fine-tune -> eval

  5g  — Exp-7 : Hallucination Mitigation (decode-time, on best checkpoint) [NEW]
  8   — Final Comparison (all systems, both directions)

LANGUAGE SELECTION (NEW):
  --lang {mni,as,brx}   selects English<->Manipuri, English<->Assamese, or
                        English<->Bodo. Sets the MT_LANG env var, which
                        config.py reads to namespace all data/output/
                        checkpoint paths so the three languages never
                        collide with each other. Default: mni.

BEYOND THIS SCRIPT (separate tools, run after the steps above):
  run_seed_sweep.py       — re-runs chosen Exp-N under extra random seeds,
                            for significance testing (see step9 below).
  step9_significance.py   — aggregates multi-seed results into mean+/-std
                            tables and runs paired-bootstrap significance
                            tests between any two systems.
  See RUNBOOK.md for the full recommended sequence.

Usage:
    python main.py --lang mni                     # full pipeline, Manipuri
    python main.py --lang as  --step 3             # single step, Assamese
    python main.py --lang brx --steps 5a 5b 5c 8   # specific steps, Bodo
    python main.py --lang mni --from-step 5a       # from step onwards
    python main.py --lang mni --aug-only           # 5a->5b->5c->8  (original exps)
    python main.py --lang mni --new-aug-only       # 5d->5e->5f->8  (new techniques)
    python main.py --lang mni --all-aug            # 5a->..->5f->8  (all aug + compare)
    python main.py --lang mni --all-aug-plus-hallu # 5a->..->5f->5g->8 (+ Exp-7)
"""

import argparse
import os
import sys
import time
import traceback
import importlib

# ── Language MUST be set before any module that imports config.py ─────────

def _preparse_lang() -> str:
    """
    Peek at --lang / MT_LANG without fully parsing args (argparse hasn't
    run yet, and config.py gets imported the moment any step module is
    imported). Falls back to 'mni' if not given, matching the original
    single-language behaviour.
    """
    if "--lang" in sys.argv:
        idx = sys.argv.index("--lang")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.environ.get("MT_LANG", "mni")


os.environ["MT_LANG"] = _preparse_lang()

# ── Step registry ──────────────────────────────────────────────────────────────
STEPS = {
    "1"  : ("Dataset Analysis, Cleaning & Bidirectional Prep (pre-split files)", "step1_analysis"),
    "2"  : ("Zero-Shot Baseline Evaluation (both directions)",         "step2_baseline"),
    "3"  : ("R1 LoRA Fine-Tuning (bidirectional)",                    "step3_finetune"),
    "4"  : ("Evaluate R1 + Compare with Baseline",                    "step4_evaluate"),
    "5a" : ("Exp-1: BT augmentation -> fine-tune -> eval",              "step5a_aug_round1"),
    "5b" : ("Exp-2: BT+FT augmentation -> fine-tune -> eval",           "step5b_aug_round2"),
    "5c" : ("Exp-3: BT+FT+Iterative Retraining -> eval",              "step5c_aug_round3"),
    "5d" : ("Exp-4: BT+FT+Synthetic Noise Injection -> eval",          "step5d_aug_round4"),
    "5e" : ("Exp-5: BT+FT+Subword Regularization -> eval",            "step5e_aug_round5"),
    "5f" : ("Exp-6: BT+FT+English-side Paraphrasing -> eval",         "step5f_aug_round6"),
    "5g" : ("Exp-7: Hallucination Mitigation (decode-time)",           "step5g_hallucination_mitigation"),
    "8"  : ("Final Comparison — all systems, both directions",         "step8_final_comparison"),
}

STEP_ORDER              = ["1", "2", "3", "4", "5a", "5b", "5c", "5d", "5e", "5f", "8"]
STEP_ORDER_WITH_HALLU   = ["1", "2", "3", "4", "5a", "5b", "5c", "5d", "5e", "5f", "5g", "8"]
AUG_STEPS                = ["5a", "5b", "5c", "8"]                      # original aug experiments
NEW_AUG_STEPS            = ["5d", "5e", "5f", "8"]                      # new aug experiments
ALL_AUG_STEPS            = ["5a", "5b", "5c", "5d", "5e", "5f", "8"]     # all aug + comparison
ALL_AUG_STEPS_PLUS_HALLU = ["5a", "5b", "5c", "5d", "5e", "5f", "5g", "8"]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_step(step_key: str) -> None:
    desc, module_name = STEPS[step_key]
    bar = "█" * 65
    print(f"\n{bar}")
    print(f"  STEP {step_key}: {desc}   [lang={os.environ['MT_LANG']}]")
    print(f"{bar}")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  ✓ Step {step_key} completed in {elapsed:.1f}s "
              f"({elapsed/60:.1f} min)")
    except Exception as exc:
        print(f"\n  ✗ Step {step_key} FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eng<->Indic Low-Resource MT Pipeline — LoRA + Augmentation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--lang", type=str, default="mni", choices=["mni", "as", "brx"],
                   help="Language pair: mni (Manipuri), as (Assamese), brx (Bodo). "
                        "Already applied before arg-parsing via MT_LANG env var; "
                        "this flag exists mainly for --help / validation.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step",         type=str,
                   help="Run a single step, e.g. --step 5d")
    g.add_argument("--steps",        type=str, nargs="+",
                   help="Run specific steps, e.g. --steps 5d 5e 5f 8")
    g.add_argument("--from-step",    type=str, dest="from_step",
                   help="Run from this step to the end, e.g. --from-step 5d")
    g.add_argument("--aug-only",     action="store_true",
                   help="Run original aug experiments: 5a->5b->5c->8")
    g.add_argument("--new-aug-only", action="store_true",
                   help="Run new aug experiments: 5d->5e->5f->8")
    g.add_argument("--all-aug",      action="store_true",
                   help="Run all aug experiments: 5a->5b->5c->5d->5e->5f->8")
    g.add_argument("--all-aug-plus-hallu", action="store_true",
                   help="All aug experiments + Exp-7 hallucination mitigation: "
                        "5a->..->5f->5g->8")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Re-assert in case --lang differed from the pre-parsed env value
    # (shouldn't happen given _preparse_lang, but keep it bullet-proof).
    if os.environ.get("MT_LANG") != args.lang:
        print(f"  ERROR: --lang {args.lang} was parsed but MT_LANG env var is "
              f"'{os.environ.get('MT_LANG')}'. This should not happen — "
              f"please report. Re-run with only --lang set.")
        sys.exit(1)

    if args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = args.steps
    elif args.from_step:
        order = STEP_ORDER_WITH_HALLU if args.from_step == "5g" else STEP_ORDER
        if args.from_step not in order:
            print(f"  ERROR: --from-step '{args.from_step}' not recognised.")
            print(f"  Valid steps: {order}")
            sys.exit(1)
        start        = order.index(args.from_step)
        steps_to_run = order[start:]
    elif args.aug_only:
        steps_to_run = AUG_STEPS
    elif args.new_aug_only:
        steps_to_run = NEW_AUG_STEPS
    elif args.all_aug:
        steps_to_run = ALL_AUG_STEPS
    elif args.all_aug_plus_hallu:
        steps_to_run = ALL_AUG_STEPS_PLUS_HALLU
    else:
        steps_to_run = STEP_ORDER   # full pipeline (Exp-1..6 + compare; run
                                     # step 5g separately once you know which
                                     # checkpoint won, or use --all-aug-plus-hallu)

    # Validate steps before starting
    invalid = [s for s in steps_to_run if s not in STEPS]
    if invalid:
        print(f"  ERROR: unrecognised step(s): {invalid}")
        print(f"  Valid steps: {list(STEPS.keys())}")
        sys.exit(1)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Eng<->Indic MT  |  LoRA + Augmentation Pipeline")
    print(f"  Language     : {args.lang}")
    print(f"  Steps to run : {steps_to_run}")
    print(f"{sep}")
    print("\n  Step descriptions:")
    for s in steps_to_run:
        print(f"    {s:3s} — {STEPS[s][0]}")

    t_start = time.time()
    for step in steps_to_run:
        run_step(step)

    total = time.time() - t_start
    print(f"\n{sep}")
    print(f"  ALL STEPS COMPLETE  |  Total wall-time: {total/60:.1f} min")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
