"""
step5c_aug_round3.py — Experiment 3: BT + FT + One Iterative Retraining Cycle
------------------------------------------------------
"""

import os
import pandas as pd, numpy as np, torch, random
from config import (
    TRAIN_PATH,
    EXP2_DATASET, EXP2_ADAPTER, EXP2_PREDS_FILE,
    EXP3_DATASET, EXP3_ADAPTER, EXP3_PREDS_FILE,
    SEED,
)
from aug_techniques import generate_bt, generate_ft_aug, generate_iter
from aug_utils import combine_and_save
from aug_finetune_eval import run_finetune_and_eval

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def _exp2_adapter_ready():
    """True if Exp-2 adapter directory contains a saved model."""
    return os.path.isdir(EXP2_ADAPTER) and any(
        f.endswith((".bin", ".safetensors"))
        for f in os.listdir(EXP2_ADAPTER))


def main():
    print("\n" + "="*65)
    print("  STEP 5c — EXP-3: BT + FT + ONE ITERATIVE RETRAINING CYCLE")
    print("="*65)

    train_df = pd.read_excel(TRAIN_PATH)
    train_df = train_df.dropna(subset=["eng_src","meti_tgt"]).reset_index(drop=True)
    print(f"\n  Training rows (bidirectional): {len(train_df):,}")

    # ── Phase A: Build BT + FT dataset ───────────────────────────────────────
    print(f"\n  PHASE A — Build BT + FT augmented dataset")

    if os.path.exists(EXP2_DATASET):
        print(f"  [RESUME] EXP2_DATASET already exists: {EXP2_DATASET}")
        print("  Skipping BT/FT generation — delete file to regenerate.")
        bt_df = ft_df = None
    else:
        print(f"  ── Back-Translation ──────────────────────────────────────")
        bt_df = generate_bt(train_df)
        print(f"  BT pairs: {len(bt_df):,}")

        print(f"  ── Forward Translation ───────────────────────────────────")
        ft_df = generate_ft_aug(train_df)
        print(f"  FT-Aug pairs: {len(ft_df):,}")

        combine_and_save(
            orig_df    = train_df,
            aug_frames = [bt_df, ft_df],
            save_path  = EXP2_DATASET,
            tag        = "exp2_bt_ft",
        )

    # ── Phase B: Initial fine-tune (reuse Exp-2 if available) ────────────────
    print(f"\n  PHASE B — Initial fine-tune on BT+FT data (from R1)")
    if _exp2_adapter_ready():
        print(f"  ✓ Exp-2 adapter already exists at {EXP2_ADAPTER} — reusing.")
    else:
        print(f"  Exp-2 adapter not found — training now ...")
        run_finetune_and_eval(
            score_key    = "exp2",
            dataset_file = EXP2_DATASET,
            adapter_out  = EXP2_ADAPTER,
            pred_file    = EXP2_PREDS_FILE,
            label        = "Exp-2 (Phase B for Exp-3)",
        )

    # ── Phase C: Pseudo-label generation ─────────────────────────────────────
    print(f"\n  PHASE C — Generate pseudo-labels with Phase B model")
    print(f"  (Model: {EXP2_ADAPTER})")
    pseudo_df = generate_iter(train_df, iter_model_dir=EXP2_ADAPTER)
    print(f"  Pseudo-label pairs kept: {len(pseudo_df):,}")

    # ── Phase D: Final fine-tune on Original + BT + FT + Pseudo ──────────────
    print(f"\n  PHASE D — Build final dataset and fine-tune from R1")
    if os.path.exists(EXP3_DATASET):
        print(f"  [RESUME] EXP3_DATASET already exists: {EXP3_DATASET}")
        print("  Skipping combine_and_save — delete file to regenerate.")
    else:
        aug_frames = [f for f in [bt_df, ft_df, pseudo_df] if f is not None]
        combine_and_save(
            orig_df    = train_df,
            aug_frames = aug_frames,
            save_path  = EXP3_DATASET,
            tag        = "exp3_bt_ft_iter",
        )

    print(f"\n  ── Fine-Tune & Evaluate (both directions) ────────────────")
    metrics = run_finetune_and_eval(
        score_key    = "exp3",
        dataset_file = EXP3_DATASET,
        adapter_out  = EXP3_ADAPTER,
        pred_file    = EXP3_PREDS_FILE,
        label        = "Exp-3 (BT+FT+Iter)",
    )

    print(f"\n  Exp-3 Primary Results (Eng→Target): {metrics}")
    print("\n  STEP 5c COMPLETE")


if __name__ == "__main__":
    main()
