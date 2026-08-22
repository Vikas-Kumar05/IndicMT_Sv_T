"""
step5e_aug_round5.py — Experiment 5: BT + FT + Subword Regularization
------------------------------------------------------
"""

import os
import random

import numpy as np
import pandas as pd
import torch

from config import (
    TRAIN_PATH,
    EXP5_DATASET, EXP5_ADAPTER, EXP5_PREDS_FILE,
    SEED,
)
from aug_techniques import generate_bt, generate_ft_aug, generate_subword_reg
from aug_utils import combine_and_save
from aug_finetune_eval import run_finetune_and_eval

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main() -> None:
    print("\n" + "=" * 65)
    print("  STEP 5e — EXP-5: BT + FT + SUBWORD REGULARIZATION")
    print("  Augmentation : Original + BT + FT + SubwordReg (English source)")
    print("  Fine-tune    : from R1 LoRA checkpoint (bidirectional)")
    print("  Guard        : LaBSE post-filter on resegmented pairs")
    print("=" * 65)

    train_df = pd.read_excel(TRAIN_PATH)
    train_df = (train_df
                .dropna(subset=["eng_src", "meti_tgt"])
                .reset_index(drop=True))
    print(f"\n  Training rows loaded (bidirectional): {len(train_df):,}")

    # ── Phase 1: Back-Translation ─────────────────────────────────────────────
    print("\n  ── PHASE 1: Back-Translation ─────────────────────────────")
    bt_df = generate_bt(train_df)
    print(f"  BT pairs        : {len(bt_df):,}")

    # ── Phase 2: Forward Translation ─────────────────────────────────────────
    print("\n  ── PHASE 2: Forward Translation ──────────────────────────")
    ft_df = generate_ft_aug(train_df)
    print(f"  FT-Aug pairs    : {len(ft_df):,}")

    # ── Phase 3: Subword Regularization ──────────────────────────────────────
    print("\n  ── PHASE 3: Subword Regularization (English source) ─────")
    print("  Stochastic SentencePiece resegmentation → novel surface forms")
    sr_df = generate_subword_reg(train_df)
    print(f"  SubwordReg pairs: {len(sr_df):,}")

    # ── Phase 4: Combine, Dedup, Cap, Save ───────────────────────────────────
    print("\n  ── PHASE 4: Combine & Save ───────────────────────────────")
    if os.path.exists(EXP5_DATASET):
        print(f"  [RESUME] Augmented dataset already exists: {EXP5_DATASET}")
        print("  Skipping combine_and_save — delete file to regenerate.")
    else:
        combine_and_save(
            orig_df    = train_df,
            aug_frames = [bt_df, ft_df, sr_df],
            save_path  = EXP5_DATASET,
            tag        = "exp5_bt_ft_subword_reg",
        )

    # ── Phase 5: Fine-Tune & Evaluate ────────────────────────────────────────
    print("\n  ── PHASE 5: Fine-Tune & Evaluate (both directions) ───────")
    metrics = run_finetune_and_eval(
        score_key    = "exp5",
        dataset_file = EXP5_DATASET,
        adapter_out  = EXP5_ADAPTER,
        pred_file    = EXP5_PREDS_FILE,
        label        = "Exp-5 (BT+FT+SubwordReg)",
    )

    print(f"\n  Exp-5 Primary Results (Eng→Target): {metrics}")
    print("\n  STEP 5e COMPLETE")


if __name__ == "__main__":
    main()
