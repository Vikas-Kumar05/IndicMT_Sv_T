"""
step5a_aug_round1.py — Experiment 1: Back-Translation
------------------------------------------------------
"""

import pandas as pd, numpy as np, torch, random
from config import (
    TRAIN_PATH, EXP1_DATASET, EXP1_ADAPTER, EXP1_PREDS_FILE, SEED,
)
from aug_techniques import generate_bt
from aug_utils import combine_and_save
from aug_finetune_eval import run_finetune_and_eval
import os
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def main():
    print("\n" + "="*65)
    print("  STEP 5a — EXP-1: BACK-TRANSLATION AUGMENTATION")
    print("  Augmentation : Original + BT")
    print("  Fine-tune    : from R1 LoRA checkpoint")
    print("="*65)

    train_df = pd.read_excel(TRAIN_PATH)
    train_df = train_df.dropna(subset=["eng_src","meti_tgt"]).reset_index(drop=True)
    print(f"\n  Training rows (bidirectional): {len(train_df):,}")

    print(f"\n  ── Back-Translation ──────────────────────────────────────")
    """bt_df = generate_bt(train_df)
    print(f"  BT pairs: {len(bt_df):,}")

    combine_and_save(
        orig_df    = train_df,
        aug_frames = [bt_df],
        save_path  = EXP1_DATASET,
        tag        = "exp1_bt",
    )"""
    

    # ---------------------------------------------------------
    # BT generation
    # ---------------------------------------------------------

    if os.path.exists(EXP1_DATASET):

        print(f"\n  [RESUME] Augmented dataset already exists:")
        print(f"  {EXP1_DATASET}")

        print("  Skipping BT regeneration + combine step.")

    else:
        print(f"\n  ── Back-Translation ──────────────────────────────────────")

        bt_df = generate_bt(train_df)

        print(f"  BT pairs: {len(bt_df):,}")

        combine_and_save(
            orig_df    = train_df,
            aug_frames = [bt_df],
            save_path  = EXP1_DATASET,
            tag        = "exp1_bt",
        )    

    print(f"\n  ── Fine-Tune & Evaluate (both directions) ────────────────")
    metrics = run_finetune_and_eval(
        score_key    = "exp1",
        dataset_file = EXP1_DATASET,
        adapter_out  = EXP1_ADAPTER,
        pred_file    = EXP1_PREDS_FILE,
        label        = "Exp-1 (BT)",
    )

    print(f"\n  Exp-1 Primary Results (Eng→Target): {metrics}")
    print("\n  STEP 5a COMPLETE")


if __name__ == "__main__":
    main()
