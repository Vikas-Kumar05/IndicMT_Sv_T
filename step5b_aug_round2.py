"""
step5b_aug_round2.py — Experiment 2: BT + Forward Translation
------------------------------------------------------
"""

import pandas as pd, numpy as np, torch, random, os
from config import (
    TRAIN_PATH, EXP2_DATASET, EXP2_ADAPTER, EXP2_PREDS_FILE, SEED,
)
from aug_techniques import generate_bt, generate_ft_aug
from aug_utils import combine_and_save
from aug_finetune_eval import run_finetune_and_eval

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def main():
    print("\n" + "="*65)
    print("  STEP 5b — EXP-2: BT + FORWARD TRANSLATION AUGMENTATION")
    print("  Augmentation : Original + BT + FT-Aug")
    print("  Fine-tune    : from R1 LoRA checkpoint")
    print("="*65)

    train_df = pd.read_excel(TRAIN_PATH)
    train_df = train_df.dropna(subset=["eng_src","meti_tgt"]).reset_index(drop=True)
    print(f"\n  Training rows (bidirectional): {len(train_df):,}")

    print(f"\n  ── Back-Translation ──────────────────────────────────────")
    bt_df = generate_bt(train_df)
    print(f"  BT pairs: {len(bt_df):,}")

    print(f"\n  ── Forward Translation ───────────────────────────────────")
    ft_df = generate_ft_aug(train_df)
    print(f"  FT-Aug pairs: {len(ft_df):,}")

    if os.path.exists(EXP2_DATASET):
        print(f"\n  [RESUME] Augmented dataset already exists: {EXP2_DATASET}")
        print("  Skipping combine_and_save — delete file to regenerate.")
    else:
        combine_and_save(
            orig_df    = train_df,
            aug_frames = [bt_df, ft_df],
            save_path  = EXP2_DATASET,
            tag        = "exp2_bt_ft",
        )

    print(f"\n  ── Fine-Tune & Evaluate (both directions) ────────────────")
    metrics = run_finetune_and_eval(
        score_key    = "exp2",
        dataset_file = EXP2_DATASET,
        adapter_out  = EXP2_ADAPTER,
        pred_file    = EXP2_PREDS_FILE,
        label        = "Exp-2 (BT+FT)",
    )

    print(f"\n  Exp-2 Primary Results (Eng→Target): {metrics}")
    print("\n  STEP 5b COMPLETE")


if __name__ == "__main__":
    main()
