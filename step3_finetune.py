"""
step3_finetune.py — R1 LoRA Fine-Tuning on Bidirectional Data
==============================================================
Fine-tunes sarvam-translate (Gemma3-4B-IT) with LoRA on the bidirectional
training split. Both directions trained from a single adapter.

"""

from transformers import DataCollatorForSeq2Seq
import os, json, warnings, random
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    EarlyStoppingCallback, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from config import (
    MODEL_NAME, MAX_SEQ_LEN, SEED, MT_LANG,
    OUTPUT_DIR, FIGURES_DIR, ADAPTER_DIR,
    TRAIN_PATH, DEV_PATH,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, USE_RSLORA,
    NUM_TRAIN_EPOCHS_R1, PER_DEVICE_TRAIN_BATCH, PER_DEVICE_EVAL_BATCH,
    GRAD_ACCUM_STEPS, LEARNING_RATE, LR_SCHEDULER, WARMUP_RATIO,
    WEIGHT_DECAY, MAX_GRAD_NORM, SAVE_TOTAL_LIMIT, LOGGING_STEPS,
    EARLY_STOPPING_PATIENCE, PROMPT_TEMPLATES,
    LABEL_SMOOTHING, OPTIMIZER,
)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

DIRECTION_TO_PROMPT = {
    "eng2tgt": PROMPT_TEMPLATES["eng2tgt"],
    "tgt2eng": PROMPT_TEMPLATES["tgt2eng"],
}
IGNORE_INDEX = -100   # CrossEntropyLoss ignores this label value


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _adapter_exists(adapter_dir: str) -> bool:
    """Return True if a FINAL adapter (not just a checkpoint) is saved."""
    if not os.path.isdir(adapter_dir):
        return False
    return any(
        f.endswith((".bin", ".safetensors"))
        for f in os.listdir(adapter_dir)
    )


def _find_latest_checkpoint(output_dir: str):
    """
    Find the latest Trainer checkpoint-NNNN directory for auto-resume.
    Returns the path string, or None if no checkpoints exist.
    """
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not ckpts:
        return None
    # Sort numerically by step number
    ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    latest = ckpts[-1]
    print(f"  [RESUME] Found checkpoint: {latest}")
    return latest


# ── Prompt-masked tokenisation ─────────────────────────────────────────────────

def format_and_tokenize(row, tok) -> dict:
    """
    Tokenise one training example with prompt masking.

    Structure:
        <prompt_ids>  → labels = IGNORE_INDEX  (no loss on prompt)
        <target_ids>  → labels = token_id      (loss computed here)
        <eos>         → labels = eos_id        (loss computed here)

    This is the standard instruction fine-tuning practice.
    Signal efficiency: ~40-60% (was ~20-30% before fix).
    """
    src       = str(row["eng_src"])
    tgt       = str(row["meti_tgt"])
    direction = str(row.get("direction", "eng2tgt"))
    sys_msg   = DIRECTION_TO_PROMPT.get(direction, PROMPT_TEMPLATES["eng2tgt"])

    # Build the prompt without the target
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": src},
    ]
    prompt_str = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)

    # Tokenise prompt and target separately to find the boundary precisely
    prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
    target_ids = tok(tgt + tok.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids

    # Truncate to MAX_SEQ_LEN — always preserve target; trim prompt if needed
    if len(input_ids) > MAX_SEQ_LEN:
        target_ids = target_ids[:MAX_SEQ_LEN]
        prompt_ids = prompt_ids[:MAX_SEQ_LEN - len(target_ids)]
        input_ids  = prompt_ids + target_ids

    # Labels: IGNORE_INDEX for prompt portion, actual ids for target
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)

    # Pad to MAX_SEQ_LEN
    pad_len        = MAX_SEQ_LEN - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids      = list(input_ids) + [tok.pad_token_id] * pad_len
    labels         = labels          + [IGNORE_INDEX]     * pad_len

    return {
        "input_ids"      : input_ids,
        "attention_mask" : attention_mask,
        "labels"         : labels,
    }


def df_to_hf(df: pd.DataFrame, tok) -> Dataset:
    records = [format_and_tokenize(row, tok) for _, row in df.iterrows()]
    ds = Dataset.from_list(records)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ── Loss / overfitting plots ──────────────────────────────────────────────────

def plot_loss(history, fname: str, title: str) -> None:
    train_pts = [(h["epoch"], h["loss"])      for h in history if "loss"      in h]
    eval_pts  = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    if train_pts: ax.plot(*zip(*train_pts), label="Train Loss", color="#4C72B0", lw=2)
    if eval_pts:  ax.plot(*zip(*eval_pts),  label="Val Loss",   color="#DD8452",
                          lw=2, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()

    ax2 = axes[1]
    if train_pts and eval_pts:
        t_dict = dict(train_pts); e_dict = dict(eval_pts)
        common = sorted(set(t_dict) & set(e_dict))
        if common:
            gaps   = [e_dict[ep] - t_dict[ep] for ep in common]
            colors = ["#C44E52" if g > 0 else "#55A868" for g in gaps]
            bars   = ax2.bar(common, gaps, color=colors, width=0.3, edgecolor="white")
            ax2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax2.axhline(0, color="black", lw=0.8, linestyle=":")
            ax2.set_title("Overfitting Gap (Val − Train Loss)", fontsize=12,
                          fontweight="bold")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Gap")
            ax2.annotate("Red = overfitting  |  Green = underfitting",
                         xy=(0.5, 0.02), xycoords="axes fraction",
                         ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, fname)
    plt.savefig(p); plt.close()
    print(f"  ✓ Loss + gap curves → {p}")


def _print_overfit_summary(history) -> None:
    train_pts = {h["epoch"]: h["loss"]      for h in history if "loss"      in h}
    eval_pts  = {h["epoch"]: h["eval_loss"] for h in history if "eval_loss" in h}
    common    = sorted(set(train_pts) & set(eval_pts))
    if not common:
        return
    print("\n  ┌─────────┬────────────┬────────────┬────────────┐")
    print(  "  │  Epoch  │ Train Loss │  Val Loss  │    Gap     │")
    print(  "  ├─────────┼────────────┼────────────┼────────────┤")
    for ep in common:
        t = train_pts[ep]; v = eval_pts[ep]; gap = v - t
        flag = " ⚠ OVERFIT" if gap > 0.05 else ""
        print(f"  │  {ep:5.2f}  │  {t:8.4f}  │  {v:8.4f}  │  {gap:+8.4f}{flag}")
    print(  "  └─────────┴────────────┴────────────┴────────────┘")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "="*65)
    print("  STEP 3 — R1 LoRA FINE-TUNING (BIDIRECTIONAL)")
    print(f"  Train          : {TRAIN_PATH}")
    print(f"  Dev            : {DEV_PATH}")
    print(f"  Backbone       : Gemma3-4B-IT (sarvam-translate)")
    print(f"  Optimizer      : {OPTIMIZER}")
    print(f"  Language       : {MT_LANG}  (LoRA config sized to this corpus)")
    print(f"  LoRA R         : {LORA_R},  Alpha: {LORA_ALPHA},  use_rslora={USE_RSLORA}")
    print(f"  LoRA Dropout   : {LORA_DROPOUT}")
    print(f"  Weight Decay   : {WEIGHT_DECAY}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}  (0.0 = no smoothing for R1)")
    print(f"  Learning Rate  : {LEARNING_RATE}  (5e-5, lowered from 1e-4)")
    print(f"  Max Epochs     : {NUM_TRAIN_EPOCHS_R1}  (ES patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Target modules : {LORA_TARGET_MODULES}")
    print(f"  PROMPT MASKING : ENABLED — loss computed on TARGET ONLY")
    print(f"  AUTO-RESUME    : ENABLED — resumes from latest checkpoint if found")
    print("="*65)

    # ── Full resume check: final adapter already saved ─────────────────────
    if _adapter_exists(ADAPTER_DIR):
        print(f"\n  [RESUME] Final adapter already at {ADAPTER_DIR}")
        print("  Skipping training. Delete the directory to retrain from scratch.")
        print("\n  STEP 3 COMPLETE (skipped — final adapter found)")
        return

    train_df = pd.read_excel(TRAIN_PATH)
    dev_df   = pd.read_excel(DEV_PATH)
    train_df = train_df.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)
    dev_df   = dev_df.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)
    print(f"  Train: {len(train_df):,}  |  Dev: {len(dev_df):,}")

    if "direction" in train_df.columns:
        print(f"  Train direction: {train_df['direction'].value_counts().to_dict()}")
    if "direction" in dev_df.columns:
        print(f"  Dev direction  : {dev_df['direction'].value_counts().to_dict()}")

    print(f"\n  Loading tokenizer from {MODEL_NAME} …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  Vocab size: {tok.vocab_size:,}")

    # ── Show prompt-masking stats on first example ─────────────────────────
    ex = format_and_tokenize(train_df.iloc[0], tok)
    lbl   = torch.tensor(ex["labels"])
    n_active = int((lbl != IGNORE_INDEX).sum().item())
    n_masked = int((lbl == IGNORE_INDEX).sum().item())
    print(f"\n  Prompt-mask example (first train row):")
    print(f"    Active label tokens (loss computed) : {n_active}")
    print(f"    Masked label tokens (loss ignored)  : {n_masked}")
    print(f"    Signal efficiency                   : {n_active/max(n_active+n_masked,1)*100:.1f}%")

    print("\n  Tokenising datasets …")
    train_ds = df_to_hf(train_df, tok)
    val_ds   = df_to_hf(dev_df,   tok)

    print(f"\n  Loading base model (bfloat16) from {MODEL_NAME} …")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    base.config.use_cache = False
    base.enable_input_require_grads()
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        target_modules = LORA_TARGET_MODULES,
        use_rslora     = USE_RSLORA,  # per-language: alpha/√r only for the
                                       # higher-rank 'as' config; mni/brx use
                                       # vanilla alpha/r scaling at low rank
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    eff_batch   = PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS
    total_steps = (len(train_ds) // eff_batch) * NUM_TRAIN_EPOCHS_R1
    print(f"\n  Eff batch  : {eff_batch}")
    print(f"  Est. steps : {total_steps:,}")

    # DataCollatorForSeq2Seq correctly pads labels with -100 (not pad_token_id)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir                  = ADAPTER_DIR,
        num_train_epochs            = NUM_TRAIN_EPOCHS_R1,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size  = PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16                        = True,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        label_smoothing_factor      = LABEL_SMOOTHING,
        optim                       = OPTIMIZER,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        logging_steps               = LOGGING_STEPS,
        logging_dir                 = os.path.join(ADAPTER_DIR, "logs"),
        report_to                   = "none",
        seed                        = SEED,
        dataloader_num_workers      = 4,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = data_collator,
        callbacks     = [EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # ── Auto-resume from latest checkpoint (partial training) ──────────────
    resume_ckpt = _find_latest_checkpoint(ADAPTER_DIR)
    if resume_ckpt:
        print(f"\n  [RESUME] Resuming from {resume_ckpt}")
    else:
        print(f"\n  Starting R1 LoRA fine-tuning ({NUM_TRAIN_EPOCHS_R1} epochs max) …")

    result = trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"  Runtime : {result.metrics['train_runtime']:.0f}s "
          f"({result.metrics['train_runtime']/60:.1f} min)")
    print(f"  Loss    : {result.metrics['train_loss']:.4f}")

    _print_overfit_summary(trainer.state.log_history)

    trainer.save_model(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"  ✓ Best adapter saved → {ADAPTER_DIR}")

    with open(os.path.join(OUTPUT_DIR, "r1_train_metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2)

    plot_loss(trainer.state.log_history,
              "r1_loss_curves.png",
              "R1 LoRA (Bidirectional) — Train vs Val Loss [prompt-masked]")

    del model, base, trainer
    torch.cuda.empty_cache()
    print("\n  STEP 3 COMPLETE")


if __name__ == "__main__":
    main()
