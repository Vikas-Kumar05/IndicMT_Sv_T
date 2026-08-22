"""
aug_finetune_eval.py — Shared Fine-Tune + Evaluate Helper for Aug Experiments
===============================================================================
Called by step5a/5b/5c/5d/5e/5f to fine-tune from R1 on an augmented
dataset, then evaluate BOTH directions on the TEST set and save predictions.

═══════════════════════════════════════════════════════════════════════════════
BUGS FIXED IN THIS VERSION:
═══════════════════════════════════════════════════════════════════════════════

BUG-1 [CRITICAL] NO PROMPT MASKING IN AUG FINE-TUNE
  OLD: _tokenize_fn set labels = input_ids.copy()
       → loss was computed on [system_prompt + user_src + target]
       → ~50-70% of gradient signal wasted on prompt tokens
       → eng2tgt (already COMET=89) regressed because noisy gradients
          overwrote already-good weights
  FIX: Replaced _tokenize_fn with format_and_tokenize() using the same
       prompt-masking logic as step3_finetune.py. Labels for the prompt
       portion are set to IGNORE_INDEX=-100. Loss is computed on target only.

BUG-2 [CRITICAL] WRONG COLLATOR BROKE THE PROMPT MASK
  OLD: DataCollatorForLanguageModeling(mlm=False)
       → This collator OVERWRITES labels=-100 set during tokenisation
          with pad_token_id, breaking the prompt mask completely
  FIX: DataCollatorForSeq2Seq(label_pad_token_id=IGNORE_INDEX)
       → Respects -100 masking, pads labels with -100 not pad_token_id

BUG-3 [CRITICAL] NO CHECKPOINT RESUME IN AUG FINE-TUNE
  OLD: trainer.train()   ← always restarts from scratch
  FIX: _find_latest_checkpoint() + trainer.train(resume_from_checkpoint=ckpt)
       Training now auto-resumes from the last saved checkpoint.

BUG-4 [MODERATE] AUG EXPERIMENTS EVALUATED ON DEV SET, NOT TEST SET
  OLD: _run_eval used DEV_PATH for evaluation
       → Dev is used during training validation; test scores were never saved
       → The reported "exp1_eng2tgt" metrics were dev scores, not test scores
  FIX: _run_eval now evaluates on TEST_PATH (the held-out test set) to give
       true generalisation metrics. Dev remains the validation set during training.

OTHER FIXES PRESERVED FROM v3:
  - METRIC KEY FIX: _compute_metrics returns uppercase {BLEU, chrF, chrF++, TER, COMET}
  - MNI2ENG EVAL FIX: correct column handling for tgt2eng direction rows
  - PROGRESS VISIBILITY FIX: VerboseProgressCallback + LOGGING_STEPS=20
  - EARLY STOPPING FIX: uses AUG_EARLY_STOPPING_PATIENCE (3, was typo 32)
  - COMET SCORE FIX: safe fallback to mean(scores)
  - _save_scores normalises metric keys to canonical uppercase form
  - gradient_checkpointing_kwargs={"use_reentrant": False} to suppress warning

FIXES ADDED IN THIS VERSION (v4):
  - BUG-5 FIX: label_smoothing_factor now uses AUG_LABEL_SMOOTHING (=0.05) not
    LABEL_SMOOTHING (=0.0). Config.py defined AUG_LABEL_SMOOTHING specifically for
    aug fine-tuning to regularise against noisy BT targets, but the old import used
    LABEL_SMOOTHING — defeating the intent entirely.
  - BUG-6 FIX: run_finetune_and_eval() now has an eval resume guard. If both
    direction score keys are present in all_scores.json AND pred_file exists, the
    expensive eval pass (including COMET) is skipped. Delete pred_file or remove
    the score keys to force re-evaluation.
  - FIX: Dev balance print now uses a proper column check instead of the fragile
    .get('direction','').eq(...) pattern that silently returned 0 when the column
    was absent.
"""

import os
import json
import warnings
import random
import time

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    EarlyStoppingCallback, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import PeftModel
from datasets import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

from config import (
    MODEL_NAME, MAX_SEQ_LEN, SEED,
    ADAPTER_DIR, OUTPUT_DIR, FIGURES_DIR, SCORES_FILE,
    NUM_TRAIN_EPOCHS_AUG, AUG_EARLY_STOPPING_PATIENCE, LEARNING_RATE_AUG,
    PER_DEVICE_TRAIN_BATCH, PER_DEVICE_EVAL_BATCH,
    GRAD_ACCUM_STEPS, LR_SCHEDULER, WARMUP_RATIO,
    WEIGHT_DECAY, MAX_GRAD_NORM, SAVE_TOTAL_LIMIT, LOGGING_STEPS,
    PROMPT_TEMPLATES, DEV_PATH, TEST_PATH,
    BATCH_SIZE_INFER, MAX_NEW_TOKENS, DIRECTIONS,
    AUG_LABEL_SMOOTHING, OPTIMIZER,
)

# SEED here is config.MODEL_SEED (see config.py). run_finetune_and_eval()
# also accepts an explicit `seed` argument so run_seed_sweep.py can fine-tune
# the SAME augmented dataset under several different training seeds without
# re-importing config for each one.
from aug_utils import batch_translate

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

IGNORE_INDEX = -100   # CrossEntropyLoss ignores this label value


# ══════════════════════════════════════════════════════════════════════════════
#  Progress callback
# ══════════════════════════════════════════════════════════════════════════════

class VerboseProgressCallback(TrainerCallback):
    HEARTBEAT_STEPS = 50

    def __init__(self, label: str = ""):
        self.label       = label
        self.epoch_start = None
        self.run_start   = time.time()
        self.last_hb     = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch) + 1
        total_eps = int(args.num_train_epochs)
        self.epoch_start = time.time()
        print(f"\n  {'─'*60}")
        print(f"  [{self.label}]  Epoch {ep}/{total_eps} — STARTED")
        print(f"  {'─'*60}")

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(state.epoch)
        elapsed = time.time() - self.epoch_start
        total_elapsed = time.time() - self.run_start
        eval_loss = None
        train_loss = None
        for entry in reversed(state.log_history):
            if eval_loss is None and "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
            if train_loss is None and "loss" in entry:
                train_loss = entry["loss"]
            if eval_loss is not None and train_loss is not None:
                break
        loss_str = ""
        if train_loss is not None:
            loss_str += f"  train_loss={train_loss:.4f}"
        if eval_loss is not None:
            loss_str += f"  val_loss={eval_loss:.4f}"
            if train_loss is not None:
                gap = eval_loss - train_loss
                flag = " ⚠ OVERFIT" if gap > 0.05 else ""
                loss_str += f"  gap={gap:+.4f}{flag}"
        print(f"\n  [{self.label}]  Epoch {ep} DONE — {elapsed:.0f}s "
              f"(total {total_elapsed:.0f}s){loss_str}")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step - self.last_hb >= self.HEARTBEAT_STEPS:
            elapsed = time.time() - self.run_start
            print(f"  [step {step:>6d}]  elapsed={elapsed:.0f}s  "
                  f"{'·'*30}", flush=True)
            self.last_hb = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        lr   = logs.get("learning_rate")
        if loss is not None:
            lr_str = f"  lr={lr:.2e}" if lr is not None else ""
            print(f"  [step {step:>6d}]  loss={loss:.4f}{lr_str}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BUG-1 FIX: Prompt-masked tokenisation (replaces old _tokenize_fn)
# ══════════════════════════════════════════════════════════════════════════════

def _format_and_tokenize(row, tok) -> dict:
    """
    FIX BUG-1: Proper prompt masking — loss computed on TARGET TOKENS ONLY.

    OLD (broken):
        text = prompt + target
        enc["labels"] = enc["input_ids"].copy()
        → Loss on ALL tokens including system prompt and source sentence

    NEW (fixed):
        prompt_ids  → labels = -100   (ignored by CrossEntropyLoss)
        target_ids  → labels = token_id (loss computed here only)
        eos_id      → labels = eos_id   (loss computed here)

    This matches the standard instruction fine-tuning practice used by
    Alpaca, Vicuna, LLaMA-Factory, and step3_finetune.py in this pipeline.
    """
    src       = str(row["eng_src"])
    tgt       = str(row["meti_tgt"])
    direction = str(row.get("direction", "eng2tgt"))
    sys_msg   = PROMPT_TEMPLATES.get(direction, PROMPT_TEMPLATES["eng2tgt"])

    # Build the prompt (everything EXCEPT the target translation)
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": src},
    ]
    prompt_str = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)

    # Tokenise prompt and target separately to find the boundary
    prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
    target_ids = tok(tgt + tok.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids

    # Truncate to MAX_SEQ_LEN — always preserve target, trim prompt if needed
    if len(input_ids) > MAX_SEQ_LEN:
        target_ids = target_ids[:MAX_SEQ_LEN]
        prompt_ids = prompt_ids[:MAX_SEQ_LEN - len(target_ids)]
        input_ids  = prompt_ids + target_ids

    # Labels: IGNORE_INDEX for prompt, actual ids for target
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


def _df_to_hf(df: pd.DataFrame, tok) -> Dataset:
    """Convert DataFrame to HuggingFace Dataset with prompt-masked labels."""
    records = [_format_and_tokenize(row, tok) for _, row in df.iterrows()]
    ds = Dataset.from_list(records)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def _adapter_exists(adapter_dir: str) -> bool:
    return os.path.isdir(adapter_dir) and any(
        f.endswith((".bin", ".safetensors"))
        for f in os.listdir(adapter_dir)
    )


def _find_latest_checkpoint(output_dir: str):
    """BUG-3 FIX: Find the latest Trainer checkpoint for auto-resume."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not ckpts:
        return None
    # Sort by step number numerically
    ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    latest = ckpts[-1]
    print(f"  [RESUME] Found checkpoint: {latest}")
    return latest


# ══════════════════════════════════════════════════════════════════════════════
#  Overfitting diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def _print_overfit_summary(history, label: str) -> None:
    train_pts = {h["epoch"]: h["loss"]      for h in history if "loss"      in h}
    eval_pts  = {h["epoch"]: h["eval_loss"] for h in history if "eval_loss" in h}
    common    = sorted(set(train_pts) & set(eval_pts))
    if not common:
        return
    print(f"\n  Overfitting summary [{label}]:")
    print(  "  ┌─────────┬────────────┬────────────┬────────────┐")
    print(  "  │  Epoch  │ Train Loss │  Val Loss  │    Gap     │")
    print(  "  ├─────────┼────────────┼────────────┼────────────┤")
    for ep in common:
        t = train_pts[ep]; v = eval_pts[ep]; gap = v - t
        flag = " ⚠ OVERFIT" if gap > 0.05 else ""
        print(f"  │  {ep:5.2f}  │  {t:8.4f}  │  {v:8.4f}  │  {gap:+8.4f}{flag}")
    print(  "  └─────────┴────────────┴────────────┴────────────┘")


def _plot_loss(history, fname: str, title: str) -> None:
    train_pts = [(h["epoch"], h["loss"])      for h in history if "loss"      in h]
    eval_pts  = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    if train_pts:
        ax.plot(*zip(*train_pts), label="Train Loss", color="#4C72B0", lw=2)
    if eval_pts:
        ax.plot(*zip(*eval_pts), label="Val Loss",   color="#DD8452", lw=2, linestyle="--")
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
            ax2.set_title("Overfitting Gap (Val − Train)", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Gap")

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, fname)
    plt.savefig(p); plt.close()
    print(f"  ✓ Loss curves → {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(refs: list, hyps: list, src_texts: list = None) -> dict:
    import sacrebleu

    bleu   = sacrebleu.corpus_bleu(hyps, [refs])
    chrf   = sacrebleu.corpus_chrf(hyps, [refs])
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    ter    = sacrebleu.corpus_ter(hyps, [refs])

    metrics = {
        "BLEU"   : round(bleu.score,   2),
        "chrF"   : round(chrf.score,   2),
        "chrF++": round(chrfpp.score, 2),
        "TER"    : round(ter.score,    2),
    }

    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint
        from config import COMET_MODEL

        print("  [COMET] Loading model …", flush=True)
        comet_path  = download_model(COMET_MODEL)
        comet_model = load_from_checkpoint(comet_path)

        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(src_texts or refs, hyps, refs)
        ]
        result = comet_model.predict(data, batch_size=8, gpus=1)

        if hasattr(result, "system_score"):
            comet_score = float(result.system_score)
        elif hasattr(result, "scores"):
            comet_score = float(np.mean(result.scores))
        else:
            comet_score = float(np.mean(list(result)))

        comet_score = round(comet_score * 100, 2)
        del comet_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [COMET] Score: {comet_score:.2f}", flush=True)

    except Exception as e:
        print(f"  [COMET] Skipped: {e}", flush=True)

    if comet_score is not None:
        metrics["COMET"] = comet_score

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_finetune_and_eval(
    score_key:    str,
    dataset_file: str,
    adapter_out:  str,
    pred_file:    str,
    label:        str,
    seed:         int = None,
) -> dict:
    """
    Fine-tune from R1 on augmented dataset (if not already done),
    then evaluate on the TEST set (both directions).

    Returns primary direction (eng2tgt) metrics dict.

    RESUME LOGIC:
      - Fine-tune is skipped if a final adapter already exists.
      - Evaluation is skipped if BOTH direction score keys are present in
        all_scores.json AND pred_file already exists — avoids re-running
        expensive COMET scoring after a crash-and-restart.

    SEED SWEEP:
      - `seed`, if given, overrides config.SEED for this run only (used by
        run_seed_sweep.py to fine-tune the SAME dataset_file under several
        seeds). `adapter_out`/`pred_file`/`score_key` must already be
        seed-suffixed by the caller so parallel seeds don't collide.
    """
    if _adapter_exists(adapter_out):
        print(f"\n  [SKIP] Adapter found at {adapter_out} — skipping fine-tune.")
        print(f"  Delete the directory to retrain from scratch.")
    else:
        print(f"\n  Fine-tuning [{label}]  →  {adapter_out}")
        print(f"  Dataset: {dataset_file}")
        _run_finetune(dataset_file, adapter_out, label, seed=seed)

    # ── Eval resume check ──────────────────────────────────────────────────────
    if os.path.exists(pred_file):
        existing = {}
        if os.path.exists(SCORES_FILE):
            with open(SCORES_FILE) as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = {}
        both_done = all(
            f"{score_key}_{d}" in existing for d, _ in DIRECTIONS
        )
        if both_done:
            print(f"\n  [SKIP] Eval already complete for [{label}] — both directions in scores.")
            print(f"  Delete {pred_file} or remove keys from all_scores.json to re-evaluate.")
            # Return cached primary metrics
            cached = existing.get(f"{score_key}_eng2tgt", {})
            return cached

    print(f"\n  Evaluating [{label}] on TEST set (both directions) ...", flush=True)
    metrics = _run_eval(adapter_out, pred_file, score_key, label)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Fine-tune
#
#  KEY DESIGN NOTES:
#  - Loads the R1 LoRA adapter in trainable mode (continues from R1).
#  - Does NOT re-apply LoraConfig — that's already saved in adapter_config.json.
#  - BUG-1 FIX: Uses _format_and_tokenize() with proper prompt masking.
#  - BUG-2 FIX: Uses DataCollatorForSeq2Seq with label_pad_token_id=-100.
#  - BUG-3 FIX: Auto-resumes from latest checkpoint via _find_latest_checkpoint().
# ══════════════════════════════════════════════════════════════════════════════

def _run_finetune(dataset_file: str, adapter_out: str, label: str,
                   seed: int = None) -> None:
    run_seed = SEED if seed is None else seed
    random.seed(run_seed); np.random.seed(run_seed)
    torch.manual_seed(run_seed); torch.cuda.manual_seed_all(run_seed)
    print(f"  Training seed for this run: {run_seed}"
          f"{'  (sweep override)' if seed is not None else '  (default)'}")

    aug_df = pd.read_excel(dataset_file)
    dev_df = pd.read_excel(DEV_PATH)
    aug_df = aug_df.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)
    dev_df = dev_df.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)

    # Add direction column if missing (backward compat with old cached datasets)
    if "direction" not in aug_df.columns:
        aug_df["direction"] = "eng2tgt"

    # Verify balance — warn if strongly imbalanced
    n_e2m = (aug_df["direction"] == "eng2tgt").sum()
    n_m2e = (aug_df["direction"] == "tgt2eng").sum()
    ratio  = n_e2m / max(n_m2e, 1)
    print(f"  Train rows : {len(aug_df):,}  "
          f"(eng2tgt={n_e2m:,}  tgt2eng={n_m2e:,}  ratio={ratio:.2f}:1)")
    if ratio > 1.3:
        print(f"  [WARN] eng2tgt is {ratio:.1f}× heavier than tgt2eng. "
              f"Consider regenerating BT cache for balanced output.")
    elif ratio < 0.77:
        print(f"  [WARN] tgt2eng is {1/ratio:.1f}× heavier than eng2tgt.")
    else:
        print(f"  Direction balance: OK")

    if "direction" in dev_df.columns:
        dev_e2m = (dev_df["direction"] == "eng2tgt").sum()
        dev_m2e = (dev_df["direction"] == "tgt2eng").sum()
    else:
        dev_e2m = dev_m2e = "n/a"
    print(f"  Dev rows   : {len(dev_df):,}  (eng2tgt={dev_e2m}  tgt2eng={dev_m2e})")

    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  BUG-1 FIX: Tokenising with PROMPT MASKING (loss on target only) …")
    train_ds = _df_to_hf(aug_df, tok)
    val_ds   = _df_to_hf(dev_df, tok)

    # Verify masking on first example
    ex = train_ds[0]
    n_active  = int((torch.tensor(ex["labels"]) != IGNORE_INDEX).sum().item())
    n_masked  = int((torch.tensor(ex["labels"]) == IGNORE_INDEX).sum().item())
    print(f"  Prompt-mask check — active labels: {n_active}  masked: {n_masked}  "
          f"efficiency: {n_active/max(n_active+n_masked,1)*100:.1f}%")

    print(f"  Loading base model ({MODEL_NAME}) …", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    base.config.use_cache = False
    base.enable_input_require_grads()
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load R1 LoRA adapter in trainable mode (continue from R1 checkpoint).
    # The adapter_config.json in ADAPTER_DIR has the correct LoRA settings.
    # We do NOT call LoraConfig here — that would add a second set of adapters.
    print(f"  Loading R1 LoRA adapter from {ADAPTER_DIR} (is_trainable=True) …")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=True)
    model.print_trainable_parameters()

    # BUG-2 FIX: DataCollatorForSeq2Seq respects label=-100 masking.
    # DataCollatorForLanguageModeling(mlm=False) was silently overwriting
    # the -100 labels with pad_token_id, breaking the prompt mask.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        label_pad_token_id=IGNORE_INDEX,  # CRITICAL: pad labels with -100
        pad_to_multiple_of=8,
    )

    eff_batch   = PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS
    total_steps = (len(train_ds) // eff_batch) * NUM_TRAIN_EPOCHS_AUG

    print(f"\n  Training plan:")
    print(f"    Effective batch size : {eff_batch}")
    print(f"    Estimated steps      : {total_steps:,}")
    print(f"    Learning rate        : {LEARNING_RATE_AUG}")
    print(f"    Epochs (max)         : {NUM_TRAIN_EPOCHS_AUG}")
    print(f"    ES patience          : {AUG_EARLY_STOPPING_PATIENCE}")
    print(f"    Optimizer            : {OPTIMIZER}")
    print(f"    Label smoothing      : {AUG_LABEL_SMOOTHING}  (AUG_LABEL_SMOOTHING — handles BT noise)")
    print(f"    Prompt masking       : ENABLED (BUG-1 FIX)")
    print(f"    Correct collator     : DataCollatorForSeq2Seq (BUG-2 FIX)")
    print(f"    Auto-resume ckpt     : ENABLED (BUG-3 FIX)", flush=True)

    t_args = TrainingArguments(
        output_dir                  = adapter_out,
        num_train_epochs            = NUM_TRAIN_EPOCHS_AUG,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size  = PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16                        = True,
        learning_rate               = LEARNING_RATE_AUG,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        label_smoothing_factor      = AUG_LABEL_SMOOTHING,
        optim                       = OPTIMIZER,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        logging_steps               = LOGGING_STEPS,
        logging_dir                 = os.path.join(adapter_out, "logs"),
        report_to                   = "none",
        seed                        = run_seed,
        dataloader_num_workers      = 4,
        remove_unused_columns       = False,   # required for custom datasets
        disable_tqdm                = False,
    )

    trainer = Trainer(
        model         = model,
        args          = t_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = data_collator,
        callbacks     = [
            EarlyStoppingCallback(early_stopping_patience=AUG_EARLY_STOPPING_PATIENCE),
            VerboseProgressCallback(label=label),
        ],
    )

    # BUG-3 FIX: auto-resume from last checkpoint
    resume_ckpt = _find_latest_checkpoint(adapter_out)
    if resume_ckpt:
        print(f"\n  [RESUME] Resuming from {resume_ckpt}", flush=True)
    else:
        print(f"\n  Starting fresh training for [{label}] …", flush=True)

    result = trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"\n  Training finished.")
    print(f"  Runtime    : {result.metrics['train_runtime']:.0f}s  "
          f"({result.metrics['train_runtime']/60:.1f} min)")
    print(f"  Train loss : {result.metrics['train_loss']:.4f}")

    _print_overfit_summary(trainer.state.log_history, label)
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
    _plot_loss(trainer.state.log_history,
               f"loss_{safe_label}.png",
               f"{label} — Train vs Val Loss")

    trainer.save_model(adapter_out)
    tok.save_pretrained(adapter_out)
    print(f"  ✓ Adapter saved → {adapter_out}", flush=True)

    del model, base, trainer
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluate
#
#  BUG-4 FIX: Evaluate on TEST_PATH (not DEV_PATH).
#  Dev is used for training-time validation only.
#  Test scores are the true generalisation metrics.
# ══════════════════════════════════════════════════════════════════════════════

def _run_eval(
    adapter_dir: str,
    pred_file: str,
    score_key: str,
    label: str,
) -> dict:
    from aug_utils import load_ft_model, free_model

    # BUG-4 FIX: Use TEST_PATH not DEV_PATH for final evaluation
    eval_df = pd.read_excel(TEST_PATH)
    eval_df = eval_df.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)

    print(f"  BUG-4 FIX: Evaluating on TEST set ({len(eval_df):,} rows)", flush=True)
    if "direction" in eval_df.columns:
        print(f"  Direction breakdown: "
              f"eng2tgt={(eval_df['direction']=='eng2tgt').sum():,}  "
              f"tgt2eng={(eval_df['direction']=='tgt2eng').sum():,}")

    model, tok = load_ft_model(adapter_dir)
    all_rows   = []

    for dir_key, _ in DIRECTIONS:
        print(f"\n  {'─'*55}", flush=True)
        print(f"  Evaluating direction: {dir_key}", flush=True)
        print(f"  {'─'*55}", flush=True)

        if "direction" in eval_df.columns:
            sub = eval_df[eval_df["direction"] == dir_key].copy()
            if len(sub) == 0:
                print(f"  [WARN] No {dir_key} rows in test set — skipping.", flush=True)
                continue
        else:
            if dir_key == "tgt2eng":
                print(f"  [WARN] No direction column, skipping tgt2eng.", flush=True)
                continue
            sub = eval_df.copy()

        sub = sub.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)
        # Column convention from step1: eng_src=source, meti_tgt=target (both dirs)
        srcs = sub["eng_src"].astype(str).tolist()
        refs = sub["meti_tgt"].astype(str).tolist()

        sys_msg = PROMPT_TEMPLATES["eng2tgt"] if dir_key == "eng2tgt" \
                  else PROMPT_TEMPLATES["tgt2eng"]

        if len(srcs) == 0:
            print(f"  [WARN] No rows for {dir_key} — skipping.", flush=True)
            continue

        print(f"  Rows      : {len(srcs):,}")
        print(f"  Prompt    : \"{sys_msg}\"")
        print(f"  Src sample: {srcs[0][:80]}")
        print(f"  Ref sample: {refs[0][:80]}", flush=True)

        print(f"  Translating {len(srcs):,} sentences …", flush=True)
        preds = batch_translate(srcs, model, tok, sys_msg)

        m = _compute_metrics(refs, preds, src_texts=srcs)
        print(f"\n  [{dir_key}] TEST Results:")
        for metric_name, val in m.items():
            if val is not None:
                ter_note = "  (lower=better)" if metric_name == "TER" else ""
                print(f"    {metric_name:8s}: {val:.2f}{ter_note}")

        print(f"\n  Sample translations (first 3):")
        for i in range(min(3, len(srcs))):
            print(f"  [{i+1}] SRC : {srcs[i][:100]}")
            print(f"       REF : {refs[i][:100]}")
            print(f"       PRED: {preds[i][:100]}")

        for s, r, p in zip(srcs, refs, preds):
            all_rows.append({
                "direction": dir_key,
                "src": s, "ref": r, "pred": p,
                "score_key": score_key,
            })

        _save_scores(score_key, dir_key, m)

    free_model(model)

    pd.DataFrame(all_rows).to_csv(pred_file, index=False)
    print(f"\n  ✓ Predictions → {pred_file}", flush=True)

    # Return primary direction (eng2tgt) metrics
    primary_rows = [r for r in all_rows if r["direction"] == "eng2tgt"]
    if primary_rows:
        refs_p  = [r["ref"]  for r in primary_rows]
        preds_p = [r["pred"] for r in primary_rows]
        srcs_p  = [r["src"]  for r in primary_rows]
        return _compute_metrics(refs_p, preds_p, src_texts=srcs_p)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  Score persistence
# ══════════════════════════════════════════════════════════════════════════════

_KEY_NORM = {
    "bleu"   : "BLEU",  "BLEU"   : "BLEU",
    "chrf"   : "chrF",  "chrF"   : "chrF",
    "chrfpp" : "chrF++", "chrF++" : "chrF++",
    "ter"    : "TER",   "TER"    : "TER",
    "comet"  : "COMET", "COMET"  : "COMET",
}


def _normalise_metrics(metrics: dict) -> dict:
    return {_KEY_NORM.get(k, k): v for k, v in metrics.items() if v is not None}


def _save_scores(score_key: str, direction: str, metrics: dict) -> None:
    scores = {}
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            try:
                scores = json.load(f)
            except Exception:
                scores = {}

    key = f"{score_key}_{direction}"
    scores[key] = _normalise_metrics(metrics)

    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  ✓ Scores saved ({key}) → {SCORES_FILE}", flush=True)
    print(f"    {scores[key]}")
