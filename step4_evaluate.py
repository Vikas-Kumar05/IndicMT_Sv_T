"""
step4_evaluate.py — Evaluate R1 LoRA Model (Eng<->Indic)
------------------------------------------------------
Evaluates R1 LoRA on the test set in BOTH directions.

"""

import os, json, warnings
import numpy as np
import pandas as pd
import torch
import sacrebleu
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

from config import (
    MODEL_NAME, MAX_SEQ_LEN, MAX_NEW_TOKENS, BATCH_SIZE_INFER,
    FIGURES_DIR, TEST_PATH, SCORES_FILE, OUTPUT_DIR,
    ADAPTER_DIR, R1_PREDS, DIRECTIONS, COMET_MODEL, PROMPT_TEMPLATES,
    SYSTEM_COLORS, DIRECTION_LABELS,
    HALL_LEN_RATIO_LOW, HALL_LEN_RATIO_HIGH, HALL_COPY_RATIO,
)

# Short prompt mapping — one sentence each
DIRECTION_PROMPT = {
    "eng2tgt": PROMPT_TEMPLATES["eng2tgt"],   # "Translate to Manipuri."
    "tgt2eng": PROMPT_TEMPLATES["tgt2eng"],   # "Translate to English."
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_sacre(preds, refs):
    bleu   = sacrebleu.corpus_bleu(preds, [refs])
    chrf   = sacrebleu.corpus_chrf(preds, [refs])
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    ter    = sacrebleu.corpus_ter(preds, [refs])
    return {"BLEU": round(bleu.score, 2),
            "chrF": round(chrf.score, 2),
            "chrF++": round(chrfpp.score, 2),
            "TER": round(ter.score, 2)}


def compute_comet(srcs, preds, refs):
    try:
        from comet import download_model, load_from_checkpoint
        path  = download_model(COMET_MODEL)
        mdl   = load_from_checkpoint(path)
        data  = [{"src": s, "mt": p, "ref": r}
                 for s, p, r in zip(srcs, preds, refs)]
        out   = mdl.predict(data, batch_size=32, gpus=1)
        score = round(float(np.mean(out.scores)) * 100, 2)
        print(f"  COMET    : {score:.2f}")
        return score
    except Exception as e:
        print(f"  [COMET] Skipped — {e}")
        return None


# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(preds, refs, label, direction, top_n=10):
    print(f"\n  ERROR ANALYSIS [{label}] [{direction}]")
    sent_bleus = []
    for p, r in zip(preds, refs):
        try:   s = sacrebleu.sentence_bleu(p, [r]).score
        except: s = 0.0
        sent_bleus.append(s)

    sb = np.array(sent_bleus)
    print(f"  Sent-BLEU  mean={sb.mean():.2f}  median={np.median(sb):.2f}"
          f"  std={sb.std():.2f}  min={sb.min():.2f}  max={sb.max():.2f}")
    print(f"  % BLEU=0  : {(sb==0).mean()*100:.1f}%")
    print(f"  % BLEU<10 : {(sb<10).mean()*100:.1f}%")
    print(f"  % BLEU≥30 : {(sb>=30).mean()*100:.1f}%")

    worst = np.argsort(sb)[:top_n]
    print(f"\n  Worst {top_n} predictions (lowest sent-BLEU):")
    for rank, i in enumerate(worst, 1):
        print(f"  [{rank}] BLEU={sb[i]:.2f}")
        print(f"       REF : {refs[i][:100]}")
        print(f"       PRED: {preds[i][:100]}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sb, bins=40, color=SYSTEM_COLORS.get("r1", "#4C72B0"),
            edgecolor="white", alpha=0.85)
    ax.axvline(sb.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Mean={sb.mean():.2f}")
    ax.set_title(f"Sentence-BLEU — {label} [{direction}]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Sentence BLEU"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"err_{label}_{direction}_sent_bleu.png"))
    plt.close()


# ── Hallucination analysis ────────────────────────────────────────────────────

def hallucination_analysis(srcs, preds, refs, label, direction):
    n = len(preds); omission = repetition = copy_hall = 0; rows = []
    for i, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        pt = pred.split() if pred.strip() else [""]
        rt = ref.split()  if ref.strip()  else [""]
        st = src.split()  if src.strip()  else [""]
        ratio = len(pt) / max(len(rt), 1)
        flags = []
        if ratio < HALL_LEN_RATIO_LOW:   omission  += 1; flags.append("omission")
        if ratio > HALL_LEN_RATIO_HIGH:  repetition += 1; flags.append("repetition")
        ps = set(t.lower() for t in pt); ss = set(t.lower() for t in st)
        if ss and len(ps & ss) / max(len(ps), 1) > HALL_COPY_RATIO:
            copy_hall += 1; flags.append("copy")
        if flags:
            rows.append({"idx": i, "src": src, "ref": ref,
                         "pred": pred, "flags": "|".join(flags),
                         "len_ratio": round(ratio, 3)})
    total = len(rows)
    print(f"\n  HALLUCINATION [{label}] [{direction}]  "
          f"suspicious: {total}/{n} ({total/max(n,1)*100:.1f}%)")
    print(f"    Omission  : {omission}")
    print(f"    Repetition: {repetition}")
    print(f"    Copy-src  : {copy_hall}")
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT_DIR, f"hallucination_{label}_{direction}.csv"),
            index=False)
        print(f"  ✓ Hallucination log saved.")


# ── Translation helpers ───────────────────────────────────────────────────────

def build_prompts(texts, tok, direction_key):
    """Short sarvam-style system prompt — one sentence."""
    sys_msg = DIRECTION_PROMPT[direction_key]
    return [tok.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user",   "content": t}],
        tokenize=False, add_generation_prompt=True) for t in texts]


def batch_translate(prompts, model, tok):
    preds = []; model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), BATCH_SIZE_INFER), desc="    Translating"):
            batch  = prompts[i: i+BATCH_SIZE_INFER]
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
            outs   = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=False, pad_token_id=tok.pad_token_id)
            new_t  = outs[:, inputs.input_ids.shape[1]:]
            preds.extend(p.strip() for p in tok.batch_decode(new_t, skip_special_tokens=True))
    return preds


def load_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f: return json.load(f)
    return {}

def save_scores(scores):
    with open(SCORES_FILE, "w") as f: json.dump(scores, f, indent=2)


# ── Per-direction evaluation ──────────────────────────────────────────────────

def evaluate_direction(test_df, model, tok, direction_key, target_lang, scores):
    """
    CHECKPOINT RESUME: If pred CSV exists and score key is in scores, skip.
    PROMPT FIX: Uses short DIRECTION_PROMPT.
    KEY FIX (columns): Always reads eng_src=source, meti_tgt=target.
    """
    score_key = f"r1_{direction_key}"
    pred_path = R1_PREDS.get(direction_key,
                    os.path.join(OUTPUT_DIR, f"preds_r1_{direction_key}.csv"))

    # ── Resume check ──────────────────────────────────────────────────────────
    if score_key in scores and os.path.exists(pred_path):
        print(f"\n  [RESUME] {score_key} already complete — skipping inference.")
        print(f"  Cached metrics: {scores[score_key]}")
        return scores[score_key]

    print(f"\n  {'─'*55}")
    print(f"  Direction : {direction_key}  →  prompt: \"{DIRECTION_PROMPT[direction_key]}\"")
    print(f"  {'─'*55}")

    if "direction" in test_df.columns:
        sub = test_df[test_df["direction"] == direction_key].copy()
    else:
        sub = test_df.copy()
        if direction_key == "tgt2eng":
            print(f"  [WARN] No direction column — skipping {direction_key}.")
            return None

    sub = sub.dropna(subset=["eng_src","meti_tgt"]).reset_index(drop=True)
    if len(sub) == 0:
        print(f"  [WARN] No rows for {direction_key} — skipping.")
        return None

    srcs = sub["eng_src"].astype(str).tolist()   # source (Meitei for tgt2eng)
    refs = sub["meti_tgt"].astype(str).tolist()  # target (English for tgt2eng)

    print(f"  Rows      : {len(srcs):,}")
    print(f"  Src sample: {srcs[0][:80]}")
    print(f"  Ref sample: {refs[0][:80]}")

    prompts = build_prompts(srcs, tok, direction_key)
    preds   = batch_translate(prompts, model, tok)
    metrics = compute_sacre(preds, refs)
    comet   = compute_comet(srcs, preds, refs)
    if comet is not None:
        metrics["COMET"] = comet

    print(f"\n  R1 RESULTS [{direction_key}]")
    print(f"  {'─'*35}")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.2f}")
    print(f"  {'─'*35}")

    # Compare with baseline
    bl_key = f"baseline_{direction_key}"
    if bl_key in scores:
        bl = scores[bl_key]
        print(f"\n  DELTA vs Baseline [{direction_key}]:")
        for m in metrics:
            if m in bl:
                delta = metrics[m] - bl[m]
                print(f"  {m:8s}: {delta:+.2f}  "
                      f"(baseline={bl[m]:.2f} → r1={metrics[m]:.2f})")

    print(f"\n  Sample translations (first 3):")
    for i in range(min(3, len(srcs))):
        print(f"  [{i+1}] SRC : {srcs[i][:100]}")
        print(f"       REF : {refs[i][:100]}")
        print(f"       PRED: {preds[i][:100]}\n")

    error_analysis(preds, refs, "r1", direction_key)
    hallucination_analysis(srcs, preds, refs, "r1", direction_key)

    out_df = pd.DataFrame({"direction": direction_key,
                            "src": srcs, "ref": refs, "pred": preds})
    out_df.to_csv(pred_path, index=False)
    print(f"  ✓ Predictions → {pred_path}")

    return metrics


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_baseline_vs_r1(scores):
    for direction_key in ["eng2tgt", "tgt2eng"]:
        bl_key = f"baseline_{direction_key}"
        r1_key = f"r1_{direction_key}"
        if bl_key not in scores or r1_key not in scores:
            continue

        metrics = [m for m in ["BLEU","chrF","chrF++","TER", "COMET"]
                   if m in scores[bl_key] or m in scores[r1_key]]
        x = np.arange(len(metrics)); w = 0.35
        systems = [(bl_key, "Baseline\n(zero-shot)", "#C44E52"),
                   (r1_key, "R1\n(LoRA fine-tuned)", "#4C72B0")]

        fig, ax = plt.subplots(figsize=(max(9, len(metrics)*2), 6))
        for i, (key, label, color) in enumerate(systems):
            vals   = [scores[key].get(m, 0) for m in metrics]
            offset = (i - len(systems)/2 + 0.5) * w
            bars   = ax.bar(x + offset, vals, w*0.9,
                            label=label, color=color, edgecolor="white")
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=10, fontweight="bold")

        dir_label = DIRECTION_LABELS.get(direction_key, direction_key)
        ax.set_title(f"Baseline vs R1 LoRA  [{dir_label}]",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylabel("Score"); ax.legend(fontsize=11)
        max_val = max(scores[k].get(m, 0)
                      for k in [bl_key, r1_key] for m in metrics)
        ax.set_ylim(0, max_val * 1.35)
        plt.tight_layout()
        p = os.path.join(FIGURES_DIR, f"r1_vs_baseline_{direction_key}.png")
        plt.savefig(p); plt.close()
        print(f"  ✓ r1_vs_baseline_{direction_key}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  STEP 4 — EVALUATE R1 LoRA (BIDIRECTIONAL)")
    print(f"  Adapter : {ADAPTER_DIR}")
    print(f"  Prompts (short, sarvam-style):")
    for k, v in DIRECTION_PROMPT.items():
        print(f"    {k}: \"{v}\"")
    print("="*60)

    test_df = pd.read_excel(TEST_PATH)
    test_df = test_df.dropna(subset=["eng_src","meti_tgt"]).reset_index(drop=True)
    print(f"  Test set: {len(test_df):,} rows")

    if "direction" in test_df.columns:
        print(f"  Direction breakdown:")
        print(f"    eng2tgt : {(test_df['direction']=='eng2tgt').sum():,}")
        print(f"    tgt2eng : {(test_df['direction']=='tgt2eng').sum():,}")

    scores = load_scores()

    # Check if all directions already done (full resume)
    all_done = all(
        f"r1_{d}" in scores and os.path.exists(
            R1_PREDS.get(d, os.path.join(OUTPUT_DIR, f"preds_r1_{d}.csv")))
        for d, _ in DIRECTIONS
    )
    if all_done:
        print("\n  [RESUME] All R1 directions already evaluated.")
        print("  Skipping model load. Regenerating comparison plots only.")
        plot_baseline_vs_r1(scores)
        print("\n  STEP 4 COMPLETE (resumed)")
        return

    print(f"\n  Loading R1 LoRA model from {ADAPTER_DIR} ...")
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    for direction_key, target_lang in DIRECTIONS:
        metrics = evaluate_direction(
            test_df, model, tok, direction_key, target_lang, scores)
        if metrics:
            scores[f"r1_{direction_key}"] = metrics
            save_scores(scores)   # save after each direction for safety

    plot_baseline_vs_r1(scores)

    del model, base; torch.cuda.empty_cache()
    print("\n  STEP 4 COMPLETE")


if __name__ == "__main__":
    main()

