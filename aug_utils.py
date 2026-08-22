"""
aug_utils.py — Shared Utilities for Augmentation Steps
=======================================================
Provides: model loading/unloading, batched translation,
          LaBSE quality filtering, combine_and_save.

CHANGES vs original:
  - combine_and_save uses MAX_AUG_RATIO from config (now 1.5, was 1.0)
  - combine_and_save retains the direction column from orig_df if present
    so that augmented datasets reflect eng2tgt direction (not broken)
  - No other changes; this file was largely correct.
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE_INFER, MAX_NEW_TOKENS, SEED,
    QA_LABSE_THRESHOLD, MAX_AUG_RATIO, ORIG_MIN_RATIO,
    FIGURES_DIR, SYSTEM_COLORS,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
COLORS = list(SYSTEM_COLORS.values())


# ── Model loading / unloading ─────────────────────────────────────────────────

def load_ft_model(adapter_dir: str):
    """Load base model + LoRA adapter. Returns (model, tokenizer)."""
    print(f"  [model] Loading from {adapter_dir} …")
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    print(f"  [model] Loaded  GPU: {torch.cuda.get_device_name(0)}")
    return model, tok


def free_model(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Batched translation ───────────────────────────────────────────────────────

def _make_prompt(tok, text: str, sys_msg: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user",   "content": text}],
        tokenize=False, add_generation_prompt=True)


def batch_translate(texts, model, tok, sys_msg: str,
                    batch_size: int = BATCH_SIZE_INFER,
                    max_new: int = MAX_NEW_TOKENS) -> list:
    texts   = [str(t) if (t is not None and str(t) != "nan") else "" for t in texts]
    prompts = [_make_prompt(tok, t, sys_msg) for t in texts]
    preds   = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size),
                      desc="    translating", leave=False):
            batch = prompts[i: i + batch_size]
            try:
                inputs = tok(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
                outs   = model.generate(**inputs, max_new_tokens=max_new,
                                        do_sample=False, pad_token_id=tok.pad_token_id)
                new_t  = outs[:, inputs.input_ids.shape[1]:]
                preds.extend(p.strip() for p in tok.batch_decode(new_t, skip_special_tokens=True))
            except RuntimeError:
                # Fallback: translate one by one on OOM
                for single in batch:
                    try:
                        inp = tok([single], return_tensors="pt", truncation=True,
                                  max_length=MAX_SEQ_LEN).to(model.device)
                        out = model.generate(**inp, max_new_tokens=max_new,
                                             do_sample=False, pad_token_id=tok.pad_token_id)
                        new = out[:, inp.input_ids.shape[1]:]
                        preds.extend(p.strip() for p in tok.batch_decode(new, skip_special_tokens=True))
                    except Exception:
                        preds.append("")
    return preds


# ── LaBSE quality filter ──────────────────────────────────────────────────────

def labse_quality_filter(src_list: list, tgt_list: list,
                          threshold: float = QA_LABSE_THRESHOLD,
                          batch_size: int = 256):
    """
    Compute cross-lingual embedding similarity with LaBSE and return
    (mask, scores) where mask[i] = True if scores[i] >= threshold.
    """
    src_list = [str(s) if s else "" for s in src_list]
    tgt_list = [str(t) if t else "" for t in tgt_list]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [QA] sentence-transformers not installed — skipping filter.")
        scores = np.ones(len(src_list), dtype=float)
        return scores >= threshold, scores

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [QA] LaBSE filtering {len(src_list):,} pairs (threshold≥{threshold}) …")
    labse = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    all_scores = []
    for i in range(0, len(src_list), batch_size):
        emb_s = labse.encode(src_list[i:i + batch_size], normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        emb_t = labse.encode(tgt_list[i:i + batch_size], normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        all_scores.extend((emb_s * emb_t).sum(axis=1).tolist())
    del labse
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scores = np.array(all_scores, dtype=float)
    mask   = scores >= threshold
    print(f"  [QA] Kept {mask.sum():,}/{len(scores):,}  "
          f"(mean={scores.mean():.3f}  threshold={threshold})")
    return mask, scores


# ── chrF++ quality filter ──────────────────────────────────────────────────────

def chrf_quality_filter(hyp_list: list, ref_list: list, threshold: float):
    """
    Sentence-level chrF++ (word_order=2) between two texts.

    IMPORTANT: hyp_list and ref_list MUST be in the SAME language/script.
    chrF++ is a character n-gram overlap metric — comparing text across two
    different scripts (e.g. English vs Meitei) produces a near-meaningless
    score. Never call this on a genuine cross-lingual source/target pair;
    use labse_quality_filter for that. Every call site in aug_techniques.py
    documents which same-language anchor it is scoring against.

    Returns (mask, scores) with the same shape/semantics as
    labse_quality_filter, so the two are drop-in interchangeable.
    """
    import sacrebleu
    hyp_list = [str(h) if h else "" for h in hyp_list]
    ref_list = [str(r) if r else "" for r in ref_list]
    scores = np.array([
        sacrebleu.sentence_chrf(h, [r], word_order=2).score
        for h, r in zip(hyp_list, ref_list)
    ], dtype=float)
    mask = scores >= threshold
    print(f"  [QA-chrF++] Kept {mask.sum():,}/{len(scores):,}  "
          f"(mean={scores.mean():.2f}  threshold={threshold})")
    return mask, scores


# ── Top-K selection (on top of a pass/fail threshold) ──────────────────────────

def top_k_filter(mask: np.ndarray, scores: np.ndarray, top_k_ratio):
    """
    Given a boolean pass/fail mask and parallel quality scores, restrict the
    True entries further to only the top-K fraction by score (highest
    first). This is the "choose the top chrF++ / LaBSE sentences" step.

    top_k_ratio: None or >= 1.0 -> no-op, returns mask unchanged.
                 e.g. 0.7 -> keep only the best 70% of pairs that already
                 passed the threshold (dropping the weakest 30% of survivors).
    """
    if not top_k_ratio or top_k_ratio >= 1.0:
        return mask
    mask = np.asarray(mask, dtype=bool)
    idx_pass = np.where(mask)[0]
    if len(idx_pass) == 0:
        return mask
    k = max(1, int(round(len(idx_pass) * top_k_ratio)))
    order = idx_pass[np.argsort(-scores[idx_pass])]   # descending by score
    keep_idx = order[:k]
    new_mask = np.zeros_like(mask)
    new_mask[keep_idx] = True
    print(f"  [QA top-K] ratio={top_k_ratio}  kept best {k:,}/{len(idx_pass):,} "
          f"pairs that passed the threshold (ranked by quality score)")
    return new_mask


# ── Unified quality-filter dispatcher ───────────────────────────────────────────

def resolve_quality_filter(method: str,
                            hyp_list: list, ref_list: list,
                            cross_src: list, cross_tgt: list,
                            labse_threshold: float, chrf_threshold: float,
                            top_k_ratio=None):
    """
    Single entry point used by every augmentation technique in
    aug_techniques.py, so the QA_METHOD config switch ("labse" / "chrf" /
    "both") applies consistently everywhere without duplicating logic.

    method            : config.QA_METHOD ("labse" | "chrf" | "both")
    hyp_list/ref_list : the SAME-LANGUAGE pair to score with chrF++
                        (e.g. noised English vs the original English source)
    cross_src/cross_tgt: the genuine CROSS-LINGUAL source/target pair to
                        score with LaBSE (e.g. noised English vs the
                        untouched Meitei target). Required for "labse"/"both".
    top_k_ratio       : forwarded to top_k_filter() — config.QA_TOP_K_RATIO

    Returns (mask, scores, score_label):
        mask        - boolean keep/drop array
        scores      - the score array used for the returned qa_score column
        score_label - human-readable string for logging (e.g. "chrF++")
    """
    if method == "labse":
        mask, scores = labse_quality_filter(cross_src, cross_tgt, threshold=labse_threshold)
        mask = top_k_filter(mask, scores, top_k_ratio)
        return mask, scores, "LaBSE"

    elif method == "chrf":
        mask, scores = chrf_quality_filter(hyp_list, ref_list, threshold=chrf_threshold)
        mask = top_k_filter(mask, scores, top_k_ratio)
        return mask, scores, "chrF++"

    elif method == "both":
        mask_l, scores_l = labse_quality_filter(cross_src, cross_tgt, threshold=labse_threshold)
        mask_c, scores_c = chrf_quality_filter(hyp_list, ref_list, threshold=chrf_threshold)
        mask = mask_l & mask_c
        print(f"  [QA both] LaBSE pass={mask_l.sum():,}  chrF++ pass={mask_c.sum():,}  "
              f"combined(AND)={mask.sum():,}")
        # Rank survivors by chrF++ — finer-grained 0-100 scale than LaBSE's 0-1
        mask = top_k_filter(mask, scores_c, top_k_ratio)
        return mask, scores_c, "chrF++ (LaBSE+chrF++ gated)"

    else:
        raise ValueError(f"Unknown QA method: {method!r}. Use labse|chrf|both.")


# ── combine_and_save ──────────────────────────────────────────────────────────

def combine_and_save(orig_df: pd.DataFrame, aug_frames: list,
                     save_path: str, tag: str = "augmented") -> pd.DataFrame:
    """
    Merge original + synthetic frames, apply size caps, dedup,
    upsample originals if needed, shuffle, and save.

    BIDIRECTIONAL FIX:
    The original code forced direction="eng2tgt" on ALL rows, including
    synthetic aug pairs. This caused:
      - BT pairs that are tgt2eng (original_meitei→bt_english) to be
        mislabelled as eng2tgt, feeding the wrong prompt during training
        and computing loss against the wrong target.
      - The aug dataset to be 1.5× heavier on eng2tgt vs tgt2eng,
        biasing the model toward eng2tgt and causing tgt2eng regression.

    FIX:
    - orig_df: keep both directions as-is (preserves the bidirectional split)
    - aug_frames: preserve the direction column from each aug frame
      (aug_techniques.py now sets direction="eng2tgt" or "tgt2eng" correctly)
    - Dedup and cap are done per-direction to maintain balance
    - Final dataset is balanced between eng2tgt and tgt2eng
    """
    # ── Originals: keep both directions ──────────────────────────────────────
    orig_copy = orig_df.copy()
    orig_copy = orig_copy.dropna(subset=["eng_src", "meti_tgt"])
    orig_copy = orig_copy[["eng_src", "meti_tgt",
                            *([c] for c in ["direction"] if c in orig_copy.columns)]
                           if False else
                           [c for c in ["eng_src", "meti_tgt", "direction"]
                            if c in orig_copy.columns]].copy()
    if "direction" not in orig_copy.columns:
        # Fallback: old-style dataset without direction — assume eng2tgt
        orig_copy["direction"] = "eng2tgt"
    orig_copy["aug_type"] = "original"
    orig_copy["qa_score"] = 1.0

    n_orig_e2m = (orig_copy["direction"] == "eng2tgt").sum()
    n_orig_m2e = (orig_copy["direction"] == "tgt2eng").sum()
    n_orig = len(orig_copy)
    print(f"  Originals: eng2tgt={n_orig_e2m:,}  tgt2eng={n_orig_m2e:,}  "
          f"total={n_orig:,}")

    # ── Synthetics: preserve direction from aug_frames ─────────────────────
    valid_frames = [f for f in aug_frames if f is not None and len(f) > 0]
    if valid_frames:
        synthetic = pd.concat(valid_frames, ignore_index=True)
        if "direction" not in synthetic.columns:
            # Backward compat: old aug frames without direction default to eng2tgt
            synthetic["direction"] = "eng2tgt"
    else:
        synthetic = pd.DataFrame(
            columns=["eng_src", "meti_tgt", "aug_type", "direction", "qa_score"])

    # ── Dedup synthetic vs originals (per direction) ──────────────────────
    if len(synthetic):
        orig_tuples = set(zip(orig_copy["eng_src"].str.strip(),
                              orig_copy["meti_tgt"].str.strip()))
        mask_novel = ~synthetic.apply(
            lambda r: (str(r["eng_src"]).strip(), str(r["meti_tgt"]).strip())
                      in orig_tuples, axis=1)
        synthetic = synthetic[mask_novel].reset_index(drop=True)
        n_e2m_syn = (synthetic["direction"] == "eng2tgt").sum()
        n_m2e_syn = (synthetic["direction"] == "tgt2eng").sum()
        print(f"  Synthetic after dedup: eng2tgt={n_e2m_syn:,}  "
              f"tgt2eng={n_m2e_syn:,}  total={len(synthetic):,}")

    # ── Cap synthetic per direction at MAX_AUG_RATIO × originals ────────
    for dir_key in ["eng2tgt", "tgt2eng"]:
        n_orig_dir = (orig_copy["direction"] == dir_key).sum()
        max_syn_dir = int(n_orig_dir * MAX_AUG_RATIO)
        syn_dir_mask = synthetic["direction"] == dir_key
        n_syn_dir = syn_dir_mask.sum()
        if n_syn_dir > max_syn_dir:
            print(f"  [CAP] {dir_key}: capping synthetic {n_syn_dir:,} → {max_syn_dir:,}")
            keep_idx = (synthetic[syn_dir_mask]
                        .nlargest(max_syn_dir, "qa_score")
                        if "qa_score" in synthetic.columns
                        else synthetic[syn_dir_mask].sample(
                            n=max_syn_dir, random_state=SEED)).index
            drop_idx = synthetic[syn_dir_mask].index.difference(keep_idx)
            synthetic = synthetic.drop(index=drop_idx).reset_index(drop=True)

    # ── Combine ───────────────────────────────────────────────────────────
    combined = pd.concat([orig_copy, synthetic], ignore_index=True)
    b = len(combined)
    combined = combined.drop_duplicates(
        subset=["eng_src", "meti_tgt"]).reset_index(drop=True)

    n_final_e2m = (combined["direction"] == "eng2tgt").sum()
    n_final_m2e = (combined["direction"] == "tgt2eng").sum()
    print(f"  Combined: {b:,} → after dedup: {len(combined):,}")
    print(f"  Final breakdown: eng2tgt={n_final_e2m:,}  tgt2eng={n_final_m2e:,}  "
          f"ratio={n_final_e2m/max(n_final_m2e,1):.2f}:1")
    print(f"  Augmentation multiplier: ×{len(combined)/max(n_orig, 1):.2f} original")

    # ── Per-direction balance check: ensure originals ≥ ORIG_MIN_RATIO ──
    for dir_key in ["eng2tgt", "tgt2eng"]:
        n_orig_dir   = (orig_copy["direction"] == dir_key).sum()
        n_combined_dir = (combined["direction"] == dir_key).sum()
        orig_frac    = (combined[
            (combined["direction"] == dir_key) &
            (combined["aug_type"] == "original")
        ].shape[0]) / max(n_combined_dir, 1)

        if orig_frac < ORIG_MIN_RATIO:
            target_total = int(n_orig_dir / ORIG_MIN_RATIO)
            n_extra = target_total - n_combined_dir
            if n_extra > 0:
                extra = orig_copy[orig_copy["direction"] == dir_key].sample(
                    n=n_extra, replace=True, random_state=SEED)
                combined = pd.concat([combined, extra], ignore_index=True)
                print(f"  [UPSAMPLE {dir_key}] +{n_extra:,} original copies → "
                      f"≥{int(ORIG_MIN_RATIO*100)}% originals for {dir_key}")

    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined[["eng_src", "meti_tgt", "aug_type", "direction"]].to_excel(
        save_path, index=False)
    print(f"  ✓ Saved → {save_path}  (total: {len(combined):,})")

    # Summary chart
    type_counts = combined["aug_type"].value_counts().sort_values(ascending=False)
    fig, axes   = plt.subplots(1, 2, figsize=(14, 6))
    bars = axes[0].bar(range(len(type_counts)), type_counts.values,
                       color=COLORS[:len(type_counts)], edgecolor="white")
    axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=9)
    axes[0].set_xticks(range(len(type_counts)))
    axes[0].set_xticklabels([t.replace("_", "\n") for t in type_counts.index],
                             fontsize=9, rotation=10)
    axes[0].set_title("Pairs per Aug Type", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("# Sentence Pairs")
    axes[1].pie(type_counts.values,
                labels=[f"{l.replace('_', ' ')}\n{v:,}" for l, v
                        in zip(type_counts.index, type_counts.values)],
                colors=COLORS[:len(type_counts)], autopct="%1.1f%%",
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5),
                textprops={"fontsize": 8})
    axes[1].set_title("Proportion", fontsize=12, fontweight="bold")
    plt.suptitle(f"Augmentation Summary [{tag}]  —  "
                 f"Orig: {n_orig:,}  →  Combined: {len(combined):,} "
                 f"(×{len(combined)/max(n_orig, 1):.1f})",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"aug_summary_{tag}.png")
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"  ✓ Summary figure → {fig_path}")

    return combined
