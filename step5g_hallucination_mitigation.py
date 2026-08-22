"""
step5g_hallucination_mitigation.py — Experiment 7: Hallucination Mitigation
=============================================================================

Variants:
  (A) BASELINE      — greedy decoding, exactly as used for all Exp-1,2,3,5 eval
                       (do_sample=False, no penalties). This is the reference
                       point; numbers should match that experiment's own
                       eval if BEST_ADAPTER_FOR_EXP7 wasn't changed.
  (B) REPETITION-PENALIZED — greedy decoding + repetition_penalty and
                       no_repeat_ngram_size (config.HALLU_MITIGATION_*).
                       Directly targets the "repetition" hallucination flag.
  (C) LABSE-RERANKED BEAM — beam search with N beams, rerank the N
                       candidates per sentence by LaBSE(src, candidate)
                       cosine similarity, keep the best-aligned one.
                       Directly targets "omission" and "copy-source" flags
                       (a candidate that drifted from src or copied it
                       verbatim will usually score lower on LaBSE than a
                       faithful translation).

"""

import os
import sys
import warnings

# ── Language MUST be set before config.py (or anything importing it) is
# imported below -- config.py reads MT_LANG from the environment at import
# time. Mirrors main.py's _preparse_lang() so this script also works when
# run standalone (not just via main.py) with --lang on the command line.

def _preparse_lang() -> str:
    if "--lang" in sys.argv:
        idx = sys.argv.index("--lang")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.environ.get("MT_LANG", "mni")


os.environ["MT_LANG"] = _preparse_lang()

import numpy as np
import pandas as pd
import torch
import sacrebleu
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

from config import (
    BEST_ADAPTER_FOR_EXP7, TEST_PATH, DIRECTIONS, PROMPT_TEMPLATES,
    MAX_SEQ_LEN, MAX_NEW_TOKENS, BATCH_SIZE_INFER,
    HALL_LEN_RATIO_LOW, HALL_LEN_RATIO_HIGH, HALL_COPY_RATIO,
    HALLU_MITIGATION_REPETITION_PENALTY, HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE,
    HALLU_MITIGATION_NUM_BEAMS, HALLU_MITIGATION_LABSE_RERANK,
    OUTPUT_DIR, FIGURES_DIR, SCORES_FILE, EXP7_PREDS_FILE, COMET_MODEL,
    MT_LANG,
    ADAPTER_DIR,
    EXP1_ADAPTER, EXP2_ADAPTER, EXP3_ADAPTER,
    EXP4_ADAPTER, EXP5_ADAPTER, EXP6_ADAPTER,
)
from aug_utils import load_ft_model, free_model

import json


# ══════════════════════════════════════════════════════════════════════════
#  Metrics (mirrors step4_evaluate.py / aug_finetune_eval.py so numbers are
#  directly comparable to every other experiment in the pipeline)
# ══════════════════════════════════════════════════════════════════════════

def compute_sacre(preds, refs):
    bleu   = sacrebleu.corpus_bleu(preds, [refs])
    chrf   = sacrebleu.corpus_chrf(preds, [refs])
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    return {"BLEU": round(bleu.score, 2), "chrF": round(chrf.score, 2),
            "chrF++": round(chrfpp.score, 2)}


def compute_comet(srcs, preds, refs):
    try:
        from comet import download_model, load_from_checkpoint
        path = download_model(COMET_MODEL)
        mdl  = load_from_checkpoint(path)
        data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
        out  = mdl.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        score = round(float(np.mean(out.scores)) * 100, 2)
        del mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score
    except Exception as e:
        print(f"  [COMET] Skipped — {e}")
        return None


def hallucination_stats(srcs, preds, refs):
    """Same rule set as step2/step4's hallucination_analysis, but returns a
    dict of counts/rates instead of only printing, so variants can be
    compared side by side in one table."""
    n = len(preds)
    omission = repetition = copy_hall = 0
    for src, pred, ref in zip(srcs, preds, refs):
        pt = pred.split() if pred.strip() else [""]
        rt = ref.split()  if ref.strip()  else [""]
        st = src.split()  if src.strip()  else [""]
        ratio = len(pt) / max(len(rt), 1)
        if ratio < HALL_LEN_RATIO_LOW:  omission   += 1
        if ratio > HALL_LEN_RATIO_HIGH: repetition += 1
        ps = set(t.lower() for t in pt); ss = set(t.lower() for t in st)
        if ss and len(ps & ss) / max(len(ps), 1) > HALL_COPY_RATIO:
            copy_hall += 1
    return {
        "n": n,
        "omission_pct":   round(omission   / max(n, 1) * 100, 2),
        "repetition_pct": round(repetition / max(n, 1) * 100, 2),
        "copy_src_pct":   round(copy_hall  / max(n, 1) * 100, 2),
        "any_flag_pct":   None,  # filled below once we know overlap; kept
                                  # simple/conservative as sum of the three
    }


# ══════════════════════════════════════════════════════════════════════════
#  LaBSE scorer (loaded once, used for reranking beam candidates)
# ══════════════════════════════════════════════════════════════════════════

class LabseScorer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("  [LaBSE] Loading reranker model …")
        self.model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    def best_of(self, src: str, candidates: list) -> str:
        """Return the candidate whose embedding is most cosine-similar to src."""
        cand = [c if c.strip() else " " for c in candidates]
        emb_src  = self.model.encode([src], normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
        emb_cand = self.model.encode(cand, normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
        sims = (emb_cand @ emb_src[0])
        return candidates[int(np.argmax(sims))]

    def free(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
#  Decoding variants
# ══════════════════════════════════════════════════════════════════════════

def _prompt(tok, text, sys_msg):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True)


def decode_baseline(texts, model, tok, sys_msg):
    """(A) Greedy decoding, no penalties — the reference point."""
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE_INFER), desc="    [A] baseline"):
            batch = texts[i:i + BATCH_SIZE_INFER]
            prompts = [_prompt(tok, t, sys_msg) for t in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
            outs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
            new_t = outs[:, inputs.input_ids.shape[1]:]
            preds.extend(p.strip() for p in tok.batch_decode(new_t, skip_special_tokens=True))
    return preds


def decode_repetition_penalized(texts, model, tok, sys_msg):
    """(B) Greedy decoding + repetition_penalty + no_repeat_ngram_size.
    Directly targets the 'repetition' hallucination flag (very long,
    looping outputs).

    NOTE: no_repeat_ngram_size is a HARD constraint (forbids ever repeating
    an n-gram anywhere earlier in the sequence). For scripts with sparse
    tokenizer coverage (e.g. mni/Meetei Mayek) this can force the decoder
    onto low-probability substitute tokens and cascade into garbled output
    -- see config.py's HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE_BY_LANG note.
    When the configured size is 0 for the active language, we omit the
    kwarg entirely rather than relying on generate() to no-op on 0."""
    preds = []
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        repetition_penalty=HALLU_MITIGATION_REPETITION_PENALTY,
    )
    if HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE and HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE > 0:
        gen_kwargs["no_repeat_ngram_size"] = HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE_INFER), desc="    [B] rep-penalty"):
            batch = texts[i:i + BATCH_SIZE_INFER]
            prompts = [_prompt(tok, t, sys_msg) for t in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
            outs = model.generate(**inputs, pad_token_id=tok.pad_token_id, **gen_kwargs)
            new_t = outs[:, inputs.input_ids.shape[1]:]
            preds.extend(p.strip() for p in tok.batch_decode(new_t, skip_special_tokens=True))
    return preds


def decode_labse_reranked_beam(texts, model, tok, sys_msg, labse: LabseScorer):
    """(C) Beam search (N beams) + LaBSE reranking of the N candidates per
    sentence. Slower (one sentence at a time, N sequences each) — this
    trades throughput for quality/faithfulness, which is the point of the
    ablation. Directly targets 'omission' and 'copy-source' flags."""
    preds = []
    n_beams = HALLU_MITIGATION_NUM_BEAMS
    with torch.no_grad():
        for t in tqdm(texts, desc="    [C] labse-rerank-beam"):
            prompt = _prompt(tok, t, sys_msg)
            inputs = tok([prompt], return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN).to(model.device)
            outs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                num_beams=n_beams, num_return_sequences=n_beams,
                do_sample=False, early_stopping=True,
                pad_token_id=tok.pad_token_id)
            new_t = outs[:, inputs.input_ids.shape[1]:]
            candidates = [p.strip() for p in tok.batch_decode(new_t, skip_special_tokens=True)]
            best = labse.best_of(t, candidates) if HALLU_MITIGATION_LABSE_RERANK \
                   else candidates[0]
            preds.append(best)
    return preds


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery — run on EVERY trained checkpoint, not just the
#  configured "best" one. config.py pre-creates every EXPn_ADAPTER folder
#  on import regardless of whether that experiment was ever trained, so
#  os.path.isdir alone is not a reliable "is this usable" signal — we
#  require real adapter files inside.
# ══════════════════════════════════════════════════════════════════════════

CANDIDATE_CHECKPOINTS = [
    ("r1",   ADAPTER_DIR),
    ("exp1", EXP1_ADAPTER),
    ("exp2", EXP2_ADAPTER),
    ("exp3", EXP3_ADAPTER),
    ("exp4", EXP4_ADAPTER),
    ("exp5", EXP5_ADAPTER),
    ("exp6", EXP6_ADAPTER),
]


def _has_trained_adapter(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = os.listdir(path)
    has_adapter_cfg = "adapter_config.json" in names
    has_weights = any(n.endswith((".safetensors", ".bin")) for n in names)
    return has_adapter_cfg and has_weights


def discover_trained_checkpoints():
    found, skipped = [], []
    for name, path in CANDIDATE_CHECKPOINTS:
        if _has_trained_adapter(path):
            found.append((name, path))
        else:
            skipped.append((name, path))
    return found, skipped


def _merge_save_csv(path: str, new_rows_df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """
    Merge new_rows_df into the CSV at `path`, REPLACING any existing rows
    whose key_cols values match a row in new_rows_df, and leaving every
    other existing row untouched. Used so a selective re-run (e.g. only
    B_rep_penalty / eng2tgt) overwrites only what it recomputed instead of
    clobbering the whole file.
    """
    if new_rows_df.empty:
        if os.path.exists(path):
            return pd.read_csv(path)
        return new_rows_df

    if os.path.exists(path):
        old_df = pd.read_csv(path)
    else:
        old_df = pd.DataFrame(columns=new_rows_df.columns)

    if not old_df.empty:
        keys_to_replace = new_rows_df[key_cols].drop_duplicates()
        mask = pd.Series(False, index=old_df.index)
        for _, key_row in keys_to_replace.iterrows():
            m = pd.Series(True, index=old_df.index)
            for c in key_cols:
                m &= (old_df[c].astype(str) == str(key_row[c]))
            mask |= m
        old_df = old_df[~mask]

    merged = pd.concat([old_df, new_rows_df], ignore_index=True)
    merged.to_csv(path, index=False)
    return merged


def _save_scores(score_key, direction, metrics):
    scores = {}
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            try: scores = json.load(f)
            except Exception: scores = {}
    scores[f"{score_key}_{direction}"] = metrics
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════
#  Per-checkpoint run — identical decoding/eval logic as before, just
#  parameterised by which checkpoint is loaded, and namespacing every
#  output file / score key by checkpoint so runs don't clobber each other.
# ══════════════════════════════════════════════════════════════════════════

ALL_VARIANT_NAMES = ["A_baseline", "B_rep_penalty", "C_labse_rerank_beam"]


def run_for_checkpoint(ckpt_name: str, ckpt_path: str, test_df: pd.DataFrame,
                        variant_filter=None, direction_filter=None) -> list:
    """
    variant_filter   : list of variant names to (re)compute, e.g.
                        ["B_rep_penalty"]. None = all variants.
    direction_filter : list of direction keys to (re)compute, e.g.
                        ["eng2tgt"]. None = both directions.
    Only the requested (direction, variant) cells are computed; everything
    else is left exactly as it already is on disk (all_scores.json keys,
    the per-checkpoint pred CSV, and the combined summary CSV are all
    MERGE-saved, not overwritten).
    """
    directions_to_run = [d for d in DIRECTIONS if direction_filter is None
                          or d[0] in direction_filter]
    variants_to_run = [v for v in ALL_VARIANT_NAMES if variant_filter is None
                        or v in variant_filter]
    need_labse = "C_labse_rerank_beam" in variants_to_run and HALLU_MITIGATION_LABSE_RERANK

    print(f"\n{'#'*65}")
    print(f"  CHECKPOINT: {ckpt_name}  →  {ckpt_path}")
    print(f"  Directions to (re)compute : {[d for d, _ in directions_to_run]}")
    print(f"  Variants to (re)compute   : {variants_to_run}")
    print(f"{'#'*65}")

    model, tok = load_ft_model(ckpt_path)
    labse = LabseScorer() if need_labse else None

    summary_rows = []
    all_rows = []

    for dir_key, _ in directions_to_run:
        sub = test_df[test_df["direction"] == dir_key].reset_index(drop=True) \
              if "direction" in test_df.columns else test_df
        if len(sub) == 0:
            print(f"  [WARN] No rows for {dir_key} — skipping.")
            continue

        srcs = sub["eng_src"].astype(str).tolist()
        refs = sub["meti_tgt"].astype(str).tolist()
        sys_msg = PROMPT_TEMPLATES.get(dir_key, PROMPT_TEMPLATES["eng2tgt"])

        print(f"\n  {'─'*55}\n  [{ckpt_name}] Direction: {dir_key}  ({len(srcs):,} rows)\n  {'─'*55}")

        variants = {}
        if "A_baseline" in variants_to_run:
            variants["A_baseline"] = decode_baseline(srcs, model, tok, sys_msg)
        if "B_rep_penalty" in variants_to_run:
            variants["B_rep_penalty"] = decode_repetition_penalized(srcs, model, tok, sys_msg)
        if "C_labse_rerank_beam" in variants_to_run and labse is not None:
            variants["C_labse_rerank_beam"] = decode_labse_reranked_beam(
                srcs, model, tok, sys_msg, labse)

        for variant_name, preds in variants.items():
            sacre = compute_sacre(preds, refs)
            comet = compute_comet(srcs, preds, refs)
            if comet is not None:
                sacre["COMET"] = comet
            hall = hallucination_stats(srcs, preds, refs)

            print(f"\n  [{ckpt_name}] [{variant_name}] [{dir_key}]")
            for k, v in sacre.items():
                print(f"    {k:8s}: {v:.2f}")
            print(f"    Hallucination — omission={hall['omission_pct']}%  "
                  f"repetition={hall['repetition_pct']}%  "
                  f"copy_src={hall['copy_src_pct']}%")

            score_key = f"exp7_{ckpt_name}_{variant_name}"
            _save_scores(score_key, dir_key, sacre)   # already merge-safe (single key update)

            summary_rows.append({
                "checkpoint": ckpt_name, "direction": dir_key, "variant": variant_name,
                **sacre, **{k: v for k, v in hall.items() if k != "any_flag_pct"},
            })
            for s, r, p in zip(srcs, refs, preds):
                all_rows.append({"checkpoint": ckpt_name, "direction": dir_key,
                                 "variant": variant_name, "src": s, "ref": r, "pred": p})

    free_model(model)
    if labse is not None:
        labse.free()

    # Merge-save: only the (direction, variant) rows we just recomputed are
    # replaced; every other row already in the file is left untouched.
    pred_file = EXP7_PREDS_FILE.replace(".csv", f"_{ckpt_name}.csv")
    new_pred_df = pd.DataFrame(all_rows)
    _merge_save_csv(pred_file, new_pred_df, key_cols=["direction", "variant"])
    print(f"\n  ✓ [{ckpt_name}] Predictions (merged) → {pred_file}")

    return summary_rows


# ══════════════════════════════════════════════════════════════════════════
#  Main — analyze every trained checkpoint, then produce a combined
#  cross-checkpoint summary table + charts.
# ══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lang", type=str, default="mni", choices=["mni", "as", "brx"],
                   help="Language pair: mni (Manipuri), as (Assamese), brx (Bodo). "
                        "Already applied before config.py was imported via the "
                        "MT_LANG env var (see _preparse_lang() near the top of "
                        "this file); this flag exists mainly for --help / "
                        "validation, same pattern as main.py.")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   choices=[n for n, _ in CANDIDATE_CHECKPOINTS],
                   help="Only (re)run these checkpoints, e.g. --checkpoints "
                        "r1 exp1 exp2 exp3 exp5. Default: every discovered "
                        "trained checkpoint.")
    p.add_argument("--variants", nargs="+", default=None,
                   choices=ALL_VARIANT_NAMES,
                   help="Only (re)run these decode variants, e.g. "
                        "--variants B_rep_penalty. Default: all variants. "
                        "Rows/score-keys for variants NOT listed are left "
                        "exactly as they already are on disk.")
    p.add_argument("--directions", nargs="+", default=None,
                   choices=["eng2tgt", "tgt2eng"],
                   help="Only (re)run these directions, e.g. --directions "
                        "eng2tgt. Default: both. Directions not listed are "
                        "left exactly as they already are on disk.")
    args = p.parse_args()

    # Re-assert in case --lang differed from the pre-parsed env value
    # (shouldn't happen given _preparse_lang, but keep it bullet-proof --
    # same safety check main.py does).
    if os.environ.get("MT_LANG") != args.lang:
        print(f"  ERROR: --lang {args.lang} was parsed but MT_LANG env var is "
              f"'{os.environ.get('MT_LANG')}'. This should not happen — "
              f"please report. Re-run with only --lang set.")
        sys.exit(1)

    is_partial_rerun = bool(args.checkpoints or args.variants or args.directions)

    print("\n" + "=" * 65)
    print("  STEP 5g — EXP-7: HALLUCINATION MITIGATION (decode-time)")
    if is_partial_rerun:
        print("  SELECTIVE RE-RUN — only the cells below are recomputed;")
        print("  everything else in all_scores.json / summary CSV / pred")
        print("  CSVs is left untouched.")
        print(f"    checkpoints : {args.checkpoints or 'ALL'}")
        print(f"    variants    : {args.variants or 'ALL'}")
        print(f"    directions  : {args.directions or 'ALL (both)'}")
    else:
        print("  Runs on EVERY trained checkpoint found (not just the")
        print("  configured 'best' one) — r1 + whichever Exp-1..6 you trained.")
        print("  Variants        : (A) baseline  (B) rep-penalty  "
              "(C) LaBSE-reranked beam")
    print("=" * 65)

    checkpoints, skipped = discover_trained_checkpoints()

    if skipped:
        print("\n  Skipping (no trained adapter found):")
        for name, path in skipped:
            print(f"    [SKIP] {name:5s} → {path}")

    if not checkpoints:
        print(f"\n  ERROR: no trained checkpoints found for lang={MT_LANG}.")
        print("  Run at least one of steps 3, 5a-5f before step 5g.")
        sys.exit(1)

    if args.checkpoints:
        wanted = set(args.checkpoints)
        missing = wanted - {n for n, _ in checkpoints}
        if missing:
            print(f"\n  ERROR: requested checkpoint(s) not trained/found: "
                  f"{sorted(missing)}")
            sys.exit(1)
        checkpoints = [(n, p) for n, p in checkpoints if n in wanted]

    print(f"\n  Running on {len(checkpoints)} checkpoint(s): "
          f"{[n for n, _ in checkpoints]}")

    test_df = pd.read_excel(TEST_PATH).dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)

    all_summary_rows = []
    for ckpt_name, ckpt_path in checkpoints:
        all_summary_rows.extend(run_for_checkpoint(
            ckpt_name, ckpt_path, test_df,
            variant_filter=args.variants, direction_filter=args.directions))

    new_summary_df = pd.DataFrame(all_summary_rows)
    summary_csv = os.path.join(
        OUTPUT_DIR, "exp7_hallucination_mitigation_summary_all_checkpoints.csv")
    # Merge-save: only rows for (checkpoint, direction, variant) combos we
    # just recomputed are replaced; every other existing row is preserved.
    summary_df = _merge_save_csv(
        summary_csv, new_summary_df, key_cols=["checkpoint", "direction", "variant"])
    print(f"\n  ✓ Combined summary table (merged) → {summary_csv}")
    print(f"\n{summary_df.to_string(index=False)}")

    if summary_df.empty:
        print("\n  STEP 5g COMPLETE (no rows produced)")
        return

    # Charts are regenerated from the full merged table (not just this run's
    # subset) so a selective re-run still produces complete, up-to-date
    # figures across every checkpoint present on disk.
    all_checkpoint_names = summary_df["checkpoint"].unique().tolist()

    # ── Per-checkpoint before/after hallucination chart ─────────────────────
    hall_cols = ["omission_pct", "repetition_pct", "copy_src_pct"]
    for ckpt_name in all_checkpoint_names:
        ck_df = summary_df[summary_df["checkpoint"] == ckpt_name]
        if ck_df.empty:
            continue
        fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(7 * len(DIRECTIONS), 5))
        if len(DIRECTIONS) == 1:
            axes = [axes]
        for ax, (dir_key, _) in zip(axes, DIRECTIONS):
            d = ck_df[ck_df["direction"] == dir_key]
            if d.empty:
                continue
            x = np.arange(len(hall_cols)); w = 0.8 / max(len(d), 1)
            for i, (_, row) in enumerate(d.iterrows()):
                vals = [row[c] for c in hall_cols]
                offset = (i - len(d) / 2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w * 0.9, label=row["variant"])
                ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(["Omission", "Repetition", "Copy-src"], fontsize=10)
            ax.set_ylabel("% of test sentences flagged")
            ax.set_title(f"Hallucination rates — {dir_key}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
        plt.suptitle(f"Exp-7 [{ckpt_name}]: Hallucination Mitigation — Before/After",
                     fontsize=13, y=1.03)
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"exp7_hallucination_before_after_{ckpt_name}.png")
        plt.savefig(fig_path, bbox_inches="tight"); plt.close()
        print(f"  ✓ {fig_path}")

    # ── Cross-checkpoint leaderboard (variant A_baseline only, so it's an
    #    apples-to-apples comparison of the checkpoints themselves) ────────
    base_df = summary_df[summary_df["variant"] == "A_baseline"]
    if not base_df.empty:
        metrics = [m for m in ["BLEU", "chrF", "chrF++", "COMET"] if m in base_df.columns]
        for dir_key, _ in DIRECTIONS:
            d = base_df[base_df["direction"] == dir_key]
            if d.empty or not metrics:
                continue
            x = np.arange(len(metrics)); w = 0.8 / max(len(d), 1)
            fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 2.2), 6))
            for i, (_, row) in enumerate(d.iterrows()):
                vals = [row.get(m, 0) for m in metrics]
                offset = (i - len(d) / 2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w * 0.9, label=row["checkpoint"])
                ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
            ax.set_ylabel("Score")
            ax.set_title(f"Checkpoint leaderboard (greedy decode) — {dir_key}",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, ncol=min(len(d), 4))
            plt.tight_layout()
            fig_path = os.path.join(FIGURES_DIR, f"exp7_checkpoint_leaderboard_{dir_key}.png")
            plt.savefig(fig_path, bbox_inches="tight"); plt.close()
            print(f"  ✓ {fig_path}")

    print("\n  STEP 5g COMPLETE")


if __name__ == "__main__":
    main()
