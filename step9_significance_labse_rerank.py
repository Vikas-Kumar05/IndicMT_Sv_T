"""
step9_significance.py — Multi-Seed Aggregation + Significance Testing
===================================================================
Run AFTER run_seed_sweep.py has produced multiple seeds for your chosen
experiments. This script:

  1. Aggregates all_scores.json entries across seeds into mean +/- std
     per metric per experiment per direction -> writes a CSV table and a
     bar chart with error bars. This is the table you put in the paper
     instead of single-run point estimates.

  2. Runs a PAIRED BOOTSTRAP SIGNIFICANCE TEST (Koehn 2004,
     https://aclanthology.org/W04-3250.pdf) between two systems you name
     (e.g. your best augmentation experiment vs R1, or vs the second-best
     experiment), using their saved prediction CSVs. This answers "is
     system A actually better than system B, or is the gap noise?" using
     the paper's own statistic: the fraction of resampled virtual test
     sets in which system A's CORPUS-level score beats system B's.
     Primary metrics: corpus BLEU and corpus chrF++ (word_order=2).
     Per-sentence n-gram match statistics are tokenized/counted ONCE per
     system (the expensive part); every bootstrap resample just sums
     cached integer arrays and applies the closed-form BLEU/chrF formula
     -- not re-tokenizing text on every iteration. This matters a lot for
     --compare-all, which multiplies the per-pair cost by every system
     pair x every direction.

Usage:
    python step9_significance.py --lang mni --experiments exp1 exp2 exp3 exp4 exp5 exp6
    python step9_significance.py --lang mni --compare exp6 r1
    python step9_significance.py --lang mni --compare exp6 exp4 --direction tgt2eng

    # NEW: run every pairwise comparison across both directions in ONE go,
    # save every row into a single combined CSV:
    python step9_significance.py --lang mni --compare-all
    python step9_significance.py --lang mni --compare-all --systems r1 exp1 exp2 exp3 exp5
    python step9_significance.py --lang mni --compare-all --directions eng2tgt
    python step9_significance.py --lang mni --compare-all --out outputs/mni/my_sig_table.csv

    # NEW: run significance testing on a DECODE-TIME VARIANT instead of
    # plain greedy decoding. By default (no --variant), this script reads
    # the ORIGINAL per-experiment prediction files, which are all Variant A
    # (greedy, no mitigation) -- the same predictions step4_evaluate.py /
    # aug_finetune_eval.py already produced. Passing --variant switches to
    # the per-checkpoint files step5g_hallucination_mitigation.py writes
    # (outputs/<lang>/preds_exp7_hallucination_mitigation_<ckpt>.csv), which
    # contain all three decode variants in one file, and filters to the
    # requested one. This lets you test whether the MAIN RESULT reported in
    # the paper's tables (LaBSE-reranked beam search, Variant C) is
    # significantly better than R1 or another checkpoint, not just whether
    # the underlying (unmitigated) checkpoints differ.
    #   A_baseline           -> greedy, no mitigation (same numbers as the
    #                            default mode, just read from the exp7 file)
    #   B_rep_penalty         -> repetition-penalised decoding
    #   C_labse_rerank_beam   -> LaBSE-reranked beam search (paper's main
    #                            decode-time result)
    # step5g must have been run first (produces the per-checkpoint exp7
    # prediction files); "baseline" is not available in this mode since
    # step5g only runs on fine-tuned checkpoints, not the zero-shot model.
    python step9_significance.py --lang mni --compare r1 exp3 --variant C_labse_rerank_beam
    python step9_significance.py --lang mni --compare-all --variant C_labse_rerank_beam
    python step9_significance.py --lang mni --compare-all --variant C_labse_rerank_beam --systems r1 exp1 exp2 exp3 exp5
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["mni", "as", "brx"])
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Experiments to aggregate across seeds, e.g. exp4 exp6. "
                        "If omitted, aggregates baseline/r1/exp1..exp6.")
    p.add_argument("--compare", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"),
                   default=None,
                   help="Run a paired bootstrap significance test between two "
                        "systems' saved predictions, e.g. --compare exp6 r1")
    p.add_argument("--compare-all", action="store_true",
                   help="Run paired bootstrap significance tests between "
                        "EVERY pair of systems (from --systems, or every "
                        "system with a discoverable pred CSV if --systems "
                        "is omitted), across the requested direction(s), "
                        "and save every result row into ONE combined CSV. "
                        "Ignores --compare when set.")
    p.add_argument("--systems", nargs="+", default=None,
                   choices=["baseline", "r1", "exp1", "exp2", "exp3",
                            "exp4", "exp5", "exp6"],
                   help="Restrict --compare-all to these systems only. "
                        "Default: auto-discover every system whose pred "
                        "CSV exists on disk.")
    p.add_argument("--direction", default="eng2tgt", choices=["eng2tgt", "tgt2eng"],
                   help="Direction for --compare (single-pair mode). Ignored "
                        "by --compare-all, which uses --directions instead.")
    p.add_argument("--directions", nargs="+", default=["eng2tgt", "tgt2eng"],
                   choices=["eng2tgt", "tgt2eng"],
                   help="Direction(s) for --compare-all. Default: both.")
    p.add_argument("--variant", default=None,
                   choices=["A_baseline", "B_rep_penalty", "C_labse_rerank_beam"],
                   help="Decode-time variant to test significance on, read "
                        "from step5g's per-checkpoint exp7 prediction files "
                        "(outputs/<lang>/preds_exp7_hallucination_mitigation_"
                        "<ckpt>.csv). Default: None, which uses the original "
                        "per-experiment prediction files (greedy decoding, "
                        "same as step4_evaluate.py / aug_finetune_eval.py). "
                        "Pass C_labse_rerank_beam to test significance on "
                        "the paper's main decode-time result. 'baseline' is "
                        "not a valid --compare/--systems entry in this mode "
                        "-- step5g does not run on the zero-shot model.")
    p.add_argument("--out", default=None,
                   help="Output CSV path for --compare-all. Default: "
                        "<OUTPUT_DIR>/significance_all_comparisons.csv")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
#  Part 1: multi-seed aggregation
# ══════════════════════════════════════════════════════════════════════════

def aggregate_seeds(scores: dict, experiments, direction: str, seeds_hint=(16, 42, 123)):
    """
    Look for score keys of the form "{exp}_{direction}" (default seed 16,
    unsuffixed) and "{exp}_s{seed}_{direction}" (sweep seeds), and compute
    mean/std per metric across whichever of those are present.
    """
    rows = []
    for exp in experiments:
        per_seed_metrics = []
        base_key = f"{exp}_{direction}"
        if base_key in scores:
            per_seed_metrics.append(scores[base_key])
        for seed in seeds_hint:
            if seed == 16:
                continue  # already covered by base_key
            k = f"{exp}_s{seed}_{direction}"
            if k in scores:
                per_seed_metrics.append(scores[k])

        if not per_seed_metrics:
            print(f"  [WARN] No scores found for {exp} [{direction}] — skipping.")
            continue

        metric_names = set()
        for m in per_seed_metrics:
            metric_names.update(m.keys())

        row = {"experiment": exp, "n_seeds": len(per_seed_metrics)}
        for mname in sorted(metric_names):
            vals = [m[mname] for m in per_seed_metrics if mname in m]
            row[f"{mname}_mean"] = round(float(np.mean(vals)), 3)
            row[f"{mname}_std"]  = round(float(np.std(vals)), 3) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_aggregate(df: pd.DataFrame, direction: str, figures_dir: str, system_colors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = [c[:-5] for c in df.columns if c.endswith("_mean")]
    if not metric_cols or df.empty:
        print("  [WARN] Nothing to plot.")
        return

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, m in zip(axes, metric_cols):
        means = df[f"{m}_mean"].values
        stds  = df[f"{m}_std"].values
        x = np.arange(len(df))
        colors = [system_colors.get(e, "#4C72B0") for e in df["experiment"]] \
                 if system_colors else "#4C72B0"
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=8, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["experiment"], rotation=30, fontsize=9)
        ax.set_title(f"{m} (mean ± std across seeds)", fontsize=11, fontweight="bold")

    plt.suptitle(f"Multi-Seed Aggregate Results [{direction}]", fontsize=13, y=1.03)
    plt.tight_layout()
    out = os.path.join(figures_dir, f"seed_aggregate_{direction}.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Part 2: paired bootstrap significance test (from saved pred CSVs)
# ══════════════════════════════════════════════════════════════════════════

def _pred_path_for_system(system: str, output_dir: str, direction: str) -> str:
    """
    Map a system name to its saved prediction CSV. Mirrors the filenames
    each step script already writes.

    NOTE: baseline/r1 save ONE CSV PER DIRECTION (preds_baseline_eng2tgt.csv /
    preds_baseline_tgt2eng.csv, and similarly for r1), unlike exp1..exp6
    which save a single CSV containing both directions distinguished by a
    "direction" column. So baseline/r1 need the direction to pick the
    right file; exp1..exp6 don't.

    This is the LEGACY path used when --variant is not passed -- it always
    resolves to Variant A (greedy, no mitigation) predictions.
    """
    if system == "baseline":
        return os.path.join(output_dir, f"preds_baseline_{direction}.csv")
    if system == "r1":
        return os.path.join(output_dir, f"preds_r1_{direction}.csv")
    if system.startswith("exp"):
        return os.path.join(output_dir, f"preds_{system}.csv")
    raise ValueError(f"Unrecognised system name: {system}")


def _pred_path_for_system_variant(system: str, output_dir: str) -> str:
    """
    Map a system name to its step5g exp7 prediction CSV -- used when
    --variant is passed. Unlike _pred_path_for_system(), this path does NOT
    depend on direction: step5g writes ONE file per checkpoint containing
    both directions and all three decode variants, distinguished by the
    "direction" and "variant" columns respectively. Filtering to a single
    (direction, variant) pair happens later, in paired_bootstrap_significance().

    step5g's checkpoint names are r1, exp1, exp2, exp3, exp4, exp5, exp6 --
    there is no "baseline" file here, since step5g only runs decode-time
    mitigation on fine-tuned checkpoints, not the zero-shot model.
    """
    if system == "baseline":
        raise ValueError(
            "'baseline' has no decode-time-variant predictions -- "
            "step5g_hallucination_mitigation.py only runs on fine-tuned "
            "checkpoints (r1, exp1..exp6), not the zero-shot baseline. "
            "Drop --variant to compare against baseline's greedy predictions, "
            "or exclude 'baseline' from --systems/--compare in variant mode.")
    if system == "r1" or system.startswith("exp"):
        return os.path.join(
            output_dir, f"preds_exp7_hallucination_mitigation_{system}.csv")
    raise ValueError(f"Unrecognised system name: {system}")


def discover_available_systems(output_dir: str, candidates=None, variant=None):
    """
    Return the subset of `candidates` (default: baseline, r1, exp1..exp6)
    that actually have a usable prediction file on disk. Used by
    --compare-all when --systems is not given, so the combined run doesn't
    hard-fail on experiments you never trained.

    If `variant` is set, checks the step5g exp7 files instead of the legacy
    per-direction files, and drops "baseline" from the candidate list
    entirely (it has no exp7 file -- see _pred_path_for_system_variant).
    """
    if candidates is None:
        candidates = ["baseline", "r1", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
    if variant is not None:
        candidates = [c for c in candidates if c != "baseline"]

    available = []
    for sys_name in candidates:
        if variant is not None:
            path = _pred_path_for_system_variant(sys_name, output_dir)
            if os.path.exists(path):
                available.append(sys_name)
            continue
        found = False
        for direction in ("eng2tgt", "tgt2eng"):
            try:
                path = _pred_path_for_system(sys_name, output_dir, direction)
            except ValueError:
                continue
            if os.path.exists(path):
                found = True
                break
        if found:
            available.append(sys_name)
    return available


_TOKENIZATION_ARTIFACT_SEEN = set()


def _check_tokenization_artifact(dedup_key, label: str, texts):
    """
    Consolidated replacement for sacrebleu's own "forgot to detokenize"
    check (sacrebleu/metrics/base.py: warns when >=100 lines end with a
    literal ' .' -- a space before the period, e.g. 'the fox jumps .'
    instead of 'the fox jumps.'). That check is BLEU-only (chrF is
    character-level and unaffected) and normally fires its generic 3-line
    message every time _extract_corpus_statistics() is called -- which
    means it repeats identically for every pair in --compare-all if the
    same reference/system text is reused across pairs (very likely, since
    the same test-set references get compared against every system).

    This version reports the SAME information -- which text source is
    affected and what fraction of its lines look pre-tokenized -- but only
    ONCE per unique text source for the whole run (tracked via dedup_key,
    e.g. the direction for references, or the prediction file path for a
    system's output), and paired_bootstrap_significance() passes
    BLEU(force=True) to suppress sacrebleu's own duplicate of this message.

    This is informational, not fatal: the metric is still computed. If a
    given source is flagged, its BLEU score (not chrF++) may not be
    comparable to a "standard" sacrebleu run on naturally detokenized text,
    since BLEU's tokenizer expects to do its own tokenization from scratch.
    """
    if dedup_key in _TOKENIZATION_ARTIFACT_SEEN:
        return
    n = len(texts)
    if n == 0:
        return
    tok_count = sum(1 for t in texts if t.endswith(" ."))
    if tok_count >= 100:
        pct = 100 * tok_count / n
        print(f"  [NOTE] {label}: {tok_count}/{n} lines ({pct:.1f}%) end with "
              f"' .' (space before the period) -- looks like pre-tokenized/"
              f"Moses-style text rather than natural detokenized text. This "
              f"can distort BLEU specifically (chrF++ is character-level and "
              f"unaffected). Worth checking the decode/postprocessing step "
              f"for this text if you want standard-comparable BLEU numbers.")
    _TOKENIZATION_ARTIFACT_SEEN.add(dedup_key)


def paired_bootstrap_significance(pred_file_a: str, pred_file_b: str,
                                   direction: str, n_bootstrap: int = 2000,
                                   seed: int = 16, variant: str = None):
    """
    Paired bootstrap resampling (Koehn, 2004 -- "Statistical Significance
    Tests for Machine Translation Evaluation", https://aclanthology.org/W04-3250.pdf)
    between two systems evaluated on the SAME test set.

    `variant`: if set (e.g. "C_labse_rerank_beam"), pred_file_a/b are
    treated as step5g exp7 files -- each containing ALL THREE decode
    variants for one checkpoint, in a "variant" column -- and rows are
    filtered to this variant before scoring, in addition to the existing
    direction filter. If None (default), pred_file_a/b are treated as the
    legacy per-experiment files, which contain only Variant A (greedy)
    predictions and have no "variant" column to filter on.

    Primary metrics: corpus BLEU and corpus chrF++ (word_order=2).

    METHODOLOGY (matches the paper, Section 6):
      For each of n_bootstrap iterations:
        1. Draw a virtual test set by sampling sentence indices WITH
           REPLACEMENT from the shared test set (paired -- same indices
           used for both systems).
        2. Recompute the CORPUS-LEVEL metric (corpus_bleu / corpus_chrf)
           for EACH system on that resampled virtual test set.
        3. Record which system scored higher on that virtual test set.
      The fraction of virtual test sets in which system A outperforms
      system B is the paper's statistic: if A wins >= 95% of the time,
      we conclude A is better with 95% statistical significance.

    IMPORTANT — PERFORMANCE: this does NOT re-tokenize the raw prediction/
    reference strings on every bootstrap iteration (that was the version
    that used to hang for 10+ minutes, and would be even worse combined
    with --compare-all). Tokenizing + counting n-gram matches per sentence
    is done ONCE per system via sacrebleu's internal
    `_extract_corpus_statistics`, giving a (n_sentences x n_stats) integer
    array. Every bootstrap resample then just sums the rows for the
    sampled indices and runs the closed-form BLEU/chrF aggregation
    (`_compute_score_from_stats`) on those sums -- this is EXACTLY the
    same computation sacrebleu's corpus_score does internally, so the
    numbers are identical, but resampling only sums cached integers
    instead of re-tokenizing text. Measured speedup: ~1000-2000x.

    NOTE: This assumes both prediction files were evaluated on the same
    TEST_PATH rows in the same order for the requested direction. If your
    two systems came from different eval runs, this still works AS LONG AS
    both eval passes used the same fixed TEST_PATH (true by default in this
    pipeline), because sentence order and content will match.
    """
    from sacrebleu.metrics import BLEU, CHRF

    df_a = pd.read_csv(pred_file_a)
    df_b = pd.read_csv(pred_file_b)

    if "direction" in df_a.columns:
        df_a = df_a[df_a["direction"] == direction].reset_index(drop=True)
    if "direction" in df_b.columns:
        df_b = df_b[df_b["direction"] == direction].reset_index(drop=True)

    if variant is not None:
        for name, df in (("A", df_a), ("B", df_b)):
            if "variant" not in df.columns:
                raise ValueError(
                    f"--variant was set to '{variant}' but pred_file_"
                    f"{name.lower()} ({pred_file_a if name == 'A' else pred_file_b}) "
                    f"has no 'variant' column. Is this a step5g exp7 file, "
                    f"not a legacy per-experiment file?")
        df_a = df_a[df_a["variant"] == variant].reset_index(drop=True)
        df_b = df_b[df_b["variant"] == variant].reset_index(drop=True)
        if len(df_a) == 0 or len(df_b) == 0:
            raise ValueError(
                f"No rows found for variant='{variant}', direction="
                f"'{direction}' after filtering. Check that step5g was run "
                f"for this checkpoint/direction/variant combination.")

    if len(df_a) != len(df_b):
        raise ValueError(
            f"Prediction files have different lengths for direction="
            f"{direction}"
            f"{', variant=' + variant if variant else ''} "
            f"({len(df_a)} vs {len(df_b)}). Cannot pair them "
            f"sentence-by-sentence. Re-run eval for both systems on the "
            f"same TEST_PATH.")

    refs    = df_a["ref"].astype(str).tolist()
    preds_a = df_a["pred"].astype(str).tolist()
    preds_b = df_b["pred"].astype(str).tolist()
    n = len(refs)

    _check_tokenization_artifact(("ref", direction), f"references [{direction}]", refs)
    _check_tokenization_artifact((pred_file_a,), f"{os.path.basename(pred_file_a)} predictions", preds_a)
    _check_tokenization_artifact((pred_file_b,), f"{os.path.basename(pred_file_b)} predictions", preds_b)

    # force=True suppresses sacrebleu's own internal version of the same
    # "forgot to detokenize" warning (see sacrebleu/metrics/base.py) --
    # we've already reported it above, once per unique text source, instead
    # of letting it repeat verbatim on every pair in --compare-all.
    # chrF++ never triggers this check in sacrebleu (it hardcodes _force=True
    # internally -- the check is BLEU-specific, tokenization doesn't affect
    # a character-level metric the same way), so nothing to pass there.
    metrics = {
        "BLEU":   BLEU(force=True),
        "chrF++": CHRF(word_order=2),
    }

    # ── Expensive part, done ONCE: tokenize + count n-gram matches per
    #    sentence for each system/metric. Shape: (n_sentences, n_stat_cols).
    #    Falls back to the slow (but always-correct) direct recomputation
    #    path if a future/older sacrebleu version doesn't expose these
    #    internal methods, so this degrades gracefully instead of crashing
    #    -- but it WILL be slow in that case, so progress is shown.
    try:
        stats_a, stats_b = {}, {}
        for name, metric in metrics.items():
            stats_a[name] = np.array(metric._extract_corpus_statistics(preds_a, [refs]))
            stats_b[name] = np.array(metric._extract_corpus_statistics(preds_b, [refs]))
        use_fast_path = True
    except AttributeError:
        print("  [WARN] sacrebleu internal stats API not found (version "
              "mismatch?) -- falling back to the slow direct-recompute "
              "path. This WILL take a while; progress is shown below.")
        use_fast_path = False

    # Observed (non-resampled) corpus scores, for reporting.
    observed = {}
    if use_fast_path:
        for name, metric in metrics.items():
            obs_a = metric._compute_score_from_stats(stats_a[name].sum(axis=0).tolist()).score
            obs_b = metric._compute_score_from_stats(stats_b[name].sum(axis=0).tolist()).score
            observed[name] = (obs_a, obs_b)
    else:
        for name, metric in metrics.items():
            observed[name] = (metric.corpus_score(preds_a, [refs]).score,
                               metric.corpus_score(preds_b, [refs]).score)

    # ── Bootstrap loop. Fast path: sum cached stats + closed-form
    #    aggregate (cheap). Slow path: re-tokenize + rescore from raw
    #    strings every iteration (expensive -- this is what used to hang).
    rng = np.random.default_rng(seed)
    a_wins = {name: 0 for name in metrics}
    boot_deltas = {name: np.empty(n_bootstrap) for name in metrics}

    iterator = range(n_bootstrap)
    if not use_fast_path:
        iterator = tqdm(iterator, desc="  Bootstrap (slow path)")

    for i in iterator:
        idx = rng.integers(0, n, n)
        if use_fast_path:
            for name, metric in metrics.items():
                agg_a = stats_a[name][idx].sum(axis=0).tolist()
                agg_b = stats_b[name][idx].sum(axis=0).tolist()
                score_a = metric._compute_score_from_stats(agg_a).score
                score_b = metric._compute_score_from_stats(agg_b).score
                boot_deltas[name][i] = score_a - score_b
                if score_a > score_b:
                    a_wins[name] += 1
        else:
            r_s  = [refs[j] for j in idx]
            pa_s = [preds_a[j] for j in idx]
            pb_s = [preds_b[j] for j in idx]
            for name, metric in metrics.items():
                score_a = metric.corpus_score(pa_s, [r_s]).score
                score_b = metric.corpus_score(pb_s, [r_s]).score
                boot_deltas[name][i] = score_a - score_b
                if score_a > score_b:
                    a_wins[name] += 1

    results = {}
    for name in metrics:
        obs_a, obs_b = observed[name]
        win_frac  = a_wins[name] / n_bootstrap
        # Koehn's statistic: confidence that the observed winner is truly
        # better = fraction of virtual test sets favouring that system.
        sig_level = max(win_frac, 1.0 - win_frac)
        deltas    = boot_deltas[name]
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        results[name] = {
            "system_a_score":       round(float(obs_a), 3),
            "system_b_score":       round(float(obs_b), 3),
            "delta":                round(float(obs_a - obs_b), 3),
            "ci_95":                [round(float(ci_lo), 3), round(float(ci_hi), 3)],
            "a_wins_fraction":      round(float(win_frac), 4),
            "significance_level":  round(float(sig_level), 4),
            "significant_at_0.95": bool(sig_level >= 0.95),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def run_all_comparisons(output_dir: str, systems, directions, n_bootstrap: int,
                         out_path: str = None, variant: str = None):
    """
    Run paired_bootstrap_significance() for EVERY unordered pair of systems
    across EVERY requested direction, and save every (pair, direction,
    metric) result row into ONE combined CSV -- so a single invocation
    of this script gives you the full comparison matrix instead of one
    pair at a time.

    Systems with a missing prediction file for a given direction are
    skipped for that direction (printed as a warning), not fatal -- so
    you don't need every experiment trained to get results for the ones
    you do have.

    `variant`: if set, resolves prediction paths via
    _pred_path_for_system_variant() (step5g exp7 files, filtered to this
    decode variant) instead of the legacy per-experiment files. "baseline"
    is dropped from the candidate/system list in this mode.
    """
    if variant is not None and systems is not None and "baseline" in systems:
        print("  [WARN] 'baseline' has no decode-time-variant predictions "
              "(step5g does not run on the zero-shot model) — dropping it "
              "from --systems for this --variant run.")
        systems = [s for s in systems if s != "baseline"]

    if systems is None:
        systems = discover_available_systems(output_dir, variant=variant)
        print(f"\n  [AUTO-DISCOVER] Found systems with prediction files: {systems}")
    if len(systems) < 2:
        print(f"  ERROR: need at least 2 systems to compare; found {systems}.")
        sys.exit(1)

    pairs = list(itertools.combinations(systems, 2))
    print(f"\n{'='*65}")
    print(f"  SIGNIFICANCE TEST — ALL COMPARISONS")
    print(f"  Systems     : {systems}")
    print(f"  Pairs       : {len(pairs)}  ({pairs})")
    print(f"  Directions  : {directions}")
    print(f"  Decode variant: {variant or 'A_baseline (legacy files, default)'}")
    print(f"  Bootstrap resamples per test: {n_bootstrap}")
    print(f"  Total tests : {len(pairs) * len(directions)} "
          f"(each producing BLEU + chrF++ rows)")
    print(f"{'='*65}")

    rows = []
    for direction in directions:
        for sys_a, sys_b in pairs:
            try:
                if variant is not None:
                    pred_a = _pred_path_for_system_variant(sys_a, output_dir)
                    pred_b = _pred_path_for_system_variant(sys_b, output_dir)
                else:
                    pred_a = _pred_path_for_system(sys_a, output_dir, direction)
                    pred_b = _pred_path_for_system(sys_b, output_dir, direction)
            except ValueError as e:
                print(f"  [SKIP] {sys_a} vs {sys_b} [{direction}] — {e}")
                continue
            if not os.path.exists(pred_a):
                print(f"  [SKIP] {sys_a} vs {sys_b} [{direction}] — "
                      f"missing pred file for {sys_a}: {pred_a}")
                continue
            if not os.path.exists(pred_b):
                print(f"  [SKIP] {sys_a} vs {sys_b} [{direction}] — "
                      f"missing pred file for {sys_b}: {pred_b}")
                continue

            print(f"\n  → {sys_a} vs {sys_b}  [{direction}] "
                  f"[{variant or 'A_baseline'}] ...")
            try:
                results = paired_bootstrap_significance(
                    pred_a, pred_b, direction, n_bootstrap=n_bootstrap,
                    variant=variant)
            except Exception as e:
                print(f"  [ERROR] {sys_a} vs {sys_b} [{direction}] failed: {e}")
                continue

            for metric, r in results.items():
                winner = sys_a if r["a_wins_fraction"] >= 0.5 else sys_b
                sig = r["significant_at_0.95"]
                print(f"    {metric:8s}: {sys_a}={r['system_a_score']}  "
                      f"{sys_b}={r['system_b_score']}  delta={r['delta']:+.3f}  "
                      f"sig_level={r['significance_level']*100:.1f}%  "
                      f"({'SIGNIFICANT' if sig else 'not significant'})")
                rows.append({
                    "system_a": sys_a, "system_b": sys_b, "direction": direction,
                    "variant": variant or "A_baseline",
                    "metric": metric,
                    "system_a_score": r["system_a_score"],
                    "system_b_score": r["system_b_score"],
                    "delta_a_minus_b": r["delta"],
                    "ci_95_low": r["ci_95"][0], "ci_95_high": r["ci_95"][1],
                    "a_wins_fraction": r["a_wins_fraction"],
                    "winner": winner,
                    "significance_level": r["significance_level"],
                    "significant_at_0.95": sig,
                })

    if not rows:
        print("\n  No comparisons could be run — check that prediction "
              "files exist for the requested systems/directions.")
        return

    result_df = pd.DataFrame(rows)
    if out_path is None:
        suffix = f"_{variant}" if variant else ""
        out_path = os.path.join(
            output_dir, f"significance_all_comparisons{suffix}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\n{'='*65}")
    print(f"  ✓ ALL {len(result_df)} comparison rows saved → {out_path}")
    print(f"{'='*65}")
    print(result_df.to_string(index=False))


def main():
    args = parse_args()
    os.environ["MT_LANG"] = args.lang
    from config import SCORES_FILE, OUTPUT_DIR, FIGURES_DIR, SYSTEM_COLORS  # noqa: E402

    if not os.path.exists(SCORES_FILE):
        print(f"  ERROR: {SCORES_FILE} not found. Run the pipeline first.")
        sys.exit(1)
    with open(SCORES_FILE) as f:
        scores = json.load(f)

    if args.experiments:
        print(f"\n{'='*65}")
        print(f"  MULTI-SEED AGGREGATION  |  lang={args.lang}  dir={args.direction}")
        print(f"{'='*65}")
        df = aggregate_seeds(scores, args.experiments, args.direction)
        if not df.empty:
            print(df.to_string(index=False))
            out_csv = os.path.join(OUTPUT_DIR, f"seed_aggregate_{args.direction}.csv")
            df.to_csv(out_csv, index=False)
            print(f"\n  ✓ Table saved → {out_csv}")
            plot_aggregate(df, args.direction, FIGURES_DIR, SYSTEM_COLORS)
        else:
            print("  No matching scores found. Did run_seed_sweep.py finish?")

    if args.compare and not args.compare_all:
        sys_a, sys_b = args.compare
        if args.variant is not None and "baseline" in (sys_a, sys_b):
            print("  ERROR: 'baseline' has no decode-time-variant predictions "
                  "(step5g does not run on the zero-shot model). Drop "
                  "--variant to compare against baseline's greedy "
                  "predictions instead.")
            sys.exit(1)
        print(f"\n{'='*65}")
        print(f"  SIGNIFICANCE TEST  |  {sys_a}  vs  {sys_b}  [{args.direction}]")
        print(f"  Decode variant: {args.variant or 'A_baseline (legacy files, default)'}")
        print(f"  Bootstrap resamples: {args.n_bootstrap}")
        print(f"{'='*65}")
        if args.variant is not None:
            pred_a = _pred_path_for_system_variant(sys_a, OUTPUT_DIR)
            pred_b = _pred_path_for_system_variant(sys_b, OUTPUT_DIR)
        else:
            pred_a = _pred_path_for_system(sys_a, OUTPUT_DIR, args.direction)
            pred_b = _pred_path_for_system(sys_b, OUTPUT_DIR, args.direction)
        for p in (pred_a, pred_b):
            if not os.path.exists(p):
                print(f"  ERROR: prediction file not found: {p}")
                if args.variant is not None:
                    print(f"  Have you run step5g_hallucination_mitigation.py "
                          f"--lang {args.lang} for this checkpoint yet?")
                sys.exit(1)
        results = paired_bootstrap_significance(
            pred_a, pred_b, args.direction, n_bootstrap=args.n_bootstrap,
            variant=args.variant)
        for metric, r in results.items():
            sig = "SIGNIFICANT" if r["significant_at_0.95"] else "not significant"
            print(f"\n  {metric}  (corpus-level, Koehn 2004 paired bootstrap):")
            print(f"    {sys_a} = {r['system_a_score']}   "
                  f"{sys_b} = {r['system_b_score']}")
            print(f"    delta = {r['delta']:+.3f}   95% CI of delta = {r['ci_95']}")
            print(f"    {sys_a} wins {r['a_wins_fraction']*100:.1f}% of "
                  f"{args.n_bootstrap} virtual test sets   "
                  f"→ significance level = {r['significance_level']*100:.1f}%  "
                  f"({sig} at the 95% level)")

    if args.compare_all:
        run_all_comparisons(OUTPUT_DIR, args.systems, args.directions,
                            args.n_bootstrap, args.out, variant=args.variant)

    if not args.experiments and not args.compare and not args.compare_all:
        print("  Nothing to do — pass --experiments, --compare, or "
              "--compare-all. See --help.")


if __name__ == "__main__":
    main()
