"""
step9_significance.py — Multi-Seed Aggregation + Significance Testing
===================================================================


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
    """
    if system == "baseline":
        return os.path.join(output_dir, f"preds_baseline_{direction}.csv")
    if system == "r1":
        return os.path.join(output_dir, f"preds_r1_{direction}.csv")
    if system.startswith("exp"):
        return os.path.join(output_dir, f"preds_{system}.csv")
    raise ValueError(f"Unrecognised system name: {system}")


def discover_available_systems(output_dir: str, candidates=None):
    """
    Return the subset of `candidates` (default: baseline, r1, exp1..exp5)
    that actually have a usable prediction file on disk for AT LEAST ONE
    direction. Used by --compare-all when --systems is not given, so the
    combined run doesn't hard-fail on experiments you never trained.
    """
    if candidates is None:
        candidates = ["baseline", "r1", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
    available = []
    for sys_name in candidates:
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
                                   seed: int = 16):
    from sacrebleu.metrics import BLEU, CHRF

    df_a = pd.read_csv(pred_file_a)
    df_b = pd.read_csv(pred_file_b)

    if "direction" in df_a.columns:
        df_a = df_a[df_a["direction"] == direction].reset_index(drop=True)
    if "direction" in df_b.columns:
        df_b = df_b[df_b["direction"] == direction].reset_index(drop=True)

    if len(df_a) != len(df_b):
        raise ValueError(
            f"Prediction files have different lengths for direction="
            f"{direction} ({len(df_a)} vs {len(df_b)}). Cannot pair them "
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
                         out_path: str = None):

    if systems is None:
        systems = discover_available_systems(output_dir)
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
    print(f"  Bootstrap resamples per test: {n_bootstrap}")
    print(f"  Total tests : {len(pairs) * len(directions)} "
          f"(each producing BLEU + chrF++ rows)")
    print(f"{'='*65}")

    rows = []
    for direction in directions:
        for sys_a, sys_b in pairs:
            pred_a = _pred_path_for_system(sys_a, output_dir, direction)
            pred_b = _pred_path_for_system(sys_b, output_dir, direction)
            if not os.path.exists(pred_a):
                print(f"  [SKIP] {sys_a} vs {sys_b} [{direction}] — "
                      f"missing pred file for {sys_a}: {pred_a}")
                continue
            if not os.path.exists(pred_b):
                print(f"  [SKIP] {sys_a} vs {sys_b} [{direction}] — "
                      f"missing pred file for {sys_b}: {pred_b}")
                continue

            print(f"\n  → {sys_a} vs {sys_b}  [{direction}] ...")
            try:
                results = paired_bootstrap_significance(
                    pred_a, pred_b, direction, n_bootstrap=n_bootstrap)
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
        out_path = os.path.join(output_dir, "significance_all_comparisons.csv")
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
        print(f"\n{'='*65}")
        print(f"  SIGNIFICANCE TEST  |  {sys_a}  vs  {sys_b}  [{args.direction}]")
        print(f"  Bootstrap resamples: {args.n_bootstrap}")
        print(f"{'='*65}")
        pred_a = _pred_path_for_system(sys_a, OUTPUT_DIR, args.direction)
        pred_b = _pred_path_for_system(sys_b, OUTPUT_DIR, args.direction)
        for p in (pred_a, pred_b):
            if not os.path.exists(p):
                print(f"  ERROR: prediction file not found: {p}")
                sys.exit(1)
        results = paired_bootstrap_significance(
            pred_a, pred_b, args.direction, n_bootstrap=args.n_bootstrap)
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
                            args.n_bootstrap, args.out)

    if not args.experiments and not args.compare and not args.compare_all:
        print("  Nothing to do — pass --experiments, --compare, or "
              "--compare-all. See --help.")


if __name__ == "__main__":
    main()
