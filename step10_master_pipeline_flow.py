"""
step10_master_pipeline_flow.py — Master Cross-Language Pipeline Flow
=============================================================================
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



LANG_META = {
    "mni": {"short": "Manipuri", "color": "#4C72B0"},
    "as":  {"short": "Assamese", "color": "#DD8452"},
    "brx": {"short": "Bodo",     "color": "#55A868"},
}

DIRECTIONS = [
    ("eng2tgt", "English → Target"),
    ("tgt2eng", "Target → English"),
]

# Pipeline stages in flow order. key_prefix must match the score_key prefix
# used when each step saved its scores (see docstring above).
STAGES = [
    ("baseline", "Zero-shot"),
    ("r1",       "R1\n(fine-tuned)"),
    ("exp1",     "Exp-1\n+BT"),
    ("exp2",     "Exp-2\n+BT+FT"),
    ("exp3",     "Exp-3\n+BT+FT+Iter"),
    ("exp5",     "Exp-5\n+BT+FT+SubReg"),
]

# Exp-7 decode-time variants, appended after whichever checkpoint Exp-7 was
# actually run against (usually exp6 -- see detect_exp7_checkpoint()).
EXP7_VARIANTS = [
    ("A_baseline",          "Exp-7\n(greedy)"),
    ("B_rep_penalty",       "Exp-7\n+RepPenalty"),
    ("C_labse_rerank_beam", "Exp-7\n+LaBSErerank"),
]

ALL_METRICS = ["BLEU", "chrF", "chrF++", "COMET"]


# ══════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=list(LANG_META.keys()),
                   choices=list(LANG_META.keys()),
                   help="Which languages to include. Default: all three.")
    p.add_argument("--metric", default="chrF++",
                   choices=ALL_METRICS + ["all"],
                   help="Metric to plot, or 'all' to produce one figure "
                        "per metric (BLEU, chrF, chrF++, COMET).")
    p.add_argument("--exp7-variants", nargs="+",
                   default=[v for v, _ in EXP7_VARIANTS],
                   choices=[v for v, _ in EXP7_VARIANTS],
                   help="Which Exp-7 decode-time variants to append to the "
                        "flow. Default: all three (greedy/rep-penalty/"
                        "LaBSE-rerank).")
    p.add_argument("--results-root", default="./outputs",
                   help="Root folder to search for <lang>/all_scores.json or "
                        "<lang>_all_scores.json.")
    p.add_argument("--scores-file", nargs="+", default=[],
                   help="Explicit overrides, e.g. --scores-file "
                        "mni=./mni_all_scores.json as=./as_all_scores.json. "
                        "Takes priority over --results-root auto-discovery.")
    p.add_argument("--out-dir", default="./outputs/_master_comparison",
                   help="Where to write the combined figures/CSV.")
    p.add_argument("--chart-type", default="bar_per_lang",
                   choices=["bar_per_lang", "bar_combined", "line_combined", "both"],
                   help="'bar_per_lang' (default) = one bar chart PER "
                        "language PER direction, every stage (baseline/r1/"
                        "exp1-6/exp7-variants) as its own bar in that single "
                        "plot -- matches the original Exp-7 checkpoint bar "
                        "chart style, extended to the whole pipeline. "
                        "'bar_combined' = one bar chart per metric with all "
                        "languages grouped together. 'line_combined' = the "
                        "cross-language flow line chart. 'both' = every view.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_scores(lang, results_root, explicit_path=None):
    """Tries several common layouts so this doesn't break just because your
    results aren't sitting under outputs/<lang>/all_scores.json:
        1) an explicit --scores-file override for this language
        2) <results_root>/<lang>/all_scores.json      (original pipeline layout)
        3) <results_root>/<lang>_all_scores.json       (flat, prefixed)
        4) ./<lang>/all_scores.json                    (folder named just 'as'/'brx'/'mni')
        5) ./<lang>_all_scores.json                    (flat, in cwd)
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates += [
        os.path.join(results_root, lang, "all_scores.json"),
        os.path.join(results_root, f"{lang}_all_scores.json"),
        os.path.join(".", lang, "all_scores.json"),
        os.path.join(".", f"{lang}_all_scores.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                print(f"  [{lang}] loaded scores from {path}")
                return json.load(f)
    print(f"  [WARN] No all_scores.json found for '{lang}' — tried:")
    for path in candidates:
        print(f"           {path}")
    print(f"  Skipping '{lang}'. Use --scores-file {lang}=/path/to/{lang}_all_scores.json "
          f"to point at it directly.")
    return None


def detect_exp7_checkpoint(scores):
    """Which checkpoint name Exp-7 was actually decoded from for this
    language. Prefers 'exp6' (the pipeline default / BEST_ADAPTER_FOR_EXP7
    in config.py), then falls back through exp5..exp1..r1 -- whichever
    checkpoint actually has exp7_<ckpt>_... rows -- so this keeps working
    even if MT_BEST_ADAPTER was overridden or exp6 was skipped for a given
    language."""
    priority = ["exp6", "exp5", "exp4", "exp3", "exp2", "exp1", "r1"]
    found = set()
    for k in scores:
        if k.startswith("exp7_"):
            # exp7_<ckpt>_<variant...>_<direction> -- ckpt is always the
            # single token right after "exp7_".
            found.add(k.split("_")[1])
    for c in priority:
        if c in found:
            return c
    return sorted(found)[0] if found else None


def build_flow_df(scores, lang, metric, exp7_variants_wanted):
    """One row per (stage, direction) for this language, in flow order."""
    ckpt = detect_exp7_checkpoint(scores)
    rows = []
    stage_order = 0

    for key_prefix, label in STAGES:
        for dir_key, _ in DIRECTIONS:
            m = scores.get(f"{key_prefix}_{dir_key}", {})
            rows.append({
                "lang": lang, "direction": dir_key,
                "stage_order": stage_order, "stage_id": key_prefix,
                "stage_label": label, "value": m.get(metric, np.nan),
            })
        stage_order += 1

    if ckpt is not None:
        for variant, vlabel in EXP7_VARIANTS:
            if variant not in exp7_variants_wanted:
                continue
            for dir_key, _ in DIRECTIONS:
                m = scores.get(f"exp7_{ckpt}_{variant}_{dir_key}", {})
                rows.append({
                    "lang": lang, "direction": dir_key,
                    "stage_order": stage_order, "stage_id": f"exp7_{variant}",
                    "stage_label": vlabel, "value": m.get(metric, np.nan),
                })
            stage_order += 1
    else:
        print(f"  [WARN] [{lang}] No Exp-7 rows found in all_scores.json — "
              f"flow will stop at the last augmentation experiment.")

    return pd.DataFrame(rows), ckpt


# ══════════════════════════════════════════════════════════════════════════
#  Plotting — one figure per metric: 2 side-by-side panels (one per
#  direction), one line per language, gaps left wherever a stage/language
#  combo has no score yet.
# ══════════════════════════════════════════════════════════════════════════

def plot_flow(all_df, metric, out_dir):
    stage_lookup = (all_df[["stage_order", "stage_label"]]
                    .drop_duplicates().sort_values("stage_order"))
    x_ticks  = stage_lookup["stage_order"].values
    x_labels = stage_lookup["stage_label"].values

    n_stages = len(x_ticks)
    fig, axes = plt.subplots(1, len(DIRECTIONS),
                              figsize=(max(11, n_stages * 1.15) * len(DIRECTIONS), 6.5),
                              sharey=True)
    if len(DIRECTIONS) == 1:
        axes = [axes]

    for ax, (dir_key, dir_title) in zip(axes, DIRECTIONS):
        d = all_df[all_df["direction"] == dir_key]
        for lang in d["lang"].unique():
            ld = d[d["lang"] == lang].sort_values("stage_order")
            color = LANG_META.get(lang, {}).get("color", "#333333")
            label = LANG_META.get(lang, {}).get("short", lang)
            # Plot with NaNs as gaps (matplotlib skips them automatically
            # when connecting points -- no interpolation across missing runs).
            ax.plot(ld["stage_order"], ld["value"], marker="o", lw=2,
                    ms=6, color=color, label=label)
            for _, r in ld.iterrows():
                if pd.notna(r["value"]):
                    ax.annotate(f"{r['value']:.1f}",
                                (r["stage_order"], r["value"]),
                                textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=7.5, color=color)
        ax.axvline(7.5, color="#999999", ls="--", lw=1, alpha=0.6)  # aug -> exp7 boundary
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8.5, rotation=20, ha="right")
        ax.set_title(dir_title, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10, title="Language")

    plt.suptitle(
        f"Full Pipeline Flow — {metric}\n"
        f"Zero-shot → R1 → Augmentation (Exp-1..6) → Exp-7 Decode-Time Mitigation",
        fontsize=14, y=1.04)
    plt.tight_layout()

    safe_metric = metric.replace("+", "p").replace(" ", "_")
    out_path = os.path.join(out_dir, f"master_pipeline_flow_{safe_metric}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  Per-stage bar colors — baseline/r1/exp1-6 reuse the same palette as the
#  rest of the pipeline (config.SYSTEM_COLORS); the three Exp-7 decode
#  variants reuse the Greedy/Rep-penalty/LaBSE-rerank colors from the
#  original single-checkpoint bar chart so the two chart families read as
#  the same visual language.
# ══════════════════════════════════════════════════════════════════════════

STAGE_COLORS = {
    "baseline": "#C44E52",
    "r1":       "#4C72B0",
    "exp1":     "#DD8452",
    "exp2":     "#55A868",
    "exp3":     "#8172B2",
    "exp4":     "#CCB974",
    "exp5":     "#64B5CD",
    "exp6":     "#E18AC0",
    "exp7_A_baseline":          "#4C72B0",   # Greedy (no mitigation)
    "exp7_B_rep_penalty":       "#DD8452",   # Rep-penalty + no-repeat n-gram
    "exp7_C_labse_rerank_beam": "#55A868",   # Beam + LaBSE-rerank
}


# ══════════════════════════════════════════════════════════════════════════
#  Plotting — ONE bar chart per (language, direction, metric): every stage
#  (baseline, r1, exp1..exp6, exp7-greedy/rep-penalty/labse-rerank) as a
#  single bar, all in one row, with value labels on top -- exactly the
#  style of the original "Exp-7 Decode-Time Mitigation by Checkpoint" chart,
#  just extended to cover the WHOLE pipeline instead of one checkpoint's
#  3 decode variants.
# ══════════════════════════════════════════════════════════════════════════

def plot_single_bar(df, lang, dir_key, dir_title, metric, out_dir):
    d = df[(df["lang"] == lang) & (df["direction"] == dir_key)].sort_values("stage_order")
    d = d[d["value"].notna()]   # skip stages that were never run -- no empty bars
    if d.empty:
        print(f"  [WARN] [{lang}] [{dir_key}] No scores at all for {metric} — skipping plot.")
        return None

    x = np.arange(len(d))
    colors = [STAGE_COLORS.get(sid, "#888888") for sid in d["stage_id"]]

    fig, ax = plt.subplots(figsize=(max(11, len(d) * 1.15), 6))
    bars = ax.bar(x, d["value"].values, 0.6, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")

    # Divider between augmentation stages and Exp-7 decode-time variants --
    # computed from stages ACTUALLY PRESENT (post-dropna), not a fixed

    # positions left.
    n_aug_stages_present = (~d["stage_id"].str.startswith("exp7")).sum()
    if 0 < n_aug_stages_present < len(d):
        ax.axvline(n_aug_stages_present - 0.5, color="#999999", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(d["stage_label"].values, fontsize=9.5, rotation=20, ha="right")
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(0, d["value"].max() * 1.25)
    ax.grid(axis="y", alpha=0.3)

    lang_short = LANG_META.get(lang, {}).get("short", lang)
    ax.set_title(
        f"{lang_short} [{dir_title}] — Full Pipeline — {metric}\n"
        f"Baseline · R1 · Augmentation (Exp-1..6) · Exp-7 Decode-Time Mitigation",
        fontsize=12.5, fontweight="bold")
    plt.tight_layout()

    safe_metric = metric.replace("+", "p").replace(" ", "_")
    out_path = os.path.join(
        out_dir, f"master_bar_{lang}_{dir_key}_{safe_metric}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  Plotting — BAR chart, all languages combined: one figure per metric, 2
#  side-by-side panels (one per direction), every stage as its own x-axis
#  group, one bar per language within each group. Kept as an OPTIONAL view
#  (--chart-type bar_combined / both) alongside the per-language default.
# ══════════════════════════════════════════════════════════════════════════

def plot_flow_bar(all_df, metric, out_dir):
    stage_lookup = (all_df[["stage_order", "stage_label"]]
                    .drop_duplicates().sort_values("stage_order"))
    x_centers = stage_lookup["stage_order"].values
    x_labels  = stage_lookup["stage_label"].values
    n_stages  = len(x_centers)

    langs_present = [l for l in LANG_META if l in set(all_df["lang"])]
    n_langs = max(len(langs_present), 1)
    bar_w = 0.8 / n_langs

    fig, axes = plt.subplots(1, len(DIRECTIONS),
                              figsize=(max(13, n_stages * 1.6) * len(DIRECTIONS), 6.5),
                              sharey=True)
    if len(DIRECTIONS) == 1:
        axes = [axes]

    for ax, (dir_key, dir_title) in zip(axes, DIRECTIONS):
        d = all_df[all_df["direction"] == dir_key]
        for i, lang in enumerate(langs_present):
            ld = d[d["lang"] == lang].sort_values("stage_order")
            color = LANG_META.get(lang, {}).get("color", "#333333")
            label = LANG_META.get(lang, {}).get("short", lang)
            offset = (i - n_langs / 2 + 0.5) * bar_w
            vals = ld["value"].values
            bars = ax.bar(ld["stage_order"].values + offset, np.nan_to_num(vals, nan=0.0),
                          bar_w * 0.92, label=label, color=color, edgecolor="white")
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v),
                                textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=6.5, fontweight="bold",
                                color=color, rotation=90 if n_stages > 8 else 0)
        ax.axvline(7.5, color="#999999", ls="--", lw=1, alpha=0.6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels, fontsize=8.5, rotation=25, ha="right")
        ax.set_title(dir_title, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10, title="Language")
        max_val = np.nanmax(d["value"].values) if d["value"].notna().any() else 1
        ax.set_ylim(0, max_val * 1.25)

    plt.suptitle(
        f"Full Pipeline Comparison — {metric}  (all languages combined)\n"
        f"Baseline · R1 · Augmentation (Exp-1..6) · Exp-7 Decode-Time Mitigation",
        fontsize=14, y=1.05)
    plt.tight_layout()

    safe_metric = metric.replace("+", "p").replace(" ", "_")
    out_path = os.path.join(out_dir, f"master_pipeline_bar_combined_{safe_metric}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def run_for_metric(metric, args):
    print(f"\n  {'─'*60}")
    print(f"  METRIC: {metric}")
    print(f"  {'─'*60}")

    all_frames = []
    ckpts_used = {}
    explicit_map = dict(pair.split("=", 1) for pair in args.scores_file) if args.scores_file else {}
    for lang in args.langs:
        scores = load_scores(lang, args.results_root, explicit_map.get(lang))
        if scores is None:
            continue
        df, ckpt = build_flow_df(scores, lang, metric, set(args.exp7_variants))
        ckpts_used[lang] = ckpt
        all_frames.append(df)

    if not all_frames:
        print("  ERROR: no languages had a usable all_scores.json. "
              "Nothing to plot for this metric.")
        return

    all_df = pd.concat(all_frames, ignore_index=True)

    safe_metric = metric.replace("+", "p").replace(" ", "_")
    csv_path = os.path.join(args.out_dir, f"master_pipeline_flow_{safe_metric}.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    for lang, ckpt in ckpts_used.items():
        print(f"  [{lang}] Exp-7 decoded from checkpoint: {ckpt or '(none found)'}")

    if args.chart_type in ("line_combined", "both"):
        plot_flow(all_df, metric, args.out_dir)
    if args.chart_type in ("bar_combined", "both"):
        plot_flow_bar(all_df, metric, args.out_dir)
    if args.chart_type in ("bar_per_lang", "both"):
        for lang in ckpts_used:
            for dir_key, dir_title in DIRECTIONS:
                plot_single_bar(all_df, lang, dir_key, dir_title, metric, args.out_dir)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  STEP 10 — MASTER CROSS-LANGUAGE PIPELINE FLOW")
    print(f"  Languages : {args.langs}")
    print(f"  Metric(s) : {args.metric}")
    print(f"  Exp-7 variants included : {args.exp7_variants}")
    print("=" * 65)

    metrics_to_run = ALL_METRICS if args.metric == "all" else [args.metric]
    for metric in metrics_to_run:
        run_for_metric(metric, args)

    print("\n  STEP 10 COMPLETE")


if __name__ == "__main__":
    main()
