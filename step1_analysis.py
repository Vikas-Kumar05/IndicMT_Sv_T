"""
step1_analysis.py — Dataset Analysis, Cleaning & Bidirectional Preparation
         | Low-Resource Machine Translation | Eng <-> Indic (mni/as/brx)
--------------------------------------------------------------
"""

import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")
import nltk
for r in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(r, quiet=True)

import unicodedata


def normalize_unicode(s):
    """
    Normalize to Unicode NFC (canonical composed form).

    """
    if pd.isna(s):
        return s
    return unicodedata.normalize("NFC", str(s))

from config import (
    RAW_TRAIN_PATH, RAW_DEV_PATH, RAW_TEST_PATH,
    OUTPUT_DIR, FIGURES_DIR,
    TRAIN_PATH, DEV_PATH, TEST_PATH,
    SEED, MT_LANG, LANG_NAME, SHORT_LANG_NAME, TARGET_COL_ALIASES,
)

random.seed(SEED); np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

PAL = {"eng": "#4C72B0", "tgt": "#DD8452", "clean": "#55A868",
       "raw": "#C44E52", "train": "#4C72B0", "dev": "#55A868", "test": "#8172B2"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(name):
    p = os.path.join(FIGURES_DIR, name)
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  ✓ {name}")

def tok_len(s):  return s.dropna().astype(str).apply(lambda x: len(x.split()))
def char_len(s): return s.dropna().astype(str).apply(len)

def vocab_stats(series, name):
    tokens = [t for sent in series.dropna().astype(str) for t in sent.split()]
    vocab  = set(tokens)
    return {"name": name, "total": len(tokens), "vocab": len(vocab),
            "ttr": round(len(vocab) / max(len(tokens), 1), 4),
            "top20": Counter(tokens).most_common(20)}


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_split(path, split_name):
    print(f"\n  Loading {split_name} from: {path}")
    # Accept both .xlsx and .csv
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names: accept common variants.
    # Target-side aliases come from config.TARGET_COL_ALIASES, which is
    # selected automatically based on MT_LANG (mni / as / brx).
    rename = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("english", "eng", "english_src", "en") and "eng_src" not in df.columns:
            rename[col] = "eng_src"
        if cl in TARGET_COL_ALIASES and "meti_tgt" not in df.columns:
            rename[col] = "meti_tgt"
    if rename:
        df = df.rename(columns=rename)
        print(f"  Renamed columns: {rename}")

    if "eng_src" not in df.columns or "meti_tgt" not in df.columns:
        raise ValueError(
            f"  ERROR: {split_name} must have an English column and a "
            f"{LANG_NAME} column (accepted target aliases for MT_LANG="
            f"'{MT_LANG}': {TARGET_COL_ALIASES}).\n"
            f"  Found: {df.columns.tolist()}")

    # Normalize BOTH columns to Unicode NFC before anything else touches
    # them -- see normalize_unicode() docstring for why this matters with
    # 3 different scripts in play.
    df["eng_src"]  = df["eng_src"].apply(normalize_unicode)
    df["meti_tgt"] = df["meti_tgt"].apply(normalize_unicode)

    print(f"  {split_name:6s} shape : {df.shape}")
    print(f"  Columns : {df.columns.tolist()}")
    print(f"  Sample  :\n{df[['eng_src','meti_tgt']].head(3).to_string(index=False)}")
    return df


# ── 2. Corpus Analysis ────────────────────────────────────────────────────────

def analyse(df, tag):
    print(f"\n{'='*60}")
    print(f"  CORPUS ANALYSIS [{tag.upper()}]")
    print(f"{'='*60}")

    print(f"  Pairs           : {len(df):,}")
    print(f"  Missing English : {df['eng_src'].isna().sum()}")
    print(f"  Missing {SHORT_LANG_NAME:12s}: {df['meti_tgt'].isna().sum()}")

    eng_l = tok_len(df["eng_src"])
    mni_l = tok_len(df["meti_tgt"])
    eng_v = vocab_stats(df["eng_src"],  "English")
    mni_v = vocab_stats(df["meti_tgt"], SHORT_LANG_NAME)

    rows = []
    for lens, name in [(eng_l, "English (src)"), (mni_l, f"{SHORT_LANG_NAME} (tgt)")]:
        rows.append({"Language": name, "Sentences": len(lens),
                     "Total Tokens": int(lens.sum()),
                     "Avg": round(lens.mean(), 2), "Median": int(lens.median()),
                     "Min": int(lens.min()), "Max": int(lens.max()),
                     "Std": round(lens.std(), 2)})
    stats_df = pd.DataFrame(rows).set_index("Language")
    print(f"\n{stats_df.to_string()}")
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f"corpus_statistics_{tag}.csv"))

    for v in [eng_v, mni_v]:
        print(f"  {v['name']:20s}: tokens={v['total']:,}  "
              f"vocab={v['vocab']:,}  TTR={v['ttr']:.4f}")
    for pct in [90, 95, 99]:
        print(f"  {pct}% coverage  Eng ≤{int(np.percentile(eng_l, pct))} tok"
              f"  |  {SHORT_LANG_NAME} ≤{int(np.percentile(mni_l, pct))} tok")

    px = f"{tag}_"

    # Length histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (lens, label, col) in zip(axes, [
            (eng_l, "English Source",                    PAL["eng"]),
            (mni_l, f"{SHORT_LANG_NAME} Target",       PAL["tgt"])]):
        ax.hist(lens, bins=50, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(lens.mean(),   color="red",   linestyle="--", lw=1.5,
                   label=f"Mean={lens.mean():.1f}")
        ax.axvline(lens.median(), color="green", linestyle=":",  lw=1.5,
                   label=f"Median={lens.median():.0f}")
        ax.set_title(f"Token Length — {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("# Tokens"); ax.set_ylabel("Frequency"); ax.legend(fontsize=9)
    plt.suptitle(f"Sentence-Length Distribution [{tag}]", fontsize=13, y=1.02)
    plt.tight_layout(); savefig(f"{px}01_length_dist.png")

    # Length ratio
    mask   = df["eng_src"].notna() & df["meti_tgt"].notna()
    ratios = (df.loc[mask, "meti_tgt"].astype(str).apply(lambda x: len(x.split())) /
              df.loc[mask, "eng_src"].astype(str).apply(lambda x: max(1, len(x.split()))))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ratios, bins=60, color=PAL["tgt"], edgecolor="white", alpha=0.85)
    ax.axvline(ratios.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Mean={ratios.mean():.2f}")
    ax.set_title(f"{SHORT_LANG_NAME}/English Token Length Ratio",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Length Ratio"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout(); savefig(f"{px}02_length_ratio.png")

    # Cumulative coverage
    fig, ax = plt.subplots(figsize=(10, 5))
    for lens, label, col in [(eng_l, "English",         PAL["eng"]),
                              (mni_l, SHORT_LANG_NAME, PAL["tgt"])]:
        sl  = np.sort(lens); cum = np.arange(1, len(sl)+1) / len(sl) * 100
        ax.plot(sl, cum, label=label, color=col, linewidth=2)
    for pct in [90, 95, 99]:
        ax.axhline(pct, color="gray", linestyle=":", lw=0.8, alpha=0.7)
    ax.set_title("Cumulative Coverage by Token Length", fontsize=13, fontweight="bold")
    ax.set_xlabel("Max Sequence Length"); ax.set_ylabel("% Covered")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(); ax.set_xlim(0); plt.tight_layout(); savefig(f"{px}03_coverage.png")

    # TTR bar
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["English", SHORT_LANG_NAME], [eng_v["ttr"], mni_v["ttr"]],
                  color=[PAL["eng"], PAL["tgt"]], edgecolor="white", width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
    ax.set_title("Type-Token Ratio (TTR)", fontsize=13, fontweight="bold")
    ax.set_ylabel("TTR"); ax.set_ylim(0, max(eng_v["ttr"], mni_v["ttr"]) * 1.2)
    plt.tight_layout(); savefig(f"{px}04_ttr.png")

    # Vocab growth
    fig, ax = plt.subplots(figsize=(10, 5))
    for series, label, col in [(df["eng_src"], "English", PAL["eng"]),
                                (df["meti_tgt"], SHORT_LANG_NAME, PAL["tgt"])]:
        rv = set(); growth = []
        for sent in series.dropna().astype(str).sample(frac=1, random_state=SEED):
            rv.update(sent.split()); growth.append(len(rv))
        ax.plot(range(1, len(growth)+1), growth, label=label, color=col, linewidth=2)
    ax.set_title("Vocabulary Growth Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("# Sentences"); ax.set_ylabel("Cumulative Vocab"); ax.legend()
    plt.tight_layout(); savefig(f"{px}05_vocab_growth.png")

    return eng_v, mni_v


# ── 3. Cleaning ───────────────────────────────────────────────────────────────

def clean(df_raw, split_name):
    print(f"\n{'='*60}")
    print(f"  CLEANING & DEDUPLICATION  [{split_name}]")
    print(f"{'='*60}")
    df = df_raw[["eng_src", "meti_tgt"]].copy()   # keep only the two text columns
    n0 = len(df); print(f"  Before    : {n0:,}")

    df = df.dropna(subset=["eng_src", "meti_tgt"])
    print(f"  Drop NaN  : {len(df):,}  (−{n0-len(df)})")

    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    b = len(df)
    df = df[(df["eng_src"] != "") & (df["meti_tgt"] != "") &
            (df["eng_src"] != "nan") & (df["meti_tgt"] != "nan")]
    print(f"  Drop empty: {len(df):,}  (−{b-len(df)})")

    b = len(df); same = (df["eng_src"] == df["meti_tgt"])
    df = df[~same]; print(f"  Drop copy : {len(df):,}  (−{b-len(df)})")

    b = len(df)
    df = df.drop_duplicates(subset=["eng_src", "meti_tgt"], keep="first")
    print(f"  Dedup     : {len(df):,}  (−{b-len(df)})")

    eng_tok = df["eng_src"].astype(str).apply(lambda x: len(x.split()))
    mni_tok = df["meti_tgt"].astype(str).apply(lambda x: len(x.split()))
    mask    = (eng_tok >= 2) & (eng_tok <= 200) & (mni_tok >= 1) & (mni_tok <= 200)
    b = len(df); df = df[mask]
    print(f"  Length flt: {len(df):,}  (−{b-len(df)})  [2≤tok≤200 both sides]")

    df = df.reset_index(drop=True)
    print(f"  ✓ Clean [{split_name}]: {len(df):,} pairs")
    return df


# ── 4. Build Bidirectional ────────────────────────────────────────────────────

def make_bidir(df, split_name):
    """
    Produce fwd + rev rows using the same two column names.
    Rev rows have values SWAPPED:
        eng_src  ← original meti_tgt  (Meitei text = source)
        meti_tgt ← original eng_src   (English text = target)
    direction column added to every row.
    """
    print(f"\n  Building bidirectional rows [{split_name}] ...")
    df = df.reset_index(drop=True)

    fwd = df[["eng_src", "meti_tgt"]].copy()
    fwd["direction"] = "eng2tgt"

    rev = df[["eng_src", "meti_tgt"]].copy()
    rev["eng_src"]   = df["meti_tgt"].values   # Meitei → source column
    rev["meti_tgt"]  = df["eng_src"].values    # English → target column
    rev["direction"] = "tgt2eng"

    combined = pd.concat([fwd, rev], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  {split_name:6s} bidir : {len(combined):,}  "
          f"(eng2tgt={(combined['direction']=='eng2tgt').sum():,}  "
          f"tgt2eng={(combined['direction']=='tgt2eng').sum():,})")
    return combined


# ── 5. Cross-split overlap check ──────────────────────────────────────────────

def check_overlap(train_clean, dev_clean, test_clean):
    print(f"\n{'='*60}")
    print(f"  CROSS-SPLIT OVERLAP CHECK")
    print(f"{'='*60}")
    train_set = set(zip(train_clean["eng_src"], train_clean["meti_tgt"]))
    dev_set   = set(zip(dev_clean["eng_src"],   dev_clean["meti_tgt"]))
    test_set  = set(zip(test_clean["eng_src"],  test_clean["meti_tgt"]))

    td = len(train_set & dev_set)
    tt = len(train_set & test_set)
    dt = len(dev_set   & test_set)
    print(f"  Train ∩ Dev  : {td}")
    print(f"  Train ∩ Test : {tt}")
    print(f"  Dev   ∩ Test : {dt}")
    if td + tt + dt == 0:
        print("  ✓ No overlap between any pair of splits.")
    else:
        print("  ⚠ WARNING: overlapping pairs found! Review source data.")


# ── Summary plots ─────────────────────────────────────────────────────────────

def plot_split_summary(train_clean, dev_clean, test_clean,
                       train_bi, dev_bi, test_bi):
    # Split sizes bar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original (unidirectional) counts
    ax = axes[0]
    names  = ["Train", "Dev", "Test"]
    sizes  = [len(train_clean), len(dev_clean), len(test_clean)]
    colors = [PAL["train"], PAL["dev"], PAL["test"]]
    bars   = ax.bar(names, sizes, color=colors, edgecolor="white", width=0.45)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Original Pairs per Split\n(Eng→Target, before bidirectional expansion)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sentence Pairs")

    # Bidirectional counts
    ax = axes[1]
    bi_sizes = [len(train_bi), len(dev_bi), len(test_bi)]
    bars = ax.bar(names, bi_sizes, color=colors, edgecolor="white", width=0.45)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Bidirectional Pairs per Split\n(Eng→Target + Target→Eng)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sentence Pairs")

    plt.tight_layout(); savefig("split_distribution.png")

    # Direction breakdown grouped bar
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3); w = 0.35
    for i, (direction, col) in enumerate([("eng2tgt", PAL["eng"]),
                                           ("tgt2eng", PAL["tgt"])]):
        counts = [df[df["direction"] == direction].shape[0]
                  for df in [train_bi, dev_bi, test_bi]]
        bars = ax.bar(x + i*w, counts, w, label=direction,
                      color=col, edgecolor="white")
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=9)
    ax.set_xticks(x + w/2); ax.set_xticklabels(["Train", "Dev", "Test"])
    ax.set_title("Bidirectional Pairs per Split", fontsize=13, fontweight="bold")
    ax.set_ylabel("# Pairs"); ax.legend()
    plt.tight_layout(); savefig("split_bidir_breakdown.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  STEP 1 — DATASET ANALYSIS, CLEANING & BIDIRECTIONAL PREP")
    print(f"  Language : {MT_LANG}  ({LANG_NAME})")
    print("  (No splitting — using pre-split train / dev / test files)")
    print("="*60)

    # ── Load raw splits ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1a — LOAD DATASETS")
    print("="*60)
    train_raw = load_split(RAW_TRAIN_PATH, "train")
    dev_raw   = load_split(RAW_DEV_PATH,   "dev")
    test_raw  = load_split(RAW_TEST_PATH,  "test")

    # ── Analyse raw ───────────────────────────────────────────────────────────
    analyse(train_raw, tag="train_raw")
    analyse(dev_raw,   tag="dev_raw")
    analyse(test_raw,  tag="test_raw")

    # ── Clean ─────────────────────────────────────────────────────────────────
    train_clean = clean(train_raw, "train")
    dev_clean   = clean(dev_raw,   "dev")
    test_clean  = clean(test_raw,  "test")

    # Save cleaned (unidirectional) for reference
    train_clean.to_excel(os.path.join(OUTPUT_DIR, "eng_mni_train_clean.xlsx"), index=False)
    dev_clean.to_excel(  os.path.join(OUTPUT_DIR, "eng_mni_dev_clean.xlsx"),   index=False)
    test_clean.to_excel( os.path.join(OUTPUT_DIR, "eng_mni_test_clean.xlsx"),  index=False)

    # ── Analyse clean ─────────────────────────────────────────────────────────
    analyse(train_clean, tag="train_clean")
    analyse(dev_clean,   tag="dev_clean")
    analyse(test_clean,  tag="test_clean")

    # ── Cross-split overlap check ─────────────────────────────────────────────
    check_overlap(train_clean, dev_clean, test_clean)

    # ── Build bidirectional splits ────────────────────────────────────────────
    train_bi = make_bidir(train_clean, "train")
    dev_bi   = make_bidir(dev_clean,   "dev")
    test_bi  = make_bidir(test_clean,  "test")

    # ── Save bidirectional splits ─────────────────────────────────────────────
    train_bi.to_excel(TRAIN_PATH, index=False)
    dev_bi.to_excel(  DEV_PATH,   index=False)
    test_bi.to_excel( TEST_PATH,  index=False)
    print(f"\n  ✓ Saved bidirectional train → {TRAIN_PATH}")
    print(f"  ✓ Saved bidirectional dev   → {DEV_PATH}")
    print(f"  ✓ Saved bidirectional test  → {TEST_PATH}")

    # ── Summary plots ─────────────────────────────────────────────────────────
    plot_split_summary(train_clean, dev_clean, test_clean,
                       train_bi, dev_bi, test_bi)

    print("\n" + "="*60)
    print("  STEP 1 COMPLETE")
    print(f"  Train : {len(train_clean):,} pairs → bidir {len(train_bi):,}")
    print(f"  Dev   : {len(dev_clean):,} pairs → bidir {len(dev_bi):,}")
    print(f"  Test  : {len(test_clean):,} pairs → bidir {len(test_bi):,}")
    print(f"  Figures → {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
