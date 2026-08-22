"""
aug_techniques.py — Augmentation Technique Generators | Eng<->Indic MT Pipeline
================================================================================
Six augmentation generators used across Exp-1 … Exp-6.

  generate_bt(train_df)                      → Exp-1 : Back-Translation
  generate_ft_aug(train_df)                  → Exp-2 : Forward Translation
  generate_noise(train_df)                   → Exp-4 : Synthetic Noise Injection
  generate_subword_reg(train_df)             → Exp-5 : Subword Regularization
  generate_eng_paraphrase(train_df)          → Exp-6 : English-side Paraphrasing
  generate_iter(train_df, iter_model_dir)    → Exp-3 : Iterative Pseudo-labelling

BUG-6 FIX:
  BT_SAMPLE_RATIO increased 0.30 → 0.60 (from config.py)
  FT_AUG_SAMPLE_RATIO increased 0.25 → 0.40
  With 16k pairs, the old 30% ratio gave only ~4800 raw BT pairs.
  After LaBSE filtering (~65% pass rate at threshold=0.75): ~3100 pairs.
  After combine_and_save cap (MAX_AUG_RATIO=1.0×16k): only 3100/32k = 10%
  augmentation — barely moves the distribution.

  New ratio: 60% × 16k = 9600 raw → ~6200 after filter.
  Combined with MAX_AUG_RATIO=1.5: up to 24k total pairs, ~38% synthetic.
  This gives meaningful augmentation signal.

All techniques:
  • Operate on the Eng→Target direction only.
  • Cache results to disk; delete the cache file to regenerate.
  • Apply LaBSE quality filtering before returning any pairs.
"""

import os
import random
import string

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet, stopwords

for _r in ["wordnet", "stopwords"]:
    nltk.download(_r, quiet=True)

from config import (
    SEED,
    BT_LABSE_THRESHOLD, BT_SAMPLE_RATIO,
    FT_AUG_SAMPLE_RATIO, AUG_SR_PROB, AUG_N_PER_SENT, QA_LABSE_THRESHOLD,
    ITER_SAMPLE_RATIO, ST_CONF_THRESHOLD,
    ADAPTER_DIR, SYNTH_CACHE, PROMPT_TEMPLATES, SHORT_LANG_NAME,
    NOISE_SAMPLE_RATIO, NOISE_WORD_DROP_PROB, NOISE_SWAP_PROB, NOISE_CHAR_PROB,
    SUBWORD_REG_SAMPLE_RATIO, SUBWORD_REG_ALPHA,
    ENG_PARA_SAMPLE_RATIO, ENG_PARA_LABSE_THRESHOLD,
    # ── Quality-selection method (chrF++ / LaBSE / both) + top-K ──────────
    QA_METHOD, QA_TOP_K_RATIO, QA_CHRF_THRESHOLD, BT_CHRF_THRESHOLD,
    SUBWORD_REG_CHRF_THRESHOLD, ENG_PARA_CHRF_THRESHOLD, ITER_CHRF_THRESHOLD,
)
from aug_utils import (
    load_ft_model, free_model, batch_translate,
    labse_quality_filter, resolve_quality_filter,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STOP_WORDS = set(stopwords.words("english"))


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _eng2tgt_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return only Eng→Target rows, drop NaN, reset index."""
    df = train_df.copy()
    if "direction" in df.columns:
        df = df[df["direction"] == "eng2tgt"]
    return df.dropna(subset=["eng_src", "meti_tgt"]).reset_index(drop=True)


def _load_or_generate(cache_key: str, train_df: pd.DataFrame,
                       generator_fn, **kwargs) -> pd.DataFrame:
    """
    Return cached DataFrame if available and valid, otherwise generate and cache.

    CACHE VALIDATION: A cached file with ≤ 5 rows is almost certainly a corrupt
    write from a previous crashed/OOM run (e.g. LaBSE keeping only 1 pair out of
    thousands means the cache was built from a bad intermediate state). Such files
    are deleted and regenerated automatically.
    """
    cache_path = SYNTH_CACHE[cache_key]
    MIN_CACHE_ROWS = 5   # fewer than this → treat as corrupt, regenerate

    if os.path.exists(cache_path):
        try:
            df = pd.read_excel(cache_path)
            if len(df) > MIN_CACHE_ROWS:
                print(f"  [CACHE HIT] {cache_key}: {len(df):,} pairs ← {cache_path}")
                return df
            else:
                print(f"  [CACHE CORRUPT] {cache_key}: only {len(df)} rows in cache "
                      f"(threshold >{MIN_CACHE_ROWS}) — deleting and regenerating …")
                os.remove(cache_path)
        except Exception as e:
            print(f"  [CACHE ERROR] {cache_key}: could not read cache ({e}) — regenerating …")
            os.remove(cache_path)

    print(f"  [CACHE MISS] {cache_key}: generating …")
    df = generator_fn(train_df, **kwargs)
    if len(df) > 0:
        df.to_excel(cache_path, index=False)
        print(f"  [CACHED] {cache_key}: {len(df):,} pairs → {cache_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Synonym utilities  (shared by Exp-2 and Exp-5 fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _get_synonyms(word: str) -> list:
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            c = lemma.name().replace("_", " ")
            if c.lower() != word.lower() and " " not in c:
                syns.add(c)
    return list(syns)


def _synonym_replace(sentence: str,
                     max_ratio: float = min(AUG_SR_PROB, 0.15)) -> str:
    words     = sentence.split()
    n_replace = max(1, int(len(words) * max_ratio))
    cands     = [i for i, w in enumerate(words)
                 if len(w) > 1 and w.lower() not in STOP_WORDS
                 and _get_synonyms(w)]
    random.shuffle(cands)
    for idx in cands[:n_replace]:
        syns = _get_synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
    return " ".join(words)


# ══════════════════════════════════════════════════════════════════════════════
#  Exp-1 : Back-Translation
#  BUG-6 FIX: BT_SAMPLE_RATIO increased 0.30→0.60 (config.py)
# ══════════════════════════════════════════════════════════════════════════════

def _run_bt(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Back-Translation with BIDIRECTIONAL output.

    Step 1 — Sample eng2tgt rows; translate Meitei → English with R1.
    Step 2 — Quality-filter the BT English with LaBSE + length ratio.
    Step 3 — For every surviving BT pair produce TWO training rows:
               • eng2tgt : (bt_english,       original_meitei)   direction="eng2tgt"
               • tgt2eng : (original_meitei,  bt_english)        direction="tgt2eng"
             The tgt2eng row uses the ORIGINAL meitei as eng_src and
             the BT english as meti_tgt, matching the column convention used
             everywhere in the pipeline (eng_src=source, meti_tgt=target,
             direction tells the prompt to use).

    WHY THIS IS IMPORTANT:
    Without this, the aug dataset has:
        ~24k eng2tgt  (original 16k + ~8k BT)
        ~16k tgt2eng  (original only — no aug signal)
    The model trains on 1.5× more eng2tgt data → biased toward eng2tgt.
    The tgt2eng direction gets no augmentation benefit and regresses.

    With bidirectional BT:
        ~24k eng2tgt  (original + bt_english→meitei pairs)
        ~24k tgt2eng  (original + meitei→bt_english pairs)
    Both directions are augmented equally; no direction bias.

    BT_SAMPLE_RATIO is 0.60 (config.py, fixed from 0.30).
    """
    df     = _eng2tgt_rows(train_df)
    n      = min(int(len(df) * BT_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED).reset_index(drop=True)

    sys_msg = PROMPT_TEMPLATES["bt_tgt2eng"]
    print(f"  [BT] Back-translating {n:,} {SHORT_LANG_NAME}→English pairs using R1 …")
    print(f"  [BT] Prompt: \"{sys_msg}\"")
    print(f"  [BT] ratio={BT_SAMPLE_RATIO}  →  output will be BIDIRECTIONAL")

    model, tok = load_ft_model(ADAPTER_DIR)
    bt_eng     = batch_translate(
        sample["meti_tgt"].astype(str).tolist(), model, tok, sys_msg)
    free_model(model)

    # ── Step 2 : Quality filters ──────────────────────────────────────────────
    # Length filter first (cheap)
    valid_mask = [bool(t.strip()) and len(t.split()) >= 3 for t in bt_eng]
    bt_v  = [bt_eng[i]                    for i in range(len(bt_eng)) if valid_mask[i]]
    mni_v = [sample["meti_tgt"].iloc[i]   for i in range(len(sample)) if valid_mask[i]]
    eng_v = [sample["eng_src"].iloc[i]    for i in range(len(sample)) if valid_mask[i]]
    print(f"  [BT] After length filter: {len(bt_v):,}/{n:,}")

    # QA_METHOD dispatcher:
    #   chrF++ anchor : bt_english vs the ORIGINAL English source (round-trip
    #                   quality check — both texts are English)
    #   LaBSE anchor  : bt_english vs the Meitei target (genuine cross-lingual
    #                   alignment check)
    mask, scores, score_label = resolve_quality_filter(
        method          = QA_METHOD,
        hyp_list        = bt_v,  ref_list  = eng_v,
        cross_src       = bt_v,  cross_tgt = mni_v,
        labse_threshold = BT_LABSE_THRESHOLD,
        chrf_threshold  = BT_CHRF_THRESHOLD,
        top_k_ratio     = QA_TOP_K_RATIO,
    )
    bt_final  = [bt_v[i]  for i in range(len(bt_v))  if mask[i]]
    mni_final = [mni_v[i] for i in range(len(mni_v)) if mask[i]]
    qa_scores = scores[mask]
    print(f"  [BT] After {score_label} filter: {len(bt_final):,}")

    # Secondary length-ratio guard (LaBSE is less reliable for low-resource Meitei)
    keep = [
        i for i in range(len(bt_final))
        if bt_final[i].strip()
        and 0.3 <= (len(bt_final[i].split()) / max(len(mni_final[i].split()), 1)) <= 3.0
    ]
    bt_final  = [bt_final[i]  for i in keep]
    mni_final = [mni_final[i] for i in keep]
    qa_scores = qa_scores[keep]
    print(f"  [BT] After length-ratio filter (0.3–3.0): {len(bt_final):,}")

    # ── Step 3 : Build BIDIRECTIONAL output ───────────────────────────────────
    # eng2tgt row: (bt_english, original_meitei)
    eng2tgt_rows = pd.DataFrame({
        "eng_src"  : bt_final,
        "meti_tgt" : mni_final,
        "aug_type" : "bt_eng2tgt",
        "qa_score" : qa_scores,
        "qa_method": score_label,
        "direction": "eng2tgt",
    })

    # tgt2eng row: (original_meitei, bt_english) — columns swapped to match
    # the pipeline convention where eng_src=source and meti_tgt=target
    tgt2eng_rows = pd.DataFrame({
        "eng_src"  : mni_final,          # source = Meitei
        "meti_tgt" : bt_final,           # target = BT English
        "aug_type" : "bt_tgt2eng",
        "qa_score" : qa_scores,
        "qa_method": score_label,
        "direction": "tgt2eng",
    })

    df_out = pd.concat([eng2tgt_rows, tgt2eng_rows], ignore_index=True)
    df_out = df_out[df_out["eng_src"].str.strip() != ""].reset_index(drop=True)

    n_e2m = (df_out["direction"] == "eng2tgt").sum()
    n_m2e = (df_out["direction"] == "tgt2eng").sum()
    print(f"  [BT] BIDIRECTIONAL output: eng2tgt={n_e2m:,}  tgt2eng={n_m2e:,}  "
          f"total={len(df_out):,}")
    return df_out


def generate_bt(train_df: pd.DataFrame) -> pd.DataFrame:
    return _load_or_generate("bt", train_df, _run_bt)


# ══════════════════════════════════════════════════════════════════════════════
#  Exp-2 : Forward Translation  (synonym-paraphrase → translate)
# ══════════════════════════════════════════════════════════════════════════════

def _run_ft_aug(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synonym-paraphrase English sources, then translate to Meitei with R1.
    Result: new (paraphrased_english, model_meitei) pairs.
    """
    df     = _eng2tgt_rows(train_df)
    n      = min(int(len(df) * FT_AUG_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED + 20).reset_index(drop=True)

    sys_msg = PROMPT_TEMPLATES["ft_eng2tgt"]
    print(f"  [FT-Aug] Forward translation on {n:,} synonym-paraphrased sources …")
    print(f"  [FT-Aug] Prompt: \"{sys_msg}\"")

    para_eng      = []
    para_gold_mni = []   # gold meti_tgt aligned 1:1 with para_eng (NOT with sample, since we drop rows)
    n_success     = 0
    n_failed      = 0
    for src, gold in tqdm(list(zip(sample["eng_src"].astype(str), sample["meti_tgt"].astype(str))),
                          desc="  Synonym paraphrase", leave=False):
        seen     = {src.strip()}
        variants = []
        for _ in range(AUG_N_PER_SENT * 6):
            aug = _synonym_replace(src)
            if aug.strip() and aug.strip() not in seen:
                variants.append(aug)
                seen.add(aug.strip())
            if variants:
                break
        if variants:
            n_success += 1
            para_eng.append(variants[0])
            para_gold_mni.append(gold)
        else:
            # FIX: no WordNet-eligible candidate word was found (all words
            # were stopwords / length<=1 / had no synonyms). Previously this
            # silently fell back to appending the UNCHANGED src, which then
            # got translated and risked producing a pair with the SAME
            # English source as an existing training row but a DIFFERENT,
            # model-generated (possibly noisier) Meitei target — since it
            # has different eng_src+meti_tgt from any original row (target
            # differs from gold), the combine_and_save dedup step could not
            # catch it as a duplicate. We now DROP these instead of faking
            # a "paraphrase" that never happened.
            n_failed += 1

    print(f"  [FT-Aug] Paraphrase success: {n_success:,}/{n:,} "
          f"({n_success/max(n,1)*100:.1f}%)  |  "
          f"skipped (no synonym candidates found): {n_failed:,}/{n:,} "
          f"({n_failed/max(n,1)*100:.1f}%)")
    if n_success == 0:
        print("  [FT-Aug] WARNING: 0 sentences were paraphrased — check "
              "AUG_SR_PROB / WordNet availability before proceeding.")
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score"])

    model, tok = load_ft_model(ADAPTER_DIR)
    mni_preds  = batch_translate(para_eng, model, tok, sys_msg)
    free_model(model)

    gold_mni = para_gold_mni   # aligned 1:1 with para_eng/mni_preds (rows with failed paraphrase already dropped)
    valid = [(p, m, g) for p, m, g in zip(para_eng, mni_preds, gold_mni)
             if p.strip() and m.strip() and len(m.split()) >= 2]
    if not valid:
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score"])

    valid_para, valid_mni, valid_gold = zip(*valid)

    # QA_METHOD dispatcher:
    #   chrF++ anchor : predicted Meitei vs the GOLD Meitei reference for
    #                   that row (both texts are Meitei — standard MT metric)
    #   LaBSE anchor  : paraphrased English vs the predicted Meitei
    #                   (genuine cross-lingual src/tgt alignment check)
    mask, scores, score_label = resolve_quality_filter(
        method          = QA_METHOD,
        hyp_list        = list(valid_mni),  ref_list  = list(valid_gold),
        cross_src       = list(valid_para), cross_tgt = list(valid_mni),
        labse_threshold = QA_LABSE_THRESHOLD,
        chrf_threshold  = QA_CHRF_THRESHOLD,
        top_k_ratio     = QA_TOP_K_RATIO,
    )
    df_out = pd.DataFrame({
        "eng_src" : [valid_para[i] for i in range(len(valid_para)) if mask[i]],
        "meti_tgt": [valid_mni[i]  for i in range(len(valid_mni))  if mask[i]],
        "aug_type": "forward_translation",
        "qa_score": scores[mask],
        "qa_method": score_label,
    }).reset_index(drop=True)
    print(f"  [FT-Aug] Final pairs after {score_label} filter: {len(df_out):,}")
    return df_out


def generate_ft_aug(train_df: pd.DataFrame) -> pd.DataFrame:
    return _load_or_generate("ft_aug", train_df, _run_ft_aug)


# ══════════════════════════════════════════════════════════════════════════════
#  Exp-4 : Synthetic Noise Injection  (English source only)
# ══════════════════════════════════════════════════════════════════════════════

def _word_drop(words: list, prob: float) -> list:
    """Drop each word independently with probability `prob`; keep ≥ 2 words."""
    kept = [w for w in words if random.random() > prob]
    return kept if len(kept) >= 2 else words[:]


def _adjacent_swap(words: list, prob: float) -> list:
    """Swap each adjacent pair (i, i+1) with probability `prob`."""
    words = words[:]
    for i in range(len(words) - 1):
        if random.random() < prob:
            words[i], words[i + 1] = words[i + 1], words[i]
    return words


def _char_typo(word: str) -> str:
    """Apply one random char-level operation: insert, delete, or substitute."""
    if len(word) < 3:
        return word
    idx = random.randint(0, len(word) - 1)
    op  = random.choice(["insert", "delete", "substitute"])
    if op == "delete":
        return word[:idx] + word[idx + 1:]
    elif op == "insert":
        return word[:idx] + random.choice(string.ascii_lowercase) + word[idx:]
    else:
        return word[:idx] + random.choice(string.ascii_lowercase) + word[idx + 1:]


def _apply_noise(sentence: str, drop_prob: float, swap_prob: float,
                 char_prob: float) -> str:
    """
    Apply exactly ONE noise operation (chosen uniformly):
      word-drop | adjacent-swap | per-word char-typo
    """
    words = sentence.split()
    if len(words) < 3:
        return sentence   # too short to noise meaningfully
    op = random.choice(["drop", "swap", "char"])
    if op == "drop":
        words = _word_drop(words, drop_prob)
    elif op == "swap":
        words = _adjacent_swap(words, swap_prob)
    else:
        words = [_char_typo(w) if random.random() < char_prob else w for w in words]
    return " ".join(words)


def _run_noise(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply synthetic noise to English sources; keep original Meitei targets.
    Primary risk guard: LaBSE post-filter.
    """
    df     = _eng2tgt_rows(train_df)
    n      = min(int(len(df) * NOISE_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED + 40).reset_index(drop=True)

    print(f"  [Noise] Applying synthetic noise to {n:,} English sources …")
    print(f"  [Noise] Ops: word-drop(p={NOISE_WORD_DROP_PROB}) | "
          f"adj-swap(p={NOISE_SWAP_PROB}) | char-typo(p={NOISE_CHAR_PROB})")

    noised_eng    = []
    original_mni  = []
    original_eng  = []
    for _, row in tqdm(sample.iterrows(), total=len(sample),
                       desc="  Noising", leave=False):
        src    = str(row["eng_src"])
        noised = _apply_noise(src, NOISE_WORD_DROP_PROB, NOISE_SWAP_PROB, NOISE_CHAR_PROB)
        if noised.strip() and noised.strip() != src.strip():
            noised_eng.append(noised)
            original_mni.append(str(row["meti_tgt"]))
            original_eng.append(src)

    print(f"  [Noise] Novel pairs: {len(noised_eng):,}/{n:,}")

    if not noised_eng:
        print("  [Noise] WARNING: no novel pairs generated — check noise params.")
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score", "direction"])

    # Guard: drop any pair where either side is empty/nan before quality-scoring
    clean_eng, clean_mni, clean_orig = [], [], []
    for e, m, o in zip(noised_eng, original_mni, original_eng):
        if str(e).strip() and str(m).strip() and str(m) != "nan" and str(e) != "nan":
            clean_eng.append(e)
            clean_mni.append(m)
            clean_orig.append(o)
    if len(clean_eng) < len(noised_eng):
        print(f"  [Noise] Dropped {len(noised_eng)-len(clean_eng):,} pairs with empty/nan text")
    noised_eng, original_mni, original_eng = clean_eng, clean_mni, clean_orig

    if not noised_eng:
        print("  [Noise] WARNING: all pairs had empty text — check training data.")
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score", "direction"])

    # QA_METHOD dispatcher:
    #   chrF++ anchor : noised English vs the ORIGINAL (un-noised) English
    #                   source — bounds how much surface-form drift the
    #                   noise introduced (both texts English)
    #   LaBSE anchor  : noised English vs the untouched Meitei target
    #                   (genuine cross-lingual alignment check)
    mask, scores, score_label = resolve_quality_filter(
        method          = QA_METHOD,
        hyp_list        = noised_eng, ref_list  = original_eng,
        cross_src       = noised_eng, cross_tgt = original_mni,
        labse_threshold = QA_LABSE_THRESHOLD,
        chrf_threshold  = QA_CHRF_THRESHOLD,
        top_k_ratio     = QA_TOP_K_RATIO,
    )

    df_out = pd.DataFrame({
        "eng_src" : [noised_eng[i]    for i in range(len(noised_eng))    if mask[i]],
        "meti_tgt": [original_mni[i]  for i in range(len(original_mni))  if mask[i]],
        "aug_type": "synthetic_noise",
        "qa_score": scores[mask],
        "qa_method": score_label,
    }).reset_index(drop=True)
    print(f"  [Noise] Final pairs after {score_label} filter: {len(df_out):,}")
    return df_out


def generate_noise(train_df: pd.DataFrame) -> pd.DataFrame:
    return _load_or_generate("noise", train_df, _run_noise)


# ══════════════════════════════════════════════════════════════════════════════
#  Exp-5 : Subword Regularization  (offline SentencePiece sampling)
# ══════════════════════════════════════════════════════════════════════════════

def _sp_resegment(texts: list, alpha: float) -> list:
    """
    Stochastic SentencePiece resegmentation. Falls back to synonym-replace
    if sentencepiece is unavailable.
    """
    try:
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download

        try:
            sp_path = hf_hub_download(
                repo_id  = "sarvamai/sarvam-translate",
                filename = "tokenizer.model",
            )
        except Exception as hub_err:
            raise FileNotFoundError(
                f"Could not fetch tokenizer.model from HF Hub: {hub_err}")

        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)

        result = []
        for text in texts:
            pieces  = sp.sample_encode(text, nbest_size=-1, alpha=alpha, out_type=str)
            decoded = sp.decode(pieces)
            result.append(decoded.strip() if decoded.strip() else text)
        return result

    except Exception as e:
        print(f"  [SubwordReg] SentencePiece unavailable ({e}). "
              f"Falling back to synonym-replace.")
        return [_synonym_replace(t) for t in texts]


def _run_subword_reg(train_df: pd.DataFrame) -> pd.DataFrame:
    df     = _eng2tgt_rows(train_df)
    n      = min(int(len(df) * SUBWORD_REG_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED + 50).reset_index(drop=True)

    print(f"  [SubwordReg] Offline SP resegmentation on {n:,} English sources …")
    print(f"  [SubwordReg] Alpha={SUBWORD_REG_ALPHA}, {SHORT_LANG_NAME} targets untouched.")

    eng_texts = sample["eng_src"].astype(str).tolist()
    mni_texts = sample["meti_tgt"].astype(str).tolist()
    reseg_eng = _sp_resegment(eng_texts, alpha=SUBWORD_REG_ALPHA)

    novel_eng, novel_mni, novel_orig_eng = [], [], []
    for orig, reseg, mni in zip(eng_texts, reseg_eng, mni_texts):
        if reseg.strip() and reseg.strip() != orig.strip():
            novel_eng.append(reseg)
            novel_mni.append(mni)
            novel_orig_eng.append(orig)

    print(f"  [SubwordReg] Novel surface forms: {len(novel_eng):,}/{n:,}")

    if not novel_eng:
        print("  [SubwordReg] WARNING: no novel forms generated.")
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score"])

    # QA_METHOD dispatcher:
    #   chrF++ anchor : resegmented English vs the ORIGINAL English surface
    #                   form — resegmentation should barely change the text,
    #                   so this floor is set high (SUBWORD_REG_CHRF_THRESHOLD)
    #   LaBSE anchor  : resegmented English vs the untouched Meitei target
    mask, scores, score_label = resolve_quality_filter(
        method          = QA_METHOD,
        hyp_list        = novel_eng, ref_list  = novel_orig_eng,
        cross_src       = novel_eng, cross_tgt = novel_mni,
        labse_threshold = QA_LABSE_THRESHOLD,
        chrf_threshold  = SUBWORD_REG_CHRF_THRESHOLD,
        top_k_ratio     = QA_TOP_K_RATIO,
    )

    df_out = pd.DataFrame({
        "eng_src" : [novel_eng[i] for i in range(len(novel_eng)) if mask[i]],
        "meti_tgt": [novel_mni[i] for i in range(len(novel_mni)) if mask[i]],
        "aug_type": "subword_reg",
        "qa_score": scores[mask],
        "qa_method": score_label,
    }).reset_index(drop=True)
    print(f"  [SubwordReg] Final pairs after {score_label} filter: {len(df_out):,}")
    return df_out


def generate_subword_reg(train_df: pd.DataFrame) -> pd.DataFrame:
    return _load_or_generate("subword_reg", train_df, _run_subword_reg)


# ══════════════════════════════════════════════════════════════════════════════
#  Exp-6 : English-side Paraphrasing  (BT-pivot: Eng → Mni → Eng)
# ══════════════════════════════════════════════════════════════════════════════

def _run_eng_paraphrase(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    BT-pivot paraphrasing:
      Step 1 — Translate English → Meitei with R1 (forward pass)
      Step 2 — Translate that Meitei → English with R1 (back pass)
      Step 3 — Pair the round-trip English with the ORIGINAL Meitei target
      Step 4 — LaBSE filter with ENG_PARA_LABSE_THRESHOLD (stricter for
               direction-asymmetry guard)
    """
    df     = _eng2tgt_rows(train_df)
    n      = min(int(len(df) * ENG_PARA_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED + 60).reset_index(drop=True)

    eng_texts = sample["eng_src"].astype(str).tolist()
    mni_refs  = sample["meti_tgt"].astype(str).tolist()

    # ── Step 1 : Eng → Mni ───────────────────────────────────────────────────
    sys_fwd = PROMPT_TEMPLATES["ft_eng2tgt"]
    print(f"  [EngPara] BT-pivot paraphrasing on {n:,} sources …")
    print(f"  [EngPara] Step 1 — Eng→Target: \"{sys_fwd}\"")
    model, tok = load_ft_model(ADAPTER_DIR)
    pivot_mni  = batch_translate(eng_texts, model, tok, sys_fwd)
    free_model(model)

    # ── Step 2 : Mni → Eng ───────────────────────────────────────────────────
    sys_back = PROMPT_TEMPLATES["bt_tgt2eng"]
    print(f"  [EngPara] Step 2 — Target→Eng: \"{sys_back}\"")
    model, tok = load_ft_model(ADAPTER_DIR)
    para_eng   = batch_translate(pivot_mni, model, tok, sys_back)
    free_model(model)

    # ── Step 3 : Novelty filter ───────────────────────────────────────────────
    valid_para, valid_mni, valid_orig = [], [], []
    for orig, para, mni in zip(eng_texts, para_eng, mni_refs):
        if (para.strip()
                and len(para.split()) >= 3
                and para.strip() != orig.strip()):
            valid_para.append(para)
            valid_mni.append(mni)
            valid_orig.append(orig)

    print(f"  [EngPara] Novel paraphrases: {len(valid_para):,}/{n:,}")

    if not valid_para:
        print("  [EngPara] WARNING: no novel paraphrases generated.")
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score"])

    # ── Step 4 : Quality filter (stricter thresholds — guards direction-
    #             asymmetry risk between the round-trip English and the
    #             ORIGINAL Meitei reference it gets paired with) ───────────
    #   chrF++ anchor : round-trip English vs the ORIGINAL English source
    #                   (both English — measures how much the BT-pivot
    #                   round-trip rewrote the sentence)
    #   LaBSE anchor  : round-trip English vs the original Meitei reference
    #                   (the direct src/tgt alignment check)
    mask, scores, score_label = resolve_quality_filter(
        method          = QA_METHOD,
        hyp_list        = valid_para, ref_list  = valid_orig,
        cross_src       = valid_para, cross_tgt = valid_mni,
        labse_threshold = ENG_PARA_LABSE_THRESHOLD,
        chrf_threshold  = ENG_PARA_CHRF_THRESHOLD,
        top_k_ratio     = QA_TOP_K_RATIO,
    )

    df_out = pd.DataFrame({
        "eng_src" : [valid_para[i] for i in range(len(valid_para)) if mask[i]],
        "meti_tgt": [valid_mni[i]  for i in range(len(valid_mni))  if mask[i]],
        "aug_type": "eng_paraphrase",
        "qa_score": scores[mask],
        "qa_method": score_label,
    }).reset_index(drop=True)
    print(f"  [EngPara] Final pairs after {score_label} filter: {len(df_out):,}")
    return df_out


def generate_eng_paraphrase(train_df: pd.DataFrame) -> pd.DataFrame:
    return _load_or_generate("eng_para", train_df, _run_eng_paraphrase)


# ══════════════════════════════════════════════════════════════════════════════
#  Exp-3 : Iterative Pseudo-labelling
# ══════════════════════════════════════════════════════════════════════════════

def _run_iter(train_df: pd.DataFrame, iter_model_dir: str) -> pd.DataFrame:
    """
    One iterative retraining cycle using the Exp-2 fine-tuned model.
    Keep only high-confidence pseudo-labels (LaBSE ≥ ST_CONF_THRESHOLD).
    """
    df     = _eng2tgt_rows(train_df)
    n      = min(int(len(df) * ITER_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED + 30).reset_index(drop=True)

    sys_msg = PROMPT_TEMPLATES["ft_eng2tgt"]
    print(f"  [Iter] Generating pseudo-labels for {n:,} sources "
          f"using {iter_model_dir} …")

    model, tok = load_ft_model(iter_model_dir)
    pseudo_mni = batch_translate(
        sample["eng_src"].astype(str).tolist(), model, tok, sys_msg)
    free_model(model)

    valid_idx = [
        i for i in range(len(pseudo_mni))
        if (pseudo_mni[i].strip()
            and len(pseudo_mni[i].split()) >= 3
            and str(sample["meti_tgt"].iloc[i]).strip()
            and pseudo_mni[i].strip() != str(sample["meti_tgt"].iloc[i]).strip())
    ]
    if not valid_idx:
        print("  [Iter] WARNING: no valid pseudo-labels generated.")
        return pd.DataFrame(columns=["eng_src", "meti_tgt", "aug_type", "qa_score"])

    v_pseudo = [pseudo_mni[i]                   for i in valid_idx]
    v_gold   = [str(sample["meti_tgt"].iloc[i]) for i in valid_idx]
    v_eng    = [str(sample["eng_src"].iloc[i])  for i in valid_idx]

    # NOTE: v_pseudo and v_gold are BOTH Meitei here (predicted vs gold
    # target), so chrF++ is a directly valid, standard MT quality metric in
    # this case — no cross-lingual mismatch concern. LaBSE also still works
    # (embedding similarity within the same language is well-defined too),
    # it's just less standard for this particular comparison than chrF++.
    mask, scores, score_label = resolve_quality_filter(
        method          = QA_METHOD,
        hyp_list        = v_pseudo, ref_list  = v_gold,
        cross_src       = v_pseudo, cross_tgt = v_gold,
        labse_threshold = ST_CONF_THRESHOLD,
        chrf_threshold  = ITER_CHRF_THRESHOLD,
        top_k_ratio     = QA_TOP_K_RATIO,
    )

    df_out = pd.DataFrame({
        "eng_src" : [v_eng[i]    for i in range(len(v_eng))    if mask[i]],
        "meti_tgt": [v_pseudo[i] for i in range(len(v_pseudo)) if mask[i]],
        "aug_type": "iter_pseudo",
        "qa_score": scores[mask],
        "qa_method": score_label,
    }).reset_index(drop=True)
    print(f"  [Iter] Pseudo-label pairs kept ({score_label} filter): "
          f"{len(df_out):,}")
    return df_out


def generate_iter(train_df: pd.DataFrame, iter_model_dir: str) -> pd.DataFrame:
    """Cache keyed to 'iter'; delete cache file to regenerate."""
    cache_path = SYNTH_CACHE["iter"]
    if os.path.exists(cache_path):
        df = pd.read_excel(cache_path)
        print(f"  [CACHE HIT] iter: {len(df):,} pairs ← {cache_path}")
        return df
    df = _run_iter(train_df, iter_model_dir)
    if len(df) > 0:
        df.to_excel(cache_path, index=False)
        print(f"  [CACHED] iter: {len(df):,} pairs → {cache_path}")
    return df
