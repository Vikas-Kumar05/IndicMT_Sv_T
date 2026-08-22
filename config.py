"""
config.py — Central Configuration | Eng<->Indic Bidirectional MT Pipeline
==========================================================================
MULTI-LANGUAGE 
------------------------------------------------------------------------

"""

import os

# ══════════════════════════════════════════════════════════════════════════
#  Language selection
# ══════════════════════════════════════════════════════════════════════════

MT_LANG = os.environ.get("MT_LANG", "mni").strip().lower()

LANGUAGES = {
    "mni": {
        "name": "Manipuri (Meitei Mayek script)",
        "short_name": "Manipuri",
        "prompt_target_name": "Manipuri in Meitei Mayek script",
        # Raw column-name variants accepted for the non-English side of
        # your raw train/dev/test files for this language.
        "target_col_aliases": [
            "meitei", "mni", "manipuri", "meitei_tgt", "mni_tgt",
        ],
    },
    "as": {
        "name": "Assamese",
        "short_name": "Assamese",
        "prompt_target_name": "Assamese",
        "target_col_aliases": [
            "assamese", "as", "asm", "as_tgt", "assamese_tgt",
        ],
    },
    "brx": {
        "name": "Bodo",
        "short_name": "Bodo",
        "prompt_target_name": "Bodo",
        "target_col_aliases": [
            "bodo", "brx", "boro", "brx_tgt", "bodo_tgt",
        ],
    },
}

if MT_LANG not in LANGUAGES:
    raise ValueError(
        f"MT_LANG='{MT_LANG}' is not configured. "
        f"Valid options: {list(LANGUAGES.keys())}. "
        f"Set MT_LANG env var or use `python main.py --lang <code>`."
    )

LANG_INFO           = LANGUAGES[MT_LANG]
LANG_NAME           = LANG_INFO["name"]
SHORT_LANG_NAME     = LANG_INFO["short_name"]
PROMPT_TARGET_NAME  = LANG_INFO["prompt_target_name"]
TARGET_COL_ALIASES  = LANG_INFO["target_col_aliases"]

# ══════════════════════════════════════════════════════════════════════════
#  Seed selection (for significance / multi-seed sweeps)
# ══════════════════════════════════════════════════════════════════════════

DATA_SEED  = 16                                   # fixed: data/augmentation RNG
MODEL_SEED = int(os.environ.get("MT_SEED", "16")) # varies across a seed sweep
SEED       = MODEL_SEED   # backward-compat alias used throughout the codebase

# The set of seeds run_seed_sweep.py / step9 use for significance testing.
SEEDS_FOR_SIGNIFICANCE = [16, 42, 123]

# When MODEL_SEED is a non-default sweep seed, fine-tune/eval artifacts get
# a "_sNN" suffix so they don't clobber the primary (default-seed) run.
_SEED_SUFFIX = "" if MODEL_SEED == 16 else f"_s{MODEL_SEED}"

# ══════════════════════════════════════════════════════════════════════════
#  Paths -- namespaced per language so the 3 languages never collide
# ══════════════════════════════════════════════════════════════════════════

# Root folder holding your raw, pre-split per-language data. Expected layout:

RAW_DATA_ROOT = os.environ.get("MT_RAW_DATA_ROOT", "./raw_data")

RAW_TRAIN_PATH = os.path.join(RAW_DATA_ROOT, MT_LANG, "train.xlsx")
RAW_DEV_PATH   = os.path.join(RAW_DATA_ROOT, MT_LANG, "dev.xlsx")
RAW_TEST_PATH  = os.path.join(RAW_DATA_ROOT, MT_LANG, "test.xlsx")

# Every language gets its own output/checkpoint tree.
OUTPUT_DIR  = os.path.join("./outputs", MT_LANG)
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
CKPT_ROOT   = os.path.join("./checkpoints", MT_LANG)

# ── Checkpoints ────────────────────────────────────────────────────────────
ADAPTER_DIR  = os.path.join(CKPT_ROOT, "sarvam_r1_lora")
EXP1_ADAPTER = os.path.join(CKPT_ROOT, f"exp1_bt{_SEED_SUFFIX}")
EXP2_ADAPTER = os.path.join(CKPT_ROOT, f"exp2_bt_ft{_SEED_SUFFIX}")
EXP3_ADAPTER = os.path.join(CKPT_ROOT, f"exp3_bt_ft_iter{_SEED_SUFFIX}")
EXP4_ADAPTER = os.path.join(CKPT_ROOT, f"exp4_noise{_SEED_SUFFIX}")
EXP5_ADAPTER = os.path.join(CKPT_ROOT, f"exp5_subword_reg{_SEED_SUFFIX}")
EXP6_ADAPTER = os.path.join(CKPT_ROOT, f"exp6_eng_para{_SEED_SUFFIX}")
# Exp-7 has no separate adapter: it reuses EXP6_ADAPTER (or BEST_ADAPTER_FOR_EXP7
# below) and only changes decoding-time behaviour.

# Which checkpoint Exp-7 (hallucination mitigation) builds on top of.
# Default: Exp-6, since that was the best-performing augmentation config in
# the original Meitei run. Override per-language after step8 tells you which
# experiment actually won for that language.
BEST_ADAPTER_FOR_EXP7 = os.environ.get("MT_BEST_ADAPTER", EXP6_ADAPTER)

# ── Augmented dataset files (shared across the seed sweep -- DATA_SEED only) ─
EXP1_DATASET = os.path.join(OUTPUT_DIR, "aug_exp1_bt.xlsx")
EXP2_DATASET = os.path.join(OUTPUT_DIR, "aug_exp2_bt_ft.xlsx")
EXP3_DATASET = os.path.join(OUTPUT_DIR, "aug_exp3_bt_ft_iter.xlsx")
EXP4_DATASET = os.path.join(OUTPUT_DIR, "aug_exp4_noise.xlsx")
EXP5_DATASET = os.path.join(OUTPUT_DIR, "aug_exp5_subword_reg.xlsx")
EXP6_DATASET = os.path.join(OUTPUT_DIR, "aug_exp6_eng_para.xlsx")

# ── Synthetic cache files ──────────────────────────────────────────────────
SYNTH_CACHE = {
    "bt"         : os.path.join(OUTPUT_DIR, "cache_synth_bt.xlsx"),
    "ft_aug"     : os.path.join(OUTPUT_DIR, "cache_synth_ft_aug.xlsx"),
    "iter"       : os.path.join(OUTPUT_DIR, "cache_synth_iter.xlsx"),
    "noise"      : os.path.join(OUTPUT_DIR, "cache_synth_noise.xlsx"),
    "subword_reg": os.path.join(OUTPUT_DIR, "cache_synth_subword_reg.xlsx"),
    "eng_para"   : os.path.join(OUTPUT_DIR, "cache_synth_eng_para.xlsx"),
}

# ── Prediction CSVs ────────────────────────────────────────────────────────
BASELINE_PREDS = {
    "eng2tgt": os.path.join(OUTPUT_DIR, "preds_baseline_eng2tgt.csv"),
    "tgt2eng": os.path.join(OUTPUT_DIR, "preds_baseline_tgt2eng.csv"),
}
R1_PREDS = {
    "eng2tgt": os.path.join(OUTPUT_DIR, "preds_r1_eng2tgt.csv"),
    "tgt2eng": os.path.join(OUTPUT_DIR, "preds_r1_tgt2eng.csv"),
}
EXP1_PREDS_FILE = os.path.join(OUTPUT_DIR, f"preds_exp1{_SEED_SUFFIX}.csv")
EXP2_PREDS_FILE = os.path.join(OUTPUT_DIR, f"preds_exp2{_SEED_SUFFIX}.csv")
EXP3_PREDS_FILE = os.path.join(OUTPUT_DIR, f"preds_exp3{_SEED_SUFFIX}.csv")
EXP4_PREDS_FILE = os.path.join(OUTPUT_DIR, f"preds_exp4{_SEED_SUFFIX}.csv")
EXP5_PREDS_FILE = os.path.join(OUTPUT_DIR, f"preds_exp5{_SEED_SUFFIX}.csv")
EXP6_PREDS_FILE = os.path.join(OUTPUT_DIR, f"preds_exp6{_SEED_SUFFIX}.csv")
EXP7_PREDS_FILE = os.path.join(OUTPUT_DIR, "preds_exp7_hallucination_mitigation.csv")

# ── Legacy aliases ─────────────────────────────────────────────────────────
BASELINE_PREDS_FILE = BASELINE_PREDS["eng2tgt"]
R1_PREDS_FILE       = R1_PREDS["eng2tgt"]

# ── Processed bidirectional split files ───────────────────────────────────
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.xlsx")
DEV_PATH   = os.path.join(OUTPUT_DIR, "dev.xlsx")
TEST_PATH  = os.path.join(OUTPUT_DIR, "test.xlsx")

# ── Global scores file (per language; seed-suffixed runs write into the
#    SAME file under seed-suffixed keys, e.g. "exp4_s42_eng2tgt") ──────────
SCORES_FILE = os.path.join(OUTPUT_DIR, "all_scores.json")

# ── Model ──────────────────────────────────────────────────────────────────

MODEL_NAME  = os.environ.get(
    "MT_MODEL_PATH", "/home/vikas1/vikas/models/sarvam-translate"
)
TARGET_LANG = PROMPT_TARGET_NAME   # kept for backward compat with old code

# ── Direction definitions ──────────────────────────────────────────────────
# Internal codes are unchanged across languages (see naming note at top).
DIRECTIONS = [
    ("eng2tgt", PROMPT_TARGET_NAME),
    ("tgt2eng", "English"),
]
PRIMARY_DIRECTION = "eng2tgt"

# Human-readable labels for each direction, correctly reflecting whichever
# language is active (MT_LANG). Use this instead of ad-hoc string surgery
# like direction.replace("2","->").replace(...) -- that approach silently
# breaks once the internal codes are language-neutral (eng2tgt/tgt2eng).
DIRECTION_LABELS = {
    "eng2tgt": f"Eng→{SHORT_LANG_NAME}",
    "tgt2eng": f"{SHORT_LANG_NAME}→Eng",
}

# ── Prompt templates ───────────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "eng2tgt"    : f"Translate the text below to {PROMPT_TARGET_NAME}.",
    "tgt2eng"    : "Translate the text below to English.",
    "bt_tgt2eng" : "Translate the text below to English.",
    "ft_eng2tgt" : f"Translate the text below to {PROMPT_TARGET_NAME}.",
}

# ── Optimizer ──────────────────────────────────────────────────────────────
OPTIMIZER = "adamw_torch_fused"

# ── LoRA target modules — hardcoded for Gemma3-4B-IT (sarvam-translate) ───
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyperparameters — R1 (original 17k data) ─────────────────────
MAX_SEQ_LEN             = 512

# ── LoRA rank / alpha / rsLoRA — PER-LANGUAGE, sized to corpus size ───────
# RESEARCH NOTE (why this is no longer one global r=64/rsLoRA=True for all 3
# languages):
#   rsLoRA (Kalajdzievski 2023, arXiv:2312.03732) fixes a real problem in
#   vanilla LoRA: the standard alpha/r scaling shrinks so fast as r grows
#   that gradients collapse and higher ranks stop improving over low ranks
#   in practice, LoRA effectively caps out around r~16-32. rsLoRA replaces
#   the scaling with alpha/sqrt(r), which keeps update variance ~constant as
#   r grows -- so bigger ranks keep paying off instead of stalling.
#   THE CATCH: that is a "make it safe to use HIGH rank" fix. At r<=16-32,
#   alpha/r and alpha/sqrt(r) don't diverge much, so rsLoRA buys little --
#   but it still makes the adapter's effective updates somewhat hotter than
#   vanilla scaling at the same alpha, which is the wrong direction on a
#   small corpus. And a high-rank adapter (more trainable params) on a small
#   dataset has more capacity to memorise the training pairs than to
#   generalise -- classic overfitting, compounded when paired with rsLoRA's
#   more aggressive scaling. Standard guidance: low rank (4-16) for small /
#   narrow data, higher rank (64+) only once the training set is large and
#   diverse enough to actually exercise that extra capacity.
#
#   Your three corpora are NOT the same size, so they should not share one
#   rank config:
#     mni (Meetei)          : ~16k pairs  -> small  -> low rank, vanilla LoRA
#     brx (Bodo)             : ~15k pairs  -> small  -> low rank, vanilla LoRA
#     as  (Assamese/Bengali) : ~54k pairs  -> ~3.5x more data & more diverse
#                               -> moderate rank, rsLoRA is defensible here
#   (Bidirectional training halves the effective per-direction signal too,
#   which is an extra reason not to over-provision capacity for mni/brx.)
LANG_LORA_CONFIG = {
    "mni": {"r": 16, "alpha": 32, "dropout": 0.10, "use_rslora": False},
    "brx": {"r": 16, "alpha": 32, "dropout": 0.10, "use_rslora": False},
    "as":  {"r": 32, "alpha": 64, "dropout": 0.10, "use_rslora": True},
}

# Allow override via env vars for quick ablation without editing this file,
# e.g.:  MT_LORA_R=8 MT_USE_RSLORA=0 python main.py --lang mni
_lora_defaults = LANG_LORA_CONFIG[MT_LANG]

LORA_R = int(os.environ.get("MT_LORA_R", _lora_defaults["r"]))
LORA_ALPHA = int(os.environ.get("MT_LORA_ALPHA", _lora_defaults["alpha"]))
LORA_DROPOUT = float(os.environ.get("MT_LORA_DROPOUT", _lora_defaults["dropout"]))

_use_rslora_env = os.environ.get("MT_USE_RSLORA")
if _use_rslora_env is not None:
    USE_RSLORA = _use_rslora_env.strip().lower() in ("1", "true", "yes")
else:
    USE_RSLORA = _lora_defaults["use_rslora"]

NUM_TRAIN_EPOCHS_R1     = 3
PER_DEVICE_TRAIN_BATCH  = 8
PER_DEVICE_EVAL_BATCH   = 8
GRAD_ACCUM_STEPS        = 4

LEARNING_RATE           = 5e-5
LR_SCHEDULER            = "cosine"
WARMUP_RATIO            = 0.06
WEIGHT_DECAY            = 0.05
MAX_GRAD_NORM           = 1.0
SAVE_TOTAL_LIMIT        = 2
LOGGING_STEPS           = 20
EARLY_STOPPING_PATIENCE = 3
LABEL_SMOOTHING         = 0.0

# ── Training — Augmented experiments ──────────────────────────────────────
NUM_TRAIN_EPOCHS_AUG         = 3
AUG_EARLY_STOPPING_PATIENCE  = 3
LEARNING_RATE_AUG            = 5e-5
AUG_LABEL_SMOOTHING          = 0.05

# ── Inference ────────────────────────────────────────────────────────────
BATCH_SIZE_INFER = 32
MAX_NEW_TOKENS   = 256

# ── Augmentation QA / mixing controls ─────────────────────────────────────
QA_LABSE_THRESHOLD  = 0.75
BT_LABSE_THRESHOLD  = 0.75
ST_CONF_THRESHOLD   = 0.85
MAX_AUG_RATIO       = 1.5
ORIG_MIN_RATIO      = 0.40

# ── Quality-selection METHOD for all augmentation techniques ──────────────
# "labse" : cross-lingual embedding similarity (LaBSE). Compares the actual
#           source/target pair in TWO DIFFERENT languages/scripts. This is
#           the only technically valid metric for checking src<->tgt
#           translation alignment across languages.
# "chrf"  : character n-gram F-score (chrF++, word_order=2, via sacrebleu).
#           Only valid when comparing two texts in the SAME language. Every
#           technique in aug_techniques.py has a natural same-language
#           anchor to score against (see comments at each _run_* function),
#           e.g. BT's back-translated English vs the ORIGINAL English source,
#           or Iter's pseudo-Meitei vs the GOLD Meitei -- so chrF++ can be
#           used everywhere as a well-scoped alternative to LaBSE. NEVER use
#           "chrf" to compare two different-language texts -- character
#           n-gram overlap across scripts is not a meaningful signal.
# "both"  : compute both signals. A pair must pass BOTH thresholds to be
#           kept; top-K ranking (below) is then done using the chrF++ score,
#           since it is a finer-grained 0-100 scale than LaBSE's 0-1 cosine
#           similarity.
# Override via env var, e.g.:  MT_QA_METHOD=chrf python main.py --lang mni
QA_METHOD = os.environ.get("MT_QA_METHOD", "chrf").strip().lower()
if QA_METHOD not in ("labse", "chrf", "both"):
    raise ValueError(
        f"MT_QA_METHOD='{QA_METHOD}' is not valid. Use one of: labse, chrf, both."
    )

# After the pass/fail threshold filter, optionally keep only the TOP-K
# fraction of surviving pairs, ranked by quality score (highest score
# first). This is the "choose the top chrF++ (or LaBSE) sentences" knob.
#   None / 1.0  -> keep every pair that passed the threshold (old behaviour)
#   0.7         -> keep only the best 70% of pairs that passed the threshold
# Override via env var, e.g.:  MT_QA_TOP_K_RATIO=0.7
QA_TOP_K_RATIO = None
_qa_topk_env = os.environ.get("MT_QA_TOP_K_RATIO", "").strip()
if _qa_topk_env:
    QA_TOP_K_RATIO = float(_qa_topk_env)

# ── chrF++ thresholds (0-100 scale) — used when QA_METHOD is "chrf"/"both" ─
# Each is calibrated to how much surface-form drift that technique is
# EXPECTED to introduce relative to its same-language anchor text (see
# table above): techniques that deliberately rewrite more aggressively
# (EngPara's BT-pivot round-trip, BT's round-trip) get a looser floor;
# techniques that should barely change the surface form (SubwordReg) or
# that are being checked against a genuine gold reference (Iter) get a
# tighter floor.
QA_CHRF_THRESHOLD       = 45.0   # generic default (FT-Aug, Noise)
BT_CHRF_THRESHOLD       = 55.0   # BT round-trip vs original English
SUBWORD_REG_CHRF_THRESHOLD = 55.0  # resegmented text should stay very close to original
ENG_PARA_CHRF_THRESHOLD = 35.0   # BT-pivot round-trip vs original English (rewrites more)
ITER_CHRF_THRESHOLD     = 30.0   # pseudo-label vs gold Meitei (standard MT chrF++ bar)

# ── Per-technique augmentation params ─────────────────────────────────────
BT_SAMPLE_RATIO     = 0.45
FT_AUG_SAMPLE_RATIO = 0.25
AUG_SR_PROB         = 0.10
AUG_N_PER_SENT      = 1
ITER_SAMPLE_RATIO   = 0.40

# ── Synthetic Noise Injection (Exp-4) ─────────────────────────────────────
NOISE_SAMPLE_RATIO   = 0.25
NOISE_WORD_DROP_PROB = 0.05
NOISE_SWAP_PROB      = 0.08
NOISE_CHAR_PROB      = 0.05

# ── Subword Regularization (Exp-5) ────────────────────────────────────────
SUBWORD_REG_SAMPLE_RATIO = 0.35
SUBWORD_REG_ALPHA        = 0.1

# ── English-side Paraphrasing / BT-pivot (Exp-6) ──────────────────────────
ENG_PARA_SAMPLE_RATIO    = 0.25
ENG_PARA_LABSE_THRESHOLD = 0.78

# ── Exp-7: Hallucination mitigation (decode-time) ─────────────────────────
# (a) Repetition-penalty / no-repeat-ngram decoding
#
# ROOT-CAUSE NOTE (mni eng2tgt collapse, all checkpoints r1..exp5):
#   A single global (repetition_penalty=1.3, no_repeat_ngram_size=3) pair
#   was applied to every language and both directions. no_repeat_ngram_size
#   is a HARD constraint -- it forbids the model from ever repeating a
#   3-gram of token IDs anywhere earlier in the sequence, regardless of
#   model confidence. Meetei Mayek (mni) has sparse coverage in the
#   sarvam-translate / Gemma3 tokenizer, so fluent mni text legitimately
#   needs to repeat the same short subword/byte fragments many times per
#   sentence (common particles/postpositions/syllables get fragmented).
#   The hard block forces the decoder onto low-probability substitute
#   tokens the moment a legitimate repeat would occur, and those errors
#   compound token-by-token into garbled output. as/brx target scripts
#   (Bengali-Assamese, Devanagari) have much better tokenizer coverage and
#   don't trigger this. tgt2eng is barely affected for any language since
#   English is the high-resource, well-covered side regardless of MT_LANG.
#   Fix: scope both knobs per (language), and effectively disable the hard
#   n-gram block for mni. Override either via env vars if you want to
#   experiment further (e.g. MT_HALLU_REP_PENALTY=1.1).
HALLU_MITIGATION_REPETITION_PENALTY_BY_LANG = {
    "mni": 1.05,   # much gentler soft penalty -- mni's fragmented tokenizer
                   # coverage makes it sensitive to aggressive penalties
    "as":  1.3,
    "brx": 1.3,
}
HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE_BY_LANG = {
    "mni": 0,      # 0 = disabled (transformers only adds the hard n-gram
                   # blocking LogitsProcessor when this is > 0); a hard
                   # block is too dangerous for mni's script coverage
    "as":  3,
    "brx": 3,
}
HALLU_MITIGATION_REPETITION_PENALTY = float(os.environ.get(
    "MT_HALLU_REP_PENALTY",
    HALLU_MITIGATION_REPETITION_PENALTY_BY_LANG.get(MT_LANG, 1.3)))
HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE = int(os.environ.get(
    "MT_HALLU_NO_REPEAT_NGRAM",
    HALLU_MITIGATION_NO_REPEAT_NGRAM_SIZE_BY_LANG.get(MT_LANG, 3)))
# (b) Length-bound decoding derived from corpus length-ratio stats (step1
#     plots meti_tgt/eng_src token ratio -- these bounds come from that
#     distribution's 5th/95th percentile as a sane default; recalibrate
#     per language from your own corpus_statistics_*.csv if very different)
HALLU_MITIGATION_MIN_LEN_RATIO = 0.5   # min_new_tokens ~= ratio * len(src)
HALLU_MITIGATION_MAX_LEN_RATIO = 2.5   # max_new_tokens ~= ratio * len(src)
# (c) LaBSE-reranked beam search
HALLU_MITIGATION_NUM_BEAMS      = 4
HALLU_MITIGATION_LABSE_RERANK   = True

# ── COMET model ────────────────────────────────────────────────────────────
COMET_MODEL = "Unbabel/wmt22-comet-da"
# NOTE for the paper: wmt22-comet-da was not trained on mni/as/brx. It is
# used here zero-shot / cross-lingually for RELATIVE ranking between your
# own systems, not as an absolute quality claim. State this explicitly as
# a limitation in any manuscript using these numbers.

# ── Hallucination thresholds ───────────────────────────────────────────────
HALL_LEN_RATIO_LOW  = 0.25
HALL_LEN_RATIO_HIGH = 4.0
HALL_COPY_RATIO     = 0.70

# ── Plot colours / labels ──────────────────────────────────────────────────
SYSTEM_LABELS = {
    "baseline" : "Baseline (zero-shot)",
    "r1"       : "R1 (LoRA fine-tuned)",
    "exp1"     : "Exp-1 (+BT)",
    "exp2"     : "Exp-2 (+BT+FT)",
    "exp3"     : "Exp-3 (+BT+FT+Iter)",
    "exp4"     : "Exp-4 (+BT+FT+Noise)",
    "exp5"     : "Exp-5 (+BT+FT+SubwordReg)",
    "exp6"     : "Exp-6 (+BT+FT+EngPara)",
    "exp7"     : "Exp-7 (+Hallucination Mitigation)",
}
SYSTEM_COLORS = {
    "baseline" : "#C44E52",
    "r1"       : "#4C72B0",
    "exp1"     : "#DD8452",
    "exp2"     : "#55A868",
    "exp3"     : "#8172B2",
    "exp4"     : "#CCB974",
    "exp5"     : "#64B5CD",
    "exp6"     : "#E18AC0",
    "exp7"     : "#937860",
}

# ── Create all required directories ───────────────────────────────────────
for _d in [
    OUTPUT_DIR, FIGURES_DIR, CKPT_ROOT,
    ADAPTER_DIR,
    EXP1_ADAPTER, EXP2_ADAPTER, EXP3_ADAPTER,
    EXP4_ADAPTER, EXP5_ADAPTER, EXP6_ADAPTER,
]:
    os.makedirs(_d, exist_ok=True)

print(f"[config] Language={MT_LANG} ({LANG_NAME})  |  Model seed={MODEL_SEED}"
      f"{'  [SWEEP RUN]' if _SEED_SUFFIX else ''}  |  Output dir={OUTPUT_DIR}")
print(f"[config] LoRA: r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}  "
      f"use_rslora={USE_RSLORA}  (sized to {MT_LANG}'s corpus — see LANG_LORA_CONFIG)")
