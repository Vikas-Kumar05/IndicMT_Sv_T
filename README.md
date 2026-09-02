# Eng ⇄ Indic Low-Resource MT

Fine-tuning [sarvam-translate](https://huggingface.co/sarvamai/sarvam-translate) (Gemma3-4B-IT) for bidirectional English ⇄ Indic translation, with a systematic ablation of data-augmentation techniques and decode-time hallucination mitigation.

**Languages:** Manipuri (`mni`), Assamese (`as`), Bodo (`brx`)

## Pipeline

```
Baseline (zero-shot) → R1 (LoRA fine-tune) → Exp-1..6 (augmentation ablations) → Exp-7 (hallucination mitigation) → Final comparison
```

| Step | Description |
|------|-------------|
| 1 | Dataset cleaning, dedup, bidirectional expansion |
| 2 | Zero-shot baseline eval |
| 3 | R1: LoRA/DoRA fine-tuning |
| 4 | Evaluate R1 vs baseline |
| 5a–5e | Exp-1: BT, Exp-2: BT+FT, Exp-3:BT+FT+Iterative retrain, Exp-5:BT+FT+Iterative+Subword reg |
| 5g | Exp-7: decode-time mitigation (rep-penalty, LaBSE-reranked beam) and hallucination analysis |
| 8 | Final comparison across all systems |
| 9 | significance testing (optional, see below) |
| 10 | Cross-language results dashboard |

Metrics: BLEU, chrF, chrF++, TER, COMET, plus a rule-based hallucination check (omission/repetition/copy-source).

> All results in this repo use the default seed (`16`) only. `run_seed_sweep.py` / `step9_significance.py` support multi-seed significance testing but haven't been run yet.

## Setup

```bash
pip install torch transformers peft datasets sacrebleu sentence-transformers \
            unbabel-comet pandas numpy matplotlib seaborn tqdm nltk sentencepiece openpyxl

hf download sarvamai/sarvam-translate --local-dir /path/to/models/sarvam-translate
export MT_MODEL_PATH=/path/to/models/sarvam-translate
```

Data layout (pre-split, per language):

```
raw_data/
├── mni/{train,dev,test}.xlsx
├── as/{train,dev,test}.xlsx
└── brx/{train,dev,test}.xlsx
```

## Usage

```bash
# Full pipeline
python main.py --lang mni

# Specific steps
python main.py --lang mni --steps 5a 5b 5c 8

# All augmentation experiments + hallucination mitigation
python main.py --lang mni --all-aug-plus-hallu

# Via bash wrapper (with logging)
bash run_step.sh full mni
bash run_step.sh all-langs full
```

## Repo Structure

```
main.py                    # orchestrator
config.py                  # paths, hyperparams, per-language config
step1..step10*.py          # pipeline stages
aug_techniques.py          # 6 augmentation generators
aug_utils.py               # shared model I/O, quality filters (LaBSE/chrF++)
aug_finetune_eval.py       # shared fine-tune + eval loop
run_seed_sweep.py / sweep_runner.py   # multi-seed sweep tooling
outputs/, checkpoints/     # generated at runtime, per-language
```


