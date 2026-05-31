<h1 align="center">SoundnessBench</h1>

<p align="center">
  <b>Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.30329">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://github.com/hosytuyen/SoundnessBench">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg" alt="Code">
  </a>
  <a href="https://huggingface.co/datasets/hosytuyen/SoundnessBench">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-ffcc4d.svg" alt="Dataset">
  </a>
  <a href="https://hosytuyen.github.io/projects/SoundnessBench/">
    <img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project">
  </a>
</p>

Abstract: Autonomous AI research agents aim to accelerate scientific discovery by automating the research pipeline, from hypothesis generation to peer review. However, existing benchmarks rarely test a fundamental bottleneck: whether Large Language Models can judge the methodological viability of a research idea before expending time and computational resources. We introduce SoundnessBench, a curated benchmark of 1,099 machine-learning research proposals reconstructed from ICLR submissions, labeled with reviewer soundness sub-scores, and audited against source papers. SoundnessBench should be interpreted as a benchmark for recoverable proposal-stage soundness rather than exact prediction of full-paper review outcomes. Across 12 frontier LLMs, we find a pervasive optimism bias: under standard prompting, models frequently rate low-soundness proposals as sound, while aggressive prompting largely shifts errors from false positives to false negatives. Additional controls for public-corpus contamination, paper-identifying phrases, surface features, and human audit quality suggest that this behavior is not explained by a single confounder. Our results indicate that current LLMs are not yet reliable as standalone first-gate evaluators for scientific rigor.


## News

- **May 30, 2026**: Released the data on HF, and eval code for standard/aggressive prompt!

## Leaderboard (Top-10 only)

Full, interactive, continuously updated leaderboard available [here](https://hosytuyen.github.io/projects/SoundnessBench/).

| Rank | Model | Eval Mode | FP Rate ↓ | FN Rate ↓ | TP Rate ↑ | TN Rate ↑ | Recall ↑ | Precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Claude-Sonnet-4-6 | Standard Prompt | 62.8% | 5.2% | 94.8% | 37.2% | 94.8% | 67.7% | 79.0% |
| 2 | claude-opus-4-6 | Standard Prompt | 71.8% | 2.8% | 97.2% | 28.2% | 97.2% | 64.8% | 77.7% |
| 3 | Gemini-3.1-Pro | Standard Prompt | 74.0% | 3.9% | 96.1% | 26.0% | 96.1% | 64.5% | 77.2% |
| 4 | Gemini-3-Flash | Standard Prompt | 80.3% | 1.9% | 98.1% | 19.7% | 98.1% | 63.1% | 76.8% |
| 5 | Gemini-2.5-Pro | Standard Prompt | 86.2% | 1.6% | 98.4% | 13.8% | 98.4% | 61.5% | 75.7% |
| 6 | Qwen3.5-27B | Standard Prompt | 76.4% | 7.5% | 92.5% | 23.6% | 92.5% | 62.9% | 74.9% |
| 7 | GPT-5.4 | Standard Prompt | 35.4% | 25.4% | 74.6% | 64.6% | 74.6% | 74.7% | 74.6% |
| 8 | Qwen3.5-122B-A10B | Standard Prompt | 73.4% | 9.4% | 90.6% | 26.6% | 90.6% | 63.4% | 74.6% |
| 9 | GPT-4o | Standard Prompt | 94.5% | 1.1% | 98.9% | 5.5% | 98.9% | 59.4% | 74.2% |
| 10 | LLaMA-3.3-70B-Instruct | Standard Prompt | 98.0% | 0.6% | 99.4% | 2.0% | 99.4% | 58.7% | 73.8% |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Data

Download the benchmark JSONL from Hugging Face into `data/`:

```bash
hf download hosytuyen/SoundnessBench --repo-type dataset --local-dir data
```

The default evaluator expects:

```text
data/soundnessbench.jsonl
```

If the downloaded file has a different name, pass it with `--pairs` or update `config/eval/eval.yaml`.

## Configure a Model

Set API credentials in your shell or in `.env`:

```bash
export OPENAI_API_KEY=...
```

Then edit `config/eval/eval.yaml` if you want a different provider or model. Supported providers are `openai`, `anthropic`, `gemini`, `vertex_ai`, and `vllm`.

## Run Evaluation

Standard prompt:

```bash
python scripts/run_evaluation.py \
  --pairs data/soundnessbench.jsonl \
  --output results/eval_results.json \
  --evaluation-mode direct_bucket
```

Aggressive prompt:

```bash
python scripts/run_evaluation.py \
  --pairs data/soundnessbench.jsonl \
  --output results/eval_results.json \
  --evaluation-mode direct_bucket_aggressive
```

The script writes model-specific result files under `results/`, including predictions, ground truth labels, accuracy, and Cohen's kappa.

## Prompts

The two public evaluation prompts are:

- Standard: `config/eval/prompt_direct_bucket.md`
- Aggressive: `config/eval/prompt_direct_bucket_aggressive.md`

To add a new prompt, create a markdown file in `config/eval/` with `## System prompt` and `## User prompt template` sections. The user template must include `{hypothesis}` and `{experiment}`, and the model should return JSON with `rigor_bucket`, plus optional `confidence` and `justification`. Add the file to `prompt_paths` in `config/eval/eval.yaml`.


## Citation

```bibtex
@article{ho2026soundnessbench,
  title={SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?},
  author={Ho, Sy-Tuyen and Liu, Minghui and Nghiem, Huy and Huang, Furong},
  journal={arXiv preprint arXiv:2605.30329},
  year={2026}
}
```
