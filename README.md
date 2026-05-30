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
