#!/usr/bin/env python3
"""
Run SoundnessBench evaluation: score hypothesis-experiment pairs with an LLM
and compute rigor-bucket metrics against ground truth.

  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --max-test-pairs 50

Model and prompt come from config/eval/eval.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_EVAL_CONFIG = project_root / "config" / "eval" / "eval.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SoundnessBench evaluation (LLM baseline vs. ground truth)")
    ap.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Path to evaluation pairs JSONL (default: configured in config/eval/eval.yaml)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write results JSON (default: configured in config/eval/eval.yaml)",
    )
    ap.add_argument(
        "--test-ratio",
        type=float,
        default=1.0,
        help="Fraction of data to use as test by paper (default: 1.0, all papers)",
    )
    ap.add_argument(
        "--max-test-pairs",
        type=int,
        default=None,
        help="Max number of pairs to score. If omitted, read from config/eval/eval.yaml. Use 0 for no limit.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test split",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between LLM calls (default 0.5)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent scoring workers (default 4)",
    )
    ap.add_argument(
        "--random-baseline",
        action="store_true",
        help="Use a uniformly random scorer instead of an LLM (chance baseline).",
    )
    ap.add_argument(
        "--evaluation-mode",
        choices=["direct_bucket", "direct_bucket_aggressive"],
        default=None,
        help="Evaluation mode override (otherwise read from config/eval/eval.yaml).",
    )
    args = ap.parse_args()

    eval_cfg: dict = {}
    if DEFAULT_EVAL_CONFIG.exists():
        with open(DEFAULT_EVAL_CONFIG, encoding="utf-8") as f:
            eval_cfg = yaml.safe_load(f) or {}

    # Resolve paths from config if not given
    pairs_path = args.pairs
    output_path = args.output
    if pairs_path is None:
        pairs_cfg = eval_cfg.get("pairs_path") or "data/soundnessbench.jsonl"
        pairs_path = project_root / pairs_cfg
    if output_path is None:
        output_cfg = eval_cfg.get("output_path") or "results/eval_results.json"
        output_path = project_root / output_cfg
    
    # Ensure eval_results directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}", file=sys.stderr)
        sys.exit(1)

    max_test = args.max_test_pairs
    if max_test is not None and max_test <= 0:
        max_test = None
    max_test_label = max_test if max_test is not None else "from eval.yaml/default"
    print(f"Pairs: {pairs_path}")
    print(f"Test ratio: {args.test_ratio}, max test pairs: {max_test_label}, seed: {args.seed}")
    print(f"Concurrency: {max(1, args.concurrency)}")

    from rigorbench.evaluation.run import run_evaluation, RandomBaseline

    if args.random_baseline:
        print("Running evaluation (random baseline)...")
        client = RandomBaseline(seed=args.seed)
    else:
        print("Running evaluation (LLM baseline)...")
        client = None

    try:
        results = run_evaluation(
            pairs_path=pairs_path,
            output_path=output_path,
            client=client,
            test_ratio=args.test_ratio,
            seed=args.seed,
            max_test_pairs=max_test,
            delay_seconds=0.0 if args.random_baseline else args.delay,
            max_concurrency=max(1, args.concurrency),
            evaluation_mode=args.evaluation_mode,
        )
    except Exception as e:  # noqa: BLE001
        err_msg = str(e).lower()
        if "connection refused" in err_msg or "connection error" in err_msg or "apiconnectionerror" in err_msg:
            print(
                "\nHint: The eval server (vLLM/Qwen) is not reachable. Either start the server on the target host, "
                "or set VLLM_BASE_URL in .env to a reachable URL (e.g. http://HOST:8000/v1).",
                file=sys.stderr,
            )
        raise

    model_name = results.get("model") or "unknown"
    print(f"\nEvaluated model: {model_name}")
    if results.get("evaluation_mode"):
        print(f"Evaluation mode: {results.get('evaluation_mode')}")

    mode = results.get("evaluation_mode")
    metrics = results.get("metrics", {})

    if mode in ("direct_bucket", "direct_bucket_aggressive"):
        summary = metrics.get("summary", {})
        print(f"{mode} metrics:")
        print(f"  Accuracy:    {summary.get('rigor_bucket_accuracy')}")
        print(f"  Cohen's κ:   {summary.get('rigor_bucket_kappa')}")
        print(f"  Valid pairs: {summary.get('total_n')}")
    else:
        print(f"Unsupported public evaluation mode: {mode}", file=sys.stderr)
        sys.exit(2)
    print(f"  Pairs evaluated: {results.get('n_pairs_evaluated')}")


if __name__ == "__main__":
    main()
