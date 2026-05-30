"""SoundnessBench public evaluation runner."""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from rigorbench.buckets import normalize_bucket
from rigorbench.extraction.extract import _format_experiments_for_eval
from .metrics import compute_bucket_metrics
from .prompt import EvaluationMode, SCORE_KEYS, build_scoring_prompt


class RandomBaseline:
    """Pseudo-client that predicts uniformly random rigor buckets."""

    model = "random_baseline"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        bucket = self._rng.choice(["low", "high"])
        confidence = self._rng.randint(1, 5)
        return json.dumps({"rigor_bucket": bucket, "confidence": confidence, "justification": "random baseline"})


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Load SoundnessBench examples from a JSONL file."""
    pairs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def get_test_split(
    pairs: list[dict[str, Any]],
    test_ratio: float = 1.0,
    seed: int = 42,
    max_test_pairs: int | None = None,
    by_paper: bool = True,
) -> list[dict[str, Any]]:
    """Return a deterministic paper-level test subset, or all pairs by default."""
    if test_ratio >= 1.0 and max_test_pairs is None:
        return list(pairs)

    rng = random.Random(seed)
    if by_paper and any("paper_id" in p for p in pairs):
        paper_ids = sorted({str(p.get("paper_id")) for p in pairs if p.get("paper_id") is not None})
        rng.shuffle(paper_ids)
        n_test = max(1, int(len(paper_ids) * test_ratio))
        selected = set(paper_ids[:n_test])
        test_pairs = [p for p in pairs if str(p.get("paper_id")) in selected]
    else:
        test_pairs = list(pairs)
        rng.shuffle(test_pairs)
        n_test = max(1, int(len(test_pairs) * test_ratio))
        test_pairs = test_pairs[:n_test]

    if max_test_pairs is not None and len(test_pairs) > max_test_pairs:
        rng.shuffle(test_pairs)
        test_pairs = test_pairs[:max_test_pairs]
    return test_pairs


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()


def _extract_first_json_object(text: str) -> str | None:
    text = _strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _clamp_confidence(value: Any) -> int | None:
    try:
        return int(round(max(1.0, min(5.0, float(value)))))
    except (TypeError, ValueError):
        return None


def _parse_prediction(response: str) -> dict[str, Any]:
    """Parse model JSON into the public prediction schema."""
    parsed: dict[str, Any] = {"rigor_bucket": None, "confidence": None, "justification": None}
    raw = _extract_first_json_object(response)
    if raw is None:
        return parsed
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        bucket_match = re.search(r'"(?:rigor_bucket|bucket)"\s*:\s*"([^"]+)"', raw, flags=re.IGNORECASE)
        conf_match = re.search(r'"(?:confidence|reviewer_confidence)"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
        if bucket_match:
            parsed["rigor_bucket"] = normalize_bucket(bucket_match.group(1))
        if conf_match:
            parsed["confidence"] = _clamp_confidence(conf_match.group(1))
        return parsed

    bucket = obj.get("rigor_bucket", obj.get("bucket"))
    parsed["rigor_bucket"] = normalize_bucket(bucket)
    parsed["confidence"] = _clamp_confidence(obj.get("confidence", obj.get("reviewer_confidence")))
    justification = obj.get("justification")
    if isinstance(justification, str) and justification.strip():
        parsed["justification"] = justification.strip()
    return parsed


def _pair_text(pair: dict[str, Any]) -> tuple[str, str]:
    hypothesis = str(pair.get("short_hypothesis") or pair.get("hypothesis") or "").strip()
    if pair.get("experiments"):
        experiment = _format_experiments_for_eval(pair.get("experiments") or [])
    else:
        experiment = str(pair.get("experiment") or "").strip()
    return hypothesis, experiment


def score_pair_with_llm(
    pair: dict[str, Any],
    client: Any,
    prompt_path: str | Path | None = None,
    mode: EvaluationMode = "direct_bucket",
) -> dict[str, Any]:
    """Score one SoundnessBench example with an LLM client."""
    hypothesis, experiment = _pair_text(pair)
    system_prompt, user_prompt = build_scoring_prompt(
        hypothesis=hypothesis,
        experiment=experiment,
        prompt_path=prompt_path,
        mode=mode,
    )
    response = client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=getattr(client, "max_tokens", 2048),
    )
    return _parse_prediction(response)


def _load_eval_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "config" / "eval" / "eval.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_output_path(output_path: Path, client: Any, mode: EvaluationMode) -> Path:
    model_slug = str(getattr(client, "model", "unknown")).replace("/", "_").replace(" ", "_")
    return output_path.with_name(f"{output_path.stem}_{model_slug}_{mode}{output_path.suffix}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_evaluation(
    pairs_path: str | Path,
    output_path: str | Path | None = None,
    client: Any | None = None,
    test_ratio: float = 1.0,
    seed: int = 42,
    max_test_pairs: int | None = None,
    delay_seconds: float = 0.0,
    max_concurrency: int = 8,
    evaluation_mode: EvaluationMode | None = None,
) -> dict[str, Any]:
    """Run standard or aggressive SoundnessBench bucket evaluation."""
    from rigorbench.llm import get_llm_client

    project_root = Path(__file__).resolve().parent.parent.parent
    eval_cfg = _load_eval_config(project_root)
    mode: EvaluationMode = evaluation_mode or eval_cfg.get("evaluation_mode", "direct_bucket")
    if mode not in ("direct_bucket", "direct_bucket_aggressive"):
        raise ValueError(f"Unsupported public evaluation mode: {mode}")

    resolved_max = max_test_pairs
    if resolved_max is None:
        try:
            resolved_max = int(eval_cfg.get("max_test_pairs", 0))
        except (TypeError, ValueError):
            resolved_max = 0
    if resolved_max is not None and resolved_max <= 0:
        resolved_max = None

    pairs = load_pairs(pairs_path)
    test_pairs = get_test_split(
        pairs,
        test_ratio=test_ratio,
        seed=seed,
        max_test_pairs=resolved_max,
        by_paper=True,
    )

    if client is None:
        config_path = project_root / "config" / "eval" / "eval.yaml"
        client = get_llm_client(config_path=config_path if config_path.exists() else None)

    prompt_paths = eval_cfg.get("prompt_paths") or {}
    prompt_path = prompt_paths.get(mode)
    if prompt_path and not Path(prompt_path).is_absolute():
        prompt_path = project_root / prompt_path

    try:
        max_workers = max(1, int(max_concurrency))
    except (TypeError, ValueError):
        max_workers = 1

    predictions: list[dict[str, Any]] = [{k: None for k in SCORE_KEYS} for _ in test_pairs]
    ground_truths = [
        {
            "soundness_score": pair.get("soundness_score"),
            "rigor_bucket": pair.get("rigor_bucket"),
        }
        for pair in test_pairs
    ]

    output_file: Path | None = None
    if output_path is not None:
        output_file = _model_output_path(Path(output_path), client, mode)

    def snapshot(metrics: dict[str, Any] | None = None) -> None:
        if output_file is None:
            return
        rows = []
        for idx, (pair, prediction, truth) in enumerate(zip(test_pairs, predictions, ground_truths)):
            rows.append(
                {
                    "pair_id": pair.get("pair_id") or f"row_{idx}",
                    "prediction": prediction,
                    "ground_truth": truth,
                    "year": pair.get("year"),
                }
            )
        _write_json(
            output_file,
            {
                "model": getattr(client, "model", None),
                "evaluation_mode": mode,
                "n_pairs_evaluated": len(test_pairs),
                "test_ratio": test_ratio,
                "seed": seed,
                "max_test_pairs": resolved_max,
                "metrics": metrics or {},
                "results": rows,
                "predictions_sample": predictions[:5],
            },
        )

    def score_one(idx: int) -> tuple[int, dict[str, Any]]:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            return idx, score_pair_with_llm(test_pairs[idx], client, prompt_path=prompt_path, mode=mode)
        except Exception as exc:  # noqa: BLE001
            pair_id = test_pairs[idx].get("pair_id") or f"row_{idx}"
            print(f"[Eval][Error] pair_id={pair_id} mode={mode} scoring failed: {exc}")
            return idx, {"rigor_bucket": None, "confidence": None, "justification": None}

    pending = list(range(len(test_pairs)))
    if max_workers == 1 or len(pending) <= 1:
        iterator = (score_one(idx) for idx in tqdm(pending, desc="Scoring pairs with LLM"))
        for idx, prediction in iterator:
            predictions[idx] = prediction
            snapshot()
    else:
        print(f"Using concurrency={max_workers} for pair scoring.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(score_one, idx): idx for idx in pending}
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Scoring pairs with LLM"):
                idx, prediction = future.result()
                predictions[idx] = prediction
                snapshot()

    metrics = compute_bucket_metrics(predictions, ground_truths)
    results = {
        "model": getattr(client, "model", None),
        "evaluation_mode": mode,
        "n_pairs_evaluated": len(test_pairs),
        "test_ratio": test_ratio,
        "seed": seed,
        "max_test_pairs": resolved_max,
        "metrics": metrics,
        "results": [
            {
                "pair_id": pair.get("pair_id") or f"row_{idx}",
                "prediction": prediction,
                "ground_truth": truth,
                "year": pair.get("year"),
            }
            for idx, (pair, prediction, truth) in enumerate(zip(test_pairs, predictions, ground_truths))
        ],
        "predictions_sample": predictions[:5],
    }
    if output_file is not None:
        _write_json(output_file, results)
        print(f"Results written to {output_file}")
    return results
