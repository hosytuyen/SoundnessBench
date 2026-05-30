"""Prompt loading and formatting for SoundnessBench evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

EvaluationMode = Literal["direct_bucket", "direct_bucket_aggressive"]
SCORE_KEYS = ["rigor_bucket", "confidence"]

SYSTEM_PROMPT_DIRECT_BUCKET = """You are an expert ML researcher and peer reviewer. Classify the scientific rigor bucket of a research idea and your assessment confidence from 1 to 5 from its hypothesis and experiment description.

Output the assessment as a JSON object, including a detailed step-by-step justification for the rigor bucket selected."""

USER_TEMPLATE_DIRECT_BUCKET = """Classify this hypothesis-experiment pair into one rigor bucket:
- "low": "Weak scientific contribution. Hypothesis is vague or trivial, experiments lack controls or baselines, metrics are weak, or methodology has fundamental flaws.",
- "high": "Strong scientific contribution. Hypothesis is clear and meaningful. Experiments are rigorous, controlled, include appropriate baselines/ablations, and use suitable metrics.",

Confidence Score Scale:
- 1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.
- 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
- 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

HYPOTHESIS:
{hypothesis}

EXPERIMENT:
{experiment}

Output format:
{{
  "justification": "<Think step-by-step, provide detailed justification>",
  "rigor_bucket": <"low" or "high">,
  "confidence": <1-5 integer>
}}

Constraints:
- rigor_bucket must be a choice in ["low", "high"]
- confidence must be an integer in [1, 5]
"""

SYSTEM_PROMPT_DIRECT_BUCKET_AGGRESSIVE = """You are a strict ML area chair applying an aggressive rigor filter. Classify scientific rigor from a hypothesis and experiment description.

Default to "low" unless the evidence clearly demonstrates strong scientific rigor with concrete controls, strong baselines, appropriate metrics, and a credible evaluation plan.

Output valid JSON only with a detailed step-by-step justification."""

USER_TEMPLATE_DIRECT_BUCKET_AGGRESSIVE = """Classify this hypothesis-experiment pair into one rigor bucket under an aggressive standard:
- "low": choose this unless there is clear, concrete, and compelling evidence of rigorous methodology.
- "high": only if the plan is explicitly strong on hypothesis clarity, experimental controls, baselines/ablations, metric validity, and methodological credibility.

Aggressive policy:
- Penalize missing controls, vague methods, missing or weak baselines, underspecified metrics, unclear evaluation protocol, or hand-wavy claims.
- If information is incomplete or ambiguous, prefer "low".
- Use "high" only when justification is unambiguous.

Confidence Score Scale:
- 1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.
- 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math or other details were not carefully checked.
- 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math or other details were not carefully checked.
- 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
- 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math or other details carefully.

HYPOTHESIS:
{hypothesis}

EXPERIMENT:
{experiment}

Output format:
{{
  "justification": "<Think step-by-step, provide detailed justification>",
  "rigor_bucket": <"low" or "high">,
  "confidence": <1-5 integer>
}}

Constraints:
- rigor_bucket must be a choice in ["low", "high"]
- confidence must be an integer in [1, 5]
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _extract_prompt_sections(text: str) -> tuple[str | None, str | None]:
    system_header = "## System prompt"
    user_header = "## User prompt template"
    if system_header not in text or user_header not in text:
        return None, None
    before_user, user_part = text.split(user_header, 1)
    system = before_user.split(system_header, 1)[-1].strip()
    user = user_part.strip()
    return system or None, user or None


def load_prompt_from_file(prompt_path: str | Path, mode: EvaluationMode) -> tuple[str, str]:
    """Load a markdown prompt file, falling back to bundled defaults."""
    if mode == "direct_bucket_aggressive":
        system = SYSTEM_PROMPT_DIRECT_BUCKET_AGGRESSIVE
        user_template = USER_TEMPLATE_DIRECT_BUCKET_AGGRESSIVE
    else:
        system = SYSTEM_PROMPT_DIRECT_BUCKET
        user_template = USER_TEMPLATE_DIRECT_BUCKET

    path = Path(prompt_path)
    if not path.is_absolute():
        path = _project_root() / path
    if not path.exists():
        return system, user_template

    file_system, file_user = _extract_prompt_sections(path.read_text(encoding="utf-8"))
    return file_system or system, file_user or user_template


def build_scoring_prompt(
    hypothesis: str,
    experiment: str,
    prompt_path: str | Path | None = None,
    mode: EvaluationMode = "direct_bucket",
) -> tuple[str, str]:
    """Return system and user prompts for one SoundnessBench pair."""
    if mode not in ("direct_bucket", "direct_bucket_aggressive"):
        raise ValueError(f"Unsupported public evaluation mode: {mode}")
    if prompt_path is None:
        if mode == "direct_bucket_aggressive":
            system, user_template = SYSTEM_PROMPT_DIRECT_BUCKET_AGGRESSIVE, USER_TEMPLATE_DIRECT_BUCKET_AGGRESSIVE
        else:
            system, user_template = SYSTEM_PROMPT_DIRECT_BUCKET, USER_TEMPLATE_DIRECT_BUCKET
    else:
        system, user_template = load_prompt_from_file(prompt_path, mode=mode)
    return system, user_template.format(
        hypothesis=hypothesis.strip() or "(none)",
        experiment=experiment.strip() or "(none)",
    )
