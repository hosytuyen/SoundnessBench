"""Formatting helper for SoundnessBench structured experiment records."""

from __future__ import annotations

from typing import Any


def _format_experiments_for_eval(experiments: list[dict[str, Any]]) -> str:
    """Render structured experiments as a readable text block for eval prompts."""
    if not experiments:
        return ""
    parts: list[str] = []
    for i, exp in enumerate(experiments, 1):
        lines = [f"Experiment {i}:"]
        if exp.get("Description"):
            lines.append(f"  Description: {exp['Description']}")
        if exp.get("Method"):
            lines.append(f"  Method: {exp['Method']}")
        if exp.get("Evaluation Metrics"):
            lines.append(f"  Evaluation Metrics: {exp['Evaluation Metrics']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
