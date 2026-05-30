"""Minimal chat clients for SoundnessBench evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class LLMClient:
    """Unified chat interface used by the evaluator."""

    model: str
    max_tokens: int
    temperature: float

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    _load_dotenv()
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "eval" / "eval.yaml"
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class OpenAIChatClient(LLMClient):
    """OpenAI API or OpenAI-compatible endpoint, including vLLM."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "dummy"
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai") from None

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        token_value = kwargs.get("max_tokens", self.max_tokens)
        if self.base_url and "openai.com" in self.base_url:
            create_kwargs["max_completion_tokens"] = token_value
        else:
            create_kwargs["max_tokens"] = token_value
            create_kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        response = client.chat.completions.create(**create_kwargs)
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()


class AnthropicChatClient(LLMClient):
    """Anthropic Messages API client."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic") from None

        system = ""
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system or None,
            messages=anthropic_messages,
        )
        if response.content:
            return (response.content[0].text or "").strip()
        return ""


class GeminiChatClient(LLMClient):
    """Google AI Studio Gemini chat client."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or ""
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai") from None

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        prompt = "\n\n".join(f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            ),
        )
        return (response.text or "").strip() if response and response.text else ""


class VertexAIChatClient(LLMClient):
    """Vertex AI Gemini chat client."""

    def __init__(
        self,
        model: str,
        project_id: str | None = None,
        location: str = "us-central1",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
        self.location = os.environ.get("VERTEX_AI_LOCATION") or location
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            import vertexai
            from vertexai.generative_models import GenerationConfig, GenerativeModel
        except ImportError:
            raise ImportError("Install google-cloud-aiplatform: pip install google-cloud-aiplatform") from None

        vertexai.init(project=self.project_id, location=self.location)
        model = GenerativeModel(self.model)
        prompt = "\n\n".join(f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            ),
        )
        if getattr(response, "text", None):
            return response.text.strip()
        return ""


def get_llm_client(
    provider: str | None = None,
    config_path: str | Path | None = None,
    model: str | None = None,
) -> LLMClient:
    """Create the configured chat client for evaluation."""
    cfg = _load_config(config_path)
    provider = (provider or cfg.get("provider") or os.environ.get("LLM_PROVIDER") or "openai").lower()
    max_tokens = int(cfg.get("max_tokens", 2048))
    temperature = float(cfg.get("temperature", 0.2))

    if provider == "openai":
        openai_cfg = cfg.get("openai") or {}
        return OpenAIChatClient(
            model=model or openai_cfg.get("model", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=openai_cfg.get("base_url"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "vllm":
        vllm_cfg = cfg.get("vllm") or {}
        return OpenAIChatClient(
            model=model or vllm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            api_key=os.environ.get("VLLM_API_KEY", "dummy"),
            base_url=vllm_cfg.get("base_url", "http://localhost:8000/v1"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "anthropic":
        anthropic_cfg = cfg.get("anthropic") or {}
        return AnthropicChatClient(
            model=model or anthropic_cfg.get("model", "claude-3-5-sonnet-20241022"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "gemini":
        google_cfg = cfg.get("google") or {}
        return GeminiChatClient(
            model=model or google_cfg.get("model", "gemini-2.5-pro"),
            api_key=os.environ.get("GOOGLE_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "vertex_ai":
        vertex_cfg = cfg.get("vertex_ai") or {}
        return VertexAIChatClient(
            model=model or vertex_cfg.get("model", "gemini-2.5-pro"),
            project_id=vertex_cfg.get("project_id"),
            location=vertex_cfg.get("location", "us-central1"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError("Unknown LLM provider. Use one of: openai, anthropic, gemini, vertex_ai, vllm")
