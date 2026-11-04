"""
AI Client abstraction for Prism.
Supports multiple LLM providers: OpenAI, Anthropic (Claude), Google (Gemini),
and local models (via Ollama, LM Studio, or local transformers).

Used for post-processing transcribed audio into actionable intents
(calendar events, todos, emails, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("ai_client")


class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    CUSTOM = "custom"


@dataclass
class AIProviderConfig:
    """Configuration for a single AI provider."""
    provider: AIProvider
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60

    def is_configured(self) -> bool:
        if self.provider == AIProvider.OLLAMA:
            return bool(self.base_url)
        if self.provider == AIProvider.LM_STUDIO:
            return bool(self.base_url)
        if self.provider == AIProvider.CUSTOM:
            return bool(self.base_url)
        return bool(self.api_key)


@dataclass
class AIResponse:
    text: str
    provider: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None


class AIClient:
    """Unified client for AI providers.

    Supports chat/completions style interface across providers.
    Falls back gracefully when provider is not configured.
    """

    def __init__(self):
        self._providers: dict[str, AIProviderConfig] = {}
        self._default_provider: str | None = None
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load provider configuration from environment variables."""
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self.add_provider("openai", AIProviderConfig(
                provider=AIProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            ))
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            self.add_provider("anthropic", AIProviderConfig(
                provider=AIProvider.ANTHROPIC,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            ))
        # Google
        if os.getenv("GOOGLE_API_KEY"):
            self.add_provider("google", AIProviderConfig(
                provider=AIProvider.GOOGLE,
                api_key=os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            ))
        # Ollama / LM Studio / Custom
        if os.getenv("OLLAMA_BASE_URL"):
            self.add_provider("ollama", AIProviderConfig(
                provider=AIProvider.OLLAMA,
                base_url=os.getenv("OLLAMA_BASE_URL"),
                model=os.getenv("OLLAMA_MODEL", "llama3"),
            ))
        if os.getenv("LM_STUDIO_BASE_URL"):
            self.add_provider("lm_studio", AIProviderConfig(
                provider=AIProvider.LM_STUDIO,
                base_url=os.getenv("LM_STUDIO_BASE_URL"),
                model=os.getenv("LM_STUDIO_MODEL", "local-model"),
            ))
        if os.getenv("CUSTOM_AI_BASE_URL"):
            self.add_provider("custom", AIProviderConfig(
                provider=AIProvider.CUSTOM,
                base_url=os.getenv("CUSTOM_AI_BASE_URL"),
                api_key=os.getenv("CUSTOM_AI_API_KEY"),
                model=os.getenv("CUSTOM_AI_MODEL", "default"),
            ))

        # Set default provider: prefer explicit PRISM_AI_PROVIDER, else first configured
        env_provider = os.getenv("PRISM_AI_PROVIDER")
        if env_provider and env_provider in self._providers:
            self._default_provider = env_provider
        elif self._providers:
            self._default_provider = next(iter(self._providers))

    def add_provider(self, name: str, config: AIProviderConfig) -> None:
        self._providers[name] = config
        if self._default_provider is None:
            self._default_provider = name

    def get_provider(self, name: str | None = None) -> AIProviderConfig | None:
        if name is None:
            name = self._default_provider
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def set_default(self, name: str) -> bool:
        if name in self._providers:
            self._default_provider = name
            return True
        return False

    async def complete(
        self,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """Generate a completion using the specified provider."""
        prov = self.get_provider(provider)
        if prov is None:
            raise ValueError(f"No AI provider available (tried: {provider})")

        # Override settings
        if model is not None:
            prov.model = model
        if temperature is not None:
            prov.temperature = temperature
        if max_tokens is not None:
            prov.max_tokens = max_tokens

        return await self._call_provider(prov, prompt)

    async def _call_provider(self, prov: AIProviderConfig, prompt: str) -> AIResponse:
        import time

        start = time.time()

        if prov.provider == AIProvider.OPENAI:
            return await self._call_openai(prov, prompt, start)
        if prov.provider == AIProvider.ANTHROPIC:
            return await self._call_anthropic(prov, prompt, start)
        if prov.provider == AIProvider.GOOGLE:
            return await self._call_google(prov, prompt, start)
        if prov.provider == AIProvider.OLLAMA:
            return await self._call_ollama(prov, prompt, start)
        if prov.provider == AIProvider.LM_STUDIO:
            return await self._call_lm_studio(prov, prompt, start)
        if prov.provider == AIProvider.CUSTOM:
            return await self._call_custom(prov, prompt, start)

        raise ValueError(f"Unknown provider: {prov.provider}")

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    async def _call_openai(self, prov: AIProviderConfig, prompt: str, start: float) -> AIResponse:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for OpenAI provider")

        headers = {"Authorization": f"Bearer {prov.api_key}"}
        payload = {
            "model": prov.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": prov.temperature,
            "max_tokens": prov.max_tokens,
        }
        async with httpx.AsyncClient(timeout=prov.timeout) as client:
            resp = await client.post(
                f"{prov.base_url or 'https://api.openai.com'}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return AIResponse(
            text=text,
            provider="openai",
            model=prov.model,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            latency_ms=int((time.time() - start) * 1000),
        )

    async def _call_anthropic(self, prov: AIProviderConfig, prompt: str, start: float) -> AIResponse:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for Anthropic provider")

        headers = {
            "Authorization": f"Bearer {prov.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": prov.model,
            "max_tokens": prov.max_tokens or 1024,
            "temperature": prov.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=prov.timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return AIResponse(
            text=text,
            provider="anthropic",
            model=prov.model,
            tokens_in=data.get("usage", {}).get("input_tokens"),
            tokens_out=data.get("usage", {}).get("output_tokens"),
            latency_ms=int((time.time() - start) * 1000),
        )

    async def _call_google(self, prov: AIProviderConfig, prompt: str, start: float) -> AIResponse:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for Google provider")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{prov.model}:generateContent?key={prov.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": prov.temperature,
                "maxOutputTokens": prov.max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=prov.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return AIResponse(
            text=text,
            provider="google",
            model=prov.model,
            latency_ms=int((time.time() - start) * 1000),
        )

    async def _call_ollama(self, prov: AIProviderConfig, prompt: str, start: float) -> AIResponse:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for Ollama provider")

        payload = {
            "model": prov.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": prov.temperature},
        }
        async with httpx.AsyncClient(timeout=prov.timeout) as client:
            resp = await client.post(
                f"{prov.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return AIResponse(
            text=data.get("response", ""),
            provider="ollama",
            model=prov.model,
            latency_ms=int((time.time() - start) * 1000),
        )

    async def _call_lm_studio(self, prov: AIProviderConfig, prompt: str, start: float) -> AIResponse:
        # LM Studio uses OpenAI-compatible API
        prov2 = AIProviderConfig(
            provider=AIProvider.OPENAI,
            api_key="lm-studio",
            base_url=prov.base_url,
            model=prov.model,
            temperature=prov.temperature,
            max_tokens=prov.max_tokens,
        )
        return await self._call_openai(prov2, prompt, start)

    async def _call_custom(self, prov: AIProviderConfig, prompt: str, start: float) -> AIResponse:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for custom provider")

        payload = {
            "model": prov.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": prov.temperature,
            "max_tokens": prov.max_tokens,
        }
        headers = {}
        if prov.api_key:
            headers["Authorization"] = f"Bearer {prov.api_key}"
        async with httpx.AsyncClient(timeout=prov.timeout) as client:
            resp = await client.post(
                f"{prov.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return AIResponse(
            text=text,
            provider="custom",
            model=prov.model,
            latency_ms=int((time.time() - start) * 1000),
        )


# Singleton instance
_ai_client: AIClient | None = None


def get_ai_client() -> AIClient:
    """Get the singleton AI client instance."""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client


def reset_ai_client() -> None:
    """Reset the AI client (useful for testing or reconfiguration)."""
    global _ai_client
    _ai_client = None
    _ai_client = AIClient()
