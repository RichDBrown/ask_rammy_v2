"""Ollama-backed final answer generation for the RAG backend."""

import os
from typing import Any, Dict, List, Tuple

import requests


class LocalGenerator:
    """Generate final chatbot responses through an Ollama-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("LOCAL_MODEL_BASE_URL", "http://ollama:11434")
        ).rstrip("/")
        self.model = model or os.getenv("LOCAL_MODEL", "llama3.2:3b")
        self.timeout = timeout if timeout is not None else float(
            os.getenv("LOCAL_GENERATION_TIMEOUT", "60")
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, int]]:
        """Return the generated answer and normalized token usage."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload: Dict[str, Any] = response.json()
        message = payload.get("message", {})
        answer = str(message.get("content", "") or "").strip()

        prompt_tokens = int(payload.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(payload.get("eval_count", 0) or 0)
        tokens = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

        return answer, tokens
