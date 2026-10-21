"""Local query-orchestration layer for the RAG backend.

The local model is always used for query planning when a question is present.
Answer generation remains owned by the configured answer model.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OrchestrationDecision:
    """The retrieval query and the reason it was selected."""

    query: str
    source: str


class LocalOrchestrator:
    """Use an Ollama-compatible local model to plan a RAG query.

    The network call is bounded. Any local model failure returns the user's
    original text so the RAG path can continue serving requests.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOCAL_MODEL_BASE_URL", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("LOCAL_MODEL", "llama3.2:3b")
        self.timeout = timeout if timeout is not None else float(os.getenv("LOCAL_ORCHESTRATION_TIMEOUT", "30"))

    def decide(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> OrchestrationDecision:
        """Return a standalone retrieval query without blocking the fallback path."""
        original = (question or "").strip()
        if not original:
            return OrchestrationDecision(original, "original")

        prompt = self._planning_prompt(original, history or [])
        try:
            import requests
        except ImportError:
            print("[orchestrator] requests is unavailable; using the original query.")
            return OrchestrationDecision(original, "fallback")

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the current message as one concise, standalone HR knowledge-base "
                                "search query. If it refers to earlier turns (e.g. 'it', 'that', 'what about "
                                "part-time?', 'yes'), use the conversation to name the specific topic and "
                                "details. Keep the user's intent and include only facts stated in the "
                                "conversation. If the message is already standalone, change it as little as "
                                "possible. Output only the query, with no quotes or explanation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            planned = " ".join(str(content).split()).strip(" \"'")
            if planned and len(planned) <= 500:
                return OrchestrationDecision(planned, "local")
        except (requests.RequestException, ValueError, TypeError, AttributeError) as error:
            print(f"[orchestrator] Local planner unavailable: {error}")

        return OrchestrationDecision(original, "fallback")

    @staticmethod
    def _clean_turn(role: str, content: str) -> str:
        text = str(content).split("(regarding:", 1)[0]
        if role == "assistant":
            text = re.sub(r"\[OPTIONS:[^\]]*\]", " ", text)
            text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        return text[:300] if role == "assistant" else text[:500]

    @classmethod
    def _planning_prompt(cls, question: str, history: List[Dict[str, Any]]) -> str:
        turns = [item for item in history if isinstance(item, dict) and item.get("content")]
        # The frontend appends the current message to history before sending.
        if turns and turns[-1].get("role") == "user" and cls._clean_turn("user", turns[-1]["content"]) == cls._clean_turn("user", question):
            turns = turns[:-1]

        recent = []
        for item in turns[-6:]:
            role = item.get("role", "user")
            text = cls._clean_turn(role, item["content"])
            if text:
                recent.append(f"{role}: {text}")
                
        context = "\n".join(recent)
        return f"Conversation:\n{context}\n\nCurrent message:\n{question}" if context else question