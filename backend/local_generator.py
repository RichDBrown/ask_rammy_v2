import os
from typing import Dict, List, Optional

import requests

class LocalGenerator:
    """Generate chatbot answers using an Ollama local model."""

    def __init__(
        self, base_url: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("LOCAL_MODERL_BASE_URL", "http://ollama:11434")
        ).rstrip("/")

        self.model = model or os.getenv("LOCAL_GENERATOR_MODEL", "llama3.2:3b")

        self.timeout = (
            timout
            if timout is not None
            else float(os.getenv(LOCAL_GENRATOR_TIMEOUT, "20"))
        )