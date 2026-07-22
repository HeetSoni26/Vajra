from __future__ import annotations

import httpx


class FoundationLMClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def complete(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        payload = {"prompt": prompt, "max_tokens": max_tokens, **kwargs}
        response = httpx.post(f"{self.base_url}/v1/completions", json=payload, headers=self.headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = {"messages": messages, **kwargs}
        response = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self.headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
