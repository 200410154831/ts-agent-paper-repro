import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import yaml


class DeepSeekClient:
    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_sec: int = 60,
        max_retries: int = 5,
        retry_backoff_sec: float = 2.0,
        default_json_output: bool = True,
        empty_content_retry_limit: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is missing.")
        # Requests encodes headers with latin-1. Non-ascii API keys (for example, placeholder text
        # like '你的key') will fail during header encoding with an unclear runtime error.
        try:
            self.api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise ValueError(
                "DEEPSEEK_API_KEY contains non-ASCII characters. "
                "Please use the real DeepSeek API key string (usually starts with 'sk-')."
            ) from e
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.default_json_output = default_json_output
        self.empty_content_retry_limit = empty_content_retry_limit

    @classmethod
    def from_yaml(cls, path: str, api_key: Optional[str] = None) -> "DeepSeekClient":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(api_key=api_key, **cfg)

    def generate(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, Any]] = None) -> str:
        effective_response_format = response_format
        if effective_response_format is None and self.default_json_output:
            effective_response_format = {"type": "json_object"}

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if effective_response_format:
            payload["response_format"] = effective_response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_err: Optional[Exception] = None
        for retry in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.retry_backoff_sec * (retry + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                if content is None:
                    content = ""
                if str(content).strip():
                    return str(content)
                if retry < min(self.max_retries - 1, self.empty_content_retry_limit):
                    time.sleep(self.retry_backoff_sec * (retry + 1))
                    continue
                raise RuntimeError("DeepSeek returned empty content.")
            except Exception as e:
                last_err = e
                time.sleep(self.retry_backoff_sec * (retry + 1))
        raise RuntimeError(f"DeepSeek API failed after retries: {last_err}")
