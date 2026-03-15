"""
Ollama API client for local LLM inference (Mistral, Llama3, etc.).
Ollama must be running locally: https://ollama.com
Pull a model first: `ollama pull mistral` or `ollama pull llama3`
"""

import requests
import json
from typing import Generator


OLLAMA_BASE = "http://localhost:11434"


class OllamaClient:
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.base = OLLAMA_BASE

    def is_available(self) -> tuple[bool, str]:
        """Check if Ollama is running and the model is pulled."""
        try:
            resp = requests.get(f"{self.base}/api/tags", timeout=5)
            resp.raise_for_status()
            raw = resp.json().get("models", [])
            # Match both full tag (mistral:7b-instruct-q4_0) and base name (mistral)
            models_full = [m["name"] for m in raw]
            models_base = [m["name"].split(":")[0] for m in raw]
            if self.model not in models_full and self.model not in models_base:
                available = ", ".join(models_full) if models_full else "none"
                return False, (
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: `ollama pull {self.model}`\n"
                    f"Available models: {available}"
                )
            return True, "ok"
        except requests.ConnectionError:
            return False, (
                "Ollama is not running. Start it with: `ollama serve`\n"
                "Install Ollama from https://ollama.com"
            )
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion from Ollama.
        Yields text tokens as they arrive.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }

        with requests.post(
            f"{self.base}/api/chat",
            json=payload,
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.5) -> str:
        """Non-streaming completion — returns full response string."""
        return "".join(self.stream_chat(system_prompt, user_prompt, temperature))