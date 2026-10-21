"""LLM + embedding clients.

Chat goes through a small provider switch (local Ollama or hosted OpenRouter); every call
returns a validated Pydantic object. Embeddings always come from Ollama, because the whole
graph is indexed with one embedding model.
"""

import json
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from synapse_rag.config import settings


class LLMError(RuntimeError):
    pass


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic JSON schema acceptable for OpenAI-style strict structured output."""
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema.get("properties", {}))
    for value in schema.values():
        if isinstance(value, dict):
            _strict_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strict_schema(item)
    return schema


class ChatClient:
    name: str

    def _complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError

    def structured[T: BaseModel](
        self,
        prompt: str,
        output_type: type[T],
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        retries: int = 1,
    ) -> T:
        """Ask for JSON matching `output_type`; on invalid output, retry with the error shown.

        `schema` overrides the schema sent to the model (e.g. a stricter, bounded one).
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        schema = schema or output_type.model_json_schema()

        last_error = ""
        for _ in range(retries + 1):
            raw = self._complete(
                messages, schema, output_type.__name__, temperature, max_tokens
            )
            try:
                return output_type.model_validate_json(raw)
            except ValidationError as exc:
                last_error = str(exc)
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Your JSON was invalid:\n{last_error}\n"
                        "Return corrected JSON only.",
                    },
                ]
        raise LLMError(f"{self.name}: no valid {output_type.__name__} JSON: {last_error}")


class OllamaChat(ChatClient):
    def __init__(self, base_url: str, model: str, num_ctx: int) -> None:
        self.name = f"ollama/{model}"
        self._model = model
        self._num_ctx = num_ctx
        self._http = httpx.Client(base_url=base_url, timeout=300)

    def _complete(self, messages, schema, schema_name, temperature, max_tokens) -> str:
        response = self._http.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                # Thinking mode is ~5x slower on a 4B model; the schemas put a
                # `reasoning` field first where step-by-step thinking is needed.
                "think": False,
                "format": schema,
                "keep_alive": "60m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": self._num_ctx,
                    "num_predict": max_tokens,
                },
            },
        )
        if response.status_code != 200:
            raise LLMError(f"{self.name}: HTTP {response.status_code}: {response.text[:300]}")
        return response.json()["message"]["content"]


class OpenRouterChat(ChatClient):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty")
        self.name = f"openrouter/{model}"
        self._model = model
        self._http = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            timeout=120,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "Synapse GraphRAG"},
        )

    def _complete(self, messages, schema, schema_name, temperature, max_tokens) -> str:
        response = self._http.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": _strict_schema(json.loads(json.dumps(schema))),
                    },
                },
            },
        )
        if response.status_code != 200:
            raise LLMError(f"{self.name}: HTTP {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"] or ""


class Embedder:
    def __init__(self, base_url: str, model: str, batch_size: int = 32) -> None:
        self.name = f"ollama/{model}"
        self._model = model
        self._batch_size = batch_size
        self._http = httpx.Client(base_url=base_url, timeout=300)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._http.post(
                "/api/embed",
                json={"model": self._model, "input": batch, "keep_alive": "60m"},
            )
            if response.status_code != 200:
                raise LLMError(f"{self.name}: HTTP {response.status_code}: {response.text[:300]}")
            vectors.extend(response.json()["embeddings"])
        return vectors


@lru_cache
def get_chat() -> ChatClient:
    if settings.llm_provider == "openrouter":
        return OpenRouterChat(settings.openrouter_api_key, settings.openrouter_model)
    return OllamaChat(settings.ollama_url, settings.ollama_model, settings.ollama_num_ctx)


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(settings.ollama_url, settings.embedding_model)
