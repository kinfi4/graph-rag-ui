from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: Literal["ollama", "openrouter"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    # One context size for every call: Ollama reloads the model when num_ctx changes.
    ollama_num_ctx: int = 8192
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.5-flash"

    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dim: int = 1024

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    data_dir: Path = ROOT / "data"

    @property
    def llm_name(self) -> str:
        if self.llm_provider == "openrouter":
            return f"openrouter/{self.openrouter_model}"
        return f"ollama/{self.ollama_model}"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


settings = Settings()
