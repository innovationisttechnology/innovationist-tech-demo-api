

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from app.ziza_chat.config import ziza_settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding


# bge-small-en-v1.5 truncates beyond this silently — no error, the tail is
# simply unsearchable. Character count can't predict it: 1200 chars is ~196
# tokens of prose but ~450 of dense code or an OCR-heavy caption.
MAX_EMBED_TOKENS = 512


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class FastEmbedEmbedder:

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: TextEmbedding | None = None

    def _get_model(self) -> "TextEmbedding":
        model = self._model
        if model is None:
            from fastembed import TextEmbedding

            model = TextEmbedding(model_name=self._model_id)
            self._model = model
        return model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._get_model().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        query_vector = next(iter(self._get_model().query_embed(text)))
        return [float(component) for component in query_vector.tolist()]

    def count_tokens(self, text: str) -> int:
        tokenizer = self._get_model().model.tokenizer  # type: ignore[attr-defined]
        return int(len(tokenizer.encode(text).ids))


@lru_cache
def get_embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(ziza_settings.embedding_model_id)


def get_embedding_dimensions(model_id: str) -> int:
    """Look up vector dimensions from fastembed's model registry — no model
    download or load needed, so it is safe to call at app startup."""
    from fastembed import TextEmbedding

    for description in TextEmbedding.list_supported_models():
        if str(description["model"]).lower() == model_id.lower():
            return int(description["dim"])
    raise ValueError(f"Model {model_id!r} is not in fastembed's supported list")
