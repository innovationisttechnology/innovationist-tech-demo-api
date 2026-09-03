from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ZizaChatSettings(BaseSettings):
    """Model + provider settings for the ziza-ai module.

    Provider API keys (e.g. ANTHROPIC_API_KEY) are read from the process
    environment by pydantic-ai itself, not from here — settings loaded by this
    class never reach os.environ, so app/main.py calls load_dotenv() to put
    .env there. Scripts that bypass main.py must do the same.

    Model names are kept configurable so the demo can swap models without code
    changes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # pydantic-ai model identifiers ("<provider>:<model>"). Defaults to Claude.
    ziza_chat_model: str = "anthropic:claude-sonnet-5"
    ziza_classifier_model: str = "anthropic:claude-haiku-4-5"

    # Local HuggingFace cross-encoder used to grade classifier outputs in the
    # eval suite (evals/). Not used on the request path.
    grader_model_id: str = "cross-encoder/nli-deberta-v3-small"

    # Local fastembed (ONNX) model that embeds knowledge-base chunks and
    # queries for the vector store. Runs on the request path, so it is kept
    # torch-free deliberately.
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"

    # Vision model that turns images into searchable descriptions at ingest
    # time. A bad caption makes a document permanently unfindable, and this is
    # off the request path, so it favours quality over latency and cost.
    # Measured on a dense text+chart screenshot: haiku-4-5 extracted every fact
    # as accurately as opus-5 for 1/12th the cost ($5 vs $65 per 1,000 images).
    # Do not pair a cheaper model with downscaled images — haiku on a 1024px
    # copy of the same screenshot hallucinated service names and figures.
    ziza_vision_model: str = "anthropic:claude-haiku-4-5"

    # Captions are cached by image content hash, scoped to the session that
    # uploaded the image and expired with it (see SESSION_TTL_SECONDS), so a
    # visitor's material never outlives their session or reaches another.

    # Where visitors are sent for anything this demo won't answer: the full
    # Ziza assistant on the main site, which does handle general questions.
    general_assistant_url: str = "https://innovationisttech.com/"

    # A public demo session is capped so one visitor cannot fill the index or
    # run up the captioning bill. Counted per distinct file or URL, not per
    # chunk, and reset when the session's chunks expire.
    max_documents_per_session: int = 10

    ziza_summary_model: str = "anthropic:claude-haiku-4-5"

    # A page summary is one chunk among many; feeding a whole long article to
    # the summariser costs input tokens without improving that one chunk.
    max_summary_input_chars: int = 24_000

    # Captioning costs one model call per image. These bound a single upload so
    # an image-heavy PDF cannot fan out unboundedly.
    max_images_per_document: int = 20
    max_concurrent_captions: int = 4

    # Optional Tavily key for the web_search tool; when unset the tool is a stub.
    tavily_api_key: str | None = None


@lru_cache
def get_ziza_settings() -> ZizaChatSettings:
    return ZizaChatSettings()


ziza_settings = get_ziza_settings()
