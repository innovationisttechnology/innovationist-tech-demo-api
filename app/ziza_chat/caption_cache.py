import hashlib
import io
import logging
from datetime import datetime

from beanie import Document
from pydantic import Field
from pymongo import IndexModel

from app.core.config import settings
from app.core.utils.time import utc_now

logger = logging.getLogger(__name__)

# Below this standard deviation in grayscale, an image is a solid fill or a
# smooth gradient — a background, a rule, or a spacer. Nothing to retrieve.
MIN_PIXEL_STDDEV = 8.0
MIN_IMAGE_EDGE_PIXELS = 64


class CachedCaption(Document):
    session_id: str
    image_hash: str
    caption: str
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "image_captions"
        indexes = [
            IndexModel(
                [("session_id", 1), ("image_hash", 1)],
                unique=True,
                name="uniq_session_image_hash",
            ),
            # Same lifetime as the chunks these describe: a caption holds the
            # image's content as text, so it must not outlive the knowledge
            # base it was built for.
            IndexModel(
                [("created_at", 1)],
                name="ttl_captions",
                expireAfterSeconds=settings.session_ttl_seconds,
            ),
        ]


def hash_image(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def carries_no_information(data: bytes) -> bool:
    """True for images with nothing worth indexing — solid fills, gradients,
    spacers — so they never reach the model."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            if min(image.size) < MIN_IMAGE_EDGE_PIXELS:
                return True
            grayscale = image.convert("L")
            _, stddev = _mean_and_stddev(grayscale.histogram())
            return stddev < MIN_PIXEL_STDDEV
    except Exception as failure:
        logger.warning("Could not inspect image, sending it anyway: %s", failure)
        return False


def _mean_and_stddev(histogram: list[int]) -> tuple[float, float]:
    total = sum(histogram)
    if total == 0:
        return 0.0, 0.0
    mean = sum(value * count for value, count in enumerate(histogram)) / total
    variance = (
        sum(count * (value - mean) ** 2 for value, count in enumerate(histogram)) / total
    )
    return mean, variance**0.5


async def get_cached_caption(session_id: str, image_hash: str) -> str | None:
    cached = await CachedCaption.find_one(
        CachedCaption.session_id == session_id,
        CachedCaption.image_hash == image_hash,
    )
    return cached.caption if cached else None


async def store_caption(session_id: str, image_hash: str, caption: str) -> None:
    try:
        await CachedCaption(
            session_id=session_id, image_hash=image_hash, caption=caption
        ).insert()
    except Exception as failure:
        # A concurrent upload of the same image can win the unique index; the
        # caption is already stored either way.
        logger.debug("Caption for %s not cached: %s", image_hash[:12], failure)


async def clear_captions(session_id: str) -> int:
    result = await CachedCaption.find(
        CachedCaption.session_id == session_id
    ).delete()
    return int(result.deleted_count) if result else 0
