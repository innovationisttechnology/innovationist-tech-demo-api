from dataclasses import dataclass, field


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedText:
    text: str
    locator: str | None = None


@dataclass(frozen=True)
class ExtractedImage:
    data: bytes
    media_type: str
    locator: str | None = None


@dataclass(frozen=True)
class LoadedDocument:
    texts: list[ExtractedText] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}
