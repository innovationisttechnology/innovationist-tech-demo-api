from typing import Callable


def _split_long_paragraph(
    paragraph: str, max_chars: int, overlap_chars: int
) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(paragraph):
        end = start + max_chars
        pieces.append(paragraph[start:end])
        if end >= len(paragraph):
            break
        start = end - overlap_chars
    return pieces


def chunk_text(
    text: str, max_chars: int = 1200, overlap_chars: int | None = None
) -> list[str]:
    if overlap_chars is None:
        overlap_chars = max_chars // 8
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be greater than overlap_chars")

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]

    chunks: list[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars, overlap_chars))
            continue
        candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        if len(candidate) > max_chars:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def enforce_token_limit(
    chunks: list[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_chars: int = 0,
) -> list[str]:
    """Re-split any chunk whose token count exceeds the embedder's limit.

    Chunking by characters can't predict token count — dense text tokenizes
    two to three times heavier than prose — and the embedder truncates silently
    rather than erroring, so an over-long chunk loses its tail with no signal.
    """
    limited: list[str] = []
    for chunk in chunks:
        limited.extend(
            _split_to_token_limit(chunk, count_tokens, max_tokens, overlap_chars)
        )
    return limited


def _split_to_token_limit(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_chars: int,
) -> list[str]:
    if len(text) < 2 or count_tokens(text) <= max_tokens:
        return [text]
    midpoint = len(text) // 2
    overlap = min(overlap_chars, midpoint // 2)
    return _split_to_token_limit(
        text[:midpoint], count_tokens, max_tokens, overlap_chars
    ) + _split_to_token_limit(
        text[midpoint - overlap :], count_tokens, max_tokens, overlap_chars
    )
