from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryImage, UserContent

from app.ziza_chat.config import ziza_settings

VISION_PROMPT = """\
You describe images so they can be found later by semantic search over the
text you produce. The description IS the search index — anything you leave out
is unfindable, and anything you invent becomes a false match.

Write for the question someone would later ask about this image. Transcribe
text exactly rather than summarising it: labels, headings, axis titles, legend
entries, and values are usually what a query matches on. For charts and tables,
read off the actual figures. Do not describe visual styling (colours, fonts,
layout) unless the styling carries the meaning.

If the image is decorative or carries no information worth retrieving, say so
plainly in the summary and leave the other fields empty.

Keep the whole description under 200 words — longer descriptions are truncated
before they reach the index."""


class ImageDescription(BaseModel):
    summary: str = Field(
        description="What the image is and what it conveys, in 1-3 sentences."
    )
    visible_text: str = Field(
        default="",
        description="Text visible in the image, transcribed verbatim. Empty if none.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="People, products, systems, or organisations shown or named.",
    )
    data_points: list[str] = Field(
        default_factory=list,
        description="Concrete values read from charts or tables, e.g. 'Q3 revenue 1.6M'.",
    )

    def to_search_text(self) -> str:
        sections = [self.summary]
        if self.visible_text:
            sections.append(f"Text shown in the image: {self.visible_text}")
        if self.entities:
            sections.append(f"Shown or referenced: {', '.join(self.entities)}")
        if self.data_points:
            sections.append(f"Values shown: {'; '.join(self.data_points)}")
        return "\n\n".join(sections)


@lru_cache
def get_vision_agent() -> Agent[None, ImageDescription]:
    return Agent[None, ImageDescription](
        ziza_settings.ziza_vision_model,
        output_type=ImageDescription,
        instructions=VISION_PROMPT,
    )


async def describe_image(data: bytes, media_type: str) -> ImageDescription:
    # noinspection PyArgumentList,PyTypeChecker
    prompt: list[UserContent] = [
        "Describe this image for a search index.",
        BinaryImage(data, media_type=media_type),
    ]
    result = await get_vision_agent().run(prompt)
    return result.output
