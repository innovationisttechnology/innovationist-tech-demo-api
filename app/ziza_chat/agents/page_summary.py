from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.ziza_chat.config import ziza_settings

SUMMARY_PROMPT = """\
You write a high-level description of a web page for a search index that also
holds the page's own text, chunk by chunk.

Those chunks already answer detailed questions. Your job is the questions they
answer badly — what this page is, who it is for, what someone would come here
to do, and how its parts fit together. Cover the whole page, including sections
a single excerpt would miss.

Describe only what is on the page. Do not add background knowledge, do not
speculate about the wider site, and do not invent detail to fill a field.
Prefer the page's own terminology over synonyms, because queries are matched
against your words. If the page is an error, a login wall, or has no real
content, say exactly that in the summary and leave the other fields empty.

Keep the whole description under 250 words."""


class PageSummary(BaseModel):
    summary: str = Field(
        description="What this page is and what it covers, in 3-6 sentences."
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Main subjects covered, in the page's own wording.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="People, products, organisations, or systems named on the page.",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Specific claims, figures, or definitions stated on the page.",
    )

    def to_search_text(self) -> str:
        sections = [self.summary]
        if self.topics:
            sections.append(f"Topics covered: {', '.join(self.topics)}")
        if self.entities:
            sections.append(f"Mentioned: {', '.join(self.entities)}")
        if self.key_facts:
            sections.append(f"Key points: {'; '.join(self.key_facts)}")
        return "\n\n".join(sections)


@lru_cache
def get_page_summary_agent() -> Agent[None, PageSummary]:
    return Agent[None, PageSummary](
        ziza_settings.ziza_summary_model,
        output_type=PageSummary,
        instructions=SUMMARY_PROMPT,
    )


async def summarize_page(title: str, url: str, text: str) -> PageSummary:
    heading = f"Page title: {title}\nURL: {url}\n\n" if title else f"URL: {url}\n\n"
    result = await get_page_summary_agent().run(
        f"{heading}{text[: ziza_settings.max_summary_input_chars]}"
    )
    return result.output
