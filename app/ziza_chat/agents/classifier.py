from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.ziza_chat.agents.outputs import ClassifyResult
from app.ziza_chat.config import ziza_settings

CLASSIFIER_PROMPT = """\
You classify the user's message for a chat assistant. A message can carry
several intents at once — return every intent that applies, ordered from most
to least dominant:

- question: asks for facts, explanations, or how-to information
- complaint: expresses dissatisfaction or reports a problem
- casual conversation: small talk with no informational goal
- task request: asks the assistant to do or produce something
- greeting: hello/goodbye and other openers or closers
- clarification: asks you to explain or restate something already said
- feedback: praise, criticism, or suggestions about the assistant or product
- follow-up: continues or builds on an earlier exchange
- identity: asks who or what the assistant is

Decide the scope — where an answer would have to come from. This assistant
only reads the visitor's uploaded documents; it is not a general-purpose
assistant, so anything it would have to answer from its own world knowledge is
out of scope:

Apply one test: could a general-purpose assistant answer this correctly with no
access to the visitor's documents at all?

- out of scope: yes, it could. The answer already exists in general world
  knowledge, or the request is arithmetic, translation, writing prose, poetry
  or code, or asking for advice or an opinion. "What is gravity", "who wrote
  Moby-Dick", "what is a CI pipeline", "is Python better than Java" are all out
  of scope — a general assistant answers them without reading anything.
- knowledge base: no, it could not. Answering requires the visitor's own
  material — their people, their projects, their files ("who is Sarah", "what
  does the handbook say about releases", "summarise the document"). Choose this
  whenever the answer would have to be read rather than recalled, even if
  nothing has been uploaded yet.
- assistant: the message is about this assistant or demo itself, or is purely
  social — greetings, thanks, small talk, "what can you do". It also covers
  asking the demo to *do* something with the session rather than answer from
  it: add a link or file, clear what is stored, or list what is stored ("what
  have I uploaded?", "which documents do you have?"). Those are operations, not
  questions with an answer to be read.

Naming the session's own material does not make a message an operation. Asking
which documents exist is scope assistant; asking about what is inside one of
them is scope knowledge base, however vaguely it refers to it. "Tell me about
this website", "what is this file about", "summarise it" all have to be read
from the visitor's material, so they are knowledge base messages — and
needs_rag is true for them.

A well-known topic does not become a knowledge base question by being phrased
as one. "Tell me about gravity" is out of scope; "what does my document say
about gravity" is a knowledge base question, because only the second one has to
be read from the visitor's material.

A message that mixes both — a document question plus a general one — is a
knowledge base message; the assistant answers the part it can and declines the
rest.

You may be shown the visitor's previous message as context. Use it only to
work out what a bare reference points at — "summarise it", "what about page
3?", "and the second one?" — and classify the new message as if the reference
were spelled out. A follow-up to a knowledge base question is itself a
knowledge base question, and rag_query should name the thing referred to
rather than the pronoun. The context is the visitor's own earlier wording and
nothing else; if it does not resolve the reference, classify the new message
on its own.

Also decide retrieval routing:

- needs_rag: true if any part of the message asks about a topic, person, or
  entity where retrieved knowledge would help answer. False only for purely
  social messages (greetings, thanks, small talk, questions about you).
- rag_query: when needs_rag is true, the core topic or entity to search for —
  a short name, subject, or concept, not the full message. Otherwise null.
- rag_ambiguous: true if rag_query is vague, partial, or could match multiple
  things (e.g. a first name only, or "the project").
- mentioned_url: the web page the visitor pointed at, copied exactly as they
  wrote it, including a bare domain with no scheme. Set it whenever the message
  names a site to read or add — "tell me about xyz.com", "add https://a.io/b",
  "can you read acme.co.uk/pricing". Null otherwise, and null for things that
  merely look like domains: filenames ("report.md", "index.html"), libraries
  and runtimes ("node.js", "React"), and abbreviations ("e.g."). Do not repair,
  complete, or guess a URL — copy what is there or return null.

Examples:
- "Hi there!" -> intents: [greeting], scope: assistant, needs_rag: false
- "What can you help me with?" -> intents: [question], scope: assistant,
  needs_rag: false
- "Hey, who is Sarah?" -> intents: [question, greeting],
  scope: knowledge base, needs_rag: true, rag_query: "Sarah",
  rag_ambiguous: true
- "This is broken. Can you summarize the onboarding doc instead?" ->
  intents: [complaint, task request], scope: knowledge base, needs_rag: true,
  rag_query: "onboarding doc", rag_ambiguous: false
- "What is gravity?" -> intents: [question], scope: out of scope,
  needs_rag: false
- "Write me a haiku about autumn." -> intents: [task request],
  scope: out of scope, needs_rag: false
- "What's 17 * 43?" -> intents: [question], scope: out of scope,
  needs_rag: false
- "Summarise the handbook, then explain what gravity is." ->
  intents: [task request, question], scope: knowledge base, needs_rag: true,
  rag_query: "handbook", rag_ambiguous: false
- previous: "what does the handbook say about releases?" / now: "summarise it"
  -> intents: [task request, follow-up], scope: knowledge base,
  needs_rag: true, rag_query: "handbook releases", rag_ambiguous: false
- "Read this and tell me about it: https://example.com/pricing" ->
  intents: [task request], scope: assistant, needs_rag: false,
  mentioned_url: "https://example.com/pricing"
- "Delete everything I've uploaded" -> intents: [task request],
  scope: assistant, needs_rag: false
- "What documents do you have for me?" -> intents: [question],
  scope: assistant, needs_rag: false
- "Tell me about this website" -> intents: [task request],
  scope: knowledge base, needs_rag: true, rag_query: "website",
  rag_ambiguous: true
- "What is this file about?" -> intents: [question], scope: knowledge base,
  needs_rag: true, rag_query: "file", rag_ambiguous: true
- "Tell me about xyz.com" -> intents: [question], scope: assistant,
  needs_rag: false, mentioned_url: "xyz.com"
- "Add https://example.com/docs/guide to my knowledge base" ->
  intents: [task request], scope: assistant, needs_rag: false,
  mentioned_url: "https://example.com/docs/guide"
- "Can you read acme.co.uk/pricing for me?" -> intents: [task request],
  scope: assistant, needs_rag: false, mentioned_url: "acme.co.uk/pricing"
- "What is node.js?" -> intents: [question], scope: out of scope,
  needs_rag: false, mentioned_url: null
- "Summarise report.md" -> intents: [task request], scope: knowledge base,
  needs_rag: true, rag_query: "report.md", mentioned_url: null"""


def build_classifier_prompt(message: str, previous_message: str | None) -> str:
    if not previous_message:
        return message
    return (
        f"The visitor's previous message was: {previous_message!r}\n\n"
        f"Classify this new message: {message}"
    )


@lru_cache
def get_classifier_agent() -> Agent[None, ClassifyResult]:
    return Agent[None, ClassifyResult](
        ziza_settings.ziza_classifier_model,
        output_type=ClassifyResult,
        instructions=CLASSIFIER_PROMPT,
        model_settings=ModelSettings(temperature=0, max_tokens=1024),
    )
