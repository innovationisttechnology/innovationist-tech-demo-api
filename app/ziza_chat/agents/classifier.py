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
  social — greetings, thanks, small talk, "what can you do".

A well-known topic does not become a knowledge base question by being phrased
as one. "Tell me about gravity" is out of scope; "what does my document say
about gravity" is a knowledge base question, because only the second one has to
be read from the visitor's material.

A message that mixes both — a document question plus a general one — is a
knowledge base message; the assistant answers the part it can and declines the
rest.

Also decide retrieval routing:

- needs_rag: true if any part of the message asks about a topic, person, or
  entity where retrieved knowledge would help answer. False only for purely
  social messages (greetings, thanks, small talk, questions about you).
- rag_query: when needs_rag is true, the core topic or entity to search for —
  a short name, subject, or concept, not the full message. Otherwise null.
- rag_ambiguous: true if rag_query is vague, partial, or could match multiple
  things (e.g. a first name only, or "the project").

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
  rag_query: "handbook", rag_ambiguous: false"""


@lru_cache
def get_classifier_agent() -> Agent[None, ClassifyResult]:
    return Agent[None, ClassifyResult](
        ziza_settings.ziza_classifier_model,
        output_type=ClassifyResult,
        instructions=CLASSIFIER_PROMPT,
        model_settings=ModelSettings(temperature=0, max_tokens=1024),
    )
