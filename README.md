# Logical Query Routing

Classify the question before retrieving, so questions the corpus cannot answer do not get hallucinated citations.

**Part of the [RAG Techniques](https://github.com/AbdelrahmanhAli/rag-techniques-collection) series — 15 self-contained LangChain v1 demos.**

---

## What it demonstrates

A router is the cheapest guardrail in a RAG system. Without one, *"what
is the weather in Cairo?"* still triggers a vector search, returns four
unrelated chunks, and the generator produces a confident answer with citations
that do not support it.

This app classifies each question with a typed schema:

```python
class Route(BaseModel):
    destination: Literal["document_search", "general"]
    reason: str
```

`Literal` makes the branch exhaustive and the output validatable — no substring
matching on free text. The classifier returns the route and a one-sentence
reason, which the script prints so routing decisions are auditable.

The `general` path calls the model without retrieval; the `document_search` path
runs the normal grounded pipeline.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OpenAI key
```

Then run:

```bash
python route.py "how does contextual compression work?"
python route.py "tell me a joke about vector databases"
```

`python <script> --help` lists the available flags. The first run builds a
persistent index in `./.chroma`; later runs reuse it. Delete that folder (or use
`--rebuild` where supported) after changing anything in `data/`.

## Project layout

```
.
├── route.py
├── core.py
├── conftest.py
├── data/
│   └── sample.txt
├── tests/
│   ├── test_core.py
│   └── test_route.py
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── LICENSE
```

## Tests

```bash
pip install pytest
pytest
```

The tests run entirely offline — no API key and no network. They cover the
shared helpers and the technique-specific logic, and use a deterministic fake
embedding model for the vector-store path.

## Notes

- One extra LLM call per request, but it prevents wasted embedding and generation calls.
- Add routes as your system grows: `web_search`, `sql`, `calculator`, `escalate_to_human`.
- A misroute is silent, so log the destination and reason and review the distribution in production.
- Structured output requires a model that supports tool/function calling.

## References

- [Query routing / logical routing in LangChain RAG from scratch](https://github.com/langchain-ai/rag-from-scratch)
- [Pydantic v2 models](https://docs.pydantic.dev/latest/)
- [LangChain v1 overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain integrations reference](https://reference.langchain.com/python/integrations/overview)
- [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)

## License

MIT — see [LICENSE](LICENSE).
