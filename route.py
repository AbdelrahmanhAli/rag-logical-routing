"""Logical query routing.

Not every question needs the document index. Classify the question first, then
send it to retrieval or to a plain model call. Routing like this keeps the
vector store honest: a question about the weather should not return three
unrelated chunks and a hallucinated citation.

The route is a typed ``Literal`` rather than a substring match, so the branch is
exhaustive and testable.
"""

from __future__ import annotations

import argparse
from typing import Literal

from pydantic import BaseModel, Field

import core


class Route(BaseModel):
    """Where a question should be answered."""

    destination: Literal["document_search", "general"] = Field(
        description=(
            "document_search when the answer should come from the local .txt/.md "
            "files; general for everything else."
        )
    )
    reason: str = Field(description="One short sentence explaining the choice.")


ROUTE_PROMPT = (
    "Classify the user's question for a RAG router.\n"
    "Use document_search when the question is about the content of the supplied "
    "documents (retrieval, indexing, chunking, embeddings, RAG techniques).\n"
    "Use general for small talk, current events, or anything the corpus cannot "
    "plausibly answer.\n\n"
    "Question: {question}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*")
    parser.add_argument("-k", type=int, default=core.DEFAULT_K)
    args = parser.parse_args()

    question = " ".join(args.question).strip() or core.question_from_args(["prog"])
    model = core.chat_model()

    route = model.with_structured_output(Route).invoke(ROUTE_PROMPT.format(question=question))
    print(f"Route : {route.destination}")
    print(f"Reason: {route.reason}\n")

    if route.destination == "document_search":
        documents = core.get_vectorstore().similarity_search(question, k=args.k)
        print(core.answer(question, documents, model=model, show_sources=True))
    else:
        print(core.plain_answer(question, model=model))


if __name__ == "__main__":
    main()
