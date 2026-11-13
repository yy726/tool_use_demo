"""Local knowledge base behind `knowledge_search`: HotpotQA paragraphs + BM25.

The corpus is built from the paragraphs that ship with each HotpotQA question in
the "distractor" setting (2 gold paragraphs + 8 distractors per question). The
scan order is the dataset's own order, so the same `n_train` / `n_val` always
yields the same corpus and the same doc ids.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

TOP_K = 3
MAX_PASSAGE_WORDS = 90


def truncate_words(text: str, max_words: int = MAX_PASSAGE_WORDS) -> str:
    words = text.split()
    return text.strip() if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


@dataclass
class Passage:
    doc_id: str
    title: str
    text: str

    def public(self) -> dict:
        """What the tool returns to the model (text is truncated to save context)."""
        return {"doc_id": self.doc_id, "title": self.title, "text": truncate_words(self.text)}


class Corpus:
    """Deduplicated passages with a BM25 index (built lazily on first search)."""

    def __init__(self, passages: Iterable[Passage] = ()):
        self.passages: list[Passage] = []
        self.by_title: dict[str, Passage] = {}
        self.by_id: dict[str, Passage] = {}
        self._retriever = None
        for p in passages:
            self._add(p)

    # --------------------------------------------------------------- building
    def _add(self, passage: Passage) -> None:
        self.passages.append(passage)
        self.by_title[passage.title] = passage
        self.by_id[passage.doc_id] = passage
        self._retriever = None

    def add(self, title: str, text: str) -> Passage:
        """Add a paragraph, keeping the first text seen for a given title."""
        existing = self.by_title.get(title)
        if existing is not None:
            return existing
        passage = Passage(doc_id=f"D{len(self.passages):06d}", title=title, text=" ".join(text.split()))
        self._add(passage)
        return passage

    def add_hotpot_row(self, row: dict) -> dict[str, str]:
        """Add all 10 paragraphs of one HotpotQA row; return {title: doc_id}."""
        ctx = row["context"]
        ids = {}
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            ids[title] = self.add(title, "".join(sentences)).doc_id
        return ids

    # ---------------------------------------------------------------- search
    def _ensure_index(self):
        if self._retriever is None:
            import bm25s

            retriever = bm25s.BM25()
            texts = [f"{p.title}. {p.text}" for p in self.passages]
            retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
            self._retriever = retriever
        return self._retriever

    def search(self, query: str, k: int = TOP_K) -> list[Passage]:
        import bm25s

        query = str(query).strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        if not self.passages:
            return []
        retriever = self._ensure_index()
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(k, len(self.passages))
        ids, scores = retriever.retrieve(tokens, k=k, show_progress=False)
        return [self.passages[int(i)] for i, s in zip(ids[0], scores[0]) if s > 0]

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps({"doc_id": p.doc_id, "title": p.title, "text": p.text}, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Corpus":
        with open(path, encoding="utf-8") as f:
            return cls(Passage(**json.loads(line)) for line in f if line.strip())

    def __len__(self) -> int:
        return len(self.passages)


def load_hotpot_rows(n_train: int, n_val: int) -> tuple[list[dict], list[dict]]:
    """Stream the first `n_train` train rows and `n_val` validation rows of HotpotQA (distractor)."""
    import itertools

    from datasets import load_dataset

    def take(split: str, n: int) -> list[dict]:
        if n <= 0:
            return []
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
        return list(itertools.islice(ds, n))

    return take("train", n_train), take("validation", n_val)
