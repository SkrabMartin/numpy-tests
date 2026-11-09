"""NumPy-only retrieval + extractive question answering (RAG building block).

Install: pip install numpy
Try: python numpy_rag.py --query "What frequencies are used for EEG alpha waves?"
Use your own data: python numpy_rag.py --docs ./documents
Only .txt and .md files are read, recursively.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_DOCUMENTS = {
    "eeg.txt": "Electroencephalography (EEG) measures electrical activity of the brain using scalp electrodes. Alpha rhythms are typically around 8 to 13 Hz and commonly appear during relaxed wakefulness with closed eyes.",
    "ecg.txt": "Electrocardiography (ECG) records electrical activity of the heart. A normal ECG typically includes the P wave, QRS complex and T wave. The QRS complex reflects ventricular depolarization.",
    "ml.txt": "Machine learning models learn patterns from training data. Overfitting occurs when a model learns training-set noise and fails to generalize to unseen samples. Cross-validation helps assess generalization.",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold(), flags=re.UNICODE)


def read_documents(directory: Path) -> dict[str, str]:
    documents = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                documents[str(path.relative_to(directory))] = content
    return documents


class NumpyRetriever:
    def __init__(self, documents: dict[str, str]):
        if not documents:
            raise ValueError("No nonempty documents were provided.")
        self.names = list(documents)
        self.texts = [documents[name] for name in self.names]
        tokens = [tokenize(text) for text in self.texts]
        self.vocab = {word: i for i, word in enumerate(sorted(set().union(*(set(t) for t in tokens))))}
        self.idf = np.ones(len(self.vocab), dtype=np.float64)
        for word, col in self.vocab.items():
            df = sum(word in set(t) for t in tokens)
            self.idf[col] = np.log((1 + len(tokens)) / (1 + df)) + 1
        self.matrix = np.stack([self._vector(t) for t in tokens])

    def _vector(self, tokens: list[str]) -> np.ndarray:
        vector = np.zeros(len(self.vocab), dtype=np.float64)
        counts = Counter(tokens)
        for word, count in counts.items():
            if word in self.vocab:
                vector[self.vocab[word]] = (1 + np.log(count)) * self.idf[self.vocab[word]]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def retrieve(self, question: str, top_k: int = 2) -> list[tuple[str, str, float]]:
        query_vector = self._vector(tokenize(question))
        similarities = self.matrix @ query_vector  # cosine similarity
        if not np.any(query_vector):
            return []
        indices = np.argsort(-similarities)[:top_k]
        return [(self.names[i], self.texts[i], float(similarities[i]))
                for i in indices if similarities[i] > 0]

    def answer_extractive(self, question: str, top_k: int = 2) -> str:
        matches = self.retrieve(question, top_k)
        if not matches:
            return "No matching passage found in the documents."
        candidate_sentences = []
        qv = self._vector(tokenize(question))
        for name, text, _ in matches:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                score = float(self._vector(tokenize(sentence)) @ qv)
                candidate_sentences.append((score, name, sentence))
        candidate_sentences.sort(reverse=True)
        _, name, sentence = candidate_sentences[0]
        return f"{sentence}\n[source: {name}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, help="Directory of your .txt/.md knowledge base")
    parser.add_argument("--query", help="Single question; omit for interactive mode")
    args = parser.parse_args()
    documents = read_documents(args.docs) if args.docs else DEFAULT_DOCUMENTS
    retriever = NumpyRetriever(documents)
    print(f"Loaded {len(documents)} document(s); vocabulary = {len(retriever.vocab)} words.")
    if args.query:
        print(retriever.answer_extractive(args.query))
    else:
        print("Ask a question (or press Enter to quit):")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            print(retriever.answer_extractive(question), "\n")


if __name__ == "__main__":
    main()
