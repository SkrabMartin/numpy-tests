"""Tiny neural character language model: pure NumPy + Python stdlib.

This is an educational feedforward neural LM, NOT a capable LLM.
Usage:
    pip install numpy
    python numpy_lm.py
    python numpy_lm.py --corpus notes.txt --steps 1200 --prompt "the "

All forward passes, gradients, training and sampling are implemented in NumPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_TEXT = ("eeg records electrical signals from the brain.\n"
                "ecg records electrical signals from the heart.\n"
                "alpha rhythms occur during relaxed wakefulness.\n"
                "machine learning finds patterns in data.\n"
                "neural networks learn representations of signals.\n") * 35


class TinyNeuralLM:
    def __init__(self, text: str, context: int = 8, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.context = context
        self.chars = sorted(set(text) | {" "})
        self.to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.ids = np.array([self.to_id[c] for c in text], dtype=np.int64)
        vocab_size = len(self.chars)
        if len(self.ids) <= context:
            raise ValueError("The training text needs more characters than context length.")
        self.parameters = {
            "E": self.rng.normal(0, 0.12, (vocab_size, 16)),
            "W1": self.rng.normal(0, 0.08, (context * 16, 64)),
            "b1": np.zeros(64),
            "W2": self.rng.normal(0, 0.08, (64, vocab_size)),
            "b2": np.zeros(vocab_size),
        }

    def train(self, steps: int = 900, batch_size: int = 64, lr: float = 0.015):
        p = self.parameters
        moment1 = {k: np.zeros_like(v) for k, v in p.items()}
        moment2 = {k: np.zeros_like(v) for k, v in p.items()}
        for step in range(1, steps + 1):
            offsets = self.rng.integers(0, len(self.ids) - self.context, batch_size)
            x = np.stack([self.ids[i:i+self.context] for i in offsets])
            y = self.ids[offsets + self.context]

            emb = p["E"][x].reshape(batch_size, -1)
            hidden = np.tanh(emb @ p["W1"] + p["b1"])
            logits = hidden @ p["W2"] + p["b2"]
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            loss = -np.log(probabilities[np.arange(batch_size), y] + 1e-12).mean()

            d_logits = probabilities.copy()
            d_logits[np.arange(batch_size), y] -= 1
            d_logits /= batch_size
            gradients = {}
            gradients["W2"] = hidden.T @ d_logits
            gradients["b2"] = d_logits.sum(axis=0)
            d_hidden = (d_logits @ p["W2"].T) * (1 - hidden**2)
            gradients["W1"] = emb.T @ d_hidden
            gradients["b1"] = d_hidden.sum(axis=0)
            gradients["E"] = np.zeros_like(p["E"])
            d_emb = (d_hidden @ p["W1"].T).reshape(batch_size, self.context, -1)
            np.add.at(gradients["E"], x, d_emb)

            # Adam optimizer, implemented from scratch.
            for key in p:
                g = np.clip(gradients[key], -2, 2)
                moment1[key] = 0.9 * moment1[key] + 0.1 * g
                moment2[key] = 0.999 * moment2[key] + 0.001 * g * g
                corrected1 = moment1[key] / (1 - 0.9**step)
                corrected2 = moment2[key] / (1 - 0.999**step)
                p[key] -= lr * corrected1 / (np.sqrt(corrected2) + 1e-8)
            if step == 1 or step % 300 == 0 or step == steps:
                print(f"step {step:4d} | loss {loss:.3f}")

    def generate(self, prompt: str = "eeg records", length: int = 160, temperature: float = 0.8) -> str:
        if temperature <= 0:
            raise ValueError("Temperature must be > 0.")
        p = self.parameters
        output = prompt
        for _ in range(length):
            context = (" " * self.context + output)[-self.context:]
            x = np.array([self.to_id.get(c, self.to_id[" "]) for c in context])
            hidden = np.tanh(p["E"][x].reshape(-1) @ p["W1"] + p["b1"])
            logits = (hidden @ p["W2"] + p["b2"]) / temperature
            prob = np.exp(logits - logits.max())
            prob /= prob.sum()
            output += self.chars[self.rng.choice(len(self.chars), p=prob)]
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="UTF-8 training text file")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--prompt", default="eeg records")
    parser.add_argument("--length", type=int, default=160)
    args = parser.parse_args()
    text = args.corpus.read_text(encoding="utf-8") if args.corpus else DEFAULT_TEXT
    lm = TinyNeuralLM(text)
    lm.train(steps=args.steps)
    print("\nGenerated (toy model; expect repetitions and errors):\n")
    print(lm.generate(args.prompt, args.length))


if __name__ == "__main__":
    main()
