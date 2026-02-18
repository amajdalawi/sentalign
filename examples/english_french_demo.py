"""Demo: align small English/French sentence lists and print scores."""

from __future__ import annotations

import hashlib

import numpy as np

from sentalign import sentalign


class KeywordHashEncoder:
    """Simple deterministic encoder suitable for local smoke tests."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, sentences):
        vectors = []
        for sentence in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in sentence.lower().split():
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
            vectors.append(vec)
        return np.vstack(vectors)


def main():
    english = [
        "Hello everyone.",
        "Today we are testing sentence alignment.",
        "The weather is beautiful.",
        "I will travel to Paris next week.",
    ]

    french = [
        "Bonjour à tous.",
        "Aujourd'hui, nous testons l'alignement de phrases.",
        "Le temps est magnifique.",
        "Je voyagerai à Paris la semaine prochaine.",
    ]

    result = sentalign(english, french, encoder=KeywordHashEncoder())

    print(f"Overall alignment quality score: {result.overall_score:.4f}")
    print(f"Average raw alignment cost: {result.average_alignment_score:.4f}\n")

    for i, block in enumerate(result.alignments, start=1):
        print(
            f"[{i}] src{block.src_indices} -> tgt{block.tgt_indices} "
            f"(cost={block.score:.4f})"
        )
        print("   EN:", " ".join(block.src_sentences) if block.src_sentences else "<deletion>")
        print("   FR:", " ".join(block.tgt_sentences) if block.tgt_sentences else "<insertion>")
        print()


if __name__ == "__main__":
    main()
