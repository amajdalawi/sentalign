"""Demo: complex French->English N:M alignments with in-memory sentence lists."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from sentalign import sentalign


class BilingualToyEncoder:
    """Tiny bilingual normalization + hashing encoder for demos.

    This is only for demonstration/testing. For real usage, pass a proper
    multilingual model (e.g. LASER or SentenceTransformers).
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.map = {
            "bonjour": "hello",
            "salut": "hello",
            "monde": "world",
            "aujourdhui": "today",
            "nous": "we",
            "testons": "test",
            "alignement": "alignment",
            "phrases": "sentences",
            "phrase": "sentence",
            "temps": "weather",
            "magnifique": "beautiful",
            "voyagerai": "travel",
            "semaine": "week",
            "prochaine": "next",
            "paris": "paris",
            "jai": "i",
            "j": "i",
            "arrive": "arrive",
            "gare": "station",
            "ce": "this",
            "matin": "morning",
            "retard": "delay",
            "train": "train",
            "cause": "because",
            "pluie": "rain",
            "forte": "heavy",
            "ensuite": "then",
            "pris": "took",
            "taxi": "taxi",
            "hotel": "hotel",
            "reunion": "meeting",
            "commence": "starts",
            "demain": "tomorrow",
            "huit": "eight",
            "heures": "o'clock",
            "preparer": "prepare",
            "presentation": "presentation",
            "ce": "this",
            "soir": "evening",
        }

    def _norm_token(self, token: str) -> str:
        token = token.lower()
        token = re.sub(r"[^\w']+", "", token)
        token = token.replace("'", "")
        if not token:
            return ""
        return self.map.get(token, token)

    def encode(self, sentences):
        vectors = []
        for sentence in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in sentence.split():
                normalized = self._norm_token(token)
                if not normalized:
                    continue
                digest = hashlib.sha1(normalized.encode("utf-8")).digest()
                vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
            vectors.append(vec)
        return np.vstack(vectors)


def main():
    # French side intentionally split differently vs English to produce N:M blocks.
    french = [
        "Bonjour à tous.",
        "J'arrive à la gare ce matin.",
        "Le train est en retard.",
        "À cause de la pluie forte.",
        "Ensuite, je prends un taxi.",
        "La réunion commence demain.",
        "À huit heures.",
        "Je prépare la présentation ce soir.",
    ]

    english = [
        "Hello everyone.",
        "I arrive at the station this morning.",
        "The train is delayed because of heavy rain.",
        "Then I take a taxi.",
        "The meeting starts tomorrow at eight o'clock.",
        "I prepare the presentation this evening.",
    ]

    result = sentalign(
        french,
        english,
        encoder=BilingualToyEncoder(),
        alignment_max_size=6,
    )

    print("Complex FR->EN demo (expect several N:M alignments)")
    print(f"Overall alignment quality score: {result.overall_score:.4f}")
    print(f"Average raw alignment cost: {result.average_alignment_score:.4f}\n")

    for i, block in enumerate(result.alignments, start=1):
        print(
            f"[{i}] FR{block.src_indices} -> EN{block.tgt_indices} "
            f"(cost={block.score:.4f})"
        )
        print("   FR:", " ".join(block.src_sentences) if block.src_sentences else "<deletion>")
        print("   EN:", " ".join(block.tgt_sentences) if block.tgt_sentences else "<insertion>")
        print()


if __name__ == "__main__":
    main()
