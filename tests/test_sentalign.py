import hashlib

import numpy as np

from sentalign import SentAlignResult, sentalign


class HashEncoder:
    """Deterministic lightweight encoder for tests."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, sentences):
        vectors = []
        for sentence in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            for idx, token in enumerate(sentence.lower().split()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                vec[bucket] += 1.0 + (idx * 0.01)
            vectors.append(vec)
        return np.vstack(vectors)


def test_sentalign_returns_structured_result():
    src = [
        "Hello world.",
        "How are you today?",
        "I like apples.",
    ]
    tgt = [
        "Bonjour le monde.",
        "Comment allez-vous aujourd'hui ?",
        "J'aime les pommes.",
    ]

    result = sentalign(src, tgt, encoder=HashEncoder())

    assert isinstance(result, SentAlignResult)
    assert len(result.alignments) > 0
    assert 0.0 <= result.overall_score <= 1.0
    assert result.average_alignment_score >= 0.0


def test_sentalign_prefers_one_to_one_for_parallel_lists():
    src = ["zero", "one", "two"]
    tgt = ["zero", "one", "two"]

    result = sentalign(src, tgt, encoder=HashEncoder())

    matched_pairs = [
        (alignment.src_indices, alignment.tgt_indices)
        for alignment in result.alignments
        if alignment.src_indices and alignment.tgt_indices
    ]
    assert matched_pairs == [([0], [0]), ([1], [1]), ([2], [2])]


def test_sentalign_supports_one_to_many_mode():
    src = ["A", "B"]
    tgt = ["A", "B", "C"]

    result = sentalign(src, tgt, encoder=HashEncoder(), one_to_many=3)

    assert len(result.alignments) > 0
    assert any(len(block.tgt_indices) > 1 for block in result.alignments)


def test_sentalign_can_produce_many_to_one_blocks():
    src = [
        "the train is delayed",
        "because of heavy rain",
        "then we took a taxi",
    ]
    tgt = [
        "the train is delayed because of heavy rain",
        "then we took a taxi",
    ]

    result = sentalign(src, tgt, encoder=HashEncoder(), alignment_max_size=6)

    assert any(block.src_indices == [0, 1] and block.tgt_indices == [0] for block in result.alignments)
