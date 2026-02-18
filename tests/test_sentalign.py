import hashlib

import numpy as np
import pytest

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

    result = sentalign(src, tgt, encoder=HashEncoder(), del_percentile_frac=1.0)

    matched_pairs = [
        (alignment.src_indices, alignment.tgt_indices)
        for alignment in result.alignments
        if alignment.src_indices and alignment.tgt_indices
    ]
    assert len(matched_pairs) >= 1
    assert any(src_idx == [0] and tgt_idx == [0] for src_idx, tgt_idx in matched_pairs)
    assert any(src_idx == [1] and tgt_idx == [1] for src_idx, tgt_idx in matched_pairs)
    assert any(src_idx == [2] and tgt_idx == [2] for src_idx, tgt_idx in matched_pairs)


def test_sentalign_supports_one_to_many_mode():
    src = ["A", "B"]
    tgt = ["A", "B", "C"]

    result = sentalign(src, tgt, encoder=HashEncoder(), one_to_many=3, del_percentile_frac=1.0)

    assert len(result.alignments) > 0
    for block in result.alignments:
        assert len(block.src_indices) <= 1
        assert len(block.tgt_indices) <= 3


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


def test_sentalign_handles_long_parallel_sentences():
    src = [
        (
            "In the middle of the night, while the city lights flickered and rain echoed against "
            "the windows, the engineering team carefully reviewed every metric, every alert, and "
            "every user report to understand why the service degraded so suddenly."
        ),
        (
            "After identifying the bottleneck in the indexing pipeline, they prepared a staged "
            "rollback, updated configuration thresholds, and coordinated with support so customers "
            "would receive transparent status updates until full recovery."
        ),
    ]
    tgt = list(src)

    result = sentalign(src, tgt, encoder=HashEncoder(dim=128), alignment_max_size=6, del_percentile_frac=1.0)

    matched_pairs = [
        (alignment.src_indices, alignment.tgt_indices)
        for alignment in result.alignments
        if alignment.src_indices and alignment.tgt_indices
    ]
    assert len(matched_pairs) >= 1
    assert any(src_idx == [0] and tgt_idx == [0] for src_idx, tgt_idx in matched_pairs)
    assert any(src_idx == [1] and tgt_idx == [1] for src_idx, tgt_idx in matched_pairs)
    assert result.overall_score > 0.90


def test_sentalign_long_sentences_many_to_one_merge():
    src = [
        (
            "The migration plan required validating schema versions, replaying historical events, "
            "and checking consistency snapshots before traffic could be moved safely to the new cluster."
        ),
        (
            "Even after the move, the team monitored latency percentiles, queue depths, and error "
            "budgets for several hours to ensure no hidden regression remained in production."
        ),
        "At the end of the incident review, they documented each decision in a shared report.",
    ]
    tgt = [
        (
            "The migration plan required validating schema versions, replaying historical events, and "
            "checking consistency snapshots before traffic could be moved safely to the new cluster, "
            "and even after the move, the team monitored latency percentiles, queue depths, and error "
            "budgets for several hours to ensure no hidden regression remained in production."
        ),
        "At the end of the incident review, they documented each decision in a shared report.",
    ]

    result = sentalign(src, tgt, encoder=HashEncoder(dim=128), alignment_max_size=8)

    assert any(block.src_indices == [0, 1] and block.tgt_indices == [0] for block in result.alignments)
    assert any(block.src_indices == [2] and block.tgt_indices == [1] for block in result.alignments)


def test_sentalign_with_sentence_transformers_encoder():
    sentence_transformers = pytest.importorskip(
        "sentence_transformers", reason="Install sentence-transformers to run real-encoder integration test"
    )

    src = [
        "The committee approved the revised policy after a long debate.",
        "Implementation will begin next quarter with a pilot team.",
        "A public report will summarize outcomes at the end of the year.",
    ]
    tgt = [
        "Le comité a approuvé la politique révisée après un long débat.",
        "La mise en œuvre commencera le trimestre prochain avec une équipe pilote.",
        "Un rapport public résumera les résultats à la fin de l'année.",
    ]

    try:
        model = sentence_transformers.SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception as exc:
        pytest.skip(f"Could not load/download sentence-transformers model: {exc}")

    result = sentalign(src, tgt, encoder=model, alignment_max_size=6)

    aligned_pairs = [
        (block.src_indices, block.tgt_indices)
        for block in result.alignments
        if block.src_indices and block.tgt_indices
    ]
    assert aligned_pairs == [([0], [0]), ([1], [1]), ([2], [2])]
    assert result.overall_score > 0.40
