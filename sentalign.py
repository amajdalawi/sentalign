"""Public library API for in-memory sentence alignment."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from random import seed as random_seed
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Union

import numpy as np

from dp_utils import (
    layer,
    make_alignment_types,
    make_norm1,
    make_one_to_many_alignment_types,
    preprocess_line,
    vecalign,
)


class EncoderProtocol(Protocol):
    """Protocol for sentence encoders used by :func:`sentalign`."""

    def encode(self, sentences: Sequence[str]) -> Union[np.ndarray, Sequence[Sequence[float]]]:
        """Return sentence embeddings where the first dimension is ``len(sentences)``."""


EncoderType = Union[Callable[[Sequence[str]], Union[np.ndarray, Sequence[Sequence[float]]]], EncoderProtocol]


@dataclass(frozen=True)
class SentenceAlignment:
    """A single alignment block between source and target sentences."""

    src_indices: List[int]
    tgt_indices: List[int]
    score: float
    src_sentences: List[str]
    tgt_sentences: List[str]


@dataclass(frozen=True)
class SentAlignResult:
    """Output of the :func:`sentalign` API."""

    alignments: List[SentenceAlignment]
    overall_score: float
    average_alignment_score: float


def _encode_sentences(encoder: EncoderType, sentences: Sequence[str]) -> np.ndarray:
    if hasattr(encoder, "encode"):
        vectors = encoder.encode(sentences)  # type: ignore[union-attr]
    else:
        vectors = encoder(sentences)  # type: ignore[misc]

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {vectors.shape}")
    if vectors.shape[0] != len(sentences):
        raise ValueError(
            "Encoder output row count does not match input sentence count: "
            f"{vectors.shape[0]} != {len(sentences)}"
        )
    return vectors


def _build_overlap_embeddings(
    sentences: Sequence[str],
    max_overlaps: int,
    encoder: EncoderType,
) -> np.ndarray:
    clean_sentences = [preprocess_line(s) for s in sentences]
    overlap_vectors = []
    for overlap in range(1, max_overlaps + 1):
        overlap_text = layer(clean_sentences, overlap)
        overlap_vecs = _encode_sentences(encoder, overlap_text)
        overlap_vectors.append(overlap_vecs)

    vecs = np.stack(overlap_vectors, axis=0)
    make_norm1(vecs)
    return vecs


def _alignment_quality(src_indices: Sequence[int], tgt_indices: Sequence[int], cost: float) -> float:
    if len(src_indices) == 0 or len(tgt_indices) == 0:
        return 0.0
    clipped = float(np.clip(cost, 0.0, 2.0))
    return 1.0 - (clipped / 2.0)


def sentalign(
    src_sentences: Sequence[str],
    tgt_sentences: Sequence[str],
    encoder: EncoderType,
    *,
    alignment_max_size: int = 4,
    one_to_many: Optional[int] = None,
    del_percentile_frac: float = 0.2,
    max_size_full_dp: int = 300,
    costs_sample_size: int = 20_000,
    num_samps_for_norm: int = 100,
    search_buffer_size: int = 5,
    random_state: int = 42,
) -> SentAlignResult:
    """Align two sentence lists using Vecalign internals.

    Args:
        src_sentences: Source document sentences.
        tgt_sentences: Target document sentences.
        encoder: Any callable/object that converts a list of strings to a 2-D
            embedding matrix.
        alignment_max_size: Search alignments where ``n + m <= alignment_max_size``.
        one_to_many: If set, restricts alignment to ``1:m`` with ``m <= one_to_many``.
        del_percentile_frac: Deletion penalty percentile knob.
        max_size_full_dp: Maximum sequence length for running full DP.
        costs_sample_size: Sample size to estimate cost distribution.
        num_samps_for_norm: Number of samples used for normalization statistics.
        search_buffer_size: Width of extra sparse-search context.
        random_state: Seed used for deterministic behavior.

    Returns:
        ``SentAlignResult`` containing per-block alignments and an aggregate score.
        ``overall_score`` is a heuristic quality estimate in ``[0, 1]`` where higher
        is better.
    """

    if alignment_max_size < 2:
        alignment_max_size = 2
    if one_to_many is not None and one_to_many < 1:
        raise ValueError("one_to_many must be >= 1 when provided")

    random_seed(random_state)
    np.random.seed(random_state)

    src_max_alignment_size = 1 if one_to_many is not None else alignment_max_size
    tgt_max_alignment_size = one_to_many if one_to_many is not None else alignment_max_size

    width_over2 = ceil(max(src_max_alignment_size, tgt_max_alignment_size) / 2.0) + search_buffer_size

    vecs0 = _build_overlap_embeddings(src_sentences, src_max_alignment_size, encoder)
    vecs1 = _build_overlap_embeddings(tgt_sentences, tgt_max_alignment_size, encoder)

    if one_to_many is not None:
        final_alignment_types = make_one_to_many_alignment_types(one_to_many)
    else:
        final_alignment_types = make_alignment_types(alignment_max_size)

    stack = vecalign(
        vecs0=vecs0,
        vecs1=vecs1,
        final_alignment_types=final_alignment_types,
        del_percentile_frac=del_percentile_frac,
        width_over2=width_over2,
        max_size_full_dp=max_size_full_dp,
        costs_sample_size=costs_sample_size,
        num_samps_for_norm=num_samps_for_norm,
    )

    final_alignments = stack[0]["final_alignments"]
    alignment_scores = stack[0]["alignment_scores"]

    output_alignments: List[SentenceAlignment] = []
    quality_weights = []
    quality_values = []
    for (src_idx, tgt_idx), score in zip(final_alignments, alignment_scores):
        block = SentenceAlignment(
            src_indices=list(src_idx),
            tgt_indices=list(tgt_idx),
            score=float(score),
            src_sentences=[src_sentences[i] for i in src_idx],
            tgt_sentences=[tgt_sentences[i] for i in tgt_idx],
        )
        output_alignments.append(block)

        block_weight = max(len(src_idx), len(tgt_idx), 1)
        quality_weights.append(block_weight)
        quality_values.append(_alignment_quality(src_idx, tgt_idx, score))

    overall_score = float(np.average(quality_values, weights=quality_weights)) if quality_values else 0.0
    mean_cost = float(np.mean(alignment_scores)) if len(alignment_scores) else 0.0

    return SentAlignResult(
        alignments=output_alignments,
        overall_score=overall_score,
        average_alignment_score=mean_cost,
    )

