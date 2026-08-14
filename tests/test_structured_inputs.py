from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np
import pytest

from sentweave import SentenceAlignment, align

align_module = importlib.import_module("sentweave.align")


class FakeEncoder:
    def encode(self, sentences):
        vectors = []
        for sentence in sentences:
            text = sentence.lower()
            if "alpha beta" in text or "beta alpha" in text:
                vectors.append([1.0, 1.0, 0.0])
            elif "alpha" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "gamma" in text:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.5, 0.5, 0.5])
        if not vectors:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class BookSentence:
    text: str
    sentence_id: int
    paragraph_id: int
    chapter_id: int


@dataclass
class SourceObject:
    text: str
    content: str
    item_id: int


@dataclass
class TargetObject:
    subtitle: str
    item_id: int


class UnsupportedObject:
    pass


def first_alignment(result):
    assert result.alignments
    return result.alignments[0]


def test_existing_strings_remain_supported():
    src = ("alpha",)
    tgt = ("alpha",)

    result = align(src, tgt, encoder=FakeEncoder())
    alignment = first_alignment(result)

    assert alignment.src_indices == [0]
    assert alignment.tgt_indices == [0]
    assert isinstance(alignment.score, float)
    assert alignment.src_sentences == ["alpha"]
    assert alignment.tgt_sentences == ["alpha"]
    assert alignment.src_items == ["alpha"]
    assert alignment.tgt_items == ["alpha"]
    assert isinstance(result.overall_score, float)
    assert isinstance(result.average_alignment_score, float)


def test_previous_sentence_alignment_constructor_still_works():
    alignment = SentenceAlignment([0], [0], 0.1, ["alpha"], ["alpha"])

    assert alignment.src_sentences == ["alpha"]
    assert alignment.tgt_sentences == ["alpha"]
    assert alignment.src_items == ["alpha"]
    assert alignment.tgt_items == ["alpha"]


def test_dictionary_inputs_preserve_metadata():
    src_item = {"text": "alpha", "start": 12.4, "end": 14.1, "subtitle_id": 10}
    tgt_item = {"text": "alpha", "start": 12.5, "end": 14.0, "subtitle_id": 20}

    result = align([src_item], [tgt_item], encoder=FakeEncoder())
    alignment = first_alignment(result)

    assert alignment.src_sentences == ["alpha"]
    assert alignment.tgt_sentences == ["alpha"]
    assert alignment.src_items == [src_item]
    assert alignment.tgt_items == [tgt_item]
    assert alignment.src_items[0]["subtitle_id"] == 10
    assert alignment.tgt_items[0]["start"] == 12.5


def test_dataclass_inputs_use_text_attribute_and_preserve_identity():
    src_item = BookSentence("alpha", sentence_id=1, paragraph_id=2, chapter_id=3)
    tgt_item = BookSentence("alpha", sentence_id=8, paragraph_id=9, chapter_id=3)

    result = align([src_item], [tgt_item], encoder=FakeEncoder())
    alignment = first_alignment(result)

    assert alignment.src_sentences == ["alpha"]
    assert alignment.tgt_sentences == ["alpha"]
    assert alignment.src_items[0] is src_item
    assert alignment.tgt_items[0] is tgt_item
    assert alignment.src_items[0].paragraph_id == 2
    assert alignment.tgt_items[0].sentence_id == 8


def test_different_source_and_target_shapes_use_separate_extractors():
    src_item = SourceObject(text="ignored", content="alpha", item_id=1)
    tgt_item = TargetObject(subtitle="alpha", item_id=2)

    result = align(
        [src_item],
        [tgt_item],
        encoder=FakeEncoder(),
        src_text_getter=lambda item: item.content,
        tgt_text_getter=lambda item: item.subtitle,
    )
    alignment = first_alignment(result)

    assert alignment.src_sentences == ["alpha"]
    assert alignment.tgt_sentences == ["alpha"]
    assert alignment.src_items[0] is src_item
    assert alignment.tgt_items[0] is tgt_item


def test_custom_getter_takes_precedence_over_text_attribute():
    src_item = SourceObject(text="beta", content="alpha", item_id=1)
    tgt_item = SourceObject(text="beta", content="alpha", item_id=2)

    result = align(
        [src_item],
        [tgt_item],
        encoder=FakeEncoder(),
        src_text_getter=lambda item: item.content,
        tgt_text_getter=lambda item: item.content,
    )
    alignment = first_alignment(result)

    assert alignment.src_sentences == ["alpha"]
    assert alignment.tgt_sentences == ["alpha"]


def test_missing_mapping_text_field_reports_side_and_index():
    with pytest.raises(ValueError, match=r"src_sentences\[1\].*'text' field"):
        align([{"text": "alpha"}, {"id": 2}], ["alpha"], encoder=FakeEncoder())


def test_missing_text_attribute_reports_side_and_index():
    with pytest.raises(ValueError, match=r"tgt_sentences\[0\].*\.text attribute"):
        align(["alpha"], [UnsupportedObject()], encoder=FakeEncoder())


def test_non_string_mapping_text_is_rejected():
    with pytest.raises(TypeError, match=r"src_sentences\[0\] text must be str, got int"):
        align([{"text": 123}], ["alpha"], encoder=FakeEncoder())


def test_non_string_custom_getter_result_is_rejected():
    with pytest.raises(TypeError, match=r"tgt_sentences\[0\] text must be str, got NoneType"):
        align(
            ["alpha"],
            [TargetObject(subtitle="alpha", item_id=1)],
            encoder=FakeEncoder(),
            tgt_text_getter=lambda item: None,
        )


def test_custom_getter_extraction_error_is_wrapped_with_context():
    with pytest.raises(ValueError, match=r"src_sentences\[0\].*missing"):
        align(
            [SourceObject(text="alpha", content="alpha", item_id=1)],
            ["alpha"],
            encoder=FakeEncoder(),
            src_text_getter=lambda item: item.missing,
        )


def test_empty_inputs_preserve_current_behavior():
    both_empty = align([], [], encoder=FakeEncoder())
    source_empty = align([], ["alpha"], encoder=FakeEncoder())
    target_empty = align(["alpha"], [], encoder=FakeEncoder())

    assert both_empty.alignments == []
    assert source_empty.alignments[0].src_items == []
    assert source_empty.alignments[0].tgt_items == ["alpha"]
    assert target_empty.alignments[0].src_items == ["alpha"]
    assert target_empty.alignments[0].tgt_items == []


def test_multi_sentence_alignment_preserves_index_text_item_order(monkeypatch):
    src = [{"text": "alpha", "id": 1}, {"text": "beta", "id": 2}]
    tgt = [{"text": "alpha", "id": 3}, {"text": "beta", "id": 4}]

    def fake_vecalign(**kwargs):
        return {
            0: {
                "final_alignments": [([0, 1], [0, 1])],
                "alignment_scores": np.asarray([0.25], dtype=np.float32),
            }
        }

    monkeypatch.setattr(align_module, "vecalign", fake_vecalign)
    result = align(src, tgt, encoder=FakeEncoder(), alignment_max_size=4)
    combined = result.alignments[0]

    assert combined.src_indices == list(range(combined.src_indices[0], combined.src_indices[-1] + 1))
    assert combined.tgt_indices == list(range(combined.tgt_indices[0], combined.tgt_indices[-1] + 1))
    assert combined.src_sentences == [src[i]["text"] for i in combined.src_indices]
    assert combined.tgt_sentences == [tgt[i]["text"] for i in combined.tgt_indices]
    assert combined.src_items == [src[i] for i in combined.src_indices]
    assert combined.tgt_items == [tgt[i] for i in combined.tgt_indices]


def test_input_objects_are_not_mutated():
    src_item = {"text": "alpha", "metadata": {"start": 1.0}}
    tgt_item = BookSentence("alpha", sentence_id=1, paragraph_id=1, chapter_id=1)
    src_before = dict(src_item)
    tgt_before = BookSentence(**tgt_item.__dict__)

    align([src_item], [tgt_item], encoder=FakeEncoder())

    assert src_item == src_before
    assert tgt_item == tgt_before
