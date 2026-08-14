# SentWeave

`sentweave` is a small Python package for aligning sentences between two languages using multilingual sentence embeddings and VecAlign-style dynamic programming.

It works fully in memory: pass two lists of sentences, get back aligned sentence blocks.

## Installation

```bash
pip install sentweave
```

For real multilingual alignment, you also need an embedding model. For example:

```bash
pip install sentence-transformers
```

## Basic usage

```python
from sentence_transformers import SentenceTransformer
from sentweave import align

src = [
    "Hello world.",
    "My name is Abdulrahman.",
    "I like machine learning.",
]

tgt = [
    "Bonjour le monde.",
    "Je m'appelle Abdulrahman.",
    "J'aime l'apprentissage automatique.",
]

encoder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

result = align(
    src,
    tgt,
    encoder=encoder,
)

print("Overall score:", result.overall_score)

for alignment in result.alignments:
    print(alignment.src_indices, alignment.tgt_indices)
    print(alignment.src_sentences)
    print(alignment.tgt_sentences)
    print(alignment.src_items)
    print(alignment.tgt_items)
    print("score:", alignment.score)
    print()
```

## Structured sentence inputs and metadata

`sentweave` also accepts sentence objects that carry metadata. The alignment
algorithm only uses extracted text for embedding and scoring; metadata is
preserved on the original objects returned in each alignment block.

Dictionary-like inputs are supported automatically when they contain a `"text"`
field:

```python
from sentweave import align

src = [
    {"text": "Where are you going?", "start": 12.4, "end": 14.1, "subtitle_id": 10},
    {"text": "I'm going home.", "start": 14.3, "end": 16.0, "subtitle_id": 11},
]

tgt = [
    {"text": "Ou vas-tu ?", "start": 12.5, "end": 14.0, "subtitle_id": 20},
    {"text": "Je rentre chez moi.", "start": 14.2, "end": 16.1, "subtitle_id": 21},
]

result = align(src, tgt, encoder=encoder)
alignment = result.alignments[0]

print(alignment.src_sentences)  # extracted strings
print(alignment.tgt_sentences)  # extracted strings
print(alignment.src_items)      # original source dictionaries
print(alignment.tgt_items)      # original target dictionaries
```

Objects with a `.text` attribute are also supported:

```python
from dataclasses import dataclass
from sentweave import align


@dataclass
class BookSentence:
    text: str
    sentence_id: int
    paragraph_id: int
    chapter_id: int


src = [BookSentence("The man left the village.", 1, 1, 1)]
tgt = [BookSentence("L'homme quitta le village.", 8, 3, 1)]

result = align(src, tgt, encoder=encoder)
source_item = result.alignments[0].src_items[0]
print(source_item.paragraph_id)
```

For different object shapes, pass separate source and target extractors:

```python
result = align(
    src_objects,
    tgt_objects,
    encoder=encoder,
    src_text_getter=lambda item: item.content,
    tgt_text_getter=lambda item: item.subtitle,
)
```

`alignment.src_sentences` and `alignment.tgt_sentences` always contain the
strings used for alignment. `alignment.src_items` and `alignment.tgt_items`
contain the original input objects, including subtitle timings, IDs, paragraph
or chapter IDs, and any user-defined fields. Extracted text must be a Python
`str`; missing text fields or non-string values raise validation errors instead
of being converted.

## Important: use a multilingual encoder

`sentweave` does not create embeddings by itself. You must pass an encoder.

For cross-language alignment, use a multilingual encoder such as:

```python
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

or a model like LaBSE.

Do not use a monolingual English-only model for English/French, English/Arabic, Hebrew/English, etc. The encoder must place sentences from both languages in the same embedding space.

## Public API

SentWeave exposes one primary operation:

```python
from sentweave import align
```

```python
result = align(
    src_sentences,
    tgt_sentences,
    encoder,
    *,
    src_text_getter=None,
    tgt_text_getter=None,
    alignment_max_size=8,
    one_to_many=None,
    del_percentile_frac=0.2,
    max_size_full_dp=300,
    costs_sample_size=20_000,
    num_samps_for_norm=100,
    search_buffer_size=5,
    random_state=42,
)
```

`align()` accepts two ordered sequences and returns an `AlignmentResult`. The
result contains the aligned blocks, an overall quality estimate, and the
average internal alignment cost.

### Inputs

The first two arguments are the source and target sentence sequences:

```python
src = ["The door opened.", "A woman entered."]
tgt = ["La porte s'ouvrit.", "Une femme entra."]
```

Their order matters. SentWeave performs monotonic alignment: it can combine,
skip, or match adjacent sentences, but it does not reorder them.

Each item can be:

- a string;
- a mapping containing a string-valued `"text"` field;
- an object with a string-valued `.text` attribute; or
- any custom object when an appropriate text getter is supplied.

For example:

```python
src = [
    {"text": "The door opened.", "page": 12},
    {"text": "A woman entered.", "page": 12},
]
```

SentWeave embeds only the extracted text. The original items and their metadata
are preserved in the result.

### Encoder contract

SentWeave does not select or download an embedding model. The caller must
provide an encoder that maps a sequence of strings to a two-dimensional numeric
matrix.

An encoder may be an object with an `encode()` method:

```python
from sentence_transformers import SentenceTransformer
from sentweave import align

encoder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

result = align(src, tgt, encoder=encoder)
```

It may also be a callable:

```python
def encode_sentences(sentences):
    # Return one numeric vector per sentence.
    return embedding_matrix

result = align(src, tgt, encoder=encode_sentences)
```

The returned matrix must have shape:

```text
(number of input strings, embedding dimension)
```

For bilingual alignment, the encoder must place both languages in the same
embedding space. A monolingual model is generally unsuitable for cross-language
alignment.

### Alignment behavior

SentWeave supports alignment blocks such as:

```text
1:1  one source sentence to one target sentence
1:2  one source sentence to two target sentences
2:1  two source sentences to one target sentence
2:2  two source sentences to two target sentences
1:0  unmatched source sentence
0:1  unmatched target sentence
```

This is useful because translations do not always preserve sentence boundaries.
A translator may split one sentence into several, combine adjacent sentences,
or omit material.

By default, SentWeave considers adjacent groups whose combined size satisfies:

```text
source count + target count <= alignment_max_size
```

With the default `alignment_max_size=8`, blocks such as `1:1`, `1:3`, `2:2`,
and `3:4` can be considered, while a block larger than the configured limit
cannot.

To restrict alignment to one source sentence and up to a fixed number of target
sentences, use `one_to_many`:

```python
result = align(
    src,
    tgt,
    encoder=encoder,
    one_to_many=3,
)
```

This restricts non-null matches to `1:1`, `1:2`, and `1:3`. It is useful for
subtitle or segmentation workflows where source boundaries must remain fixed.

### Custom text extraction

Mappings with a `"text"` field and objects with `.text` work automatically. For
other shapes, pass separate source and target getters:

```python
result = align(
    src_objects,
    tgt_objects,
    encoder=encoder,
    src_text_getter=lambda item: item.content,
    tgt_text_getter=lambda item: item.subtitle,
)
```

The getter for each side must return a Python `str`.

SentWeave raises a contextual error when:

- a mapping lacks `"text"`;
- an object lacks `.text`;
- a getter raises a common extraction error;
- extracted text is not a string;
- the encoder returns the wrong number of vectors; or
- the encoder output is not two-dimensional.

### Return value

`align()` returns an `AlignmentResult`:

```python
from sentweave import AlignmentResult

result: AlignmentResult = align(src, tgt, encoder=encoder)
```

It contains:

```python
result.alignments
result.overall_score
result.average_alignment_score
```

#### `result.alignments`

An ordered list of `SentenceAlignment` objects. Each object describes one
matched or unmatched block:

```python
for block in result.alignments:
    print(block.src_indices)
    print(block.tgt_indices)
    print(block.src_sentences)
    print(block.tgt_sentences)
    print(block.src_items)
    print(block.tgt_items)
    print(block.score)
```

| Field | Meaning |
| --- | --- |
| `src_indices` | Zero-based source indices in this block. |
| `tgt_indices` | Zero-based target indices in this block. |
| `src_sentences` | Source strings actually supplied to the encoder. |
| `tgt_sentences` | Target strings actually supplied to the encoder. |
| `src_items` | Original source objects, including metadata. |
| `tgt_items` | Original target objects, including metadata. |
| `score` | Internal alignment cost for this block. |

For an unmatched block, one side has an empty index, sentence, and item list:

```python
if not block.tgt_indices:
    print("Unmatched source text:", block.src_sentences)
```

#### `result.overall_score`

A heuristic quality estimate between `0.0` and `1.0`, where higher is better.

It is calculated from the block costs and weighted by block size. Null
alignments contribute zero quality. This makes it useful for comparing several
alignment attempts produced with similar inputs, models, and settings.

It should not be interpreted as a calibrated probability that the translation
is correct. Scores from different encoders or substantially different documents
may not be directly comparable.

#### `result.average_alignment_score`

The arithmetic mean of the raw internal alignment costs. Unlike
`overall_score`, lower costs generally indicate closer matches. Most
applications should use `overall_score` for a convenient high-level signal and
inspect individual blocks when precise quality decisions matter.

### Configuration reference

| Parameter | Default | Description |
| --- | ---: | --- |
| `src_text_getter` | `None` | Extract text from custom source objects. |
| `tgt_text_getter` | `None` | Extract text from custom target objects. |
| `alignment_max_size` | `8` | Maximum combined source/target block size considered during general alignment. |
| `one_to_many` | `None` | Restrict matches to one source item and at most this many target items. |
| `del_percentile_frac` | `0.2` | Controls the dynamically estimated penalty for unmatched items. |
| `max_size_full_dp` | `300` | Maximum sequence size for the full dynamic-programming pass. |
| `costs_sample_size` | `20_000` | Number of sampled costs used to estimate the cost distribution. |
| `num_samps_for_norm` | `100` | Number of samples used for normalization statistics. |
| `search_buffer_size` | `5` | Extra context retained around the approximate alignment path. |
| `random_state` | `42` | Seed used to make sampling and alignment behavior reproducible. |

The defaults are intended as general-purpose values. In most applications, the
first parameters to tune are:

- `alignment_max_size` when translations frequently split or combine sentences;
- `one_to_many` when one side's boundaries must remain fixed;
- the embedding model when semantic matches are weak; and
- `del_percentile_frac` when the alignment produces too many or too few
  unmatched sentences.

### Empty inputs

Empty sequences are supported:

```python
result = align([], [], encoder=encoder)
```

If both sides are empty, `result.alignments` is empty and the aggregate scores
are zero. If only one side is empty, the available sentences are returned as
unmatched blocks.

### Complete example

```python
from sentence_transformers import SentenceTransformer
from sentweave import align

src = [
    {"text": "The door opened.", "page": 12},
    {"text": "A woman entered.", "page": 12},
]

tgt = [
    {"text": "La porte s'ouvrit.", "page": 14},
    {"text": "Une femme entra.", "page": 14},
]

encoder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

result = align(
    src,
    tgt,
    encoder=encoder,
    alignment_max_size=4,
    random_state=42,
)

print(f"Overall quality: {result.overall_score:.3f}")

for block in result.alignments:
    print(
        f"{block.src_indices} -> {block.tgt_indices} "
        f"(cost={block.score:.3f})"
    )
    print("SOURCE:", " ".join(block.src_sentences))
    print("TARGET:", " ".join(block.tgt_sentences))
    print("SOURCE METADATA:", block.src_items)
    print("TARGET METADATA:", block.tgt_items)
```

## Development

Install locally in editable mode:

```bash
pip install -e .
```

Run a simple multilingual test:

```bash
pip install sentence-transformers
python tests/test_multilingual.py
```

Build the package:

```bash
python -m build
twine check dist/*
```

## Project history

This project was previously published as `sentalign`. It was renamed to
SentWeave to avoid confusion with the pre-existing academic SentAlign project.

## License

Apache-2.0
