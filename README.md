# sentalign

`sentalign` is a small Python package for aligning sentences between two languages using multilingual sentence embeddings and VecAlign-style dynamic programming.

It works fully in memory: pass two lists of sentences, get back aligned sentence blocks.

## Installation

```bash
pip install sentalign
```

For real multilingual alignment, you also need an embedding model. For example:

```bash
pip install sentence-transformers
```

## Basic usage

```python
from sentence_transformers import SentenceTransformer
from sentalign import sentalign

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

result = sentalign(
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

`sentalign` also accepts sentence objects that carry metadata. The alignment
algorithm only uses extracted text for embedding and scoring; metadata is
preserved on the original objects returned in each alignment block.

Dictionary-like inputs are supported automatically when they contain a `"text"`
field:

```python
from sentalign import sentalign

src = [
    {"text": "Where are you going?", "start": 12.4, "end": 14.1, "subtitle_id": 10},
    {"text": "I'm going home.", "start": 14.3, "end": 16.0, "subtitle_id": 11},
]

tgt = [
    {"text": "Ou vas-tu ?", "start": 12.5, "end": 14.0, "subtitle_id": 20},
    {"text": "Je rentre chez moi.", "start": 14.2, "end": 16.1, "subtitle_id": 21},
]

result = sentalign(src, tgt, encoder=encoder)
alignment = result.alignments[0]

print(alignment.src_sentences)  # extracted strings
print(alignment.tgt_sentences)  # extracted strings
print(alignment.src_items)      # original source dictionaries
print(alignment.tgt_items)      # original target dictionaries
```

Objects with a `.text` attribute are also supported:

```python
from dataclasses import dataclass
from sentalign import sentalign


@dataclass
class BookSentence:
    text: str
    sentence_id: int
    paragraph_id: int
    chapter_id: int


src = [BookSentence("The man left the village.", 1, 1, 1)]
tgt = [BookSentence("L'homme quitta le village.", 8, 3, 1)]

result = sentalign(src, tgt, encoder=encoder)
source_item = result.alignments[0].src_items[0]
print(source_item.paragraph_id)
```

For different object shapes, pass separate source and target extractors:

```python
result = sentalign(
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

`sentalign` does not create embeddings by itself. You must pass an encoder.

For cross-language alignment, use a multilingual encoder such as:

```python
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

or a model like LaBSE.

Do not use a monolingual English-only model for English/French, English/Arabic, Hebrew/English, etc. The encoder must place sentences from both languages in the same embedding space.

## API

```python
sentalign(
    src_sentences,
    tgt_sentences,
    encoder,
    *,
    src_text_getter=None,
    tgt_text_getter=None,
    alignment_max_size=8,
    one_to_many=None,
    ...
)
```

Returns a `SentAlignResult`:

```python
result.alignments
result.overall_score
result.average_alignment_score
```

Each alignment block contains:

```python
alignment.src_indices
alignment.tgt_indices
alignment.score
alignment.src_sentences  # extracted source strings
alignment.tgt_sentences  # extracted target strings
alignment.src_items      # original source objects
alignment.tgt_items      # original target objects
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

## License

Apache-2.0
