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
    print("score:", alignment.score)
    print()
```

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
sentalign(src_sentences, tgt_sentences, encoder)
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
alignment.src_sentences
alignment.tgt_sentences
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
