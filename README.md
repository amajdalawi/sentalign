# Vecalign

## New: Python library API (`sentalign`)

This repository now supports installation as a Python package and an in-memory API:

```bash
pip install .
```

```python
from sentence_transformers import SentenceTransformer
from sentalign import sentalign

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

src = ["Hallo Welt.", "Wie geht es dir?"]
tgt = ["Hello world.", "How are you?"]

result = sentalign(src, tgt, encoder=model)
print(result.overall_score)
for block in result.alignments:
    print(block.src_indices, block.tgt_indices, block.score)
```

The `encoder` argument is intentionally model-agnostic: pass any object with an
`encode(list[str]) -> 2D array` method (or a callable with the same behavior).

`overall_score` is a heuristic aggregate quality score in `[0, 1]` (higher is better),
derived from alignment costs and penalizing insertion/deletion blocks.

### Run tests and inspect English/French alignment output

A small test suite for the new in-memory API lives under `tests/`.

```bash
python -m pip install -e .[test]
python -m pytest -q

# Optional: run integration test with a real multilingual encoder
python -m pip install -e .[test-real]
python -m pytest -q -k sentence_transformers
```

If you want to force GPU usage and verify it explicitly during tests, you can run:

```bash
python - <<'PY'
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Chosen device:", device)
if device == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device,
)
print("Model device:", next(model._first_module().auto_model.parameters()).device)
PY

python -m pytest -q -k sentence_transformers
```

Tip: run `nvidia-smi -l 1` in another terminal while the test runs to confirm GPU utilization.

To quickly see alignment output on two in-memory lists (English vs French), run:

```bash
python examples/english_french_demo.py
```

For a more complex French→English example with intentional sentence splits/merges
(N:M alignment blocks), run:

```bash
python examples/complex_french_english_demo.py
```

This prints per-block alignments and an overall quality score in `[0, 1]`.
For production quality embeddings, pass your own encoder model (e.g. LASER or
SentenceTransformers) to `sentalign(...)`.

