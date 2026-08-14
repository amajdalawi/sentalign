# tests/test_multilingual.py

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

print(result)