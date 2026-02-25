import numpy as np
import pytest
from pathlib import Path

import re
from sentalign import SentAlignResult, sentalign
import torch
from sentence_transformers import SentenceTransformer


# fr_sents = [
#     "Bonjour le monde.",
#     "Comment allez-vous aujourd'hui ?",
#     "J'aime les pommes.",
# ]

# en_sents = [
#     "Hello world.",
#     "How are you today?",
#     "I like apples.",
# ]

nl_sents = Path(__file__).parent / "nl_texts.txt"
en_sents = Path(__file__).parent / "en_texts.txt"
en_wrong = Path(__file__).parent / "en_wrong.txt"

en = en_sents.read_text(encoding="utf-8").splitlines()
nl = nl_sents.read_text(encoding="utf-8").splitlines()
en_wrong = en_wrong.read_text(encoding="utf-8").splitlines()


def split_sentences(text):
    """Sentence splitter on commas, points and question marks."""
    
    sentences = re.split(r'[,.!?]+', text)
    return [sent.strip() for sent in sentences if sent.strip()]

en = [sent for line in en for sent in split_sentences(line)]
nl = [sent for line in nl for sent in split_sentences(line)]
en_wrong = [sent for line in en_wrong for sent in split_sentences(line)]


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Chosen device:", device)
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # "intfloat/multilingual-e5-base",
        device=device,
    )
    print(f"Model device: {next(model._first_module().auto_model.parameters()).device}")
    result = sentalign(en, nl, encoder=model, alignment_max_size=8)
    # print(result)
    # print("Alignments:")
    # for alignment in result.alignments:
    #     print(f"  src_indices={alignment.src_indices}, tgt_indices={alignment.tgt_indices}, score={alignment.score:.4f}")
    #     print("    src:", " | ".join(en[i] for i in alignment.src_indices))
    #     print("    tgt:", " | ".join(nl[i] for i in alignment.tgt_indices))
    #     # print final score
    print(f"Overall score: {result.overall_score:.4f}")
    print(f"Average alignment score: {result.average_alignment_score:.4f}")

    # now to compare nl with en_wrong, printing only the average alignment score for brevity and overall score
    result_wrong = sentalign(en_wrong, nl, encoder=model, alignment_max_size=8)
    print(f"\nComparing nl with en_wrong:")
    print(f"Overall score: {result_wrong.overall_score:.4f}")   
    print(f"Average alignment score: {result_wrong.average_alignment_score:.4f}")