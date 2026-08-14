"""Compute the real bigram word-transition table embedded in the
watermark-embedding artifact.

Downloads the TinyStories validation split, tokenizes to lowercase words, and
counts real (previous word -> next word) transition frequencies. This is
plain corpus statistics, not a trained model: no gradient descent, no
quantization, just counts. The artifact's core module samples from this
table and biases the sample toward a per-context "green list" of words to
demonstrate LLM watermarking (Kirchenbauer et al. 2023, arXiv:2301.10226).

Usage:
  .venv/bin/python scripts/train-watermark-lm.py
"""

import json
import os
import re
import urllib.request
from collections import Counter, defaultdict

CORPUS_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)
SCRATCHPAD = (
    "/tmp/claude-1000/-mnt-data-anyesh-github-io/"
    "c7e685af-cd83-4153-880c-2f56e650b2bf/scratchpad"
)
CORPUS_PATH = os.path.join(SCRATCHPAD, "tinystories.txt")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO_ROOT, "src", "artifacts", "data", "watermark-lm.json")

VOCAB_SIZE = 500
MAX_NEXT_PER_WORD = 40
SMOOTHING = 0.5

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?\"]")


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_table():
    text = load_corpus()
    stories = [WORD_RE.findall(s.lower()) for s in text.split("<|endoftext|>")]
    stories = [s for s in stories if len(s) >= 2]

    unigram_counts = Counter(tok for s in stories for tok in s)
    vocab = [w for w, _ in unigram_counts.most_common(VOCAB_SIZE)]
    stoi = {w: i for i, w in enumerate(vocab)}

    bigram_counts = defaultdict(Counter)
    total_pairs = 0
    for s in stories:
        for prev, nxt in zip(s, s[1:]):
            if prev not in stoi or nxt not in stoi:
                continue
            bigram_counts[stoi[prev]][stoi[nxt]] += 1
            total_pairs += 1

    bigram = {}
    for prev_idx, counter in bigram_counts.items():
        top = counter.most_common(MAX_NEXT_PER_WORD)
        bigram[str(prev_idx)] = [[next_idx, count] for next_idx, count in top]

    unigram = [unigram_counts[w] for w in vocab]

    return vocab, bigram, unigram, total_pairs


def main():
    vocab, bigram, unigram, total_pairs = build_table()

    export = {"vocab": vocab, "bigram": bigram, "unigram": unigram}
    blob = json.dumps(export, separators=(",", ":"))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        f.write(blob)

    n_covered_contexts = len(bigram)
    n_pairs_stored = sum(len(v) for v in bigram.values())
    print(f"vocab size: {len(vocab)}")
    print(f"distinct prev-word contexts with bigram data: {n_covered_contexts}")
    print(f"total real (prev, next) occurrences counted: {total_pairs}")
    print(f"(prev, next) pairs stored (top {MAX_NEXT_PER_WORD} per context): {n_pairs_stored}")
    print(f"export written to {OUT_JSON} ({len(blob)} bytes)")

    for w in ["once", "the", "she", "little"]:
        if w not in vocab:
            continue
        idx = vocab.index(w)
        top = bigram.get(str(idx), [])[:8]
        preview = ", ".join(f"{vocab[i]}({c})" for i, c in top)
        print(f"sample bigram [{w}]: {preview}")


if __name__ == "__main__":
    main()
