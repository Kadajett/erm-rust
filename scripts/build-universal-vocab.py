#!/usr/bin/env python3
"""
build-universal-vocab.py — Build a universal BPE vocabulary from all books in a corpus.

Produces a BPE vocabulary JSON file compatible with erm-core's BpeTokenizer format:
  { merges: [[left, right], ...], vocab: {str: id}, id_to_token: {id: str}, vocab_size: N }

Uses frequency-weighted word lists for efficient BPE training (O(V*m) not O(N*m)).

Usage:
  python3 build-universal-vocab.py \
    --books-dir /workspace/rust-pcn/data/books \
    --output /workspace/erm-rust/data/curriculum_vocab/bpe_vocab.json \
    --num-merges 4096 \
    --sample-mb 10
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


# Reserved tokens matching erm-core
PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
FIRST_SUBWORD_ID = 3

SENTINEL = "Ġ"  # End-of-word marker


def collect_corpus(books_dir: str, max_bytes: int) -> str:
    """Collect text from all .txt files, up to max_bytes."""
    corpus = []
    total = 0
    txt_files = sorted(Path(books_dir).glob("*.txt"))
    print(f"Found {len(txt_files)} .txt files in {books_dir}", flush=True)

    for path in txt_files:
        if total >= max_bytes:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            remaining = max_bytes - total
            if len(text) > remaining:
                text = text[:remaining]
            corpus.append(text)
            total += len(text)
        except Exception as e:
            print(f"  WARNING: skipping {path.name}: {e}", file=sys.stderr)

    result = "\n".join(corpus)
    print(f"Collected {len(result):,} bytes from {len(corpus)} files", flush=True)
    return result


def train_bpe(corpus: str, num_merges: int):
    """
    Train BPE vocabulary matching erm-core's algorithm.

    Uses frequency-weighted word dictionary for O(V*m) performance
    instead of O(N*m) where N=total word occurrences, V=unique words.
    """
    vocab = {}
    id_to_token = {}

    # Reserve special tokens
    for tok_id, name in [(PAD_ID, "<pad>"), (MASK_ID, "<mask>"), (UNK_ID, "<unk>")]:
        vocab[name] = tok_id
        id_to_token[str(tok_id)] = name

    next_id = FIRST_SUBWORD_ID

    # Collect unique chars as base vocabulary
    unique_chars = sorted(set(corpus))
    for ch in unique_chars:
        if ch not in vocab:
            vocab[ch] = next_id
            id_to_token[str(next_id)] = ch
            next_id += 1

    # Build frequency-weighted word list (much faster than tracking all occurrences)
    word_counts = Counter(corpus.split())
    print(f"  Unique words: {len(word_counts):,} (from {sum(word_counts.values()):,} total)", flush=True)

    # Convert to tokenized form: { (tok1, tok2, ...) : count }
    word_freqs = {}
    for word, count in word_counts.items():
        chars = list(word)
        if chars:
            chars[-1] = chars[-1] + SENTINEL
        tokens = tuple(chars)
        word_freqs[tokens] = word_freqs.get(tokens, 0) + count

    # Register sentinel tokens in vocab
    for tokens in word_freqs:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = next_id
                id_to_token[str(next_id)] = tok
                next_id += 1

    merges = []

    for merge_i in range(num_merges):
        if merge_i % 200 == 0:
            print(f"  Merge {merge_i}/{num_merges} (vocab_size={next_id}, words={len(word_freqs)})...", flush=True)

        # Count pairs using word frequencies
        pair_counts = Counter()
        for tokens, freq in word_freqs.items():
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += freq

        if not pair_counts:
            print(f"  No more pairs at merge {merge_i}")
            break

        # Find most frequent pair
        best_pair = pair_counts.most_common(1)[0][0]
        merged = best_pair[0] + best_pair[1]

        # Register merged token
        if merged not in vocab:
            vocab[merged] = next_id
            id_to_token[str(next_id)] = merged
            next_id += 1

        merges.append(list(best_pair))

        # Apply merge to all words — build new word_freqs dict
        new_word_freqs = {}
        for tokens, freq in word_freqs.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_key = tuple(new_tokens)
            new_word_freqs[new_key] = new_word_freqs.get(new_key, 0) + freq
        word_freqs = new_word_freqs

    vocab_size = next_id
    print(f"  BPE training complete: {len(merges)} merges, vocab_size={vocab_size}")
    return merges, vocab, id_to_token, vocab_size


def main():
    parser = argparse.ArgumentParser(description="Build universal BPE vocabulary")
    parser.add_argument("--books-dir", required=True, help="Directory containing .txt books")
    parser.add_argument("--output", required=True, help="Output path for bpe_vocab.json")
    parser.add_argument("--num-merges", type=int, default=4096, help="Number of BPE merge operations")
    parser.add_argument("--sample-mb", type=int, default=10, help="Max MB of corpus to sample")
    args = parser.parse_args()

    max_bytes = args.sample_mb * 1024 * 1024

    print(f"=== Building Universal BPE Vocabulary ===")
    print(f"Books dir: {args.books_dir}")
    print(f"Output: {args.output}")
    print(f"Num merges: {args.num_merges}")
    print(f"Sample size: {args.sample_mb} MB")

    # Collect corpus
    corpus = collect_corpus(args.books_dir, max_bytes)
    if not corpus.strip():
        print("ERROR: empty corpus", file=sys.stderr)
        sys.exit(1)

    # Train BPE
    import time
    t0 = time.time()
    print(f"\nTraining BPE ({args.num_merges} merges)...")
    merges, vocab, id_to_token, vocab_size = train_bpe(corpus, args.num_merges)
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")

    # Build output JSON matching erm-core's BpeTokenizer serialization format
    output = {
        "merges": merges,
        "vocab": vocab,
        "id_to_token": id_to_token,
        "vocab_size": vocab_size,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(args.output)
    print(f"\nVocabulary saved: {args.output} ({file_size:,} bytes)")
    print(f"Vocab size: {vocab_size}")
    print(f"Merges: {len(merges)}")

    # Quick sanity check
    with open(args.output) as f:
        loaded = json.load(f)
    assert loaded["vocab_size"] == vocab_size
    assert len(loaded["merges"]) == len(merges)
    print("Sanity check: OK")


if __name__ == "__main__":
    main()
