#!/usr/bin/env python3
"""
Prepare three sharded corpora from lsb/million-english-numbers.

Variants emitted:
1) word-copy-sharded       : plain word samples for denoise-style reconstruction
2) int-to-word-sharded     : Input integer -> Output word
3) word-to-int-sharded     : Input word -> Output integer
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VariantStats:
    examples_written: int = 0
    shard_count: int = 0
    skipped_empty: int = 0


@dataclass
class VariantWriter:
    name: str
    out_dir: Path
    shard_prefix: str
    shard_size: int
    stats: VariantStats
    shard_index: int = 1
    in_shard: int = 0
    handle: object | None = None

    def open_first(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.handle = self._open_shard(self.shard_index)
        self.stats.shard_count = 1
        print(f"[prepare-numbers] writing shard: {self.current_shard_path()}")

    def current_shard_path(self) -> Path:
        return self.out_dir / f"{self.shard_prefix}-{self.shard_index:05d}.txt"

    def _open_shard(self, shard_index: int):
        path = self.out_dir / f"{self.shard_prefix}-{shard_index:05d}.txt"
        return path.open("w", encoding="utf-8")

    def write_example(self, text: str) -> None:
        if not text:
            self.stats.skipped_empty += 1
            return

        if self.in_shard >= self.shard_size:
            self.rotate_shard()

        assert self.handle is not None
        self.handle.write(text)
        self.handle.write("\n\n")
        self.stats.examples_written += 1
        self.in_shard += 1

    def rotate_shard(self) -> None:
        assert self.handle is not None
        self.handle.close()
        self.shard_index += 1
        self.in_shard = 0
        self.handle = self._open_shard(self.shard_index)
        self.stats.shard_count += 1
        print(f"[prepare-numbers] writing shard: {self.current_shard_path()}")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def normalize_word(text: str) -> str:
    """Normalize a number word without changing token semantics."""
    collapsed = re.sub(r"\s+", " ", text.strip())
    return collapsed


def format_word_copy(word: str) -> str:
    return word


def format_int_to_word(number: int, word: str) -> str:
    return f"Input:\n{number}\n\nOutput:\n{word}\n"


def format_word_to_int(number: int, word: str) -> str:
    return f"Input:\n{word}\n\nOutput:\n{number}\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="lsb/million-english-numbers",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output directory root where variant folders will be created",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=20_000,
        help="Examples per shard file",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap for quick tests (0 = no cap)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Progress print interval",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional HF datasets cache dir",
    )
    return parser.parse_args()


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(
            "ERROR: missing dependency 'datasets'. Install with: "
            "python3 -m pip install --user datasets"
        )
        raise SystemExit(1) from exc

    dataset_kwargs = {}
    if args.cache_dir:
        dataset_kwargs["cache_dir"] = args.cache_dir

    ds = load_dataset(args.dataset, **dataset_kwargs)
    if args.split not in ds:
        print(f"ERROR: split '{args.split}' not found. Available: {list(ds.keys())}")
        return 1

    split_ds = ds[args.split]
    total_rows = len(split_ds)
    print(f"[prepare-numbers] loaded dataset={args.dataset} split={args.split} rows={total_rows}")

    word_copy = VariantWriter(
        name="word-copy-sharded",
        out_dir=out_root / "word-copy-sharded",
        shard_prefix="numbers-word-copy",
        shard_size=args.shard_size,
        stats=VariantStats(),
    )
    int_to_word = VariantWriter(
        name="int-to-word-sharded",
        out_dir=out_root / "int-to-word-sharded",
        shard_prefix="numbers-int-to-word",
        shard_size=args.shard_size,
        stats=VariantStats(),
    )
    word_to_int = VariantWriter(
        name="word-to-int-sharded",
        out_dir=out_root / "word-to-int-sharded",
        shard_prefix="numbers-word-to-int",
        shard_size=args.shard_size,
        stats=VariantStats(),
    )
    writers = [word_copy, int_to_word, word_to_int]

    for writer in writers:
        writer.open_first()

    rows_seen = 0
    try:
        for idx, row in enumerate(split_ds):
            if args.max_examples > 0 and rows_seen >= args.max_examples:
                break
            rows_seen += 1

            raw_word = row.get("text")
            if not isinstance(raw_word, str):
                for writer in writers:
                    writer.stats.skipped_empty += 1
                continue

            word = normalize_word(raw_word)
            if not word:
                for writer in writers:
                    writer.stats.skipped_empty += 1
                continue

            word_copy.write_example(format_word_copy(word))
            int_to_word.write_example(format_int_to_word(idx, word))
            word_to_int.write_example(format_word_to_int(idx, word))

            if args.progress_every > 0 and rows_seen % args.progress_every == 0:
                print(
                    "[prepare-numbers] "
                    f"seen={rows_seen} "
                    f"copy_written={word_copy.stats.examples_written} "
                    f"i2w_written={int_to_word.stats.examples_written} "
                    f"w2i_written={word_to_int.stats.examples_written}"
                )
    finally:
        for writer in writers:
            writer.close()

    root_manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "rows_total_split": total_rows,
        "rows_seen": rows_seen,
        "max_examples": args.max_examples,
        "shard_size": args.shard_size,
        "variants": {
            writer.name: {
                "out_dir": str(writer.out_dir),
                "shard_prefix": writer.shard_prefix,
                "examples_written": writer.stats.examples_written,
                "shard_count": writer.stats.shard_count,
                "skipped_empty": writer.stats.skipped_empty,
            }
            for writer in writers
        },
        "format_templates": {
            "word-copy-sharded": "<number-word>",
            "int-to-word-sharded": "Input:\\n<integer>\\n\\nOutput:\\n<number-word>",
            "word-to-int-sharded": "Input:\\n<number-word>\\n\\nOutput:\\n<integer>",
        },
    }
    write_manifest(out_root / "manifest.json", root_manifest)

    for writer in writers:
        variant_manifest = {
            "dataset": args.dataset,
            "split": args.split,
            "variant": writer.name,
            "out_dir": str(writer.out_dir),
            "shard_prefix": writer.shard_prefix,
            "shard_size": args.shard_size,
            "rows_seen": rows_seen,
            "examples_written": writer.stats.examples_written,
            "shard_count": writer.stats.shard_count,
            "skipped_empty": writer.stats.skipped_empty,
        }
        write_manifest(writer.out_dir / "manifest.json", variant_manifest)

    print("[prepare-numbers] done")
    print(json.dumps(root_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
