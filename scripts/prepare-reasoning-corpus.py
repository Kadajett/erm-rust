#!/usr/bin/env python3
"""
Prepare a plain-text Q/A corpus from nohurry/Opus-4.6-Reasoning-3000x-filtered.

This extractor intentionally ignores chain-of-thought ("thinking") and keeps only:
  - problem  -> Question
  - solution -> Answer

Output is sharded .txt files suitable for ERM streaming training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/nohurry/"
    "Opus-4.6-Reasoning-3000x-filtered/raw/main/"
    "distilled_corpus_400k_with_cot-filtered.jsonl"
)


@dataclass
class Stats:
    lines_seen: int = 0
    examples_written: int = 0
    skipped_json_error: int = 0
    skipped_missing_fields: int = 0
    skipped_too_short: int = 0
    skipped_duplicate: int = 0
    shard_count: int = 0


def normalize_text(text: str) -> str:
    """Normalize whitespace lightly while preserving content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def format_example(problem: str, solution: str) -> str:
    """Render one plain-text QA training sample."""
    return (
        "Question:\n"
        f"{problem}\n\n"
        "Answer:\n"
        f"{solution}\n"
    )


def open_shard(out_dir: Path, shard_index: int) -> tuple[Path, "TextIO"]:
    path = out_dir / f"reasoning-qa-{shard_index:05d}.txt"
    handle = path.open("w", encoding="utf-8")
    return path, handle


def iter_lines(source_url: str):
    with urllib.request.urlopen(source_url) as resp:
        for raw in resp:
            yield raw.decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help=f"Source JSONL URL (default: {DEFAULT_SOURCE_URL})",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for sharded .txt corpus",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=20_000,
        help="Examples per output shard (.txt)",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap for quick tests (0 = no cap)",
    )
    p.add_argument(
        "--min-problem-chars",
        type=int,
        default=8,
        help="Minimum problem text length after normalization",
    )
    p.add_argument(
        "--min-solution-chars",
        type=int,
        default=8,
        help="Minimum solution text length after normalization",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Progress print interval (lines seen)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    seen_ids: set[str] = set()

    shard_idx = 1
    examples_in_shard = 0
    shard_path, shard_file = open_shard(out_dir, shard_idx)
    stats.shard_count = 1
    print(f"[prepare-reasoning] writing shard: {shard_path}")

    try:
        for line in iter_lines(args.source_url):
            stats.lines_seen += 1
            s = line.strip()
            if not s:
                continue

            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                stats.skipped_json_error += 1
                continue

            # Prefer dataset hash/id for dedupe, fallback to content key.
            dedupe_key = (
                str(row.get("hash"))
                if row.get("hash") is not None
                else f"{row.get('problem', '')}\n{row.get('solution', '')}"
            )
            if dedupe_key in seen_ids:
                stats.skipped_duplicate += 1
                continue
            seen_ids.add(dedupe_key)

            problem_raw = row.get("problem")
            solution_raw = row.get("solution")
            if not isinstance(problem_raw, str) or not isinstance(solution_raw, str):
                stats.skipped_missing_fields += 1
                continue

            # Explicitly ignore chain-of-thought field "thinking".
            problem = normalize_text(problem_raw)
            solution = normalize_text(solution_raw)
            if len(problem) < args.min_problem_chars or len(solution) < args.min_solution_chars:
                stats.skipped_too_short += 1
                continue

            text = format_example(problem, solution)
            shard_file.write(text)
            shard_file.write("\n\n")

            stats.examples_written += 1
            examples_in_shard += 1

            if args.max_examples > 0 and stats.examples_written >= args.max_examples:
                break

            if examples_in_shard >= args.shard_size:
                shard_file.close()
                shard_idx += 1
                examples_in_shard = 0
                shard_path, shard_file = open_shard(out_dir, shard_idx)
                stats.shard_count += 1
                print(f"[prepare-reasoning] writing shard: {shard_path}")

            if args.progress_every > 0 and stats.lines_seen % args.progress_every == 0:
                print(
                    "[prepare-reasoning] "
                    f"seen={stats.lines_seen} written={stats.examples_written} "
                    f"skip_json={stats.skipped_json_error} "
                    f"skip_missing={stats.skipped_missing_fields} "
                    f"skip_short={stats.skipped_too_short} "
                    f"skip_dupe={stats.skipped_duplicate}"
                )
    finally:
        shard_file.close()

    manifest = {
        "source_url": args.source_url,
        "out_dir": str(out_dir),
        "shard_size": args.shard_size,
        "max_examples": args.max_examples,
        "min_problem_chars": args.min_problem_chars,
        "min_solution_chars": args.min_solution_chars,
        "lines_seen": stats.lines_seen,
        "examples_written": stats.examples_written,
        "skipped_json_error": stats.skipped_json_error,
        "skipped_missing_fields": stats.skipped_missing_fields,
        "skipped_too_short": stats.skipped_too_short,
        "skipped_duplicate": stats.skipped_duplicate,
        "shard_count": stats.shard_count,
        "format": {
            "sample_template": "Question:\\n...\\n\\nAnswer:\\n...",
            "thinking_included": False,
            "fields_used": ["problem", "solution"],
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[prepare-reasoning] done")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
