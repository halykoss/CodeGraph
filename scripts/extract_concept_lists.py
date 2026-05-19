#!/usr/bin/env python3
"""Extract normalized concept lists from CodeGraph annotation outputs."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable


AXES = ("domains", "algorithms", "paradigms", "design_patterns")


def normalize_text(text: object) -> str:
    """Canonical text normalization matching build_dataset.py normalize_text()."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = text.replace("c++", "cpp").replace("c#", "csharp").replace("f#", "fsharp")
    text = re.sub(r"[-_/\\|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def parse_semantic_value(value: Any) -> Any:
    """Parse JSON/Python-like semantic values while preserving plain strings."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return ""

    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except (TypeError, json.JSONDecodeError):
            pass
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError, TypeError):
            return value

    return value


def concept_from_object(item: dict[str, Any]) -> str:
    """Extract only the concept name from an annotation object."""
    name = item.get("name")
    return name if isinstance(name, str) else ""


def iter_concepts(axis: str, value: Any) -> Iterable[str]:
    """Yield raw concept strings from one semantic cell."""
    parsed = parse_semantic_value(value)

    if isinstance(parsed, str):
        yield parsed
        return

    if isinstance(parsed, dict):
        concept = concept_from_object(parsed)
        if concept:
            yield concept
        return

    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                concept = concept_from_object(item)
                if concept:
                    yield concept
            elif axis == "domains":
                continue
            else:
                continue


def add_row_concepts(row: dict[str, Any], concepts: dict[str, set[str]]) -> None:
    for axis in AXES:
        if axis not in row:
            continue
        try:
            for raw in iter_concepts(axis, row[axis]):
                normalized = normalize_text(raw)
                if normalized:
                    concepts[axis].add(normalized)
        except Exception:
            continue


def iter_json_rows(value: Any) -> Iterable[dict[str, Any]]:
    """Yield likely annotation rows from legacy JSON shard structures."""
    if isinstance(value, dict):
        if any(axis in value for axis in AXES):
            yield value
            return

        for key in ("rows", "results", "data", "samples", "annotations"):
            nested = value.get(key)
            if isinstance(nested, (list, dict)):
                yield from iter_json_rows(nested)
                return

        for nested in value.values():
            if isinstance(nested, (list, dict)):
                yield from iter_json_rows(nested)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_rows(item)


def read_json_file(path: Path, concepts: dict[str, set[str]], verbose: bool) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
        return

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        for row in iter_json_rows(parsed):
            add_row_concepts(row, concepts)
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed_line = json.loads(stripped)
        except json.JSONDecodeError:
            if verbose:
                print(f"Skipping malformed JSON line {path}:{line_number}", file=sys.stderr)
            continue
        for row in iter_json_rows(parsed_line):
            add_row_concepts(row, concepts)


def read_arrow_file(path: Path, concepts: dict[str, set[str]], verbose: bool) -> None:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Arrow/Feather inputs") from exc

    try:
        dataset = ds.dataset(str(path), format="ipc")
        columns = [axis for axis in AXES if axis in dataset.schema.names]
        if not columns:
            return
        for batch in dataset.scanner(columns=columns).to_batches():
            for row in batch.to_pylist():
                add_row_concepts(row, concepts)
    except Exception as exc:
        if verbose:
            print(f"Skipping unreadable Arrow/Feather file {path}: {exc}", file=sys.stderr)


def discover_input_files(input_dir: Path, max_files: int) -> list[Path]:
    suffixes = {".arrow", ".feather", ".json"}
    files = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in suffixes)
    if max_files > 0:
        return files[:max_files]
    return files


def write_outputs(output_root: Path, concepts: dict[str, set[str]]) -> None:
    for axis in AXES:
        output_path = output_root / axis / f"{axis}_list.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = sorted(concepts[axis])
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized, deduplicated semantic concept lists."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("output"), type=Path)
    parser.add_argument("--max-files", default=0, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_root = args.output_root

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if args.max_files < 0:
        print("--max-files must be >= 0", file=sys.stderr)
        return 2

    concepts: dict[str, set[str]] = {axis: set() for axis in AXES}
    files = discover_input_files(input_dir, args.max_files)

    for path in files:
        if args.verbose:
            print(f"Reading {path}", file=sys.stderr)
        suffix = path.suffix.lower()
        if suffix == ".json":
            read_json_file(path, concepts, args.verbose)
        elif suffix in {".arrow", ".feather"}:
            read_arrow_file(path, concepts, args.verbose)

    write_outputs(output_root, concepts)

    for axis in AXES:
        print(f"{axis}: {len(concepts[axis])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
