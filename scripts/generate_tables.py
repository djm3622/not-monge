"""Aggregate baseline benchmark outputs into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import generate_tables_from_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", default=str(ROOT / "outputs"))
    parser.add_argument("--table-root", default=str(ROOT / "artifacts" / "tables"))
    args = parser.parse_args()
    grouped = generate_tables_from_results(args.search_root, args.table_root)
    summary = {key: len(value) for key, value in grouped.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
