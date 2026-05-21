"""cleanup.py — Remove generated artifacts (cache, logs, __pycache__, etc.)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

TARGETS = {
    "cache": [
        PROJECT_ROOT / "data" / "cache",
        PROJECT_ROOT / "__pycache__",
    ],
    "logs": [PROJECT_ROOT / "logs"],
    "vectordb": [PROJECT_ROOT / "data" / "vectordb"],
    "embeddings": [PROJECT_ROOT / "data" / "embeddings"],
    "pycache": list(PROJECT_ROOT.rglob("__pycache__")),
    "pytest": [
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / "htmlcov",
        PROJECT_ROOT / ".coverage",
    ],
}


def _remove(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"  [dry-run] would remove: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"  Removed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up generated project artifacts.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=list(TARGETS) + ["all"],
        default=["cache", "logs", "pycache", "pytest"],
        help="Which artifact groups to remove (default: cache logs pycache pytest).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    selected = list(TARGETS.keys()) if "all" in args.targets else args.targets
    print(f"Cleaning: {', '.join(selected)}" + (" [dry-run]" if args.dry_run else ""))

    for target in selected:
        paths = TARGETS.get(target, [])
        # Re-evaluate pycache glob each run
        if target == "pycache":
            paths = list(PROJECT_ROOT.rglob("__pycache__"))
        for path in paths:
            _remove(path, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
