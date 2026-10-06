"""Fail if `make reproduce` changed any committed file other than figures.

PNG files and the images embedded in executed notebooks are excepted, because fonts and
anti-aliasing differ by OS. Every other file (tables, JSON, CSV, parquet, notebook text
and numbers) must match the commit byte for byte, and no untracked file may appear.
"""

from __future__ import annotations

import json
import subprocess
import sys

IMAGE_MIMES = ("image/png", "image/svg+xml", "image/jpeg")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def strip_images(notebook: dict) -> dict:
    for cell in notebook.get("cells", []):
        for out in cell.get("outputs", []):
            for mime in IMAGE_MIMES:
                out.get("data", {}).pop(mime, None)
    return notebook


def main() -> int:
    problems = []
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        status, path = line[:2], line[3:]
        if path.endswith(".png"):
            continue
        if status.strip() == "M" and path.endswith(".ipynb"):
            before = strip_images(json.loads(git("show", f"HEAD:{path}")))
            with open(path) as f:
                after = strip_images(json.load(f))
            if before == after:
                continue
        problems.append(f"{status} {path}")
    if problems:
        print("Reproduced files differ from the commit:\n  " + "\n  ".join(problems))
        print(git("diff", "--stat", "--", ".", ":(exclude)*.png"))
        return 1
    print("All reproduced results match the commit (figures excepted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
