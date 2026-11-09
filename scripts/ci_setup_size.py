"""Print setup exe size and fail when it is far from the v3.19 installer."""
from __future__ import annotations

import os
import sys

# Rapid.Scribe.v3.19-Setup.exe on the v3.19 GitHub release.
V3_19_BYTES = 60_906_958
MIN_BYTES = 40_000_000
MAX_BYTES = 120_000_000


def size_error(size: int) -> str | None:
    if size < MIN_BYTES or size > MAX_BYTES:
        return f"Setup size {size} is outside the {MIN_BYTES}-{MAX_BYTES} sanity window"
    return None


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: ci_setup_size.py <setup.exe>")
    path = sys.argv[1]
    size = os.path.getsize(path)
    print(f"ASSET {path}")
    print(f"SIZE_BYTES {size}")
    print(f"V3_19_BYTES {V3_19_BYTES}")
    print(f"DELTA_BYTES {size - V3_19_BYTES}")
    error = size_error(size)
    if error:
        sys.exit(error)


if __name__ == "__main__":
    main()
