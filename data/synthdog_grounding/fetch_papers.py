#!/usr/bin/env python3
"""Download paper textures listed in resources/paper/papers.yaml.

Checks each texture's SHA-256 before downloading; skips files that already
match.  Use ``--force`` to re-download everything.

If an entry has ``center_crop: <fraction>`` the downloaded image is cropped to
that fraction of its width and height (centred) after SHA-256 verification.
The SHA-256 in the manifest is always for the *raw downloaded* file, so
integrity is checked before the crop is applied.

Usage:
    uv run python fetch_papers.py
    uv run python fetch_papers.py --force
    uv run python fetch_papers.py --manifest path/to/papers.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

MANIFEST_DEFAULT = Path(__file__).resolve().parent / "resources" / "paper" / "papers.yaml"
PAPER_ROOT = Path(__file__).resolve().parent / "resources" / "paper"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* using only the stdlib."""
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_papers/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)


def _center_crop(path: Path, fraction: float) -> None:
    """Crop *path* in-place to the centre *fraction* of its dimensions."""
    from PIL import Image

    with Image.open(path) as img:
        w, h = img.size
        new_w = int(w * fraction)
        new_h = int(h * fraction)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        cropped.save(path)


def fetch_papers(manifest: Path, *, force: bool = False) -> bool:
    """Download paper textures from *manifest*.  Returns True if all succeeded."""
    with open(manifest, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    papers = data.get("papers", [])
    if not papers:
        print("No papers listed in manifest.")
        return True

    skipped = 0
    downloaded = 0
    failed = 0

    for entry in papers:
        filename: str = entry["filename"]
        url: str = entry["url"]
        expected_sha: str = entry["sha256"]
        raw_sha: str | None = entry.get("raw_sha256")
        center_crop: float | None = entry.get("center_crop")

        dest = PAPER_ROOT / filename

        # Skip check uses sha256 — the final on-disk file (post-crop if any).
        if not force and dest.exists():
            actual = _sha256(dest)
            if actual == expected_sha:
                skipped += 1
                continue
            print(f"  {filename}: checksum mismatch, re-downloading")

        print(f"  {filename} …", end=" ", flush=True)

        try:
            _download(url, dest)

            # Verify raw download against raw_sha256 if provided, else sha256.
            verify_sha = raw_sha if raw_sha is not None else expected_sha
            actual = _sha256(dest)
            if actual != verify_sha:
                print(f"FAILED (sha256 {actual[:12]}… != {verify_sha[:12]}…)")
                dest.unlink(missing_ok=True)
                failed += 1
                continue

            if center_crop is not None:
                _center_crop(dest, center_crop)

            print("OK")
            downloaded += 1

        except Exception as exc:
            print(f"FAILED ({exc})")
            failed += 1

    print()
    print(f"Done: {downloaded} downloaded, {skipped} up-to-date, {failed} failed")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download paper textures for SynthDoG")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DEFAULT,
        help="Path to papers.yaml manifest (default: resources/paper/papers.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all textures even if checksums match",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    ok = fetch_papers(args.manifest, force=args.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
