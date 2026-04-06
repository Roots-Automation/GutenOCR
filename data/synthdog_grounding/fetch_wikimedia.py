#!/usr/bin/env python3
"""Download resources listed in a SynthDoG YAML manifest via the Wikimedia API.

Replaces fetch_papers.py and fetch_backgrounds.py with a single polite script
that resolves thumbnail URLs through the Wikimedia action API (never hits
upload.wikimedia.org directly), enforces serial downloads with a 2 s gap, and
validates every download with PIL before writing to disk.

Manifest schema
---------------
Each entry must have either ``wikimedia_title`` (resolved via the API) or a
direct ``url``.  All other fields are identical to the old manifests.

  - filename:        target filename under ``--root``
  - wikimedia_title: "File:Foo.jpg"   # Wikimedia Commons file title
  - thumbnail_width: 1920             # px; passed to iiurlwidth
  - url:             https://…        # direct URL (non-Wikimedia sources)
  - sha256:          null | <hex>     # null → first-time download
  - raw_sha256:      null | <hex>     # hash before center_crop (optional)
  - center_crop:     0.9              # crop to centre fraction (optional)
  - license:         CC0-1.0
  - source:          https://…
  - notes: >
      …

sha256: null workflow
---------------------
* File missing  → download, validate, save, print "sha256 missing — computed: <hash>"
* File present  → print "sha256 missing — computed: <hash>" and skip re-download
In both cases the manifest is NOT auto-updated; the human pastes the value in.

Usage
-----
    cd data/synthdog_grounding
    uv run python fetch_wikimedia.py --manifest resources/paper/papers.yaml \\
                                     --root resources/paper
    uv run python fetch_wikimedia.py --manifest resources/background/backgrounds.yaml \\
                                     --root resources/background
    uv run python fetch_wikimedia.py --force ...
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

_UA = "GutenOCR-fetch/1.0 (https://github.com/Roots-Automation/GutenOCR/; hunter.heidenreich@roots.ai)"
_SLEEP = 2.0  # seconds between network requests


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _wikimedia_thumbnail_url(title: str, width: int) -> str:
    """Return the thumbnail URL for *title* at *width* px via the action API."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": str(width),
            "format": "json",
        }
    )
    api_url = f"https://commons.wikimedia.org/w/api.php?{params}"
    req = urllib.request.Request(api_url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page["imageinfo"][0]["thumburl"]


def _download_bytes(url: str) -> bytes:
    """Download *url* and return the raw bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _validate_image(data: bytes) -> None:
    """Raise if *data* is not a valid image (uses PIL.Image.verify)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.verify()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


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


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------


def fetch_resources(manifest: Path, root: Path, *, force: bool = False) -> bool:
    """Download all resources described in *manifest* into *root*.

    Returns True if all entries succeeded (or were up-to-date).
    """
    with open(manifest, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # The top-level key differs between manifests ("papers", "backgrounds", …).
    entries: list[dict] = []
    for v in (data or {}).values():
        if isinstance(v, list):
            entries = v
            break

    if not entries:
        print("No entries found in manifest.")
        return True

    skipped = downloaded = failed = 0
    need_sleep = False  # sleep before the *next* network request

    for entry in entries:
        filename: str = entry["filename"]
        dest = root / filename
        wikimedia_title: str | None = entry.get("wikimedia_title")
        direct_url: str | None = entry.get("url")
        expected_sha: str | None = entry.get("sha256")
        raw_sha: str | None = entry.get("raw_sha256")
        center_crop: float | None = entry.get("center_crop")
        thumbnail_width: int = int(entry.get("thumbnail_width", 1920))

        # ── sha256: null workflow ──────────────────────────────────────────
        if expected_sha is None:
            if dest.exists() and not force:
                computed = _sha256_file(dest)
                print(f"  {filename}: sha256 missing — computed: {computed}")
                skipped += 1
                continue
            # Fall through to download below.

        else:
            # ── Normal skip / verify ───────────────────────────────────────
            if not force and dest.exists():
                actual = _sha256_file(dest)
                if actual == expected_sha:
                    skipped += 1
                    continue
                print(f"  {filename}: checksum mismatch, re-downloading")

        # ── Download ───────────────────────────────────────────────────────
        print(f"  {filename} …", end=" ", flush=True)

        if need_sleep:
            time.sleep(_SLEEP)
        need_sleep = True  # next entry will sleep before its request

        try:
            # Resolve the final URL.
            if wikimedia_title is not None:
                url = _wikimedia_thumbnail_url(wikimedia_title, thumbnail_width)
            elif direct_url is not None:
                url = direct_url
            else:
                raise ValueError("entry has neither 'wikimedia_title' nor 'url'")

            raw_data = _download_bytes(url)
            _validate_image(raw_data)

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw_data)

            # sha256: null → print computed hash, don't verify against manifest.
            if expected_sha is None:
                computed = _sha256_bytes(raw_data)
                if center_crop is not None:
                    _center_crop(dest, center_crop)
                    computed_after = _sha256_file(dest)
                    print(f"OK  (raw_sha256: {computed}  sha256: {computed_after})")
                else:
                    print(f"OK  (sha256: {computed})")
                downloaded += 1
                continue

            # Verify hash.
            verify_sha = raw_sha if raw_sha is not None else expected_sha
            actual = _sha256_bytes(raw_data)
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
            dest.unlink(missing_ok=True)
            failed += 1

    print()
    print(f"Done: {downloaded} downloaded, {skipped} up-to-date, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SynthDoG resources via the Wikimedia action API")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a resource YAML manifest (papers.yaml or backgrounds.yaml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory where downloaded files are saved",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all files even if checksums match",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    ok = fetch_resources(args.manifest, args.root, force=args.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
