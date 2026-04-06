#!/usr/bin/env python3
"""Download fonts listed in resources/font/fonts.yaml.

Checks each font's SHA-256 before downloading; skips files that already
match.  Use ``--force`` to re-download everything.

Usage:
    uv run python fetch_fonts.py
    uv run python fetch_fonts.py --force
    uv run python fetch_fonts.py --manifest path/to/fonts.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

MANIFEST_DEFAULT = Path(__file__).resolve().parent / "resources" / "font" / "fonts.yaml"
FONT_ROOT = Path(__file__).resolve().parent / "resources" / "font"

# Reuse downloaded archives within a single run so we don't fetch the same
# archive once per font that lives inside it.
_archive_cache: dict[str, Path] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* using only the stdlib."""
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_fonts/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)


def _download_cached(url: str, tmpdir: Path) -> Path:
    """Download *url* once per run, returning the cached local path."""
    if url in _archive_cache:
        return _archive_cache[url]
    fname = url.rsplit("/", 1)[-1]
    dest = tmpdir / fname
    print(f"  Downloading archive {fname} …")
    _download(url, dest)
    _archive_cache[url] = dest
    return dest


def _extract_from_zip(archive: Path, zip_path: str) -> bytes:
    """Extract *zip_path* from a zip archive."""
    with zipfile.ZipFile(archive) as zf:
        # Try exact path first, then fall back to basename match
        try:
            return zf.read(zip_path)
        except KeyError:
            basename = zip_path.rsplit("/", 1)[-1]
            for name in zf.namelist():
                if name.endswith("/" + basename) or name == basename:
                    return zf.read(name)
    raise FileNotFoundError(f"{zip_path} not found in {archive}")


def fetch_fonts(manifest: Path, *, force: bool = False) -> bool:
    """Download fonts from *manifest*.  Returns True if all succeeded."""
    with open(manifest, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    fonts = data.get("fonts", [])
    if not fonts:
        print("No fonts listed in manifest.")
        return True

    skipped = 0
    downloaded = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for entry in fonts:
            if entry.get("bundled"):
                continue

            filename: str = entry["filename"]
            lang: str = entry["lang"]
            url: str = entry["url"]
            expected_sha: str = entry["sha256"]
            zip_path: str | None = entry.get("zip_path")

            dest = FONT_ROOT / lang / filename

            # Check existing file
            if not force and dest.exists():
                actual = _sha256(dest)
                if actual == expected_sha:
                    skipped += 1
                    continue
                print(f"  {lang}/{filename}: checksum mismatch, re-downloading")

            print(f"  {lang}/{filename} …", end=" ", flush=True)

            try:
                if zip_path:
                    # Font lives inside a zip archive
                    local_archive = _download_cached(url, tmpdir_path)
                    font_bytes = _extract_from_zip(local_archive, zip_path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(font_bytes)
                else:
                    # Direct download
                    _download(url, dest)

                actual = _sha256(dest)
                if actual != expected_sha:
                    print(f"FAILED (sha256 {actual[:12]}… != {expected_sha[:12]}…)")
                    dest.unlink(missing_ok=True)
                    failed += 1
                    continue

                print("OK")
                downloaded += 1

            except Exception as exc:
                print(f"FAILED ({exc})")
                failed += 1

    print()
    print(f"Done: {downloaded} downloaded, {skipped} up-to-date, {failed} failed")
    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fonts for SynthDoG")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DEFAULT,
        help="Path to fonts.yaml manifest (default: resources/font/fonts.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all fonts even if checksums match",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    ok = fetch_fonts(args.manifest, force=args.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
