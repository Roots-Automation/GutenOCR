"""
Goal: Convert TABME++ to a standard, shard-by-*document* format, streaming ("online") as we go.

Input layout:
/mnt/mldata/tabmepp/
    {document_id}/
        0000.jpg
        0000.json
        0001.jpg
        0001.json
        ...

Output layout (1024 documents per shard):
/mnt/mldata/tabmepp-standard/
    train-00000.tar
        {document_id}_{page_id}.jpg|png|tif
        {document_id}_{page_id}.json
    train-00001.tar
    ...

Standard JSON per page:
{
  "text": {
    "words": [{"text": str, "box": [x1, y1, x3, y3]}],
    "lines": [{"text": str, "box": [x1, y1, x3, y3]}]
  },
  "image": {
    "path": "{document_id}_{page_id}.<ext>",  # path inside the tar shard
    "width": int,
    "height": int,
    "dpi": int | null
  }
}

Key improvements vs. previous script:
- Shards are grouped by *documents* (folders), not pages. Each shard contains all pages for up to N documents.
- Streaming writer: we write pages directly into the current shard as each document is processed (no global indexing list).
- Dry-run: reports the number of documents and pages separately, and estimates shards by document count.
- Image extension preserved (jpg/png/tif...). JSON's image.path matches the file name stored *inside* the tar.
- Coordinates rounded to 3 decimals (x1,y1,x3,y3) per requirement.
"""

import argparse
import io
import json
import logging
import os
import tarfile
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def extract_box_coordinates(item: Dict[str, Any]) -> List[float]:
    """Return [x1, y1, x3, y3] rounded to 3 decimals from TABME++ polygon.

    TABME++ provides X1,Y1 (top-left) clockwise to X4,Y4. We only need TL and BR.
    """
    x1 = round(float(item["X1"]), 3)
    y1 = round(float(item["Y1"]), 3)
    x3 = round(float(item["X3"]), 3)
    y3 = round(float(item["Y3"]), 3)
    return [x1, y1, x3, y3]


def get_image_info(image_path: Path) -> Dict[str, Any]:
    """Extract width/height/dpi from an image on disk."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            dpi = img.info.get("dpi")
            dpi_val: Optional[int] = int(dpi[0]) if isinstance(dpi, (tuple, list)) else (
                int(dpi) if isinstance(dpi, (int, float)) else None
            )
            return {
                "width": int(width),
                "height": int(height),
                "dpi": dpi_val,
            }
    except Exception as e:
        logger.error(f"Failed to read image info for {image_path}: {e}")
        return {"width": None, "height": None, "dpi": None}


def convert_tabmepp_to_standard(
    json_data: Dict[str, Any], image_arcname: str, image_src_path: Path
) -> Dict[str, Any]:
    """Convert one TABME++ page json to the standard schema."""
    # words
    words: List[Dict[str, Any]] = []
    for w in json_data.get("words_data", []) or []:
        words.append({"text": w.get("Word", ""), "box": extract_box_coordinates(w)})

    # lines
    lines: List[Dict[str, Any]] = []
    for ln in json_data.get("lines_data", []) or []:
        lines.append({"text": ln.get("Word", ""), "box": extract_box_coordinates(ln)})

    # image
    meta = get_image_info(image_src_path)
    image = {"path": image_arcname, **meta}

    return {"text": {"words": words, "lines": lines}, "image": image}


# -----------------------------------------------------------------------------
# Core processing (streaming / online)
# -----------------------------------------------------------------------------

def find_document_folders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_page_images(doc_folder: Path) -> List[Path]:
    imgs = [p for p in doc_folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def open_shard(output_dir: Path, shard_index: int) -> tarfile.TarFile:
    shard_name = f"train-{shard_index:05d}.tar"
    shard_path = output_dir / shard_name
    logger.info(f"Opening {shard_name}")
    return tarfile.open(shard_path, mode="w")


def add_json_to_tar(tar: tarfile.TarFile, arcname: str, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def process_one_document(
    tar: tarfile.TarFile, doc_folder: Path, strict_pairing: bool = True
) -> Tuple[int, int]:
    """Write all pages of a single document folder into the open shard.

    Returns (pages_written, pages_skipped).
    """
    pages_written = 0
    pages_skipped = 0

    doc_id = doc_folder.name
    page_images = list_page_images(doc_folder)

    for img_path in page_images:
        stem = img_path.stem
        # JSON assumed to share the same stem (case-sensitive .json)
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            msg = f"Missing JSON for image: {img_path}"
            if strict_pairing:
                logger.warning(msg + " — skipping page")
                pages_skipped += 1
                continue
            else:
                logger.warning(msg + " — writing image without text")

        try:
            json_data: Dict[str, Any] = {}
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

            # Target names inside the tar
            image_arcname = f"{doc_id}_{stem}{img_path.suffix.lower()}"
            json_arcname = f"{doc_id}_{stem}.json"

            # Convert & write json
            standard = convert_tabmepp_to_standard(json_data, image_arcname, img_path)
            add_json_to_tar(tar, json_arcname, standard)

            # Write image
            tar.add(img_path, arcname=image_arcname)

            pages_written += 1
        except Exception as e:
            logger.error(f"Error processing page {img_path}: {e}")
            pages_skipped += 1

    return pages_written, pages_skipped


def stream_standardize_tabmepp(
    input_path: Path,
    output_path: Path,
    documents_per_shard: int = 4096,
    strict_pairing: bool = True,
    dry_run: bool = False,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    doc_folders = find_document_folders(input_path)
    if dry_run:
        # Accurate counts by *documents*, plus total pages
        page_total = 0
        samples: List[Tuple[str, int]] = []
        for d in doc_folders:
            n_pages = len(list_page_images(d))
            page_total += n_pages
            if len(samples) < 10:
                samples.append((d.name, n_pages))

        num_docs = len(doc_folders)
        est_shards = ceil(num_docs / max(1, documents_per_shard))

        logger.info("DRY RUN")
        logger.info(f"  Documents: {num_docs}")
        logger.info(f"  Pages: {page_total}")
        logger.info(f"  Shards (by documents): ~{est_shards}")
        for name, n in samples:
            logger.info(f"    {name}: {n} pages")
        if num_docs > len(samples):
            logger.info(f"    ... and {num_docs - len(samples)} more documents")
        return

    # Streaming writing
    shard_index = 0
    docs_in_current = 0
    total_docs = 0
    total_pages_written = 0
    total_pages_skipped = 0

    tar = open_shard(output_path, shard_index)

    try:
        for doc_folder in tqdm(doc_folders, desc="Processing documents"):
            pages_written, pages_skipped = process_one_document(
                tar, doc_folder, strict_pairing=strict_pairing
            )

            total_pages_written += pages_written
            total_pages_skipped += pages_skipped

            total_docs += 1
            docs_in_current += 1

            if docs_in_current >= documents_per_shard:
                tar.close()
                shard_index += 1
                docs_in_current = 0
                tar = open_shard(output_path, shard_index)
    finally:
        # Close the last tar if it has any content
        try:
            tar.close()
        except Exception:
            pass

    logger.info("Done")
    logger.info(f"  Documents processed: {total_docs}")
    logger.info(f"  Pages written: {total_pages_written}")
    logger.info(f"  Pages skipped: {total_pages_skipped}")
    logger.info(f"  Shards created: {shard_index + (1 if docs_in_current > 0 else 0)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Standardize TABME++ (streaming)")
    p.add_argument("input_path", type=Path, help="Path to TABME++ dataset root")
    p.add_argument("output_path", type=Path, help="Path to write shards")
    p.add_argument(
        "--documents-per-shard",
        type=int,
        default=4096,
        help="Max documents per shard (default: 4096)",
    )
    p.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="If set, write images even when matching JSON is missing",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts (documents/pages/shards) without writing files",
    )
    args = p.parse_args()

    stream_standardize_tabmepp(
        input_path=args.input_path,
        output_path=args.output_path,
        documents_per_shard=args.documents_per_shard,
        strict_pairing=not args.allow_unpaired,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
