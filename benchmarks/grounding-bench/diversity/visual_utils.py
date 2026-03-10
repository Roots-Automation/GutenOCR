import logging
import os
from collections.abc import Iterator

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

SIGLIP_ID = "google/siglip2-so400m-patch16-naflex"


def get_image_files(directory: str) -> list[str]:
    """Returns sorted list of image filenames in directory (recursively)."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if os.path.splitext(f.lower())[1] in valid_exts:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, directory)
                files.append(rel_path)
    files.sort()
    return files


def load_siglip_model(device: str) -> tuple[AutoProcessor, AutoModel]:
    """Loads SigLIP model and processor."""
    logger.info(f"Loading SigLIP model: {SIGLIP_ID}...")
    processor = AutoProcessor.from_pretrained(SIGLIP_ID)
    model = AutoModel.from_pretrained(SIGLIP_ID, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def batch_iterator(iterator: Iterator, batch_size: int) -> Iterator[list]:
    """Yields batches from an iterator."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_embeddings(
    image_iterator: Iterator[Image.Image],
    total_count: int = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Computes normalized embeddings for an iterator of PIL Images.

    Args:
        image_iterator: Iterator yielding PIL.Image objects.
        total_count: Optional total count for progress bar.
        batch_size: Batch size for inference.
        device: 'cuda' or 'cpu'.
    """
    processor, model = load_siglip_model(device)
    all_embeds = []

    # Calculate number of batches for tqdm if total_count is known
    total_batches = (total_count + batch_size - 1) // batch_size if total_count else None

    batched_iter = batch_iterator(image_iterator, batch_size)

    for batch_imgs in tqdm(batched_iter, total=total_batches, desc="Embedding Images"):
        # Validate and convert images to RGB
        clean_imgs = []
        for img in batch_imgs:
            if img.mode != "RGB":
                img = img.convert("RGB")
            clean_imgs.append(img)

        if not clean_imgs:
            continue

        inputs = processor(images=clean_imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy())

    if not all_embeds:
        return np.empty((0, 0))

    return np.vstack(all_embeds)
