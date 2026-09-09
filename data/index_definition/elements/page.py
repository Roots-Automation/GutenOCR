"""Paper and background rendering for index_definition generator."""

from pathlib import Path

import numpy as np
from PIL import Image


class PageRenderer:
    def __init__(self, config: dict, project_root: Path):
        bg_cfg = config.get("background", {})
        self._bg_paths = self._collect_images(bg_cfg.get("paths", []), project_root)
        self._bg_prob = float(bg_cfg.get("prob", 0.25))
        self._bg_opacity = bg_cfg.get("opacity", [0.03, 0.08])
        self._paper_paths = self._collect_images(config.get("paper", {}).get("texture_paths", []), project_root)
        self._paper_color_cfg = config.get("paper", {}).get("color", {})
        self._texture_alpha = config.get("paper", {}).get("texture_alpha", [0.0, 0.15])

    @staticmethod
    def _collect_images(paths: list[str], root: Path) -> list[Path]:
        result = []
        for p in paths:
            d = (root / p).resolve()
            if d.is_dir():
                result.extend(sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")))
        return result

    def generate(self, width: int, height: int, rng: np.random.Generator) -> tuple[Image.Image, tuple[int, int, int]]:
        # Paper color
        paper_rgb: tuple[int, int, int] = (255, 255, 255)
        color_cfg = self._paper_color_cfg
        if rng.random() < color_cfg.get("prob", 0.5):
            rgb_ranges = color_cfg.get("rgb", [[200, 255], [200, 255], [185, 255]])
            paper_rgb = tuple(int(rng.integers(r[0], r[1] + 1)) for r in rgb_ranges)  # type: ignore[assignment]

        paper = Image.new("RGB", (width, height), paper_rgb)

        # Paper texture overlay (subtle grain, max 10% blend)
        if self._paper_paths:
            tex_path = self._paper_paths[int(rng.integers(len(self._paper_paths)))]
            tex = Image.open(tex_path).convert("L").resize((width, height), Image.BILINEAR)
            alpha = float(rng.uniform(*self._texture_alpha))
            tex_rgb = Image.merge("RGB", [tex, tex, tex])
            paper = Image.blend(paper, tex_rgb, alpha)

        # Background: optionally composite a background photo at very low opacity
        # so the page still reads as paper, not a photo.
        bg_prob = self._bg_prob
        if self._bg_paths and rng.random() < bg_prob:
            bg_path = self._bg_paths[int(rng.integers(len(self._bg_paths)))]
            bg = Image.open(bg_path).convert("RGB").resize((width, height), Image.BILINEAR)
            bg_opacity = float(rng.uniform(*self._bg_opacity))
            page = Image.blend(paper, bg, bg_opacity)
        else:
            page = paper

        # Compute median paper color for adaptive text coloring
        arr = np.array(paper)
        median_rgb = tuple(int(np.median(arr[:, :, c])) for c in range(3))  # type: ignore[assignment]

        return page.convert("RGBA"), median_rgb  # type: ignore[return-value]
