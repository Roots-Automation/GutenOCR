"""Content distribution for synthetic table cell text.

Provides word sampling from a bundled distribution (out-of-box) or an
optional UNLV-derived pickle for higher-fidelity output.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

_RESOURCES_DIR = Path(__file__).parent / "resources"
_DEFAULT_DISTRIBUTION = _RESOURCES_DIR / "sample_words.json"


class ContentDistribution:
    """Word pool for sampling synthetic table cell text.

    Args:
        distribution_path: Path to a UNLV pickle (list of word strings) or
            a JSON file with ``{"words": [...], "numbers": [...], "symbols": [...]}``
            keys. If None, loads the bundled sample_words.json.
    """

    def __init__(self, distribution_path: Path | None = None) -> None:
        path = Path(distribution_path) if distribution_path else _DEFAULT_DISTRIBUTION

        if path.suffix == ".pkl":
            with path.open("rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, list):
                self._words = [str(w) for w in raw]
                self._numbers: list[str] = []
                self._symbols: list[str] = ["-", "N/A"]
            elif isinstance(raw, dict):
                self._words = [str(w) for w in raw.get("words", [])]
                self._numbers = [str(n) for n in raw.get("numbers", [])]
                self._symbols = [str(s) for s in raw.get("symbols", ["-", "N/A"])]
            else:
                raise ValueError(f"Unexpected pickle type: {type(raw)}")
        else:
            with path.open() as f:
                data = json.load(f)
            self._words = data.get("words", [])
            self._numbers = data.get("numbers", [])
            self._symbols = data.get("symbols", ["-", "N/A"])

        if not self._words:
            raise ValueError(f"Distribution at {path} contains no words")

        self._all_tokens = self._words + self._numbers + self._symbols

    def sample_cell_content(self, rng: random.Random, *, max_words: int = 3) -> str:
        """Sample a short text string suitable for a table cell.

        Args:
            rng: Seeded random instance.
            max_words: Maximum number of tokens to include.

        Returns:
            A space-joined string of 1–max_words tokens drawn from the pool.
        """
        n = rng.randint(1, max(1, max_words))
        tokens = rng.choices(self._all_tokens, k=n)
        return " ".join(tokens)

    def sample_header_content(self, rng: random.Random) -> str:
        """Sample a short header label (typically a single word).

        Args:
            rng: Seeded random instance.

        Returns:
            A single word from the words-only pool.
        """
        return rng.choice(self._words)
