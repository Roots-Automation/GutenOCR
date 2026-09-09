"""WordNet-based definition corpus for index_definition generator."""

import numpy as np


class DefinitionCorpus:
    def __init__(self, entries: list[tuple[str, str]]):
        self._entries = entries

    @classmethod
    def load(cls, min_definition_length: int = 10) -> "DefinitionCorpus":
        import nltk

        try:
            from nltk.corpus import wordnet as wn

            wn.synsets("test")  # probe — raises LookupError if not downloaded
        except LookupError:
            nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet as wn

        entries = []
        for synset in wn.all_synsets():
            defn = synset.definition()
            if not defn or len(defn) < min_definition_length:
                continue
            lemmas = synset.lemmas()
            if not lemmas:
                continue
            headword = lemmas[0].name().replace("_", " ")
            entries.append((headword, defn))
        return cls(entries)

    def shuffled_batch(self, rng: np.random.Generator) -> list[tuple[str, str]]:
        """Return all entries in rng-determined random (non-alphabetical) order."""
        idx = rng.permutation(len(self._entries))
        return [self._entries[i] for i in idx]

    def __len__(self) -> int:
        return len(self._entries)
