# SynthDoG Resources

This directory contains the resource files required for synthetic document generation.
All resources are organized by type and language where applicable.

## Directory Structure

```
resources/
├── background/     # Background texture images
├── corpus/         # Text corpus files for each language
├── font/           # Font files organized by language
│   ├── en/         # English fonts
│   ├── ja/         # Japanese fonts
│   ├── ko/         # Korean fonts
│   ├── zh/         # Chinese fonts
│   └── fonts.yaml  # Font manifest (source of truth)
└── paper/          # Paper texture images
```

## Resource Types

### Background (`background/`)

Background images used for the document background layer. These appear behind
the paper/document itself, simulating desk surfaces, shadows, etc.

**Supported formats:** PNG, JPG, JPEG, TIFF, BMP

**Requirements:**
- Images should be large enough to cover typical document sizes (recommended: 1024x1024 or larger)
- RGB or RGBA format
- Various textures work well: wood grain, fabric, solid colors, gradients

**Adding custom backgrounds:**
Simply add image files to the `background/` directory. All images will be
randomly selected during generation.

---

### Paper (`paper/`)

Paper texture images that simulate the document surface. These are overlaid
with the text content.

**Supported formats:** PNG, JPG, JPEG, TIFF, BMP

**Requirements:**
- Should be tileable or large enough to cover document sizes
- Light-colored textures work best (white, cream, light gray)
- Can include paper grain, slight discoloration, or aging effects

**Adding custom paper textures:**
Add image files to the `paper/` directory.

---

### Fonts (`font/<language>/`)

TrueType (`.ttf`) or OpenType (`.otf`) font files for text rendering,
organized by language. Font binaries are **not stored in git** — they are
downloaded on demand from the URLs listed in `font/fonts.yaml`.

**Supported formats:** TTF, OTF

**Language directories:**
- `en/` - English and Latin-script fonts
- `ja/` - Japanese fonts (must support Hiragana, Katakana, and Kanji)
- `ko/` - Korean fonts (must support Hangul)
- `zh/` - Chinese fonts (must support Simplified/Traditional characters)

**Downloading fonts:**

```bash
cd data/synthdog_grounding
uv run python fetch_fonts.py
```

The script reads `resources/font/fonts.yaml`, downloads any missing fonts, and
verifies SHA-256 checksums. Use `--force` to re-download all fonts.

**Adding new fonts:**

1. Add the font file to the appropriate `font/<lang>/` directory
2. Add an entry to `font/fonts.yaml` with the filename, language, download URL,
   SHA-256 checksum (`shasum -a 256 <file>`), license, and source
3. The font file itself is gitignored — only the manifest entry is tracked

**Included fonts (downloaded on demand):**
- **Noto Sans** — Clean sans-serif; Regular, Bold, Italic, Bold Italic, Condensed (×4) (Google, OFL-1.1)
- **Noto Serif** — Classic serif; Regular, Bold, Italic, Bold Italic (Google, OFL-1.1)
- **Noto Sans Mono** — Monospace for code and technical text (Google, OFL-1.1)
- **Open Sans** — Humanist sans-serif; Regular, Bold, Italic, Bold Italic (Google, OFL-1.1)
- **Roboto** — Geometric sans-serif; Regular, Bold, Italic, Bold Italic, Condensed (×4) (Google, Apache-2.0)
- **Source Serif 4** — Transitional serif; Regular, Bold, Italic, Bold Italic (Adobe, OFL-1.1)
- **Courier Prime** — Typewriter-style monospace; Regular, Bold, Italic, Bold Italic (Quote-Unquote Apps, OFL-1.1)
- **Noto Sans JP** / **Noto Serif JP** — Japanese, regular and bold (Google, OFL-1.1)
- **Noto Sans KR** / **Noto Serif KR** — Korean, regular and bold (Google, OFL-1.1)
- **Noto Sans SC** / **Noto Serif SC** — Chinese Simplified, regular and bold (Google, OFL-1.1)

**Included fonts (bundled in git):**
- **[Erratic Cursive](https://www.fontspace.com/erratic-cursive-font-f121261)** — Handwritten cursive with irregular letterforms (GGBotNet, CC0-1.0)
- **[Gib Font Plox](https://www.fontspace.com/gib-font-plox-f22438)** — Stylized display font (Cannot Into Space Fonts, Public Domain)
- **[Public Pixel](https://www.fontspace.com/public-pixel-font-f72305)** — Bitmap pixel font (GGBotNet, CC0-1.0)
- **[Scabber](https://www.fontspace.com/scabber-font-f140130)** — Casual hand-drawn typeface (GGBotNet, CC0-1.0)

---

### Corpus (`corpus/`)

Plain text files containing source text for document generation.
Each language has its own corpus file.

**File format:** UTF-8 encoded plain text (`.txt`)

**Files:**
- `enwiki.txt` - English text (from Wikipedia)
- `jawiki.txt` - Japanese text
- `kowiki.txt` - Korean text
- `zhwiki.txt` - Chinese text

**Requirements:**
- UTF-8 encoding (no BOM)
- One sentence or paragraph per line works well
- Remove special characters that aren't typical for documents
- Larger corpora provide more variety in generated documents

**Creating custom corpora:**
1. Prepare a plain text file with diverse, representative text
2. Ensure proper encoding (UTF-8)
3. Name it appropriately and add to `corpus/`
4. Update the config YAML to reference the new corpus file

**Alternative: HuggingFace Datasets**

Instead of static corpus files, you can use HuggingFace datasets for streaming
text. See `config/config_huggingface.yaml` for an example configuration.

---

## Configuration

Resources are referenced in the config YAML files under `config/`.
For example, in `config_en.yaml`:

```yaml
document:
  content:
    font:
      paths: ["resources/font/en"]
    corpus:
      paths: ["resources/corpus/enwiki.txt"]
background:
  path: "resources/background"
paper:
  path: "resources/paper"
```

## Obtaining Resources

### Fonts
Fonts are managed via the manifest at `font/fonts.yaml`. Run `fetch_fonts.py`
to download them. See the Fonts section above for details.

### Backgrounds & Paper
- Create your own using photo editing software
- Use texture sites (ensure license permits your use case)

### Corpora
- Wikipedia dumps: https://dumps.wikimedia.org/
- HuggingFace Datasets: https://huggingface.co/datasets

## Notes

- Empty directories will cause generation to fail
- At minimum, you need at least one file in each resource directory
- For best results, include a variety of each resource type
- Larger resource pools produce more diverse outputs
