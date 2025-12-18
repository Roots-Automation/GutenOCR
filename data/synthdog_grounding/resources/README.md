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
│   └── zh/         # Chinese fonts
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
organized by language.

**Supported formats:** TTF, OTF

**Language directories:**
- `en/` - English and Latin-script fonts
- `ja/` - Japanese fonts (must support Hiragana, Katakana, and Kanji)
- `ko/` - Korean fonts (must support Hangul)
- `zh/` - Chinese fonts (must support Simplified/Traditional characters)

**Requirements:**
- Fonts must support the character set of the target language
- Both serif and sans-serif fonts are recommended for variety
- Consider including monospace fonts for technical document simulation
- Font licensing must permit synthetic data generation

**Adding custom fonts:**
Add `.ttf` or `.otf` files to the appropriate language directory.

**Included English fonts:**
- `NotoSans-Regular.ttf` - Clean sans-serif (Google)
- `NotoSerif-Regular.ttf` - Classic serif (Google)
- Various stylized fonts for variety

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
- **Noto Fonts**: https://fonts.google.com/noto (free, open source)
- **Google Fonts**: https://fonts.google.com/ (free)

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
