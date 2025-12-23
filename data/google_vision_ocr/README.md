# Google Vision OCR Module

> **Module**: `data/google_vision_ocr/` - OCR processing with Google Cloud Vision API
>
> This module provides OCR functionality using Google Cloud Vision API with support for multiple detection modes and document types.

## Overview

The Google Vision OCR module provides OCR capabilities through Google Cloud Vision API. It supports three distinct modes for different use cases:

-   **Text Detection**: Basic text recognition in images (printed text)
-   **Document Detection**: Advanced text recognition including handwriting and complex layouts
-   **PDF Processing**: Asynchronous text extraction from PDF/TIFF files via Google Cloud Storage

## Features

### Core Capabilities

-   **Multiple OCR Modes**: Text, document, and PDF processing modes
-   **Compatible Output Format**: Standardized output structure for easy integration
-   **Advanced Line Clustering**: Intelligent word-to-line grouping based on vertical alignment
-   **Multi-level Data**: Word, line, and paragraph-level text extraction
-   **Bounding Box Support**: Precise 4-point polygon coordinates for all detected text
-   **Confidence Scores**: Text detection confidence when available
-   **Multi-page Support**: PDF processing with page-level organization
-   **Error Handling**: Comprehensive error reporting and graceful failure handling

### Supported Formats

-   **Images**: PNG, JPG, GIF, BMP, WebP, RAW, ICO, PDF, TIFF
-   **Documents**: PDF and TIFF files via Google Cloud Storage
-   **Text Types**: Printed text, handwritten text, mixed layouts

## Installation

### Prerequisites

1.  **Google Cloud Project** with Vision API enabled
2.  **Authentication** via service account or application default credentials
3.  **Python Dependencies** for Google Cloud libraries

### Install Dependencies

Add the Google Cloud dependencies to your project:

```bash
# Install Google Cloud Vision and Storage
pip install google-cloud-vision google-cloud-storage

# Or add to requirements
uv add google-cloud-vision google-cloud-storage
```

### Authentication Setup

Choose one of these authentication methods:

**Method 1: Service Account Key File**

```bash
# Download service account JSON from Google Cloud Console
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Method 2: Application Default Credentials**

```bash
# If running on Google Cloud (App Engine, Cloud Run, etc.)
gcloud auth application-default login
```

**Method 3: Environment Variables**

```python
# Pass credentials path directly to the extractor
extractor = GoogleVisionOCRExtractor(credentials_path="/path/to/credentials.json")
```

## Usage

### Basic Usage

```python
from data.google_vision_ocr import GoogleVisionOCRExtractor

# Initialize the extractor
extractor = GoogleVisionOCRExtractor()

# Basic text detection (good for printed text)
result = extractor.extract_ocr("document.png", mode="text")

# Document detection (good for handwriting and complex layouts)
result = extractor.extract_ocr("handwritten.jpg", mode="document")

# Access results
lines = result["lines_data"]
words = result["words_data"]
metadata = result["image_metadata"]
success = result["success"]
```

### OCR Modes

#### 1. Text Detection Mode (`mode="text"`)

Best for: Printed text, clean documents, simple layouts

```python
extractor = GoogleVisionOCRExtractor()
result = extractor.extract_ocr("invoice.png", mode="text")

# Returns individual text elements with bounding boxes
for text in result["words_data"]:
    print(f"Text: {text['Word']}")
    print(f"Bounds: ({text['X1']}, {text['Y1']}) to ({text['X3']}, {text['Y3']})")
```

#### 2. Document Detection Mode (`mode="document"`)

Best for: Handwriting, complex layouts, mixed text types

```python
extractor = GoogleVisionOCRExtractor()
result = extractor.extract_ocr("handwritten_notes.jpg", mode="document")

# Returns hierarchical text structure (paragraphs -> words)
for line in result["lines_data"]:
    print(f"Line: {line['Word']}")
    print(f"Confidence: {line['Confidence']}")

for word in result["words_data"]:
    print(f"Word: {word['Word']}")
    print(f"Confidence: {word['Confidence']}")
```

#### 3. PDF Processing Mode (`mode="pdf"`)

Best for: Multi-page PDFs, batch processing, large documents

```python
extractor = GoogleVisionOCRExtractor()

# Requires Google Cloud Storage URIs
result = extractor.extract_ocr(
    image_path="",  # Not used for PDF mode
    mode="pdf",
    gcs_source_uri="gs://my-bucket/document.pdf",
    gcs_destination_uri="gs://my-bucket/ocr-output/",
    batch_size=2,
    timeout=300
)

# Returns text from all pages
for line in result["lines_data"]:
    print(f"Page {line['page']}: {line['Word']}")
```

### Integration with ArXiv Pipeline

For ArXiv paper processing, the module integrates seamlessly:

```python
# In arxiv processing pipeline
from data.google_vision_ocr import GoogleVisionOCRExtractor

def process_arxiv_paper_with_google_vision(image_path):
    extractor = GoogleVisionOCRExtractor()
    
    # Use text mode for clean academic papers
    result = extractor.extract_ocr(image_path, mode="text")
    
    if result["success"]:
        return result
    else:
        print(f"OCR failed: {result['error']}")
        return None
```

## Output Format

The module returns comprehensive OCR data with intelligent line clustering:

```json
{
  "lines_data": [
    {
      "Word": "Complete line of text clustered from words",
      "Confidence": 0.95,
      "X1": 100, "Y1": 50,
      "X2": 300, "Y2": 50,
      "X3": 300, "Y3": 70,
      "X4": 100, "Y4": 70,
      "page": 1,
      "Index": 0
    }
  ],
  "words_data": [
    {
      "Word": "individual",
      "Confidence": 0.97,
      "X1": 100, "Y1": 50,
      "X2": 170, "Y2": 50,
      "X3": 170, "Y3": 70,
      "X4": 100, "Y4": 70,
      "page": 1,
      "Index": 0
    }
  ],
  "paragraph_data": [
    {
      "Word": "Full paragraph text (document mode only)",
      "Confidence": 0.94,
      "X1": 100, "Y1": 50,
      "X2": 400, "Y2": 50,
      "X3": 400, "Y3": 120,
      "X4": 100, "Y4": 120,
      "page": 1,
      "Index": 0
    }
  ],
  "image_metadata": {
    "width": 1700,
    "height": 2200,
    "format": "PNG",
    "mode": "RGB",
    "dpi": [200, 200]
  },
  "success": true
}
```

### Coordinate System

-   **Format**: 4-point polygon (X1,Y1) → (X2,Y2) → (X3,Y3) → (X4,Y4)
-   **Units**: Actual pixel coordinates
-   **Origin**: Top-left corner (0,0)
-   **Rotation**: Supports rotated text bounding boxes

## Advanced Configuration

### Custom Credentials

```python
# Use specific service account
extractor = GoogleVisionOCRExtractor(
    credentials_path="/path/to/service-account.json"
)
```

### PDF Processing Options

```python
# Configure PDF processing
result = extractor.extract_ocr(
    image_path="",
    mode="pdf",
    gcs_source_uri="gs://bucket/large-document.pdf",
    gcs_destination_uri="gs://bucket/output/",
    batch_size=5,    # Pages per output file
    timeout=600      # Wait up to 10 minutes
)
```

### Error Handling

```python
result = extractor.extract_ocr("document.png", mode="text")

if not result["success"]:
    error_msg = result["error"]
    print(f"OCR failed: {error_msg}")
    
    # Common error types:
    # - "Image file not found"
    # - "Google Vision API error: [details]"
    # - "Failed to initialize Google Vision client"
    # - "Invalid mode: [mode]"
```

## Line Clustering Algorithm

The Google Vision OCR module includes an intelligent line clustering algorithm that groups words into lines based on their vertical alignment:

### How It Works

1.  **Vertical Position Analysis**: Groups words by their Y1 (top) and Y3 (bottom) coordinates
2.  **Tolerance-Based Clustering**: Words within a configurable tolerance (default: 5 pixels) are grouped together
3.  **Horizontal Ordering**: Words within each line are sorted left-to-right by X1 coordinate
4.  **Bounding Box Calculation**: Line bounding boxes span from leftmost to rightmost word

### Configuration

```python
# Customize line clustering tolerance
def _cluster_words_into_lines(self, words_data, tolerance=5):
    # tolerance: pixel tolerance for vertical alignment
    # Lower values = stricter line grouping
    # Higher values = more permissive grouping
```

### Output Structure

-   **lines_data**: Clustered lines with combined text and spanning bounding boxes
-   **words_data**: Original word-level detections
-   **paragraph_data**: Document structure paragraphs (document mode only)

## Use Cases

### 1. Academic Paper Processing (ArXiv Integration)

```python
# Process academic papers with clean printed text
extractor = GoogleVisionOCRExtractor()
result = extractor.extract_ocr("arxiv_paper_page.png", mode="text")
```

### 2. Handwritten Document Analysis

```python
# Extract text from handwritten notes or forms
extractor = GoogleVisionOCRExtractor()
result = extractor.extract_ocr("handwritten_form.jpg", mode="document")
```

### 3. Large PDF Processing

```python
# Process multi-page PDFs asynchronously
extractor = GoogleVisionOCRExtractor()
result = extractor.extract_ocr(
    "", mode="pdf",
    gcs_source_uri="gs://docs/book.pdf",
    gcs_destination_uri="gs://docs/ocr-output/"
)
```

### 4. Mixed Content Documents

```python
# Handle documents with both printed and handwritten text
extractor = GoogleVisionOCRExtractor()
result = extractor.extract_ocr("mixed_document.png", mode="document")
```

## Limitations

### API Limits

-   **Images**: Max 20MB per image
-   **PDF**: Max 2000 pages per document
-   **Rate Limits**: Subject to Google Cloud Vision API quotas

### Supported Formats

-   **PDF Mode**: Requires Google Cloud Storage (no local PDF processing)
-   **Image Formats**: Standard formats only (PNG, JPG, etc.)

### Confidence Scores

-   **Text Mode**: Fixed confidence (0.95) - Google Vision doesn't provide scores
-   **Document Mode**: Actual confidence scores when available
-   **PDF Mode**: Actual confidence scores when available

## 🔗 Integration Examples

### With ArXiv Pipeline

```python
# Modify arxiv download script to use Google Vision
from data.google_vision_ocr import GoogleVisionOCRExtractor

def process_arxiv_with_google_vision(image_path, mode="text"):
    extractor = GoogleVisionOCRExtractor()
    return extractor.extract_ocr(image_path, mode=mode)
```

### Standalone Script

```python
#!/usr/bin/env python3
"""Standalone Google Vision OCR script"""

from data.google_vision_ocr import GoogleVisionOCRExtractor
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> [mode]")
        return
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "text"
    
    extractor = GoogleVisionOCRExtractor()
    result = extractor.extract_ocr(image_path, mode=mode)
    
    if result["success"]:
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {result['error']}", file=sys.stderr)

if __name__ == "__main__":
    main()
```
