"""
Google Vision OCR module for document text extraction.

This module provides OCR capabilities using Google Cloud Vision API with support for:
- Basic text detection in images
- Handwriting/document text detection in images  
- Text detection in PDF/TIFF files via Google Cloud Storage
"""

from .google_vision_ocr_extraction import GoogleVisionOCRExtractor, create_google_vision_ocr_extractor

__all__ = ['GoogleVisionOCRExtractor', 'create_google_vision_ocr_extractor']
