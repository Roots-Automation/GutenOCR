#!/usr/bin/env python3
"""
OCR JSON Formatter for PubMed Processing

This module provides a formatter to standardize Google Vision OCR JSON responses
into a consistent format with simplified bounding box coordinates.

The formatter converts from the original Google Vision format:
- lines_data, words_data, paragraph_data with 8-coordinate bounding boxes (X1,Y1,X2,Y2,X3,Y3,X4,Y4)

To the standardized format:
- lines, words, paragraphs with 4-coordinate bounding boxes [X1,Y1,X3,Y3] (top-left, bottom-right)
"""

from typing import Dict, List, Any, Optional
from pathlib import Path


class OCRJSONFormatter:
    """
    Formatter for standardizing OCR JSON responses.
    
    Converts Google Vision OCR output format to a standardized format with:
    - Simplified bounding boxes using [X1,Y1,X3,Y3] format
    - Consistent structure for lines, words, and paragraphs
    - Preserved image metadata
    """
    
    def __init__(self):
        pass
    
    def format_ocr_response(self, ocr_data: Dict, image_path: Optional[str] = None, preserve_metadata: bool = False) -> Dict:
        """
        Convert OCR response to standardized format.
        
        Args:
            ocr_data: Original Google Vision OCR response dict
            image_path: Optional path to override image path in metadata
            preserve_metadata: If True, preserve additional keys from original items
            
        Returns:
            Formatted OCR response dict with standardized structure
        """
        try:
            # Handle error responses
            if not ocr_data.get('success', True):
                return self._create_empty_response(
                    error=ocr_data.get('error', 'OCR processing failed'),
                    image_path=image_path
                )
            
            # Extract and format text data
            lines = self._format_text_items(ocr_data.get('lines_data', []), preserve_metadata)
            words = self._format_text_items(ocr_data.get('words_data', []), preserve_metadata)
            paragraphs = self._format_text_items(ocr_data.get('paragraph_data', []), preserve_metadata)
            
            # Extract and format image metadata
            image_metadata = self._format_image_metadata(
                ocr_data.get('image', {}), 
                image_path
            )
            
            return {
                "text": {
                    "lines": lines,
                    "words": words,
                    "paragraphs": paragraphs
                },
                "image": image_metadata
            }
            
        except Exception as e:
            return self._create_empty_response(
                error=f"Formatting error: {str(e)}",
                image_path=image_path
            )
    
    def _format_text_items(self, items_data: List[Dict], preserve_metadata: bool = False) -> List[Dict]:
        """
        Format text items (lines, words, or paragraphs) to standardized format.
        
        Args:
            items_data: List of text item dictionaries from Google Vision
            preserve_metadata: If True, preserve additional keys from original items
            
        Returns:
            List of formatted text items with simplified bounding boxes
        """
        formatted_items = []
        
        for item in items_data:
            try:
                # Extract text content
                text = item.get('Word', '')
                
                # Convert 8-coordinate bounding box to 4-coordinate format
                box = self._convert_bounding_box(item)
                
                # Create formatted item
                formatted_item = {
                    "text": text,
                    "box": box
                }
                
                # If preserving metadata, add all other keys from original item
                if preserve_metadata:
                    # Keys to exclude (already converted to 'text' and 'box')
                    excluded_keys = {'Word', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'}
                    
                    for key, value in item.items():
                        if key not in excluded_keys:
                            formatted_item[key] = value
                
                formatted_items.append(formatted_item)
                
            except Exception as e:
                # Skip malformed items but log the issue
                print(f"Warning: Skipping malformed text item: {e}")
                continue
        
        return formatted_items
    
    def _convert_bounding_box(self, item: Dict) -> List[int]:
        """
        Convert 8-coordinate bounding box to 4-coordinate format.
        
        Args:
            item: Text item dict with X1,Y1,X2,Y2,X3,Y3,X4,Y4 coordinates
            
        Returns:
            List of 4 coordinates [X1,Y1,X3,Y3] representing top-left and bottom-right
        """
        try:
            # Extract all 8 coordinates
            x1 = item.get('X1', 0)
            y1 = item.get('Y1', 0)
            x2 = item.get('X2', 0)
            y2 = item.get('Y2', 0)
            x3 = item.get('X3', 0)
            y3 = item.get('Y3', 0)
            x4 = item.get('X4', 0)
            y4 = item.get('Y4', 0)
            
            # Find bounding rectangle coordinates
            min_x = min(x1, x2, x3, x4)
            min_y = min(y1, y2, y3, y4)
            max_x = max(x1, x2, x3, x4)
            max_y = max(y1, y2, y3, y4)
            
            return [min_x, min_y, max_x, max_y]
            
        except (TypeError, ValueError):
            # Return default box if coordinates are invalid
            return [0, 0, 0, 0]
    
    def _format_image_metadata(self, image_data: Dict, override_path: Optional[str] = None) -> Dict:
        """
        Format image metadata to standardized format.
        
        Args:
            image_data: Original image metadata dict
            override_path: Optional path to override the original path
            
        Returns:
            Formatted image metadata dict
        """
        # Extract image properties with defaults
        width = image_data.get('width')
        height = image_data.get('height')
        dpi = image_data.get('dpi')
        
        # Use override path if provided, otherwise use original path
        path = override_path if override_path is not None else image_data.get('path')
        
        # Handle DPI format (could be tuple or single value)
        if isinstance(dpi, (list, tuple)) and len(dpi) >= 2:
            dpi = dpi[0]  # Use first DPI value
        
        return {
            "path": path,
            "width": width,
            "height": height,
            "dpi": dpi
        }
    
    def _create_empty_response(self, error: str = None, image_path: Optional[str] = None) -> Dict:
        """
        Create an empty formatted response for error cases.
        
        Args:
            error: Optional error message
            image_path: Optional image path
            
        Returns:
            Empty formatted response dict
        """
        response = {
            "text": {
                "lines": [],
                "words": [],
                "paragraphs": []
            },
            "image": {
                "path": image_path,
                "width": None,
                "height": None,
                "dpi": None
            }
        }
        
        if error:
            response["error"] = error
        
        return response
    
    def format_batch(self, ocr_responses: Dict[str, Dict], image_paths: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        """
        Format a batch of OCR responses.
        
        Args:
            ocr_responses: Dictionary mapping identifiers to OCR response dicts
            image_paths: Optional dictionary mapping identifiers to image paths
            
        Returns:
            Dictionary mapping identifiers to formatted OCR responses
        """
        formatted_responses = {}
        
        for identifier, ocr_data in ocr_responses.items():
            image_path = None
            if image_paths and identifier in image_paths:
                image_path = image_paths[identifier]
            
            formatted_responses[identifier] = self.format_ocr_response(ocr_data, image_path)
        
        return formatted_responses


def create_ocr_formatter() -> OCRJSONFormatter:
    """
    Convenience function to create an OCRJSONFormatter instance.
    
    Returns:
        OCRJSONFormatter instance
    """
    return OCRJSONFormatter()


# Example usage
if __name__ == "__main__":
    # Example Google Vision OCR response
    sample_ocr_data = {
        "lines_data": [
            {
                "Word": "Sample text line",
                "Confidence": 0.95,
                "X1": 100, "Y1": 200, "X2": 300, "Y2": 200,
                "X3": 300, "Y3": 220, "X4": 100, "Y4": 220,
                "page": 1, "Index": 0
            }
        ],
        "words_data": [
            {
                "Word": "Sample",
                "Confidence": 0.98,
                "X1": 100, "Y1": 200, "X2": 150, "Y2": 200,
                "X3": 150, "Y3": 220, "X4": 100, "Y4": 220,
                "page": 1, "Index": 0
            },
            {
                "Word": "text",
                "Confidence": 0.96,
                "X1": 160, "Y1": 200, "X2": 190, "Y2": 200,
                "X3": 190, "Y3": 220, "X4": 160, "Y4": 220,
                "page": 1, "Index": 1
            }
        ],
        "paragraph_data": [],
        "image": {
            "width": 800,
            "height": 600,
            "format": "PNG",
            "mode": "RGB",
            "dpi": (300, 300),
            "path": "/path/to/image.png"
        },
        "success": True
    }
    
    # Format the response
    formatter = create_ocr_formatter()
    formatted = formatter.format_ocr_response(sample_ocr_data)
    
    import json
    print(json.dumps(formatted, indent=2))
