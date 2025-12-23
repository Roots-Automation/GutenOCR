"""
Google Vision OCR Extractor for document images and PDFs.

This module provides OCR capabilities using Google Cloud Vision API with support for:
- Basic text detection in images
- Handwriting/document text detection in images
- Text detection in PDF/TIFF files via GCS
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import tempfile

from google.cloud import vision
from google.cloud import storage
from PIL import Image


class GoogleVisionOCRExtractor:
    """
    OCR extractor using Google Cloud Vision API.
    
    This class provides OCR capabilities with support for:
    - Basic text detection in images (mode='text')
    - Handwriting/document text detection in images (mode='document') 
    - Text detection in PDF/TIFF files via GCS (mode='pdf')
    - Standardized output format for integration
    
    Usage:
        extractor = GoogleVisionOCRExtractor()
        result = extractor.extract_ocr("image.png", mode="text")
        
    Modes:
        - 'text': Basic text detection for printed text in images
        - 'document': Document text detection for handwriting and complex layouts
        - 'pdf': Async PDF/TIFF processing via Google Cloud Storage
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the Google Vision OCR extractor.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var
                            or default application credentials.
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)
        
        try:
            self.vision_client = vision.ImageAnnotatorClient()
            self.storage_client = storage.Client()
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Google Vision client: {e}\n"
                "Make sure you have valid Google Cloud credentials set up."
            )
    
    def extract_ocr(
        self,
        image_path: Union[str, Path],
        mode: str = 'text',
        gcs_source_uri: Optional[str] = None,
        gcs_destination_uri: Optional[str] = None,
        batch_size: int = 1,
        timeout: int = 420
    ) -> Dict:
        """
        Extract OCR data from a document image or PDF.
        
        Args:
            image_path: Path to the image/PDF file
            mode: OCR mode - 'text', 'document', or 'pdf'
            gcs_source_uri: GCS URI for PDF mode (e.g., 'gs://bucket/file.pdf')
            gcs_destination_uri: GCS URI for PDF output (e.g., 'gs://bucket/output/')
            batch_size: Pages per output file for PDF mode
            timeout: Timeout in seconds for PDF mode
            
        Returns:
            Dictionary containing OCR results in standardized format:
            {
                "lines_data": [...],     # Line-level text with bounding boxes (clustered from words)
                "words_data": [...],     # Word-level text with bounding boxes
                "paragraph_data": [...], # Paragraph-level text with bounding boxes (document mode only)
                "image_metadata": {...}, # Image/document metadata
                "success": bool,
                "error": str or None
            }
        """
        image_path = Path(image_path)
        
        try:
            if mode == 'pdf':
                return self._extract_pdf_ocr(
                    gcs_source_uri, gcs_destination_uri, batch_size, timeout
                )
            elif mode in ['text', 'document']:
                return self._extract_image_ocr(image_path, mode)
            else:
                return self._error_response(f"Invalid mode: {mode}. Use 'text', 'document', or 'pdf'")
                
        except Exception as e:
            return self._error_response(f"OCR processing failed: {str(e)}")
    
    def _extract_image_ocr(self, image_path: Path, mode: str) -> Dict:
        """Extract OCR from image using specified mode."""
        if not image_path.exists():
            return self._error_response(f"Image file not found: {image_path}")
        
        # Load image for metadata
        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
            image_format = image.format or 'Unknown'
            image_mode = image.mode
            dpi_info = image.info.get('dpi', None)
        except Exception as e:
            return self._error_response(f"Failed to load image metadata: {e}")
        
        # Read image file
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
        except Exception as e:
            return self._error_response(f"Failed to read image file: {e}")
        
        vision_image = vision.Image(content=content)
        
        try:
            if mode == 'text':
                response = self.vision_client.text_detection(image=vision_image)
                return self._process_text_response(response, image_width, image_height, image_format, image_mode, dpi_info, image_path)
            elif mode == 'document':
                response = self.vision_client.document_text_detection(image=vision_image)
                return self._process_document_response(response, image_width, image_height, image_format, image_mode, dpi_info, image_path)
        except Exception as e:
            return self._error_response(f"Google Vision API call failed: {e}")

    def _extract_pdf_ocr(self, gcs_source_uri: str, gcs_destination_uri: str, batch_size: int=1, timeout: int=420) -> Dict:
        """Extract OCR from PDF using async GCS processing."""
        if not gcs_source_uri or not gcs_destination_uri:
            return self._error_response("PDF mode requires both gcs_source_uri and gcs_destination_uri")
        
        try:
            # Configure the request
            feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            gcs_source = vision.GcsSource(uri=gcs_source_uri)
            input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
            
            gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
            output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=batch_size)
            
            async_request = vision.AsyncAnnotateFileRequest(
                features=[feature], input_config=input_config, output_config=output_config
            )
            
            # Start the async operation
            operation = self.vision_client.async_batch_annotate_files(requests=[async_request])
            print(f"[info] Starting PDF OCR operation, waiting up to {timeout} seconds...")
            operation.result(timeout=timeout)
            
            # Process results from GCS
            return self._process_pdf_results(gcs_destination_uri)
            
        except Exception as e:
            return self._error_response(f"PDF OCR processing failed: {e}")
    
    def _process_text_response(self, response, img_width: int, img_height: int, img_format: str, img_mode: str, dpi_info, image_path: Path) -> Dict:
        """Process basic text detection response."""
        if response.error and response.error.message:
            return self._error_response(f"Google Vision API error: {response.error.message}")
        
        texts = response.text_annotations
        if not texts:
            return self._create_empty_response(img_width, img_height, img_format, img_mode, dpi_info, image_path)
        
        # First annotation contains the full text, skip it for individual text blocks
        individual_texts = texts[1:] if len(texts) > 1 else []
        
        words_data = []
        
        for i, text in enumerate(individual_texts):
            # Convert bounding poly to 4-point coordinates
            vertices = text.bounding_poly.vertices
            if len(vertices) >= 4:
                coords = self._vertices_to_coordinates(vertices)
                
                # Create word-level entry (Google Vision returns word/phrase level)
                word_entry = {
                    "Word": text.description,
                    "Confidence": 0.95,  # Google Vision doesn't provide confidence for text_detection
                    "X1": coords[0], "Y1": coords[1],
                    "X2": coords[2], "Y2": coords[3],
                    "X3": coords[4], "Y3": coords[5],
                    "X4": coords[6], "Y4": coords[7],
                    "page": 1,
                    "Index": i
                }
                words_data.append(word_entry)
        
        # Generate lines_data by clustering words based on their vertical alignment
        lines_data = self._cluster_words_into_lines(words_data)
        
        return {
            "lines_data": lines_data,
            "words_data": words_data,
            "image": {
                "width": img_width,
                "height": img_height,
                "format": img_format,
                "mode": img_mode,
                "dpi": dpi_info,
                "path": str(image_path)
            },
            "success": True
        }
    
    def _process_document_response(self, response, img_width: int, img_height: int, img_format: str, img_mode: str, dpi_info, image_path: Path) -> Dict:
        """Process document text detection response."""
        if response.error and response.error.message:
            return self._error_response(f"Google Vision API error: {response.error.message}")
        
        if not response.full_text_annotation:
            return self._create_empty_response(img_width, img_height, img_format, img_mode, dpi_info, image_path)
        
        paragraph_data = []
        words_data = []
        word_index = 0
        paragraph_index = 0

        for i, page in enumerate(response.full_text_annotation.pages):
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    # Collect all words in paragraph for paragraph-level data
                    paragraph_words = []
                    paragraph_bounds = []
                    
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        if word.bounding_box and word.bounding_box.vertices:
                            coords = self._vertices_to_coordinates(word.bounding_box.vertices)
                            
                            word_entry = {
                                "Word": word_text,
                                "Confidence": getattr(word, 'confidence', 0.95),
                                "X1": coords[0], "Y1": coords[1],
                                "X2": coords[2], "Y2": coords[3],
                                "X3": coords[4], "Y3": coords[5],
                                "X4": coords[6], "Y4": coords[7],
                                "page": i+1,
                                "Index": word_index,
                                "paragraph_id": paragraph_index
                            }
                            words_data.append(word_entry)
                            paragraph_words.append(word_text)
                            paragraph_bounds.extend([coords[0], coords[1], coords[4], coords[5]])
                            word_index += 1
                    
                    # Create paragraph entry
                    if paragraph_words and paragraph_bounds:
                        paragraph_text = " ".join(paragraph_words)
                        min_x = min(paragraph_bounds[::2])  # All x coordinates
                        min_y = min(paragraph_bounds[1::2])  # All y coordinates
                        max_x = max(paragraph_bounds[::2])
                        max_y = max(paragraph_bounds[1::2])
                        
                        paragraph_entry = {
                            "Word": paragraph_text,
                            "Confidence": getattr(paragraph, 'confidence', 0.95),
                            "X1": min_x, "Y1": min_y,
                            "X2": max_x, "Y2": min_y,
                            "X3": max_x, "Y3": max_y,
                            "X4": min_x, "Y4": max_y,
                            "page": i+1,
                            "Index": paragraph_index
                        }
                        paragraph_data.append(paragraph_entry)
                        paragraph_index += 1
        
        # Generate lines_data by clustering words based on their vertical alignment
        lines_data = self._cluster_words_into_lines(words_data, paragraph_data=paragraph_data)
        
        return {
            "lines_data": lines_data,
            "words_data": words_data,
            "paragraph_data": paragraph_data,
            "image": {
                "width": img_width,
                "height": img_height,
                "format": img_format,
                "mode": img_mode,
                "dpi": dpi_info,
                "path": str(image_path)
            },
            "success": True
        }
    
    def _process_pdf_results(self, gcs_destination_uri: str) -> Dict:
        """Process PDF OCR results from GCS."""
        try:
            # Parse GCS URI
            match = re.match(r"gs://([^/]+)/(.+)", gcs_destination_uri)
            if not match:
                return self._error_response("Invalid GCS destination URI format")
            
            bucket_name = match.group(1)
            prefix = match.group(2)
            
            bucket = self.storage_client.get_bucket(bucket_name)
            
            # List output files
            blob_list = [
                blob for blob in list(bucket.list_blobs(prefix=prefix))
                if not blob.name.endswith("/")
            ]
            
            if not blob_list:
                return self._error_response("No output files found in GCS destination")
            
            # Process the first output file
            output = blob_list[0]
            json_string = output.download_as_bytes().decode("utf-8")
            response = json.loads(json_string)
            
            # Extract text from each page separately and return as dictionary with page numbers as keys
            page_results = {}
            
            for page_idx, page_response in enumerate(response.get("responses", [])):
                page_num = page_idx + 1
                
            
                
                if "fullTextAnnotation" in page_response:
                    page_data = self._extract_page_data(page_response["fullTextAnnotation"], page_num)
                    
                    # Create individual page result
                    page_result = {
                        "lines_data": page_data["lines"],
                        "words_data": page_data["words"],
                        "paragraph_data": page_data["paragraphs"],
                        "image": {
                            "width": None,  # Not available for PDF processing
                            "height": None,
                            "format": "PDF",
                            "mode": "PDF",
                            "dpi": None,
                            "path": None,  # No individual image path for PDF pages
                            "page": page_num
                        },
                        "success": True
                    }
                    page_results[page_num] = page_result
                else:
                    # Create empty result for pages without text annotation
                    page_result = {
                        "lines_data": [],
                        "words_data": [],
                        "paragraph_data": [],
                        "image": {
                            "width": None,
                            "height": None,
                            "format": "PDF",
                            "mode": "PDF",
                            "dpi": None,
                            "path": None,
                            "page": page_num
                        },
                        "success": True
                    }
                    page_results[page_num] = page_result
            
            return page_results
            
        except Exception as e:
            return self._error_response(f"Failed to process PDF results: {e}")
    
    def _extract_page_data(self, annotation: Dict, page_num: int) -> Dict:
        """Extract lines, words, and paragraphs data from a page annotation."""
        paragraph_data = []
        words_data = []
        word_index = 0
        paragraph_index = 0
        
        for page in annotation.get("pages", []):
            for block in page.get("blocks", []):
                for paragraph in block.get("paragraphs", []):
                    paragraph_words = []
                    paragraph_bounds = []
                    
                    for word in paragraph.get("words", []):
                        word_text = "".join([symbol.get("text", "") for symbol in word.get("symbols", [])])
                        
                        if "boundingBox" in word and "vertices" in word["boundingBox"]:
                            vertices = word["boundingBox"]["vertices"]
                            # Convert dict vertices to object-like structure for consistency
                            vertex_objects = []
                            for v in vertices:
                                class VertexObj:
                                    def __init__(self, x, y):
                                        self.x = x
                                        self.y = y
                                vertex_objects.append(VertexObj(v.get("x", 0), v.get("y", 0)))
                            
                            coords = self._vertices_to_coordinates(vertex_objects)
                            
                            word_entry = {
                                "Word": word_text,
                                "Confidence": word.get("confidence", 0.95),
                                "X1": coords[0], "Y1": coords[1],
                                "X2": coords[2], "Y2": coords[3],
                                "X3": coords[4], "Y3": coords[5],
                                "X4": coords[6], "Y4": coords[7],
                                "page": page_num,
                                "Index": word_index,
                                "paragraph_id": paragraph_index
                            }
                            words_data.append(word_entry)
                            paragraph_words.append(word_text)
                            paragraph_bounds.extend([coords[0], coords[1], coords[4], coords[5]])
                            word_index += 1
                    
                    # Create paragraph entry
                    if paragraph_words and paragraph_bounds:
                        paragraph_text = " ".join(paragraph_words)
                        min_x = min(paragraph_bounds[::2])
                        min_y = min(paragraph_bounds[1::2])
                        max_x = max(paragraph_bounds[::2])
                        max_y = max(paragraph_bounds[1::2])
                        
                        paragraph_entry = {
                            "Word": paragraph_text,
                            "Confidence": paragraph.get("confidence", 0.95),
                            "X1": min_x, "Y1": min_y,
                            "X2": max_x, "Y2": min_y,
                            "X3": max_x, "Y3": max_y,
                            "X4": min_x, "Y4": max_y,
                            "page": page_num,
                            "Index": paragraph_index
                        }
                        paragraph_data.append(paragraph_entry)
                        paragraph_index += 1
        
        # Generate lines_data by clustering words based on their vertical alignment
        lines_data = self._cluster_words_into_lines(words_data, paragraph_data=paragraph_data)
        
        return {"lines": lines_data, "words": words_data, "paragraphs": paragraph_data}
    
    def _vertices_to_coordinates(self, vertices) -> List[int]:
        """
        Convert Google Vision vertices to standardized 8-coordinate format.
        
        Output format: [X1,Y1,X2,Y2,X3,Y3,X4,Y4] where:
        - X1,Y1 = top-left corner
        - X2,Y2 = top-right corner  
        - X3,Y3 = bottom-right corner
        - X4,Y4 = bottom-left corner
        """
        if len(vertices) < 4:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Extract x,y coordinates from vertices
        points = []
        for vertex in vertices[:4]:
            x = getattr(vertex, 'x', 0)
            y = getattr(vertex, 'y', 0)
            points.append((x, y))
        
        # Sort points to ensure consistent ordering
        # First, find top-left (minimum x+y sum) and bottom-right (maximum x+y sum)
        points_with_sum = [(x, y, x + y) for x, y in points]
        points_with_sum.sort(key=lambda p: p[2])  # Sort by x+y sum
        
        top_left = (points_with_sum[0][0], points_with_sum[0][1])
        bottom_right = (points_with_sum[-1][0], points_with_sum[-1][1])
        
        # Find top-right and bottom-left from remaining points
        remaining = [points_with_sum[1], points_with_sum[2]]
        
        # Top-right has larger x but smaller y compared to bottom-left
        if remaining[0][0] > remaining[1][0]:  # First point has larger x
            top_right = (remaining[0][0], remaining[0][1])
            bottom_left = (remaining[1][0], remaining[1][1])
        else:
            top_right = (remaining[1][0], remaining[1][1])
            bottom_left = (remaining[0][0], remaining[0][1])
        
        # Return in standardized order: top-left, top-right, bottom-right, bottom-left
        coords = [
            top_left[0], top_left[1],        # X1, Y1
            top_right[0], top_right[1],      # X2, Y2
            bottom_right[0], bottom_right[1], # X3, Y3
            bottom_left[0], bottom_left[1]   # X4, Y4
        ]
        
        return coords
    
    def _cluster_words_into_lines(self, words_data: List[Dict], tolerance: int = 5, paragraph_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Cluster words into lines based on their vertical alignment and paragraph membership.
        
        Args:
            words_data: List of word dictionaries with bounding box coordinates
            tolerance: Pixel tolerance for considering words on the same line
            paragraph_data: Optional list of paragraph data to prevent cross-column clustering
            
        Returns:
            List of line dictionaries with combined text and bounding boxes
        """
        if not words_data:
            return []
        
        # Group words by their vertical position (using Y1 as top coordinate)
        lines_groups = []
        
        for word in words_data:
            y1 = word['Y1']  # Top coordinate
            y3 = word['Y3']  # Bottom coordinate
            
            # Find existing line group that this word belongs to
            found_group = False
            for group in lines_groups:
                # Check if this word's vertical position aligns with existing group
                group_y1 = group['y1']
                group_y3 = group['y3']
                
                # Words are on the same line if their top and bottom coordinates are within tolerance
                if (abs(y1 - group_y1) <= tolerance and abs(y3 - group_y3) <= tolerance):
                    group['words'].append(word)
                    found_group = True
                    break
            
            if not found_group:
                # Create new line group
                lines_groups.append({
                    'y1': y1,
                    'y3': y3,
                    'words': [word]
                })
        
        # Sort line groups by their vertical position (top to bottom)
        lines_groups.sort(key=lambda group: group['y1'])
        
        # If paragraph data is available, split groups that span multiple paragraphs
        if paragraph_data is not None:
            lines_groups = self._split_cross_paragraph_groups(lines_groups)
        
        # Create line entries from grouped words
        lines_data = []
        for line_index, group in enumerate(lines_groups):
            words = group['words']
            
            # Sort words in the line by their horizontal position (left to right)
            words.sort(key=lambda w: w['X1'])
            
            # Calculate line bounding box
            min_x1 = min(word['X1'] for word in words)  # Leftmost X1
            shared_y1 = words[0]['Y1']  # Shared top coordinate
            max_x2 = max(word['X2'] for word in words)  # Rightmost X2
            shared_y3 = words[0]['Y3']  # Shared bottom coordinate
            
            # Combine words into line text
            line_text = " ".join(word['Word'] for word in words)
            
            # Calculate average confidence
            avg_confidence = sum(word['Confidence'] for word in words) / len(words)
            
            # Use the same page number as the constituent words
            page_num = words[0]['page']
            
            # Create line entry
            line_entry = {
                "Word": line_text,
                "Confidence": avg_confidence,
                "X1": min_x1, "Y1": shared_y1,           # Top-left
                "X2": max_x2, "Y2": shared_y1,           # Top-right  
                "X3": max_x2, "Y3": shared_y3,           # Bottom-right
                "X4": min_x1, "Y4": shared_y3,           # Bottom-left
                "page": page_num,
                "Index": line_index
            }
            lines_data.append(line_entry)
        
        return lines_data
    
    def _split_cross_paragraph_groups(self, lines_groups: List[Dict]) -> List[Dict]:
        """
        Split line groups that contain words from different paragraphs.
        
        This prevents creating continuous lines across columns in multi-column documents
        by ensuring all words in a line belong to the same paragraph.
        
        Args:
            lines_groups: List of line group dictionaries with 'words' list
            
        Returns:
            List of split line groups where each group contains words from only one paragraph
        """
        split_groups = []
        
        for group in lines_groups:
            words = group['words']
            
            # Group words by their paragraph_id
            paragraph_groups = {}
            words_without_paragraph = []
            
            for word in words:
                if 'paragraph_id' in word:
                    paragraph_id = word['paragraph_id']
                    if paragraph_id not in paragraph_groups:
                        paragraph_groups[paragraph_id] = []
                    paragraph_groups[paragraph_id].append(word)
                else:
                    # Handle words that don't have paragraph_id (from basic text mode)
                    words_without_paragraph.append(word)
            
            # If all words belong to the same paragraph (or no paragraph data), keep as one group
            if len(paragraph_groups) <= 1 and not words_without_paragraph:
                split_groups.append(group)
            elif not paragraph_groups and words_without_paragraph:
                # Basic text mode - no paragraph splitting
                split_groups.append(group)
            else:
                # Split into separate groups for each paragraph
                for paragraph_id, paragraph_words in paragraph_groups.items():
                    if paragraph_words:  # Only create group if it has words
                        # Calculate group bounds from the words in this paragraph
                        y1_coords = [w['Y1'] for w in paragraph_words]
                        y3_coords = [w['Y3'] for w in paragraph_words]
                        
                        split_group = {
                            'y1': min(y1_coords) if y1_coords else group['y1'],
                            'y3': max(y3_coords) if y3_coords else group['y3'],
                            'words': paragraph_words
                        }
                        split_groups.append(split_group)
                
                # Handle words without paragraph_id separately
                if words_without_paragraph:
                    y1_coords = [w['Y1'] for w in words_without_paragraph]
                    y3_coords = [w['Y3'] for w in words_without_paragraph]
                    
                    split_group = {
                        'y1': min(y1_coords) if y1_coords else group['y1'],
                        'y3': max(y3_coords) if y3_coords else group['y3'],
                        'words': words_without_paragraph
                    }
                    split_groups.append(split_group)
        
        return split_groups
    
    def _create_empty_response(self, img_width: int, img_height: int, img_format: str, img_mode: str, dpi_info, image_path: Path) -> Dict:
        """Create empty response when no text is detected."""
        return {
            "lines_data": [],
            "words_data": [],
            "paragraph_data": [],
            "image": {
                "width": img_width,
                "height": img_height,
                "format": img_format,
                "mode": img_mode,
                "dpi": dpi_info,
                "path": str(image_path)
            },
            "success": True
        }
    
    def _error_response(self, error_msg: str) -> Dict:
        """Create a standardized error response."""
        return {
            "lines_data": [],
            "words_data": [],
            "paragraph_data": [],
            "success": False,
            "error": error_msg
        }


def create_google_vision_ocr_extractor(credentials_path: Optional[str] = None) -> GoogleVisionOCRExtractor:
    """
    Convenience function to create a GoogleVisionOCRExtractor instance.
    
    Args:
        credentials_path: Path to Google Cloud credentials JSON file.
        
    Returns:
        GoogleVisionOCRExtractor instance
    """
    return GoogleVisionOCRExtractor(credentials_path=credentials_path)
