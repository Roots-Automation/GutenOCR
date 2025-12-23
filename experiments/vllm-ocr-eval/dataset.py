"""Dataset handling for OCR evaluation tasks."""

import json
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OCREntry:
    """Represents a single OCR document entry."""
    
    filename: str
    image_path: str
    text: str
    text2d: str
    latex: List[Dict[str, Any]]
    lines: List[Dict[str, Any]]
    words: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    image_width: int
    image_height: int
    metadata: Dict[str, Any]
    shard_file: str
    
    @classmethod
    def from_json_data(cls, json_data: Dict[str, Any], filename: str, image_path: str, shard_file: str) -> 'OCREntry':
        """Create OCREntry from JSON data."""
        text_data = json_data.get('text', {})
        image_data = json_data.get('image', {})
        
        return cls(
            filename=filename,
            image_path=image_path,
            text=text_data.get('text', ''),
            text2d=text_data.get('text2d', ''),
            latex=text_data.get('latex', []),
            lines=text_data.get('lines', []),
            words=text_data.get('words', []),
            paragraphs=text_data.get('paragraphs', []),
            image_width=image_data.get('width', 0),
            image_height=image_data.get('height', 0),
            metadata=json_data.get('metadata', {}),
            shard_file=shard_file
        )
    
    def to_prompt_builder_format(self) -> Dict[str, Any]:
        """Convert OCREntry to format expected by prompt_builder.py."""
        return {
            'text': {
                'text': self.text,
                'text2d': self.text2d,
                'lines': [
                    {
                        'text': line.get('text', ''),
                        'box': line.get('box', line.get('bbox', []))
                    }
                    for line in self.lines
                ],
                'words': [
                    {
                        'text': word.get('text', ''),
                        'box': word.get('box', word.get('bbox', []))
                    }
                    for word in self.words
                ],
                'paragraphs': [
                    {
                        'text': para.get('text', ''),
                        'box': para.get('box', para.get('bbox', []))
                    }
                    for para in self.paragraphs
                ],
                'latex': [
                    {
                        'text': item.get('text', ''),
                        'box': item.get('box', item.get('bbox', []))
                    }
                    for item in self.latex
                ]
            },
            'image': {
                'width': self.image_width,
                'height': self.image_height,
                'path': self.image_path
            },
            'metadata': self.metadata
        }


class TarShardDataset:
    """Dataset handler for TAR-based OCR data shards."""
    
    def __init__(self, shard_path: str):
        """
        Initialize the dataset with a TAR shard file.
        
        Args:
            shard_path: Path to the TAR file containing image-JSON pairs.
        """
        self.shard_path = Path(shard_path)
        self.shard_name = self.shard_path.name
        
        if not self.shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        
        # Cache the entries for efficient access
        self._entries = None
        self._temp_dir = None
        
        logger.info(f"Initialized TarShardDataset with shard: {self.shard_name}")
    
    def _load_entries(self) -> List[OCREntry]:
        """Load all entries from the TAR file."""
        if self._entries is not None:
            return self._entries
        
        entries = []
        
        with tarfile.open(self.shard_path, 'r') as tar:
            # Get all JSON files
            json_files = [m for m in tar.getmembers() if m.name.endswith('.json')]
            
            for json_member in json_files:
                # Find corresponding image file (PNG or JPG)
                base_name = json_member.name.replace('.json', '')
                image_name = None
                
                # Check for PNG first, then JPG
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_image = base_name + ext
                    if any(m.name == potential_image for m in tar.getmembers()):
                        image_name = potential_image
                        break
                
                if image_name is None:
                    logger.warning(f"No corresponding image found for {json_member.name}")
                    continue
                
                try:
                    # Extract JSON data
                    json_file = tar.extractfile(json_member)
                    json_data = json.load(json_file)
                    
                    # Create entry
                    entry = OCREntry.from_json_data(
                        json_data=json_data,
                        filename=os.path.basename(json_member.name),
                        image_path=image_name,
                        shard_file=self.shard_name
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    logger.warning(f"Failed to load entry {json_member.name}: {e}")
        
        self._entries = entries
        logger.info(f"Loaded {len(entries)} entries from {self.shard_name}")
        return self._entries
    
    def get_entries(self, limit: Optional[int] = None) -> List[OCREntry]:
        """Get all entries, optionally limited to a certain number."""
        entries = self._load_entries()
        if limit is not None:
            return entries[:limit]
        return entries
    
    def extract_images_to_temp(self, entries: List[OCREntry]) -> Dict[str, str]:
        """
        Extract images for the given entries to temporary files.
        
        Args:
            entries: List of OCREntry objects to extract images for.
            
        Returns:
            Dictionary mapping filename to temporary file path.
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix='ocr_eval_')
            logger.info(f"Created temporary directory: {self._temp_dir}")
        
        temp_paths = {}
        
        with tarfile.open(self.shard_path, 'r') as tar:
            for entry in entries:
                try:
                    # Extract image to temporary file
                    image_member = tar.getmember(entry.image_path)
                    image_file = tar.extractfile(image_member)
                    
                    # Get the correct file extension from the image path
                    image_ext = os.path.splitext(entry.image_path)[1]
                    temp_path = os.path.join(self._temp_dir, entry.filename.replace('.json', image_ext))
                    
                    # Write image data
                    with open(temp_path, 'wb') as f:
                        f.write(image_file.read())
                    
                    temp_paths[entry.filename] = temp_path
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image for {entry.filename}: {e}")
        
        logger.info(f"Extracted {len(temp_paths)} images to temporary files")
        return temp_paths
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)
            logger.info(f"Cleaned up temporary directory: {self._temp_dir}")
            self._temp_dir = None
    
    def __len__(self) -> int:
        """Return the number of entries in the dataset."""
        return len(self._load_entries())
    
    def __getitem__(self, idx: int) -> OCREntry:
        """Get entry by index."""
        entries = self._load_entries()
        return entries[idx]



