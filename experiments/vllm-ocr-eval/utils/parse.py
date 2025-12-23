import logging
import re
from typing import Any
from json_repair import repair_json

logger = logging.getLogger(__name__)


def parse_json_output(output_text: str) -> list[Any] | dict[str, Any] | None:
    """
    Safely parse JSON output from model predictions with repair capabilities.
    
    Args:
        output_text: Raw text output from the model.
        
    Returns:
        Parsed JSON data or None if parsing fails.
    """
    try:
        return repair_json(output_text, return_objects=True)
    except Exception:
        return None


def normalize_structured_keys(data: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Normalize keys in structured data.
    - Keys containing "text" → "text"
    - Keys containing "box" → "bbox"
    
    Args:
        data: Dictionary or list of dictionaries to normalize
        
    Returns:
        Normalized data structure
    """
    def _normalize_dict(item: dict[str, Any]) -> dict[str, Any]:
        normalized = {}
        for key, value in item.items():
            # Normalize key names
            new_key = key
            if "text" in key.lower() or "label" in key.lower():
                new_key = "text"
            elif "box" in key.lower():
                new_key = "bbox"
            
            # Recursively normalize nested structures
            if isinstance(value, dict):
                normalized[new_key] = _normalize_dict(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                normalized[new_key] = [_normalize_dict(v) for v in value]
            else:
                normalized[new_key] = value
        
        return normalized

    try:
        if isinstance(data, dict):
            return _normalize_dict(data)
        elif isinstance(data, list) and data:
            # Handle list of dictionaries (normal case)
            if isinstance(data[0], dict):
                return [_normalize_dict(item) for item in data if isinstance(item, dict)]
            # Handle flat coordinate lists like [[x1,y1,x2,y2]] -> [{"bbox": [x1,y1,x2,y2]}]
            elif isinstance(data[0], list) and len(data[0]) == 4 and all(isinstance(x, (int, float)) for x in data[0]):
                return [{"bbox": item} for item in data if isinstance(item, list) and len(item) == 4]
            # Handle other list types
            else:
                return data
        else:
            return data
    except Exception as e:
        logger.warning(f"Failed to normalize structured keys: {e} {data}")
        raise 


def extract_boxes_from_normalized_json(normalized_json: Any) -> list[list[int]]:
    """
    Extract bounding boxes from normalized JSON data.
    Handles various formats:
    - Empty lists: returns empty list
    - List of integers: wraps in extra list 
    - List of dicts with 'bbox': extracts bbox values
    """
    if isinstance(normalized_json, list):
        if len(normalized_json) == 0:
            return normalized_json
        elif isinstance(normalized_json[0], dict):
            return [x.get("bbox", [0, 0, 0, 0]) for x in normalized_json]
        else:
            return [normalized_json]  # wrap in an extra list
    else:
        return []
