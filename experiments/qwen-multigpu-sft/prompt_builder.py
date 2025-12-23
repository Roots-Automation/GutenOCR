"""Generate prompts for OCR training from structured data and task definitions."""

import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_keys(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize 'box' to 'bbox' in a text item (or 'latex' to 'text') for internal processing."""
    if isinstance(item, dict) and 'box' in item and 'bbox' not in item:
        item = item.copy()
        item['bbox'] = item.pop('box')

    if isinstance(item, dict) and 'latex' in item and 'text' not in item:
        item = item.copy()
        item['text'] = item.pop('latex')

    return item


def generate_fake_bounding_boxes(
    image_width: int, 
    image_height: int, 
    granularity: str = 'word', 
    num_boxes: int = None
) -> List[List[int]]:
    """Generate plausible fake bounding boxes for empty text scenarios.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels  
        granularity: Either 'word', 'line', or 'paragraph' to determine box sizing
        num_boxes: Number of boxes to generate (defaults based on granularity)
    
    Returns:
        List of bounding boxes in format [x1, y1, x2, y2] in absolute pixel coordinates
    """
    if num_boxes is None:
        if granularity == 'word':
            num_boxes = random.randint(3, 8)
        elif granularity == 'paragraph':
            num_boxes = random.randint(1, 3)
        else:  # line
            num_boxes = random.randint(2, 5)
    
    boxes = []
    for _ in range(num_boxes):
        if granularity == 'word':
            # Word boxes: smaller and tighter
            x1 = random.randint(int(0.1 * image_width), int(0.8 * image_width))
            y1 = random.randint(int(0.1 * image_height), int(0.9 * image_height))
            
            # Word dimensions based on typical ratios
            word_width = random.randint(int(0.03 * image_width), int(0.17 * image_width))
            word_height = random.randint(int(0.01 * image_height), int(0.025 * image_height))
            
            x2 = min(x1 + word_width, image_width - 5)
            y2 = min(y1 + word_height, image_height - 5)
        elif granularity == 'paragraph':
            # Paragraph boxes: larger blocks spanning multiple lines
            x1 = random.randint(int(0.05 * image_width), int(0.3 * image_width))
            y1 = random.randint(int(0.1 * image_height), int(0.6 * image_height))
            
            # Paragraph dimensions based on typical ratios
            para_width = random.randint(int(0.4 * image_width), int(0.8 * image_width))
            para_height = random.randint(int(0.08 * image_height), int(0.25 * image_height))
            
            x2 = min(x1 + para_width, image_width - 5)
            y2 = min(y1 + para_height, image_height - 5)
        else:  # line
            # Line boxes: wider and span more horizontally
            x1 = random.randint(int(0.05 * image_width), int(0.7 * image_width))
            y1 = random.randint(int(0.1 * image_height), int(0.9 * image_height))
            
            # Line dimensions based on typical ratios
            line_width = random.randint(int(0.14 * image_width), int(0.57 * image_width))
            line_height = random.randint(int(0.015 * image_height), int(0.035 * image_height))
            
            x2 = min(x1 + line_width, image_width - 5)
            y2 = min(y1 + line_height, image_height - 5)
        
        # Ensure valid box dimensions
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    
    return boxes


# =============================================================================
# TEXT CONTENT VALIDATION
# =============================================================================

def has_any_text_content(example: Dict[str, Any]) -> bool:
    """Check if the example has any actual text content."""
    text_data = example.get('text', {})
    
    # Check list fields (should be non-empty lists)
    list_fields = ['words', 'lines', 'latex', 'paragraphs']
    for field in list_fields:
        if text_data.get(field) and len(text_data[field]) > 0:
            return True
    
    # Check string fields (should be non-empty after stripping)
    string_fields = ['text', 'text2d']
    for field in string_fields:
        if text_data.get(field) and text_data[field].strip():
            return True
    
    return False


def get_available_text_types(example: Dict[str, Any]) -> List[str]:
    """Get list of available text types in the example."""
    text_data = example.get('text', {})
    available_types = list(text_data.keys())
    available_types.append('image')
    if 'lines' in available_types or 'words' in available_types or 'latex' in available_types or 'paragraphs' in available_types:
        available_types.append('box')
    return available_types


# =============================================================================
# TASK COMPATIBILITY
# =============================================================================

def check_task_compatibility(task: Dict[str, str], available_types: List[str]) -> bool:
    """Check if a task is compatible with available data types (both input and output)."""
    
    def _parse_maybe_list(type_str: str) -> List[str]:
        """Parse a type string that may be a list (e.g. "[words, box]") or single type."""
        type_str = type_str.strip()
        if type_str.startswith('[') and type_str.endswith(']'):
            return [t.strip() for t in type_str[1:-1].split(',')]
        return [type_str]

    input_type = set(_parse_maybe_list(task['input_type']))
    output_type = set(_parse_maybe_list(task['output_type']))
    available_types_set = set(available_types)

    if not input_type.issubset(available_types_set) or not output_type.issubset(available_types_set):
        return False
    
    return True


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_answer(
    example: Dict[str, Any], 
    output_type: str, 
    input_type: str, 
    task_type: str, 
    sampled_item: Optional[Dict[str, Any]] = None
) -> Any:
    """Extract the appropriate answer based on output_type, input_type, and task_type context."""
    text_data = example.get('text', {})
    input_type_clean = input_type.strip()
    
    # Handle list output types - return entire dataset
    if output_type == '[words, box]':
        grounded_output = text_data.get('words', [])
    elif output_type == '[lines, box]':
        grounded_output = text_data.get('lines', [])
    elif output_type == '[latex, box]':
        grounded_output = text_data.get('latex', [])
    elif output_type == '[paragraphs, box]':
        grounded_output = text_data.get('paragraphs', [])
    else:
        grounded_output = None

    if grounded_output is not None:
        grounded_output = [normalize_keys(item) for item in grounded_output]
        # sort by y1, then x1 for consistent ordering (conservative, just in case)
        grounded_output = sorted(grounded_output, key=lambda x: (x.get('bbox', [0,0,0,0])[1], x.get('bbox', [0,0,0,0])[0]))
        return grounded_output
    
    # Handle detection tasks (output_type == 'box')
    if output_type == 'box':
        if task_type == 'conditional_detection':
            # Find ALL instances of the sampled text in the specified granularity only
            if sampled_item:
                target_text = sampled_item.get('text', '').lower().strip()
                if target_text:
                    matching_boxes = []

                    # Parse the input_type to determine which granularity to search
                    input_type_clean = input_type.strip()
                    search_granularities = []
                    
                    if input_type_clean.startswith('[') and input_type_clean.endswith(']'):
                        # Extract granularity from input like "[words, image]" or "[lines, image]"
                        input_parts = [part.strip() for part in input_type_clean[1:-1].split(',')]
                        for part in input_parts:
                            # For conditional detection, only words and lines are valid input types
                            if part in ['words', 'lines']:
                                search_granularities.append(part)
                    
                    # If we couldn't parse the input_type, fall back to words and lines only
                    if not search_granularities:
                        search_granularities = ['words', 'lines']
                    
                    # Search only in the specified granularities
                    for data_type in search_granularities:
                        for item in text_data.get(data_type, []):
                            if item.get('text', '').lower().strip() == target_text:
                                box = normalize_keys(item).get('bbox')
                                if box:
                                    matching_boxes.append(box)
                    if matching_boxes:
                        return sorted(matching_boxes, key=lambda b: (b[1], b[0]))  # sort by y1, then x1
            return []
        else:
            # Regular detection - return ALL bounding boxes for the specified granularity
            # Parse the input_type to determine which granularity to use
            input_type_clean = input_type.strip()
            if input_type_clean.startswith('[') and input_type_clean.endswith(']'):
                # Extract granularity from input like "[paragraphs, image]"
                input_parts = [part.strip() for part in input_type_clean[1:-1].split(',')]
                # Find the granularity part (not "image")
                granularity = None
                for part in input_parts:
                    if part in ['words', 'lines', 'latex', 'paragraphs']:
                        granularity = part
                        break
                
                if granularity and text_data.get(granularity):
                    boxes = [
                        normalize_keys(item)['bbox']
                        for item in text_data[granularity]
                    ]
                    return sorted(boxes, key=lambda x: (x[1], x[0]))  # sort by y1, then x1
            
            # Fallback to original logic if we can't parse the input_type
            available_granularities = [
                name for name in ['words', 'lines', 'latex', 'paragraphs'] 
                if text_data.get(name)
            ]
            
            if not available_granularities:
                return []
            
            chosen_granularity = random.choice(available_granularities)
            boxes = [
                normalize_keys(item)['bbox']
                for item in text_data[chosen_granularity]
            ]
            return sorted(boxes, key=lambda x: (x[1], x[0]))  # sort by y1, then x1

    # Check if this is a localized task
    is_localized = input_type_clean.startswith('[') and input_type_clean.endswith(']')
    
    if is_localized:
        # Localized tasks - work with sampled items
        return sampled_item.get('text', '') if sampled_item else ''
    else:
        # Full image tasks - return entire data objects
        if output_type in ['text', 'text2d']:
            return text_data.get(output_type) or fallback_text(text_data.get("lines", []), output_type)
        elif output_type == 'latex':
            latex_data = text_data.get('latex', [])
            latex_data = [normalize_keys(item) for item in latex_data]
            return sorted(latex_data, key=lambda x: (x.get('bbox', [0,0,0,0])[1], x.get('bbox', [0,0,0,0])[0]))
        elif output_type == 'lines':
            lines_data = text_data.get('lines', [])
            lines_data = [normalize_keys(item) for item in lines_data]
            return sorted(lines_data, key=lambda x: (x.get('bbox', [0,0,0,0])[1], x.get('bbox', [0,0,0,0])[0]))
        elif output_type == 'paragraphs':
            paragraphs_data = text_data.get('paragraphs', [])
            paragraphs_data = [normalize_keys(item) for item in paragraphs_data]
            return sorted(paragraphs_data, key=lambda x: (x.get('bbox', [0,0,0,0])[1], x.get('bbox', [0,0,0,0])[0]))
        # elif output_type == 'table':
        #     return {'table': text_data.get('table', [])}
        else:
            # return {output_type: text_data.get(output_type, '')}
            return text_data.get(output_type, '')


def text_1d(detections: list[dict[str, str | list[float]]]) -> str:
    """
    Extract and concatenate text from a list of detection dictionaries.
    Each dictionary is expected to have a 'text' key.
    """
    if not detections:
        return ""
    
    if 'bbox' not in detections[0] and 'box' not in detections[0]:
        raise ValueError("Detections must contain 'bbox' or 'box' key for sorting")

    detections_sorted = sorted(detections, key=lambda x: (x.get('bbox', x.get('box', []))[1], x.get('bbox', x.get('box', []))[0]))
    return '\n'.join(d['text'] for d in detections_sorted if 'text' in d)


def text_2d(detections: list[dict[str, str | list[float]]], p: float = 0.8, max_newlines: int = 2) -> str:
    """
    Extract and concatenate text from a list of detection dictionaries in a 2D layout.
    Each dictionary is expected to have a 'text' key and either 'bbox' or 'box' key with coordinates.
    
    This function uses robust line clustering and character density estimation to preserve 
    the 2D spatial layout of text by:
    1. Computing line height and character density using percentile-based estimation
    2. Clustering text blocks into horizontal lines using line height tolerance
    3. Sorting blocks within each line by horizontal position
    4. Adding appropriate spacing between blocks based on absolute position on page
    5. Adding newlines between lines based on vertical gaps
    
    Args:
        detections: List of detection dictionaries, each containing:
            - 'text': The text content
            - 'bbox' or 'box': Bounding box coordinates [x1, y1, x2, y2] or similar format
        p: Percentile for character density estimation (default 0.8)
        max_newlines: Maximum number of newlines to add between blocks (default 2)
    
    Returns:
        String with text arranged in 2D layout preserving spatial relationships
    """
    if not detections:
        return ""
    
    # Normalize bounding box format and extract coordinates
    normalized_blocks = []
    for detection in detections:
        text = detection['text'].strip()
        if not text:  # Skip empty text blocks
            continue
            
        bbox = detection.get('bbox', detection.get('box', []))
        
        if len(bbox) >= 4:
            # Assume format is [x1, y1, x2, y2] or similar
            x1, y1 = float(bbox[0]), float(bbox[1])
            x2, y2 = float(bbox[2]), float(bbox[3])
            
            # Ensure x1 <= x2 and y1 <= y2
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            
            width = right - left
            height = bottom - top
            
            # Skip degenerate boxes
            if width <= 0 or height <= 0:
                continue
            
            normalized_blocks.append({
                'text': text,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
                'center_y': (top + bottom) / 2,
                'center_x': (left + right) / 2,
                'width': width,
                'height': height,
                'char_ratio': len(text) / width  # characters per unit width
            })
    
    if not normalized_blocks:
        return ""
    
    # Compute line height and character density using robust estimation
    line_height, char_density = _compute_line_height_char_density(normalized_blocks, p)
    
    # Find page bounds for absolute positioning
    page_left = min(block['left'] for block in normalized_blocks)
    page_right = max(block['right'] for block in normalized_blocks)
    page_width = page_right - page_left
    
    # Cluster blocks into lines using line height tolerance
    lines = _cluster_blocks_robust(normalized_blocks, line_height)
    
    # Sort lines by vertical position (top of the line)
    lines = sorted(lines.items(), key=lambda x: x[0])
    
    # Build the output string with absolute positioning
    result_lines = []
    
    for _, line_blocks in lines:
        # Sort blocks in line by horizontal position
        line_blocks.sort(key=lambda b: b['left'])
        
        # Build the line with proper absolute spacing
        line_text = ""
        line_length = 0  # Track current position in characters
        
        for i, block in enumerate(line_blocks):
            # Calculate absolute position on the page as character position
            block_col_pos = (block['left'] - page_left) * char_density
            target_pos = int(round(block_col_pos))
            
            # Add spaces to reach the target position
            spaces_needed = max(0, target_pos - line_length)
            line_text += " " * spaces_needed
            line_length += spaces_needed
            
            # Add the text
            line_text += block['text']
            line_length += len(block['text'])
        
        result_lines.append(line_text.rstrip())
    
    # Join lines with newlines, adding extra newlines for large vertical gaps
    if len(result_lines) <= 1:
        return '\n'.join(result_lines)
    
    final_text = result_lines[0]
    
    for i in range(1, len(result_lines)):
        # Calculate vertical gap between lines
        prev_line_blocks = dict(lines)[list(dict(lines).keys())[i-1]]
        curr_line_blocks = dict(lines)[list(dict(lines).keys())[i]]
        
        prev_line_bottom = max(b['bottom'] for b in prev_line_blocks)
        curr_line_top = min(b['top'] for b in curr_line_blocks)
        gap = curr_line_top - prev_line_bottom
        
        # Add extra newlines for large gaps (more than 1.5x line height)
        if gap > 1.5 * line_height:
            num_newlines = min(max_newlines, max(1, int(round(gap / line_height))))
        else:
            num_newlines = 1
            
        final_text += '\n' * num_newlines + result_lines[i]
    
    return final_text


def _compute_line_height_char_density(blocks: list[dict], p: float) -> tuple[float, float]:
    """
    Compute line height and character density using robust percentile-based estimation.
    
    Args:
        blocks: List of normalized text blocks
        p: Percentile for character density estimation
        
    Returns:
        Tuple of (line_height, char_density)
    """
    # Line height is the minimum height of any text block
    line_height = min(block['height'] for block in blocks)
    
    # Character density is the p-th percentile of character-to-width ratios
    # Higher percentiles correspond to denser text (more chars per unit width)
    char_ratios = [block['char_ratio'] for block in blocks]
    char_density = _percentile(char_ratios, p * 100)
    
    return line_height, char_density


def _cluster_blocks_robust(blocks: list[dict], line_height: float) -> dict:
    """
    Cluster blocks into lines using line height tolerance, updating cluster centroids.
    
    Args:
        blocks: List of normalized text blocks
        line_height: Estimated line height for clustering tolerance
        
    Returns:
        Dictionary of lines keyed by average y-position
    """
    lines = {}
    
    for block in blocks:
        block_center_y = block['center_y']
        placed = False
        
        # Search for a nearby line to merge with
        for line_y in list(lines.keys()):  # Use list() to avoid dict mutation during iteration
            if abs(block_center_y - line_y) <= line_height:
                # Add block to existing line
                line_blocks = lines.pop(line_y)
                line_blocks.append(block)
                
                # Recompute line centroid
                new_line_y = sum(b['center_y'] for b in line_blocks) / len(line_blocks)
                lines[new_line_y] = line_blocks
                placed = True
                break
        
        # If no nearby line found, create new line
        if not placed:
            lines[block_center_y] = [block]
    
    return lines


def _percentile(data: list[float], percentile: float) -> float:
    """
    Calculate the percentile of a list of values.
    
    Args:
        data: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if percentile <= 0:
        return sorted_data[0]
    if percentile >= 100:
        return sorted_data[-1]
    
    # Use linear interpolation for percentiles
    index = (percentile / 100) * (n - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, n - 1)
    
    if lower_index == upper_index:
        return sorted_data[lower_index]
    
    # Linear interpolation
    weight = index - lower_index
    return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight


def fallback_text(lines: list[dict], output_type: str) -> str:
    """Fallback to concatenated lines text if specific output type is missing."""
    if output_type == 'text':
        return text_1d(lines)
    elif output_type == 'text2d':
        return text_2d(lines)
    return ""


# =============================================================================
# QUESTION FORMATTING
# =============================================================================

def format_question(
    question: str, 
    example: Dict[str, Any], 
    input_type: str,
    task_type: str = None,
    output_type: str = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Format the question template with actual values from the example.
    
    Returns:
        Tuple of (formatted_question, sampled_item_for_answer_extraction)
    """
    formatted_question = question
    sampled_item = None
    
    # Replace {image} placeholder with varied reference words
    if '{image}' in formatted_question:
        ref_words = ['the', 'my', 'this', 'that', 'the above', 'the attached', 
                    'the following', 'the provided', 'the included', 'the displayed', 'the shown']
        image_words = ['document', 'image', 'picture', 'photo', 'file', 'scan', 
                      'attachment', 'graphic', 'visual', 'snapshot', 'illustration']
        sampled_phrase = f"{random.choice(ref_words)} {random.choice(image_words)}"
        formatted_question = formatted_question.replace('{image}', sampled_phrase)
    
    # Handle localized tasks (list input types)
    input_type_clean = input_type.strip()
    if input_type_clean.startswith('[') and input_type_clean.endswith(']'):
        required_types = [t.strip() for t in input_type_clean[1:-1].split(',')]
    text_data = example.get('text', {})
    
    # Parse the input_type to determine which granularities are allowed for sampling
    allowed_granularities = []
    input_type_clean = input_type.strip()
    if input_type_clean.startswith('[') and input_type_clean.endswith(']'):
        # Extract granularities from input like "[words, image]"
        input_parts = [part.strip() for part in input_type_clean[1:-1].split(',')]
        for part in input_parts:
            # For conditional detection, only words and lines are valid
            if task_type == 'conditional_detection':
                if part in ['words', 'lines']:
                    allowed_granularities.append(part)
            else:
                # For other tasks, all granularities are valid
                if part in ['words', 'lines', 'latex', 'paragraphs']:
                    allowed_granularities.append(part)
    
    # If we couldn't parse or no granularities specified, set defaults based on task type
    if not allowed_granularities:
        if task_type == 'conditional_detection':
            allowed_granularities = ['words', 'lines']
        else:
            allowed_granularities = ['words', 'lines', 'latex', 'paragraphs']
    
    # Handle {text} placeholder
    if '{text}' in formatted_question:
        # Sample from allowed granularities only
        for data_type in allowed_granularities:
            if (data_type in required_types or data_type[:-1] in required_types) and text_data.get(data_type):
                sampled_item = random.choice(text_data[data_type])
                formatted_question = formatted_question.replace('{text}', sampled_item['text'] if 'text' in sampled_item else sampled_item['latex'])
                break
        
        # Fallback for general text requirement - but still respect allowed granularities
        if sampled_item is None and 'text' in required_types:
            for data_type in allowed_granularities:  # Use allowed granularities instead of hardcoded list
                if text_data.get(data_type):
                    sampled_item = random.choice(text_data[data_type])
                    formatted_question = formatted_question.replace('{text}', sampled_item['text'] if 'text' in sampled_item else sampled_item['latex'])
                    break
    
    # Handle {box} placeholder
    if '{box}' in formatted_question:
        # For localized_reading tasks, sample box from the output_type granularity
        box_sampling_granularities = []
        
        if task_type == 'localized_reading' and output_type:
            # For localized reading, the box should come from the output granularity
            if output_type in ['lines', 'words', 'paragraphs', 'latex']:
                box_sampling_granularities = [output_type]
            else:
                # Fallback to allowed granularities if output_type doesn't specify granularity
                box_sampling_granularities = allowed_granularities
        else:
            # For other task types, use the allowed granularities from input_type
            box_sampling_granularities = allowed_granularities
        
        if sampled_item is None:  # Sample if we didn't already do so for text
            for data_type in box_sampling_granularities:
                if text_data.get(data_type):
                    sampled_item = random.choice(text_data[data_type])
                    break
        
        if sampled_item:
            sample_box = normalize_keys(sampled_item).get('bbox')
            if sample_box:
                formatted_question = formatted_question.replace('{box}', str(sample_box))
        
        # # Handle other specialized placeholders
        # for placeholder in ['{text2d}', '{markdown}', '{html}', '{smiles}', '{latex}']:
        #     if placeholder in formatted_question:
        #         key = placeholder[1:-1]  # Remove curly braces
        #         if key in text_data and text_data[key]:
        #             if isinstance(text_data[key], list) and text_data[key]:
        #                 sampled_item = random.choice(text_data[key])
        #                 sample_value = sampled_item.get(key, '')
        #             else:
        #                 sample_value = str(text_data[key])
        #             formatted_question = formatted_question.replace(placeholder, sample_value)
    
    return formatted_question, sampled_item


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_empty_text_prompt(
    example: Dict[str, Any], 
    allowed_tasks: Optional[List[str]] = None, 
    allowed_output_types: Optional[List[str]] = None
) -> Tuple[str, Any]:
    """Generate a prompt for images with no text content using CSV tasks and fake bounding boxes."""
    # Load tasks CSV
    current_dir = Path(__file__).parent
    tasks_csv_path = current_dir / 'tasks.csv'
    tasks = pd.read_csv(tasks_csv_path).to_dict(orient='records')
    
    image_info = example.get('image', {})
    width = image_info.get('width', 800)
    height = image_info.get('height', 1000)
    
    available_text_keys = set(example.get('text', {}).keys())
    
    # Determine suitable task types for empty text
    suitable_task_types = ['reading', 'localized_reading']
    if allowed_tasks is not None:
        allowed_tasks_lower = [task.lower() for task in allowed_tasks]
        suitable_task_types = [task for task in suitable_task_types if task.lower() in allowed_tasks_lower]
        
        if not suitable_task_types:
            raise ValueError(f"No suitable empty text task types found for allowed tasks: {allowed_tasks}")
    
    selected_task_type = random.choice(suitable_task_types)
    
    if selected_task_type == 'reading':
        # Full document reading
        reading_tasks = []
        for task in tasks:
            if task['task'] == 'reading' and task['input_type'].strip() == 'image':
                output_type = task['output_type'].strip()
                # Check if output format exists in example
                if ((output_type == '[words, box]' and 'words' in available_text_keys) or
                    (output_type == '[lines, box]' and 'lines' in available_text_keys) or
                    (output_type == '[paragraphs, box]' and 'paragraphs' in available_text_keys) or
                    (output_type in available_text_keys)):
                    reading_tasks.append(task)
        
        # Filter by allowed output types
        if allowed_output_types is not None:
            allowed_output_types_lower = [ot.lower() for ot in allowed_output_types]
            reading_tasks = [task for task in reading_tasks 
                           if task['output_type'].strip().lower() in allowed_output_types_lower]
        
        if not reading_tasks:
            error_msg = f"No suitable reading tasks found for available keys: {available_text_keys}"
            if allowed_output_types:
                error_msg += f" and allowed output types: {allowed_output_types}"
            raise ValueError(error_msg)
        
        selected_task = random.choice(reading_tasks)
        question = selected_task['question'].replace('{image}', 'image')
        
        # Generate empty answer based on output type
        output_type = selected_task['output_type']
        answer_mapping = {
            '[words, box]': [],
            '[lines, box]': [],
            '[paragraphs, box]': [],
            'text': '',
            'text2d': '',
            # 'markdown': '',
            # 'html': '',
            # 'smiles': '',
            'latex': [],
            # 'table': {'table': []}
        }
        answer = answer_mapping.get(output_type, '')

    else:  # localized_reading
        localized_tasks = [
            task for task in tasks 
            if task['task'] == 'localized_reading' and task['input_type'].strip() == '[box, image]'
        ]
        
        if allowed_output_types is not None:
            allowed_output_types_lower = [ot.lower() for ot in allowed_output_types]
            localized_tasks = [task for task in localized_tasks 
                             if task['output_type'].strip().lower() in allowed_output_types_lower]
        
        if not localized_tasks:
            error_msg = "No suitable localized reading tasks found"
            if allowed_output_types:
                error_msg += f" for allowed output types: {allowed_output_types}"
            raise ValueError(error_msg)
        
        selected_task = random.choice(localized_tasks)
        output_type = selected_task['output_type'].strip()
        
        # Determine granularity and generate fake box
        granularity = 'word' if output_type == 'words' else 'line' if output_type == 'lines' else 'paragraph' if output_type == 'paragraphs' else random.choice(['word', 'line'])
        fake_box = generate_fake_bounding_boxes(width, height, granularity=granularity, num_boxes=1)[0]
        x1, y1, x2, y2 = fake_box
        box_str = f"[{x1}, {y1}, {x2}, {y2}]"
        
        question = selected_task['question'].replace('{image}', 'image').replace('{box}', box_str)
        answer = '' if output_type in ['text', 'text2d'] else []
    
    return question, answer


def generate_prompt(
    example: Dict[str, Any], 
    tasks_csv_path: Optional[str] = None, 
    seed: Optional[int] = None, 
    allowed_tasks: Optional[List[str]] = None, 
    allowed_output_types: Optional[List[str]] = None
) -> Tuple[str, Any]:
    """Generate a prompt and answer for training based on available data types.
    
    Args:
        example: The data example containing text and image information
        tasks_csv_path: Path to the tasks.csv file (defaults to relative path)
        seed: Random seed for reproducibility
        allowed_tasks: Optional list of task types to restrict to
        allowed_output_types: Optional list of output types to restrict to
    
    Returns:
        Tuple of (formatted_question, expected_answer)
    """
    if seed is not None:
        random.seed(seed)
    
    # Set default tasks CSV path
    if tasks_csv_path is None:
        current_dir = Path(__file__).parent
        tasks_csv_path = current_dir / 'tasks.csv'
    
    # Load tasks
    tasks = pd.read_csv(tasks_csv_path).to_dict(orient='records')
    
    # Handle empty text examples
    if not has_any_text_content(example):
        try:
            return generate_empty_text_prompt(example, allowed_tasks, allowed_output_types)
        except Exception as e:
            return "Recognize all text in this image and return it as a JSON object in TEXT format.", ""

    # Get available text types and find compatible tasks
    available_types = get_available_text_types(example)
    compatible_tasks = [task for task in tasks if check_task_compatibility(task, available_types)]
    
    if not compatible_tasks:
        raise ValueError(f"No compatible tasks found for available types: {available_types}")
    
    # Filter by allowed tasks
    if allowed_tasks is not None:
        allowed_tasks_lower = [task.lower() for task in allowed_tasks]
        compatible_tasks = [task for task in compatible_tasks if task['task'].lower() in allowed_tasks_lower]
        
        if not compatible_tasks:
            available_categories = list(set(task['task'] for task in tasks if check_task_compatibility(task, available_types)))
            raise ValueError(f"No compatible tasks found for allowed task types: {allowed_tasks}. Available: {available_categories}")
    
    # Filter by allowed output types
    if allowed_output_types is not None:
        allowed_output_types_lower = [ot.lower() for ot in allowed_output_types]
        compatible_tasks = [task for task in compatible_tasks 
                          if task['output_type'].strip().lower() in allowed_output_types_lower]
        
        if not compatible_tasks:
            available_output_types = list(set(task['output_type'] for task in tasks))
            raise ValueError(f"No compatible tasks found for allowed output types: {allowed_output_types}. Available: {available_output_types}")
    
    # Sample task category equally, then random task within category
    task_categories = list(set(task['task'] for task in compatible_tasks))
    selected_task_category = random.choice(task_categories)
    tasks_in_category = [task for task in compatible_tasks if task['task'] == selected_task_category]
    selected_task = random.choice(tasks_in_category)
    
    # Format question and extract answer
    formatted_question, sampled_item = format_question(selected_task['question'], example, selected_task['input_type'], selected_task['task'], selected_task['output_type'])
    expected_answer = extract_answer(example, selected_task['output_type'], selected_task['input_type'], selected_task['task'], sampled_item)
    
    return formatted_question, expected_answer


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Example usage of the generate_prompt function."""
    # Example data structure
    example = {
        "text": {
            "lines": [
                {"text": "Hello World", "box": [80, 60, 280, 90]}
            ],
            "latex": [
                {"text": "\\omega = \\Pi_{i=1} x_i^\\tau", "box": [80, 160, 140, 190]}
            ],
            "text": "Hello World",
            "text2d": "    Hello\n  World"
        },
        "image": {
            "path": "example.jpg",
            "width": 800,
            "height": 600
        }
    }
    
    cnt = 0
    try:
        while True:
            question, answer = generate_prompt(example, seed=42 + cnt)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print()

            user_input = input("Generate another prompt? (y/n): ").strip().lower()
            if user_input != 'y':
                break
            print()
            cnt += 1
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
