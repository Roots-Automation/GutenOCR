"""
Usage:
    This script compiles LaTeX math expressions into cropped PDFs, places them onto
    blank pages with controlled rotations, and generates corresponding JSON files
    with bounding boxes. It is designed to run in parallel for large datasets.

Inputs:
    - wmf_texvc_inputs.json
        A JSON file containing unique LaTeX expressions (values only).
    wmf_texvc_inputs.json is a file containing all mathematical formulas from all Wikipedia projects. It can be downloaded from here: https://zenodo.org/records/15162182
    - Dependencies: pdflatex, pdfcrop, (optionally ghostscript for fallback).

Outputs:
    - dataset/<folder>/<file>.pdf and .json
        Each expression is saved as both:
            <id>.pdf       (equation at 0° rotation)
            <id>.json      (bbox + latex string)
            <id>_rotX.pdf  (rotated variant, X ∈ {90,180,270} or small rotation)
            <id>_rotX.json (corresponding bbox + latex string)
    - equation_errors.log
        Log file with detailed errors during LaTeX compilation or cropping.

Features:
    - Preprocessing and retry logic to fix common LaTeX issues (\rm, \bf, °, braces).
    - Cropping with pdfcrop, fallback to ghostscript if available.
    - Places equations at a consistent random center with safe margins.
    - Generates variants with controlled rotations or small random tilt.
    - Multiprocessing with up to 70 processes for throughput.
    - Automatic cleanup of temp files and orphaned PDFs/JSONs.

How to run:
    1. Ensure pdflatex, pdfcrop, and ghostscript are installed and on PATH.
    2. Place `wmf_texvc_inputs.json` in the working directory.
    3. Run the script directly:
           python script_name.py
    4. Outputs will be saved under `dataset/` in subfolders of 1000 samples.
"""




import os
import json
import math
import random
import subprocess
import time
import multiprocessing as mp
import warnings
import logging
import tempfile
import shutil
import re
import argparse
from functools import partial
from reportlab.lib.pagesizes import letter
from pdfrw import PdfReader as PdfrwReader, PdfWriter
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.pdfgen import canvas

# Default configuration (can be overridden via CLI arguments)
DEFAULT_INPUT_FILE = "wmf_texvc_inputs.json"
DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_WORKERS = None  # Auto-detect

# Suppress pdfrw warnings about PDF stream length attributes
warnings.filterwarnings("ignore", message=".*stream /Length attribute.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfrw")

# Suppress pdfrw logging messages about stream length attributes
pdfrw_logger = logging.getLogger('pdfrw')
pdfrw_logger.setLevel(logging.CRITICAL)  # Only show critical errors, not warnings

# Also suppress any other pdfrw-related loggers
for logger_name in ['pdfrw.tokens', 'pdfrw.pdfobjects', 'pdfrw.pdfreader']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# ------------------------
# Helpers
# ------------------------

def make_blank_pdf(path="blank.pdf"):
    """Create a blank Letter-sized PDF page once."""
    c = canvas.Canvas(path, pagesize=letter)
    c.showPage()
    c.save()

def compile_and_crop(expr, folder, equation_index, retry_with_fixes=False, skip_crop=False):
    """Compile LaTeX expression to cropped PDF using pdflatex + pdfcrop."""
    expr_clean = expr.strip()
    
    # Preprocessing to fix common LaTeX issues
    if expr_clean.startswith(r"\begin"):
        latex_body = expr_clean
    else:
        latex_body = f"${expr_clean}$"
    
    # Apply basic fixes on first attempt, more comprehensive fixes on retry
    if retry_with_fixes:
        # Complex fixes only on retry (keep existing comprehensive fixes)
        latex_body = latex_body.replace("\\rm ", "\\mathrm{")  # Fix \rm command
        latex_body = latex_body.replace("\\lang", "\\langle")  # Fix angle brackets
        latex_body = latex_body.replace("\\rang", "\\rangle")  # Fix angle brackets
        latex_body = latex_body.replace("\\bf ", "\\mathbf{")  # Fix \bf command
        latex_body = latex_body.replace("\\it ", "\\mathit{")  # Fix \it command
        latex_body = latex_body.replace("\\cal ", "\\mathcal{")  # Fix \cal command
        latex_body = latex_body.replace("\\over", " \\over ")  # Add spaces around \over
        latex_body = latex_body.replace("\\choose", " \\choose ")  # Add spaces around \choose
        
        # Fix degree symbols
        latex_body = latex_body.replace("°", "^\\circ")  # Convert degree symbol
        latex_body = latex_body.replace("\\degree", "^\\circ")  # Convert degree command
        
        # Quick fix for unmatched braces
        font_commands = ["\\mathrm{", "\\mathbf{", "\\mathit{", "\\mathcal{"]
        for cmd in font_commands:
            if cmd in latex_body:
                open_count = latex_body.count(cmd)
                close_count = latex_body.count('}')
                if open_count > close_count:
                    latex_body += '}' * (open_count - close_count)
        
        # Handle incomplete environments
        if "\\begin{align}" in latex_body and "\\end{align}" not in latex_body:
            latex_body += "\n\\end{align}"
    else:
        # Apply most common fixes on first attempt to reduce retry rate
        latex_body = latex_body.replace("\\rm ", "\\mathrm{")  # Fix \rm command (common)
        latex_body = latex_body.replace("\\bf ", "\\mathbf{")  # Fix \bf command (common)
        latex_body = latex_body.replace("\\it ", "\\mathit{")  # Fix \it command (common)
        latex_body = latex_body.replace("\\cal ", "\\mathcal{")  # Fix \cal command (common)
        latex_body = latex_body.replace("°", "^\\circ")  # Convert degree symbol (common)
        
        # Fix common spacing issues that cause compilation errors
        latex_body = latex_body.replace("\\text{Value of}\\ \\ ", "\\text{Value of } ")  # Fix double spaces
        latex_body = latex_body.replace("\\ \\ ", " ")  # Fix double backslash spaces
        
        # Fix problematic curly braces in expressions like v({i,j})
        latex_body = re.sub(r'([a-zA-Z])\(\{([^}]+)\}\)', r'\1(\2)', latex_body)

    tex_content = rf"""
\documentclass[border=1pt,varwidth]{{standalone}}
\usepackage{{amsmath,amssymb,mathtools}}
\usepackage{{textcomp}}
\usepackage{{gensymb}}
\begin{{document}}
{latex_body}
\end{{document}}
"""

    tex_path = os.path.join(folder, f"equation_{equation_index}_{os.getpid()}.tex")
    pdf_path = os.path.join(folder, f"equation_{equation_index}_{os.getpid()}.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # Run pdflatex with better error handling
    latex_result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", folder, tex_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if latex_result.returncode != 0:
        # LaTeX compilation failed - try with enhanced preprocessing if this was first attempt
        if not retry_with_fixes:
            return compile_and_crop(expr, folder, equation_index, retry_with_fixes=True, skip_crop=skip_crop)
        else:
            # Even enhanced preprocessing failed
            error_msg = f"LaTeX compilation failed (exit code {latex_result.returncode})"
            if latex_result.stderr:
                error_msg += f": {latex_result.stderr[:200]}"
            raise subprocess.CalledProcessError(latex_result.returncode, "pdflatex", error_msg)
    
    # Check if PDF was actually created and is valid
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"pdflatex succeeded but no PDF was created: {pdf_path}")
    
    # Check PDF file size and basic validity
    try:
        pdf_size = os.path.getsize(pdf_path)
        if pdf_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")
        if pdf_size < 100:  # PDFs smaller than 100 bytes are likely corrupted
            raise ValueError(f"PDF file too small ({pdf_size} bytes), likely corrupted: {pdf_path}")
    except OSError as e:
        raise FileNotFoundError(f"Cannot access PDF file {pdf_path}: {e}")
    
    # Run pdfcrop with better error handling - use separate output file to avoid overwrite
    if skip_crop:
        # Skip cropping for speed - use original PDF
        cropped_pdf_path = pdf_path
    else:
        cropped_pdf_path = os.path.join(folder, f"equation_{equation_index}_{os.getpid()}_cropped.pdf")
        crop_result = subprocess.run(
            ["pdfcrop", pdf_path, cropped_pdf_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        if crop_result.returncode != 0:
            # pdfcrop failed - try alternative approach using pdftk or ghostscript
            # First log the failure
            with open("crop_warnings.log", "a", encoding="utf-8") as crop_log:
                crop_log.write(f"pdfcrop failed for {pdf_path} (exit code {crop_result.returncode})\n")
            
            # Try alternative cropping with ghostscript (if available)
            try:
                gs_result = subprocess.run([
                    "gs", "-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH", "-dSAFER",
                    "-dEPSCrop", "-dAutoRotatePages=/None",
                    f"-sOutputFile={cropped_pdf_path}", pdf_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if gs_result.returncode != 0 or not os.path.exists(cropped_pdf_path):
                    # Ghostscript also failed, use original uncropped PDF
                    cropped_pdf_path = pdf_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Ghostscript not available or failed, use original uncropped PDF
                cropped_pdf_path = pdf_path
        # else: pdfcrop succeeded, use cropped version

    # Clean up temporary files (when skipping crop, don't remove the original PDF)
    aux_file = os.path.splitext(tex_path)[0] + ".aux"
    log_file = os.path.splitext(tex_path)[0] + ".log"
    cleanup_files = [tex_path, aux_file, log_file]
    if not skip_crop and cropped_pdf_path != pdf_path:
        cleanup_files.append(pdf_path)  # Only remove original if we created a cropped version
    
    for temp_file in cleanup_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return expr_clean, cropped_pdf_path

def draw_bbox_from_json(pdf_path, json_path):
    """
    Add a colored bounding box to an existing PDF using coordinates from its JSON file.
    This verifies that the JSON coordinates accurately represent the content location.
    """
    # Read the JSON coordinates
    with open(json_path, 'r') as f:
        data = json.load(f)
    bbox = data['bbox']
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox
    
    # Create a new PDF with the bbox overlay
    temp_pdf = pdf_path + "_temp"
    c = canvas.Canvas(temp_pdf, pagesize=letter)
    
    # First, copy the original PDF content
    reader = PdfrwReader(pdf_path)
    page = reader.pages[0]
    xobj = pagexobj(page)
    c.doForm(makerl(c, xobj))
    
    # Draw the bounding box using JSON coordinates
    c.saveState()
    c.setStrokeColorRGB(1, 0, 0)  # RED color for visibility
    c.setLineWidth(2)  # Thick line
    c.setFillColorRGB(1, 0, 0, alpha=0.1)  # Semi-transparent red fill
    
    # Convert to ReportLab coordinates (bottom-left origin)
    # bbox is in top-left origin: [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    PAGE_HEIGHT = letter[1]
    
    # Convert top-left origin bbox to ReportLab bottom-left origin
    rl_x = x_top_left  # X coordinate stays the same
    rl_y = PAGE_HEIGHT - y_bottom_right  # Bottom edge in ReportLab coordinates
    width = x_bottom_right - x_top_left
    height = y_bottom_right - y_top_left
    
    # Draw rectangle using converted coordinates
    c.rect(rl_x, rl_y, width, height, stroke=1, fill=1)
    c.restoreState()
    
    c.save()
    
    # Replace original with annotated version
    os.replace(temp_pdf, pdf_path)

def place_equation_on_page(equation_pdf, output_pdf, rotation=0, base_position=None, base_dims=None):
    """
    Overlay cropped equation PDF onto a blank Letter page with rotation around its center.
    A single center point is used for all rotations to keep them consistent.
    """
    PAGE_WIDTH, PAGE_HEIGHT = letter

    # Use pdfrw to read the equation PDF with error handling
    try:
        eq_reader = PdfrwReader(equation_pdf)
        if not eq_reader.pages or len(eq_reader.pages) == 0:
            raise ValueError("Empty PDF file!")
        eq_page = eq_reader.pages[0]
    except Exception as e:
        raise ValueError(f"Could not read PDF file {equation_pdf}: {e}")
    
    # Get original equation size from its mediabox and scale by 25%
    scale_factor = 1.25
    eq_width = float(eq_page.MediaBox[2]) * scale_factor
    eq_height = float(eq_page.MediaBox[3]) * scale_factor

    # Calculate rotated bounding box dimensions
    if rotation in [90, 270]:
        box_w, box_h = eq_height, eq_width  # Swapped for 90°/270°
    elif rotation in [0, 180]:
        box_w, box_h = eq_width, eq_height  # Same for 0°/180°
    else:
        # For arbitrary angles, calculate the actual rotated bounding box
        angle_rad = math.radians(abs(rotation))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        box_w = abs(eq_width * cos_a) + abs(eq_height * sin_a)
        box_h = abs(eq_width * sin_a) + abs(eq_height * cos_a)

    # SIMPLE: Use ONE center for everything
    if base_position:
        # Use the stored center from the 0° rotation
        center_x, center_y = base_position
    else:
        # First time (0° rotation): pick a safe center that works for all rotations
        # Make sure any rotation fits on the page
        # NOTE: center_y is in TOP-LEFT origin coordinate system (0 = top of page)
        safe_margin = max(eq_width, eq_height) / 2
        center_x = random.uniform(safe_margin, PAGE_WIDTH - safe_margin)
        center_y = random.uniform(safe_margin, PAGE_HEIGHT - safe_margin)
        base_position = (center_x, center_y)  # Store just the center point
        base_dims = (eq_width, eq_height)

    # Calculate bbox around the center (TOP-LEFT origin coordinates)
    x_top_left = center_x - box_w / 2
    y_top_left = center_y - box_h / 2
    x_bottom_right = x_top_left + box_w
    y_bottom_right = y_top_left + box_h

    # Draw the equation (convert center_y to ReportLab bottom-left origin)
    draw_center_y_rl = PAGE_HEIGHT - center_y  # Convert to ReportLab coordinates
    
    c = canvas.Canvas(output_pdf, pagesize=letter)
    eq_xobj = pagexobj(eq_page)

    c.saveState()
    c.translate(center_x, draw_center_y_rl)  # Move to center
    c.scale(scale_factor, scale_factor)      # Scale
    if rotation != 0:
        c.rotate(rotation)                   # Rotate around center
    c.translate(-float(eq_page.MediaBox[2]) / 2, -float(eq_page.MediaBox[3]) / 2)  # Center the content
    c.doForm(makerl(c, eq_xobj))
    c.restoreState()
    
    c.save()

    # Return bbox in [topleft_x, topleft_y, bottomright_x, bottomright_y] format
    bbox = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    return bbox, base_position, base_dims

def process_equation(args):
    """Worker function to process a single equation - designed for multiprocessing"""
    i, expr, base_output_dir = args
    
    # Set unique random seed for each process to avoid identical random numbers
    random.seed(i + os.getpid())
    
    # Define ALL file paths upfront for cleanup
    folder_id = i // 1000
    file_id = i % 1000
    folder = f"{folder_id:04d}"
    filename = f"{file_id:04d}"
    out_dir = os.path.join(base_output_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    
    # Define all possible output files upfront
    out_pdf = os.path.join(out_dir, f"{filename}.pdf")
    out_json = os.path.join(out_dir, f"{filename}.json")
    
    # Calculate variant info upfront
    if i % 4 == 0:
        variant = "smallrot"
    else:
        rotations = [90, 180, 270]
        non_skip_index = i - (i // 4) - 1
        rotation_angle = rotations[non_skip_index % 3]
        variant = f"rot{rotation_angle}"
    
    out_pdf_variant = os.path.join(out_dir, f"{filename}_{variant}.pdf")
    out_json_variant = os.path.join(out_dir, f"{filename}_{variant}.json")
    
    # List of all files this function might create
    all_output_files = [out_pdf, out_json, out_pdf_variant, out_json_variant]
    
    # SKIP CHECK: Only consider completed if ALL 4 files exist
    if all(os.path.exists(f) for f in all_output_files):
        return f"Skipped: {i} (already completed)"
    
    cropped_pdf = None

    try:
        expr_clean, cropped_pdf = compile_and_crop(expr, out_dir, i, skip_crop=True)

        # Calculate rotations upfront
        if i % 4 == 0:
            if random.random() < 0.5:
                second_rotation = random.uniform(-10, -3)
            else:
                second_rotation = random.uniform(3, 10)
        else:
            rotations = [90, 180, 270]
            non_skip_index = i - (i // 4) - 1
            second_rotation = rotations[non_skip_index % 3]

        # Create ALL files in temporary locations first for perfect atomicity
        temp_pdf = out_pdf + ".tmp"
        temp_pdf_variant = out_pdf_variant + ".tmp"
        temp_json = out_json + ".tmp"
        temp_json_variant = out_json_variant + ".tmp"
        
        # Create PDFs in temporary locations
        bbox, base_position, base_dims = place_equation_on_page(cropped_pdf, temp_pdf, rotation=0)
        bbox_variant, _, _ = place_equation_on_page(
            cropped_pdf, temp_pdf_variant, rotation=second_rotation, 
            base_position=base_position, base_dims=base_dims
        )
        
        # Verify BOTH temp PDFs were created successfully
        if not os.path.exists(temp_pdf) or not os.path.exists(temp_pdf_variant):
            raise Exception("Temporary PDF creation failed")
        
        # Create JSONs in temporary locations
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump({"bbox": bbox, "latex": expr_clean}, f, indent=2)
        
        with open(temp_json_variant, "w", encoding="utf-8") as f:
            json.dump({"bbox": bbox_variant, "latex": expr_clean}, f, indent=2)
        
        # Verify ALL temp files were created successfully
        temp_files = [temp_pdf, temp_pdf_variant, temp_json, temp_json_variant]
        if not all(os.path.exists(f) for f in temp_files):
            raise Exception("Temporary file creation failed")
        
        # Atomically move ALL files to final locations in one batch
        # This ensures monitoring scripts see either 0 files or all 4 files
        os.rename(temp_pdf, out_pdf)
        os.rename(temp_pdf_variant, out_pdf_variant)
        os.rename(temp_json, out_json)
        os.rename(temp_json_variant, out_json_variant)
        
        # Final verification that ALL files exist
        if not all(os.path.exists(f) for f in all_output_files):
            raise Exception("Final verification failed - not all files exist")

        return f"Success: {i}"

    except Exception as e:
        # CRITICAL: Clean up ALL files on ANY error (including all temp files)
        temp_files = [
            out_pdf + ".tmp", out_pdf_variant + ".tmp", 
            out_json + ".tmp", out_json_variant + ".tmp"
        ]
        for file_path in all_output_files + temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        
        # Return appropriate error message
        if isinstance(e, subprocess.CalledProcessError):
            if "pdflatex" in str(e.cmd):
                return f"Error {i} (LaTeX compilation): {e} | Expression: {repr(expr[:100])}"
            elif "pdfcrop" in str(e.cmd):
                return f"Error {i} (PDF crop): {e} | Expression: {repr(expr[:100])}"
            else:
                return f"Error {i} (LaTeX/PDF): {e} | Expression: {repr(expr[:100])}"
        elif isinstance(e, FileNotFoundError):
            return f"Error {i} (File missing): {e} | Expression: {repr(expr[:100])}"
        else:
            return f"Error {i} (General): {e} | Expression: {repr(expr[:100])}"
    
    finally:
        # Always clean up temporary cropped PDF
        if cropped_pdf and os.path.exists(cropped_pdf):
            try:
                os.remove(cropped_pdf)
            except OSError:
                pass

# ------------------------
# Main loop
# ------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile LaTeX math expressions into cropped PDFs with bounding boxes."
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_FILE,
        help=f"Path to JSON file with LaTeX expressions (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated PDFs/JSONs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Number of parallel processes (default: auto-detected)"
    )
    parser.add_argument(
        "--max-equations", type=int, default=None,
        help="Maximum equations to process (default: all)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    output_dir = args.output
    input_file = args.input
    
    # Clean up any orphaned files from previous runs
    print("Cleaning up orphaned files...")
    try:
        removed_count = 0
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                should_remove = False
                
                if file.endswith(".pdf"):
                    # Check for temporary equation PDFs (contain process ID in name)
                    if "equation_" in file and "_" in file:
                        # More comprehensive pattern matching for temporary files
                        parts = file.replace(".pdf", "").split("_")
                        # Pattern: equation_index_processid.pdf or equation_index_processid_cropped.pdf
                        if len(parts) >= 3:
                            # Check if last part or second-to-last part is a process ID (all digits, 6+ chars)
                            if (parts[-1].isdigit() and len(parts[-1]) >= 6) or \
                               (len(parts) >= 4 and parts[-2].isdigit() and len(parts[-2]) >= 6):
                                should_remove = True
                    
                    # Also check for regular PDFs missing their JSON
                    if not should_remove:
                        json_path = file_path.replace(".pdf", ".json")
                        if not os.path.exists(json_path):
                            should_remove = True
                
                if should_remove:
                    os.remove(file_path)
                    removed_count += 1
        
        if removed_count > 0:
            print(f"Removed {removed_count} orphaned files")
    except OSError:
        pass
    
    # Ensure blank template exists
    if not os.path.exists("blank.pdf"):
        make_blank_pdf("blank.pdf")

    # Load input file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        exit(1)
        
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total unique expressions: {len(data)}")
    
    if len(data) == 0:
        print(f"Error: No equations found in {input_file}")
        exit(1)
    
    # Create work items: (index, expression, output_directory)
    work_items = [(i, expr, output_dir) for i, expr in enumerate(data.values())]
    
    # Limit equations if requested
    if args.max_equations is not None:
        work_items = work_items[:args.max_equations]
        print(f"Limited to {args.max_equations} equations")
    
    # Determine optimal number of processes for file I/O heavy workload
    total_cores = mp.cpu_count()
    
    if args.workers is not None:
        num_processes = args.workers
    else:
        # Scale based on disk performance and available space
        # Conservative: 40 processes (good for most SSDs)
        # Aggressive: 60-80 processes (if you have fast NVMe and plenty of disk space)
        optimal_processes = min(70, total_cores * 3)  # Limited to 70 processes for stability
        num_processes = max(1, optimal_processes)
    
    print(f"Using {num_processes} processes on {total_cores} CPU cores")
    print(f"Processing {len(work_items)} equations into folders of 1000...")
    
    start_time = time.time()
    
    # Process equations in parallel with chunked approach for better file system performance
    completed = 0
    skipped = 0
    error_count = 0
    last_progress_time = start_time
    
    # Open error log file to track errors
    with open("equation_errors.log", "w", encoding="utf-8") as error_log:
        
        with mp.Pool(processes=num_processes) as pool:
            # Use chunk size of 1 for immediate feedback (minimal overhead for I/O-bound tasks)
            print(f"Starting processing with immediate feedback...")
            print("Processing equations (one dot = 50 equations):", end="", flush=True)
            
            for result in pool.imap_unordered(process_equation, work_items, chunksize=1):
                completed += 1
                
                # Track different result types
                if result.startswith("Skipped"):
                    skipped += 1
                elif result.startswith("Error"):
                    error_count += 1
                    error_log.write(f"{result}\n")
                    error_log.flush()  # Ensure errors are written immediately
                
                # Show immediate progress with dots
                if completed % 50 == 0:
                    print(".", end="", flush=True)
                
                # Progress reporting every 1000 equations or every 60 seconds
                current_time = time.time()
                if completed % 1000 == 0 or (current_time - last_progress_time) >= 60:
                    elapsed = current_time - start_time
                    processed_count = completed - skipped  # Actual new equations processed
                    equations_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    remaining_count = len(work_items) - completed
                    total_time_est = remaining_count / equations_per_sec / 60 if equations_per_sec > 0 else 0
                    progress_pct = (completed / len(work_items)) * 100
                    
                    print(f"\n[✓] Progress: {completed}/{len(work_items)} ({progress_pct:.1f}%) | "
                          f"Processed: {processed_count} | Skipped: {skipped} | "
                          f"Speed: {equations_per_sec:.1f} eq/s | ETA: {total_time_est:.1f} min | Errors: {error_count}")
                    last_progress_time = current_time

    # Final statistics (after error log file is closed)
    total_time = time.time() - start_time
    processed_count = len(work_items) - skipped
    avg_speed = processed_count / total_time if total_time > 0 else 0
    print(f"\nCompleted! Processed {len(work_items)} equations in {total_time/60:.1f} minutes")
    print(f"New equations processed: {processed_count} | Skipped (already complete): {skipped}")
    print(f"Average speed: {avg_speed:.1f} new equations/second")
    print(f"Total errors: {error_count} (see equation_errors.log for details)")
