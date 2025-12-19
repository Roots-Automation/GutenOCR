#!/bin/bash
set -euo pipefail

# Fast bash script to cleanup incomplete equation sets
# Removes any equation files that don't have all 4 parts
# Also removes temporary equation_* files from LaTeX compilation
#
# Usage: bash cleanup_incomplete.sh [dataset_dir]
#   dataset_dir: Directory containing equation folders (default: ./dataset)

DATASET_DIR="${1:-dataset}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Directory '$DATASET_DIR' not found"
    exit 1
fi

echo "Cleaning up incomplete equation sets and temporary files in $DATASET_DIR..."

total_removed=0
total_sets_checked=0
total_temp_files_removed=0
total_folders_processed=0

# Process each folder - remove temp files and check incomplete sets in one pass
for folder in "$DATASET_DIR"/*/; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    folder_sets_checked=0
    folder_incomplete_sets=0
    folder_temp_files=0
    
    # Remove all temporary equation_* files first
    for temp_file in "$folder"equation_*; do
        if [ -f "$temp_file" ]; then
            rm "$temp_file"
            ((folder_temp_files++))
            ((total_temp_files_removed++))
        fi
    done
    
    # Find all base equation files (without underscore variants)
    for pdf_file in "$folder"[0-9][0-9][0-9][0-9].pdf; do
        if [ ! -f "$pdf_file" ]; then
            continue
        fi
        
        base_name=$(basename "$pdf_file" .pdf)
        ((folder_sets_checked++))
        ((total_sets_checked++))
        
        # Define the 4 expected files
        base_pdf="${folder}${base_name}.pdf"
        base_json="${folder}${base_name}.json"
        
        # Find variant files (any with underscore)
        variant_pdf=$(find "$folder" -maxdepth 1 -name "${base_name}_*.pdf" | head -1)
        
        if [ -z "$variant_pdf" ]; then
            # No variant found - incomplete set
            ((folder_incomplete_sets++))
            [ -f "$base_pdf" ] && rm "$base_pdf" && ((total_removed++))
            [ -f "$base_json" ] && rm "$base_json" && ((total_removed++))
            continue
        fi
        
        variant_name=$(basename "$variant_pdf" .pdf)
        variant_json="${folder}${variant_name}.json"
        
        # Check all 4 files exist
        missing_files=()
        [ ! -f "$base_pdf" ] && missing_files+=("$base_name.pdf")
        [ ! -f "$base_json" ] && missing_files+=("$base_name.json")
        [ ! -f "$variant_pdf" ] && missing_files+=("$variant_name.pdf")
        [ ! -f "$variant_json" ] && missing_files+=("$variant_name.json")
        
        # If any missing, remove the whole set
        if [ ${#missing_files[@]} -gt 0 ]; then
            ((folder_incomplete_sets++))
            [ -f "$base_pdf" ] && rm "$base_pdf" && ((total_removed++))
            [ -f "$base_json" ] && rm "$base_json" && ((total_removed++))
            [ -f "$variant_pdf" ] && rm "$variant_pdf" && ((total_removed++))
            [ -f "$variant_json" ] && rm "$variant_json" && ((total_removed++))
        fi
    done
    
    # Show progress for this folder
    ((total_folders_processed++))
    complete_sets=$((folder_sets_checked - folder_incomplete_sets))
    
    if [ $folder_sets_checked -gt 0 ]; then
        echo "Folder $folder_name: $complete_sets/$folder_sets_checked complete ($folder_incomplete_sets incomplete, $folder_temp_files temp files removed)"
    else
        if [ $folder_temp_files -gt 0 ]; then
            echo "Folder $folder_name: No equation sets, removed $folder_temp_files temp files"
        else
            echo "Folder $folder_name: Empty"
        fi
    fi
done

echo ""
echo "Cleanup complete!"
echo "Processed $total_folders_processed folders"
echo "Checked $total_sets_checked equation sets total"
echo "Removed $total_removed files from incomplete sets"
echo "Removed $total_temp_files_removed temporary equation_* files"

# Calculate and show completion percentage
if [ $total_sets_checked -gt 0 ]; then
    incomplete_sets=$((total_removed / 4))  # Each incomplete set has up to 4 files
    complete_sets=$((total_sets_checked - incomplete_sets))
    completion_percent=$((complete_sets * 100 / total_sets_checked))
    echo "Dataset completion: $complete_sets/$total_sets_checked equations complete ($completion_percent%)"
fi
