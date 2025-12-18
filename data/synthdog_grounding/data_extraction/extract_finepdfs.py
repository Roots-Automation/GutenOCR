#!/usr/bin/env python3
"""
Extract text from FinePDFs dataset and save to a .txt file.
This script loads the first 10 million rows from the eng_Latn subset of FinePDFs.
"""

from datasets import load_dataset
from tqdm import tqdm

def extract_finepdfs_text():
    """Extract text from FinePDFs dataset and save to txt file."""
    
    print("Loading FinePDFs dataset (eng_Latn subset)...")
    
    # Load the dataset - using streaming to handle large dataset efficiently
    ds = load_dataset("HuggingFaceFW/finepdfs", "eng_Latn", streaming=True)
    
    # Get the train split (streaming)
    train_ds = ds['train']
    
    output_file = "finepdfs_eng_latn_1M.txt"
    target_samples = 1_000_000
    
    print(f"Extracting {target_samples:,} ASCII-only samples to {output_file}...")
    
    valid_samples = 0
    processed_rows = 0
    
    # Create progress bar for valid samples
    pbar = tqdm(total=target_samples, desc="Valid samples", unit="samples")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in train_ds:
            processed_rows += 1
            
            # Extract the text content from the sample
            # The text field in FinePDFs is typically called 'text'
            text = sample.get('text', '')
            
            # Clean the text: replace internal newlines with spaces to maintain one sample per line
            # This ensures each sample is on its own line for easier processing later
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Skip empty texts
            if not cleaned_text:
                continue
            
            # Check if text contains only ASCII characters
            try:
                cleaned_text.encode('ascii')
            except UnicodeEncodeError:
                # Skip samples with non-ASCII characters
                continue
            
            # Write valid ASCII-only text
            f.write(cleaned_text)
            f.write('\n')
            valid_samples += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'processed_rows': f"{processed_rows:,}",
                'success_rate': f"{valid_samples/processed_rows*100:.1f}%"
            })
            
            # Stop when we reach target
            if valid_samples >= target_samples:
                break
    
    pbar.close()
    print(f"Extraction complete! Saved {valid_samples:,} ASCII-only samples to {output_file}")
    print(f"Total rows processed: {processed_rows:,}")
    print(f"Success rate: {valid_samples/processed_rows*100:.2f}%")

if __name__ == "__main__":
    extract_finepdfs_text()
