#!/bin/bash
set -eo pipefail

# Dataset integrity checker - ONLY checks actual dataset files you care about
#
# Usage: bash count_success.sh [dataset_dir]
#   dataset_dir: Directory containing equation folders (default: ./dataset)

DATASET_DIR="${1:-dataset}"

echo "🔍 Dataset Integrity Check"
echo "$(date)"
echo "=============================="

# Change to the dataset directory  
cd "$DATASET_DIR" 2>/dev/null || { echo "❌ Error: dataset directory '$DATASET_DIR' not found"; exit 1; }

echo "Checking ONLY dataset files (0000.pdf, 0000_*.pdf patterns)..."

# Count ONLY your actual dataset files - ignore ALL temporary/equation files
dataset_pdfs=$(find . -name "[0-9][0-9][0-9][0-9].pdf" -o -name "[0-9][0-9][0-9][0-9]_*.pdf" | wc -l)
dataset_jsons=$(find . -name "[0-9][0-9][0-9][0-9].json" -o -name "[0-9][0-9][0-9][0-9]_*.json" | wc -l)

# Also count temporary files for reference
temp_count=$(find . -name "equation_*_*.pdf" | wc -l)
temp_json_count=$(find . -name "*.tmp" | wc -l)

# Calculate mismatch for ACTUAL dataset files only
mismatch=$((dataset_pdfs - dataset_jsons))

# Display results
echo ""
echo "📊 DATASET FILE STATUS:"
echo "📄 Dataset PDFs:  $dataset_pdfs"
echo "📋 Dataset JSONs: $dataset_jsons"

if [ "$temp_count" -gt 0 ] || [ "$temp_json_count" -gt 0 ]; then
    echo "🗑️  Temp files: $temp_count PDFs, $temp_json_count .tmp files (ignored)"
fi

if [ "$mismatch" -eq 0 ]; then
    echo "✅ PERFECT MATCH! All dataset files properly paired"
    match_status="✅ PERFECT"
elif [ "$mismatch" -gt 0 ]; then
    echo "⚠️  MISMATCH: $mismatch more dataset PDFs than JSONs"
    match_status="⚠️  MISMATCH"
    
    # For active processing, small mismatches are normal
    if [ "$mismatch" -lt 1000 ]; then
        echo "📝 This small mismatch is normal during active processing"
    fi
    
    # Find actual missing JSON pairs
    echo "🔍 Sample missing JSONs:"
    find . -name "[0-9][0-9][0-9][0-9].pdf" -o -name "[0-9][0-9][0-9][0-9]_*.pdf" | head -10 | while read pdf; do
        json="${pdf%.pdf}.json"
        if [ ! -f "$json" ]; then
            echo "   Missing: $json"
        fi
    done | head -3
else
    mismatch_abs=$((0 - mismatch))
    echo "📝 STATUS: $mismatch_abs more JSONs than PDFs"
    
    # For active processing, having more JSONs is normal
    if [ "$mismatch_abs" -lt 1000 ]; then
        echo "✅ This is NORMAL - JSONs are written before PDFs finish rendering"
        echo "🔄 Your system is actively processing files"
        match_status="✅ NORMAL"
    else
        echo "⚠️  Large imbalance detected - likely processing bottleneck"
        match_status="⚠️  BOTTLENECK"
        
        # Check for LaTeX process backlog
        latex_procs=$(ps aux | grep -E "(latex|pdflatex)" | grep -v grep | wc -l)
        if [ "$latex_procs" -gt 20 ]; then
            echo "🔄 LaTeX bottleneck: $latex_procs LaTeX processes running"
            echo "📝 PDFs are being rendered but LaTeX can't keep up with JSON creation"
            echo "✅ This is a performance issue, not a data integrity problem"
            match_status="⚠️  LATEX_BACKLOG"
        else
            # Find actual orphaned JSON files - search more thoroughly
            echo "🔍 Searching for orphaned JSON files..."
            find . -name "[0-9][0-9][0-9][0-9].json" -o -name "[0-9][0-9][0-9][0-9]_*.json" | head -20 | while read json; do
                pdf="${json%.json}.pdf"
                if [ ! -f "$pdf" ]; then
                    echo "   Orphaned: $json (missing $pdf)"
                fi
            done | head -5
        fi
    fi
fi

# Show progress estimate
highest_folder=$(find . -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9]" | sed 's|./||' | sort -n | tail -1)
if [ -n "$highest_folder" ]; then
    echo "📈 Current folder: $highest_folder"
    
    # Estimate equations processed based on dataset PDFs
    processed_equations=$((dataset_pdfs / 2))
    echo "🎯 Equations processed: ~$processed_equations"
    
    # Check recent activity - files created in last 5 minutes
    recent_pdfs=$(find . -name "[0-9][0-9][0-9][0-9].pdf" -o -name "[0-9][0-9][0-9][0-9]_*.pdf" -newermt "5 minutes ago" | wc -l)
    recent_jsons=$(find . -name "[0-9][0-9][0-9][0-9].json" -o -name "[0-9][0-9][0-9][0-9]_*.json" -newermt "5 minutes ago" | wc -l)
    
    if [ "$recent_pdfs" -gt 0 ] || [ "$recent_jsons" -gt 0 ]; then
        echo "🕐 Last 5 min: $recent_jsons JSONs, $recent_pdfs PDFs created"
        if [ "$recent_jsons" -gt "$recent_pdfs" ]; then
            rate_diff=$((recent_jsons - recent_pdfs))
            echo "⚠️  PDF rendering is $rate_diff files behind JSON creation rate"
        fi
        
        # Calculate processing rate
        equations_per_5min=$((recent_jsons / 2))
        equations_per_second=$(echo "scale=1; $equations_per_5min / 300" | bc -l 2>/dev/null || echo "$(($equations_per_5min / 300))")
        echo "📊 Current rate: ~$equations_per_second equations/second"
        
        # Check system load
        load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
        echo "⚙️  System load: $load_avg"
        
        # Warn about critical performance levels
        load_num=$(echo $load_avg | cut -d. -f1)
        if [ "$load_num" -gt 100 ]; then
            echo "🚨 CRITICAL: System severely overloaded (load > 100)"
        elif [ "$load_num" -gt 80 ]; then
            echo "⚠️  WARNING: Very high system load (load > 80)"
        fi
        
        # Performance status
        rate_num=$(echo $equations_per_second | cut -d. -f1)
        if [ "$rate_num" -lt 10 ]; then
            echo "🐌 SLOW: Processing rate critically low (< 10 eq/sec)"
        elif [ "$rate_num" -lt 50 ]; then
            echo "⏳ DEGRADED: Processing rate low (< 50 eq/sec)"
        fi
    fi
fi

echo ""
echo "🚦 Dataset Status: $match_status"

# Final summary
if [ "$mismatch" -eq 0 ]; then
    echo "✨ Your dataset is perfect! Every PDF has its matching JSON."
elif [ "$mismatch_abs" -lt 1000 ] && [ "$mismatch" -lt 0 ]; then
    echo "✨ Your dataset is healthy! Small JSON lead is normal during processing."
elif [ "$match_status" = "⚠️  LATEX_BACKLOG" ]; then
    echo "⚙️  LaTeX rendering backlog detected - PDFs will catch up when processing completes"
else
    echo "🚨 ACTION NEEDED: Dataset integrity issue detected!"
fi
