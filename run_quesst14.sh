#!/bin/sh
# QbE-STD Retrieval Pipeline for QUESST14 Dataset
# Usage:
#   sh run_quesst14.sh        # Run both steps (feature extraction + eval retrieval)
#   sh run_quesst14.sh 1      # Run only feature extraction

stage=${1:-0}

# Validate stage argument
if [ "$stage" != "0" ] && [ "$stage" != "1" ]; then
    echo "================================================"
    echo "Invalid stage: $stage"
    echo "================================================"
    echo "Usage:"
    echo "  sh run_quesst14.sh        # Run both steps"
    echo "  sh run_quesst14.sh 1      # Run only feature extraction"
    echo ""
    echo "Note: Stage 2 (retrieval) must always run after stage 1 (feature extraction)"
    echo "      to ensure consistent vocabulary across all datasets."
    exit 1
fi

# Step 1: Feature Extraction
if [ "$stage" -eq 0 ] || [ "$stage" -eq 1 ]; then
    echo "================================================"
    echo "Step 1: Extracting features"
    echo "================================================"
    
    echo "Extracting audio features..."
    pixi run python ./src/feature_extraction.py --input_dir=data/quesst14Database/quesst14Database/Audio --output_csv=data/quesst14_audio.csv
    
    echo "Extracting dev query features..."
    pixi run python ./src/feature_extraction.py --input_dir=data/quesst14Database/quesst14Database/dev_queries --output_csv=data/quesst14_dev_queries.csv
    
    echo "Extracting eval query features..."
    pixi run python ./src/feature_extraction.py --input_dir=data/quesst14Database/quesst14Database/eval_queries --output_csv=data/quesst14_eval_queries.csv
    
    if [ "$stage" -eq 1 ]; then
        echo "Feature extraction complete."
        exit 0
    fi
fi

# Step 2: Run Retrieval (Eval)
if [ "$stage" -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Step 2: Running retrieval (eval)"
    echo "================================================"
    
    # Clean up stale HNSW index to force rebuild with current vocabulary
    rm -f hnsw_index.bin
    
    pixi run python main.py --audio_csv=data/quesst14_audio.csv --query_csv=data/quesst14_eval_queries.csv --output_csv=data/quesst14_eval_results.csv --top_k 10
    
    if [ $? -eq 0 ]; then
        echo "✓ Eval retrieval complete. Results: data/quesst14_eval_results.csv"
    else
        echo "✗ Eval retrieval failed"
        exit 1
    fi
fi

echo ""
echo "================================================"
echo "Pipeline complete!"
echo "================================================"
