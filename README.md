# QbE-STD Retrieval Pipeline

This repository contains code for H-QuEST: Accelerating Query-by-Example Spoken Term Detection with Hierarchical Indexing featuring:

- Feature extraction from audio using Wav2Vec2 (`src/feature_extraction.py`)
- HQuEST retrieval with HNSW + Smith-Waterman (`src/hquest_retrieval.py`)
- Baseline retrieval methods including TF-IDF, DTW, and a BigTable-style inverted index (`src/baseline_retrievals.py`)
- Main pipeline to run retrieval and output results (`main.py`)

## Installation
1. Install dependencies using [pixi](https://pixi.prefix.dev/latest/installation/):
```bash
pixi install
```
2. You can run the python under the pixi environment that can be activated with:
```bash
pixi shell
```

## DTW C++ Implementation
A high-performance C++ DTW module optimized with pybind11. Build with `pixi run build_dtw` and verify with `pixi run python test_dtw_verification.py`.

## Usage
Step 1: Extract Features
python feature_extraction.py --input_dir path/to/audio --output_csv features.csv

Step 2: Run Retrieval
python main.py --audio_csv features.csv --query_csv queries.csv --output_csv results.csv --top_k 10

