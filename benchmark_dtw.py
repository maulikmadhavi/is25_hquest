"""
Performance benchmark script for DTW implementations.
Compares C++ vs Python implementations.
"""

import importlib.util
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def benchmark_dtw():
    """Benchmark DTW implementations"""

    # Check if C++ version is available
    have_cpp = importlib.util.find_spec("dtw_cpp") is not None
    if have_cpp:
        print("✓ C++ DTW module found\n")
    else:
        print("✗ C++ DTW module not found (install with: python build_dtw.py)\n")

    # Import Python implementation
    from baseline_retrieval import compute_dtw_distance, compute_dtw_distance_python

    # Test cases with different sizes
    test_cases = [
        (10, 10, "Small (10x10)"),
        (50, 50, "Medium (50x50)"),
        (100, 100, "Medium-Large (100x100)"),
        (200, 200, "Large (200x200)"),
        (500, 500, "Very Large (500x500)"),
    ]

    print("=" * 70)
    print("DTW Performance Benchmark")
    print("=" * 70)
    print()

    all_results = []

    for query_len, audio_len, label in test_cases:
        print(f"\nTest: {label}")
        print("-" * 70)

        # Generate random test data
        np.random.seed(42)
        query_tokens = np.random.uniform(0, 100, query_len).astype(np.float64)
        audio_tokens = np.random.uniform(0, 100, audio_len).astype(np.float64)

        # Test Python implementation
        python_times = []
        for _ in range(3):  # 3 runs
            start = time.perf_counter()
            python_dist = compute_dtw_distance_python(query_tokens, audio_tokens)
            elapsed = time.perf_counter() - start
            python_times.append(elapsed)

        python_time = np.mean(python_times) * 1000  # ms
        python_std = np.std(python_times) * 1000

        print(f"  Python:   {python_time:8.3f}ms (±{python_std:6.3f}ms) -> {python_dist:.2f}")

        # Test C++ implementation if available
        if have_cpp:
            cpp_times = []
            for _ in range(3):
                start = time.perf_counter()
                cpp_dist = compute_dtw_distance(query_tokens, audio_tokens)
                elapsed = time.perf_counter() - start
                cpp_times.append(elapsed)

            cpp_time = np.mean(cpp_times) * 1000  # ms
            cpp_std = np.std(cpp_times) * 1000
            speedup = python_time / cpp_time

            print(f"  C++:      {cpp_time:8.3f}ms (±{cpp_std:6.3f}ms) -> {cpp_dist:.2f}")
            print(f"  Speedup:  {speedup:8.1f}x")

            all_results.append((label, python_time, cpp_time, speedup))
        else:
            print("  C++:      Not available")

    # Summary
    if have_cpp and all_results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Test':<25} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
        print("-" * 70)

        for label, python_time, cpp_time, speedup in all_results:
            print(f"{label:<25} {python_time:>10.3f}ms {cpp_time:>14.3f}ms {speedup:>8.1f}x")

        avg_speedup = np.mean([r[3] for r in all_results])
        print("-" * 70)
        print(f"{'Average Speedup':<25} {'':<15} {'':<15} {avg_speedup:>8.1f}x")
        print()


def benchmark_batch():
    """Benchmark batch processing"""

    if importlib.util.find_spec("dtw_cpp") is None:
        print("C++ DTW module not available for batch benchmark")
        return

    from baseline_retrieval import retrieve_dtw

    print("\n" + "=" * 70)
    print("Batch Processing Benchmark")
    print("=" * 70)
    print()

    test_cases = [
        (100, 10, 100, "10 sequences of length 100"),
        (100, 100, 100, "100 sequences of length 100"),
        (100, 1000, 100, "1000 sequences of length 100"),
    ]

    np.random.seed(42)
    query_tokens = [str(int(x)) for x in np.random.uniform(1, 1000, 20)]

    for seq_len, num_seqs, query_len, label in test_cases:
        audio_filenames = [f"audio_{i}.wav" for i in range(num_seqs)]
        audio_sequences = [",".join(str(int(x)) for x in np.random.uniform(1, 1000, seq_len)) for _ in range(num_seqs)]

        print(f"\nTest: {label}")
        print(f"  Query length: {len(query_tokens)} tokens")
        print(f"  Sequence length: {seq_len} tokens")
        print(f"  Number of sequences: {num_seqs}")

        start = time.perf_counter()
        retrieve_dtw(query_tokens, audio_sequences, audio_filenames, top_k=5)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Time: {elapsed:.2f}ms")
        print(f"  Throughput: {num_seqs / (elapsed / 1000):.0f} sequences/sec")


if __name__ == "__main__":
    benchmark_dtw()
    benchmark_batch()
