"""Verification script to ensure Python and C++ DTW implementations produce identical results.
Tests various scenarios to catch any discrepancies.
"""

import importlib.util
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_implementations():
    """Test that Python and C++ produce identical results"""

    from baseline_retrieval import compute_dtw_distance_python, compute_dtw_distance

    # Check if C++ version is available
    if importlib.util.find_spec("dtw_cpp") is None:
        print("✗ C++ DTW module not found")
        print("  Run: python build_dtw.py\n")
        return False

    import dtw_cpp

    print("✓ C++ DTW module found\n")

    print("=" * 80)
    print("DTW Implementation Verification - Python vs C++")
    print("=" * 80)
    print()

    all_passed = True
    test_cases = []

    # Test Case 1: Identical sequences
    print("Test 1: Identical sequences")
    print("-" * 80)
    query = [1.0, 2.0, 3.0, 4.0, 5.0]
    audio = [1.0, 2.0, 3.0, 4.0, 5.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result}")
    print(f"  C++    result: {cpp_result}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Identical sequences", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Identical sequences", False))
        all_passed = False

    # Test Case 2: Different sequences
    print("Test 2: Different sequences")
    print("-" * 80)
    query = [1.0, 2.0, 3.0]
    audio = [1.5, 2.5, 3.5]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result}")
    print(f"  C++    result: {cpp_result}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Different sequences", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Different sequences", False))
        all_passed = False

    # Test Case 3: Longer sequences
    print("Test 3: Longer sequences (50 tokens)")
    print("-" * 80)
    np.random.seed(42)
    query = np.random.uniform(0, 100, 50).tolist()
    audio = np.random.uniform(0, 100, 50).tolist()

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query length: {len(query)}")
    print(f"  Audio length: {len(audio)}")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Longer sequences (50×50)", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Longer sequences (50×50)", False))
        all_passed = False

    # Test Case 4: Different length sequences
    print("Test 4: Different length sequences")
    print("-" * 80)
    query = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    audio = [1.0, 2.0, 3.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query} (length={len(query)})")
    print(f"  Audio: {audio} (length={len(audio)})")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Different length sequences (6×3)", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Different length sequences (6×3)", False))
        all_passed = False

    # Test Case 5: Very small sequences
    print("Test 5: Very small sequences")
    print("-" * 80)
    query = [1.0]
    audio = [1.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result}")
    print(f"  C++    result: {cpp_result}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Very small sequences (1×1)", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Very small sequences (1×1)", False))
        all_passed = False

    # Test Case 6: Zero values
    print("Test 6: Sequences with zero values")
    print("-" * 80)
    query = [0.0, 0.0, 0.0]
    audio = [0.0, 0.0, 0.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result}")
    print(f"  C++    result: {cpp_result}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Zero values", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Zero values", False))
        all_passed = False

    # Test Case 7: Negative values
    print("Test 7: Sequences with negative values")
    print("-" * 80)
    query = [-1.0, -2.0, -3.0]
    audio = [-1.5, -2.5, -3.5]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Negative values", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Negative values", False))
        all_passed = False

    # Test Case 8: Large values
    print("Test 8: Sequences with large values")
    print("-" * 80)
    query = [1000.0, 2000.0, 3000.0]
    audio = [1500.0, 2500.0, 3500.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result:.1f}")
    print(f"  C++    result: {cpp_result:.1f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Large values", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Large values", False))
        all_passed = False

    # Test Case 9: Random sequences (100×100)
    print("Test 9: Random sequences (100×100)")
    print("-" * 80)
    np.random.seed(123)
    query = np.random.uniform(-10, 10, 100).tolist()
    audio = np.random.uniform(-10, 10, 100).tolist()

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query length: {len(query)}")
    print(f"  Audio length: {len(audio)}")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Random sequences (100×100)", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Random sequences (100×100)", False))
        all_passed = False

    # Test Case 10: Batch processing
    print("Test 10: Batch processing consistency")
    print("-" * 80)
    np.random.seed(456)
    query = [1.0, 2.0, 3.0, 4.0, 5.0]
    audio_sequences = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.5, 2.5, 3.5, 4.5, 5.5],
        [0.5, 1.5, 2.5, 3.5, 4.5],
    ]

    print(f"  Query: {query}")
    print(f"  Number of sequences: {len(audio_sequences)}")

    # Get individual results (Python)
    individual_results = []
    for audio in audio_sequences:
        result = compute_dtw_distance_python(query, audio)
        individual_results.append(result)

    # Get individual results (C++)
    cpp_individual_results = []
    for audio in audio_sequences:
        result = compute_dtw_distance(query, audio)
        cpp_individual_results.append(result)

    # Get batch results (C++)
    query_array = np.array(query, dtype=np.float64)
    audio_arrays = [np.array(audio, dtype=np.float64) for audio in audio_sequences]
    computer = dtw_cpp.DTWComputer()
    batch_results = computer.compute_dtw_batch(query_array, audio_arrays)

    print(f"  Python individual results: {[f'{r:.6f}' for r in individual_results]}")
    print(f"  C++    individual results: {[f'{r:.6f}' for r in cpp_individual_results]}")
    print(f"  C++    batch results:      {[f'{r:.6f}' for r in batch_results]}")

    # Check all match
    all_match = True
    for i, (py_ind, cpp_ind, batch) in enumerate(zip(individual_results, cpp_individual_results, batch_results)):
        if not (np.isclose(py_ind, cpp_ind, rtol=1e-10) and np.isclose(cpp_ind, batch, rtol=1e-10)):
            all_match = False
            print(f"  ✗ Sequence {i}: Mismatch detected")

    if all_match:
        print("  ✓ PASS - All results match\n")
        test_cases.append(("Batch processing consistency", True))
    else:
        print("  ✗ FAIL - Results don't match\n")
        test_cases.append(("Batch processing consistency", False))
        all_passed = False

    # Test Case 11: Query in the middle of audio sequence
    print("Test 11: Query embedded in middle of audio sequence")
    print("-" * 80)
    query = [5.0, 6.0, 7.0]
    # Create audio with query in the middle: [prefix] + query + [suffix]
    audio = [1.0, 2.0, 3.0, 4.0] + [5.0, 6.0, 7.0] + [8.0, 9.0, 10.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio} (query at middle)")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Query in middle of audio", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Query in middle of audio", False))
        all_passed = False

    # Test Case 12: Query in middle with longer sequences
    print("Test 12: Query in middle of longer sequence (50×30)")
    print("-" * 80)
    np.random.seed(789)
    query = np.random.uniform(0, 100, 30).tolist()
    prefix = np.random.uniform(0, 100, 20).tolist()
    suffix = np.random.uniform(0, 100, 20).tolist()
    audio = prefix + query + suffix

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query length: {len(query)}")
    print(f"  Audio length: {len(audio)} (prefix={len(prefix)}, query={len(query)}, suffix={len(suffix)})")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Query in middle (long sequence)", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Query in middle (long sequence)", False))
        all_passed = False

    # Test Case 13: Query in middle with small noise
    print("Test 13: Query in middle with slight noise")
    print("-" * 80)
    query = [10.0, 20.0, 30.0, 40.0, 50.0]
    # Add small noise to the embedded query
    query_noisy = [x + 0.1 for x in query]
    audio = [1.0, 2.0, 3.0] + query_noisy + [60.0, 70.0, 80.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Noisy query in audio: {query_noisy}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Query in middle with noise", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Query in middle with noise", False))
        all_passed = False

    # Test Case 14: Multiple instances of query pattern in audio
    print("Test 14: Query pattern appears multiple times in audio")
    print("-" * 80)
    query = [1.0, 2.0, 3.0]
    # Pattern appears at position 3, 8, and 13
    audio = [0.0, 0.0] + [1.0, 2.0, 3.0] + [5.0, 6.0] + [1.0, 2.0, 3.0] + [9.0] + [1.0, 2.0, 3.0]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio} (pattern at indices 2, 7, 11)")
    print(f"  Python result: {python_result:.6f}")
    print(f"  C++    result: {cpp_result:.6f}")

    match = np.isclose(python_result, cpp_result, rtol=1e-10)
    if match:
        print("  ✓ PASS - Results match\n")
        test_cases.append(("Query pattern repeated in audio", True))
    else:
        print(f"  ✗ FAIL - Difference: {abs(python_result - cpp_result)}\n")
        test_cases.append(("Query pattern repeated in audio", False))
        all_passed = False

    # Test Case 15: Query at different positions (performance test)
    print("Test 15: Query at different positions in audio")
    print("-" * 80)
    query = [5.0, 6.0, 7.0, 8.0]
    positions = {
        "start": query + [20.0, 30.0, 40.0, 50.0],
        "middle": [10.0, 20.0] + query + [50.0, 60.0],
        "end": [10.0, 20.0, 30.0, 40.0] + query,
    }

    print(f"  Query: {query}")
    results = {}
    for position_name, audio in positions.items():
        python_result = compute_dtw_distance_python(query, audio)
        cpp_result = compute_dtw_distance(query, audio)
        results[position_name] = (python_result, cpp_result)

        match = np.isclose(python_result, cpp_result, rtol=1e-10)
        print(f"  {position_name:<10}: Python={python_result:.6f}, C++={cpp_result:.6f}, Match={match}")

    all_match = all(np.isclose(py, cpp, rtol=1e-10) for py, cpp in results.values())
    if all_match:
        print("  ✓ PASS - All positions match\n")
        test_cases.append(("Query at different positions", True))
    else:
        print("  ✗ FAIL - Position results don't match\n")
        test_cases.append(("Query at different positions", False))
        all_passed = False

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Test':<40} {'Status':<10}")
    print("-" * 80)

    for test_name, passed in test_cases:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<40} {status:<10}")

    total = len(test_cases)
    passed_count = sum(1 for _, passed in test_cases if passed)
    print("-" * 80)
    print(f"{'TOTAL':<40} {passed_count}/{total}")
    print()

    if all_passed:
        print("✓ All tests passed! Python and C++ implementations are identical.")
        return True
    else:
        print("✗ Some tests failed! There are discrepancies between implementations.")
        return False


def test_numerical_precision():
    """Test numerical precision and edge cases"""

    from baseline_retrieval import compute_dtw_distance_python, compute_dtw_distance

    if importlib.util.find_spec("dtw_cpp") is None:
        print("C++ module not available")
        return

    print("\n" + "=" * 80)
    print("Numerical Precision Tests")
    print("=" * 80)
    print()

    # Test with very small differences
    print("Precision Test 1: Very small differences")
    print("-" * 80)
    query = [1.0, 2.0, 3.0]
    audio = [1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10]

    python_result = compute_dtw_distance_python(query, audio)
    cpp_result = compute_dtw_distance(query, audio)

    print(f"  Query: {query}")
    print(f"  Audio: {audio}")
    print(f"  Python result: {python_result:.15f}")
    print(f"  C++    result: {cpp_result:.15f}")
    print(f"  Relative difference: {abs(python_result - cpp_result) / (python_result + 1e-100):.2e}")

    if np.isclose(python_result, cpp_result, rtol=1e-9):
        print("  ✓ PASS - Results match within tolerance\n")
    else:
        print("  ✗ WARNING - Numerical difference detected\n")


if __name__ == "__main__":
    success = test_implementations()
    test_numerical_precision()

    sys.exit(0 if success else 1)
