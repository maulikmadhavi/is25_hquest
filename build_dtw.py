#!/usr/bin/env python
"""
Build script for DTW C++ extension.
Usage: python build_dtw.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.absolute()
    
    print("=" * 60)
    print("Building DTW C++ Extension")
    print("=" * 60)
    
    # Step 1: Check if pybind11 is installed
    print("\n[1/3] Checking dependencies...")
    try:
        import pybind11
        print(f"  ✓ pybind11 found at: {pybind11.get_include()}")
    except ImportError:
        print("  ✗ pybind11 not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.6.0"])
    
    # Step 2: Build using setup.py
    print("\n[2/3] Building extension module...")
    build_cmd = [sys.executable, str(root_dir / "setup.py"), "build_ext", "--inplace"]
    
    try:
        subprocess.check_call(build_cmd, cwd=str(root_dir))
        print("  ✓ Build successful")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Build failed with error code {e.returncode}")
        return 1
    
    # Step 3: Verify the module
    print("\n[3/3] Verifying module...")
    sys.path.insert(0, str(root_dir))
    try:
        import dtw_cpp
        print("  ✓ Module imported successfully")
        
        # Test the module
        import numpy as np
        computer = dtw_cpp.DTWComputer()
        query = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        distance = computer.compute_dtw_distance(query, audio)
        print(f"  ✓ Test computation successful (DTW distance: {distance})")
        
    except Exception as e:
        print(f"  ✗ Module verification failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Build completed successfully!")
    print("=" * 60)
    print("\nUsage in Python:")
    print("  import dtw_cpp")
    print("  computer = dtw_cpp.DTWComputer()")
    print("  distance = computer.compute_dtw_distance(query_array, audio_array)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
