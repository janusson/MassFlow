
"""
Script to verify MSP library loading and processing.
"""
import argparse
import sys
import os
from MassFlow import processing, io

def verify_library(msp_path: str) -> None:
    print(f"Testing MSP loading from: {msp_path}")

    try:
        # 1. Test Listing
        data_dir = os.path.dirname(msp_path)
        # If path is just a file in cwd, dirname is empty string, which glob might handle or might need '.'
        if not data_dir:
            data_dir = "."
            
        found_libs = io.list_msp_libraries(data_dir)
        print(f"Libraries found in directory: {found_libs}")
        
        abs_path = os.path.abspath(msp_path)
        if msp_path not in found_libs and abs_path not in found_libs:
             print("WARNING: list_msp_libraries didn't list the exact file (might be path formatting), but continuing...")

        # 2. Test Cleaning/Loading
        print("\nAttempting to clean/load library...")
        spectra = processing.clean_msp_library(msp_path)
        
        print(f"\nSUCCESS: Loaded {len(spectra)} spectra.")
        if len(spectra) > 0:
            print("First spectrum metadata:")
            print(spectra[0].metadata)
            print(f"First spectrum peaks: {len(spectra[0].peaks.mz)}")

    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify MSP library.")
    parser.add_argument("path", help="Path to MSP file")
    args = parser.parse_args()
    
    verify_library(args.path)
