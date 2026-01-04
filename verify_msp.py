
import sys
import os
from SpectralMetricMS import processing, io

# Path provided by user
msp_path = "/Users/ericjanusson/Programming/SpectralMetricMS/original_source/data/GNPS-NIH-CLINICALCOLLECTION1.msp"

print(f"Testing MSP loading from: {msp_path}")

try:
    # 1. Test Listing (mimicking how the CLI finds files)
    # The file is in a 'data' directory, let's see if list_msp_libraries finds it if we point to the dir
    data_dir = os.path.dirname(msp_path)
    found_libs = io.list_msp_libraries(data_dir)
    print(f"Libraries found in directory: {found_libs}")
    if msp_path not in found_libs and os.path.abspath(msp_path) not in found_libs:
         print("WARNING: list_msp_libraries didn't list the exact file (might be path formatting), but continuing...")

    # 2. Test Cleaning/Loading (The core check)
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
