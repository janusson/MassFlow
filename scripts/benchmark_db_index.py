import time
from pathlib import Path

import numpy as np
from matchms import Spectrum

from MassFlow.database import SpectralDatabase


def create_mock_spectrum(mz):
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        metadata={"precursor_mz": mz, "name": f"Spec_{mz}"},
    )


def benchmark():
    db_path = Path("benchmark_temp.db")
    if db_path.exists():
        db_path.unlink()

    db = SpectralDatabase(db_path)

    print("Generating 100,000 mock spectra...")
    mzs = np.linspace(50.0, 1000.0, 100000)
    spectra = (create_mock_spectrum(mz) for mz in mzs)

    start = time.time()
    db.add_spectra(spectra)
    print(f"Added 100,000 spectra in {time.time() - start:.2f}s")

    # Test 1: Full scan search (simulated by current get_spectra logic)
    print("\nTest 1: Full scan (current logic)")
    start = time.time()
    count = 0
    target_min, target_max = 500.0, 500.1
    for spec in db.get_spectra():
        mz = spec.get("precursor_mz")
        if mz is not None and target_min <= mz <= target_max:
            count += 1
    print(f"Found {count} spectra in {time.time() - start:.4f}s")

    # Test 2: Add index and use SQL range query
    print("\nTest 2: Adding index and using SQL range query")
    assert db.conn is not None
    cursor = db.conn.cursor()
    cursor.execute("CREATE INDEX idx_precursor_mz ON spectra(precursor_mz)")
    db.conn.commit()

    start = time.time()
    cursor.execute(
        "SELECT * FROM spectra WHERE precursor_mz BETWEEN ? AND ?",
        (target_min, target_max),
    )
    rows = cursor.fetchall()
    count = len(rows)
    print(f"Found {count} spectra in {time.time() - start:.4f}s")

    db.close()
    db_path.unlink()


if __name__ == "__main__":
    benchmark()
