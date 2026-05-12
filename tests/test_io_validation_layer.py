import logging
from pathlib import Path

import pytest

from MassFlow.io import UnsupportedVendorFormatError, load_spectra
from MassFlow.log_config import setup_structured_logging

MALFORMED_MGF_CONTENT = """
BEGIN IONS
PEPMASS=100.0
SCANS=1
TITLE=Valid Spectrum 1
101.0 10.0
102.0 20.0
END IONS

BEGIN IONS
PEPMASS=200.0
SCANS=2
TITLE=Invalid Spectrum - Non-positive Intensity
201.0 10.0
202.0 -5.0
END IONS

BEGIN IONS
PEPMASS=300.0
SCANS=3
TITLE=Invalid Spectrum - Unsorted M/Z
302.0 10.0
301.0 20.0
END IONS

BEGIN IONS
PEPMASS=0
SCANS=4
TITLE=Invalid Spectrum - Zero Precursor
401.0 10.0
402.0 20.0
END IONS

BEGIN IONS
SCANS=5
TITLE=Invalid Spectrum - Missing Precursor
501.0 10.0
502.0 20.0
END IONS

BEGIN IONS
PEPMASS=600.0
SCANS=6
TITLE=Valid Spectrum 2
601.0 10.0
602.0 20.0
END IONS
"""


@pytest.fixture
def malformed_mgf_file(tmp_path: Path) -> Path:
    """Creates a temporary MGF file with both valid and invalid spectra."""
    mgf_file = tmp_path / "test.mgf"
    mgf_file.write_text(MALFORMED_MGF_CONTENT)
    return mgf_file


def test_validation_layer_quarantines_junk_spectra(
    malformed_mgf_file: Path, tmp_path: Path
):
    """
    Verify that the I/O validation layer correctly identifies, logs,
    and filters out malformed spectra.
    """
    # Configure logging to capture quarantine logs in the temp directory
    setup_structured_logging()
    quarantine_log_path = tmp_path / "massflow_quarantine.log"
    if quarantine_log_path.exists():
        quarantine_log_path.unlink()

    # Link quarantine logger to a file in the temp directory
    q_logger = logging.getLogger("quarantine")
    # Remove existing handlers to avoid duplicates from previous test runs
    for handler in list(q_logger.handlers):
        q_logger.removeHandler(handler)
    q_handler = logging.FileHandler(quarantine_log_path)
    q_logger.addHandler(q_handler)

    # Load spectra from the malformed file
    spectra = list(load_spectra(malformed_mgf_file))

    # 1. Assert that only the valid spectra were loaded
    assert len(spectra) == 2
    valid_scans = {s.get("scans") for s in spectra}
    assert valid_scans == {"1", "6"}

    # 2. Assert that the quarantine log was created and contains the correct messages
    assert quarantine_log_path.exists()
    log_content = quarantine_log_path.read_text()

    assert "Quarantined Spectrum" in log_content
    assert "SCANS=2 | Reason: Contains non-positive intensity values" in log_content
    assert (
        "SCANS=3 | Reason: M/Z values are not monotonically increasing" in log_content
    )
    assert "SCANS=4 | Reason: Non-positive precursor_mz: 0.0" in log_content
    assert "SCANS=5 | Reason: Missing precursor_mz" in log_content

    # Clean up the log file and handler
    quarantine_log_path.unlink()
    q_logger.removeHandler(q_handler)


def test_unsupported_vendor_format_raises_error(tmp_path):
    """Test that loading a proprietary file format raises the correct error."""
    raw_file = tmp_path / "test.raw"
    raw_file.touch()
    with pytest.raises(UnsupportedVendorFormatError):
        list(load_spectra(raw_file))


def test_load_spectra_file_not_found():
    """Test that a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        list(load_spectra(Path("nonexistent.mgf")))
