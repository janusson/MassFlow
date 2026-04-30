import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class MSConvertNotFoundError(Exception):
    pass


class ConversionError(Exception):
    pass


def check_msconvert_installed() -> bool:
    """Check if msconvert is available on the system PATH."""
    return shutil.which("msconvert") is not None


def get_vendor_files(input_dir: Path) -> List[Path]:
    """Find all vendor raw files (.raw, .d) in the directory."""
    vendor_files: List[Path] = []
    if not input_dir.exists():
        return vendor_files

    for f in input_dir.iterdir():
        if f.suffix.lower() == ".raw" and f.is_file():
            vendor_files.append(f)
        elif f.suffix.lower() == ".d" and f.is_dir():
            vendor_files.append(f)

    return sorted(vendor_files)


def run_conversion(
    input_path: Path, output_dir: Path, output_format: str = "mzML"
) -> None:
    """
    Run msconvert on a single vendor file or directory.
    """
    if not check_msconvert_installed():
        raise MSConvertNotFoundError(
            "msconvert was not found on the system path. "
            "Please install ProteoWizard and add it to your PATH."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "msconvert",
        str(input_path),
        "-o",
        str(output_dir),
        f"--{output_format.lower()}",
        "--64",
        "--zlib",
        "--filter",
        "peakPicking true 1-",
    ]

    logger.info(f"Running msconvert for {input_path.name}...")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully converted {input_path.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"msconvert failed for {input_path.name}. Error output:")
        logger.error(e.stderr)
        raise ConversionError(f"Failed to convert {input_path.name}") from e


def convert_directory(
    input_dir: Path, output_dir: Path, output_format: str = "mzML"
) -> int:
    """
    Find and convert all vendor files in a directory.
    Returns the number of successfully converted files.
    """
    if not check_msconvert_installed():
        raise MSConvertNotFoundError(
            "msconvert was not found on the system path. "
            "Please install ProteoWizard and add it to your PATH."
        )

    vendor_files: List[Path] = get_vendor_files(input_dir)
    if not vendor_files:
        logger.warning(f"No vendor files (.raw, .d) found in {input_dir}")
        return 0

    logger.info(f"Found {len(vendor_files)} vendor files to convert.")

    success_count = 0
    for file_path in vendor_files:
        try:
            run_conversion(file_path, output_dir, output_format)
            success_count += 1
        except ConversionError:
            logger.error(f"Skipping {file_path.name} due to conversion errors.")
            continue

    return success_count
