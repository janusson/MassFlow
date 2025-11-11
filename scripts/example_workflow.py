"""
Example script for running Yogimass cleaning workflows for both MGF and MSP libraries.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yogimass import pipeline
from yogimass.utils.logging import get_logger


def _clean_and_export_single_library(library_path, cleaner, saver, output_dir):
    logger = get_logger(__name__)
    cleaned = cleaner(library_path)
    if not cleaned:
        logger.warning(f"No spectra produced for {library_path}")
        return
    export_name = f"{Path(library_path).stem}_cleaned"
    saver(cleaned, str(output_dir), export_name)
    logger.info("Saved cleaned spectra to %s", output_dir / f"{export_name}{_EXTENSION_LOOKUP[saver]}")


_EXTENSION_LOOKUP = {
    pipeline.save_spectra_to_mgf: ".mgf",
    pipeline.save_spectra_to_msp: ".msp",
}


def main(data_dir="./data", output_dir="./out"):
    logger = get_logger(__name__)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mgf_files = pipeline.list_mgf_libraries(data_dir)
    if mgf_files:
        logger.info("Cleaning first MGF library for demonstration.")
        _clean_and_export_single_library(
            Path(mgf_files[0]),
            pipeline.clean_mgf_library,
            pipeline.save_spectra_to_mgf,
            output_dir,
        )
        pipeline.batch_clean_mgf_libraries(
            data_dir,
            output_dir / "batch_mgf",
            export_formats=("mgf", "json"),
        )
    else:
        logger.warning(f"No MGF files found in {data_dir}")

    msp_files = pipeline.list_msp_libraries(data_dir)
    if msp_files:
        logger.info("Cleaning first MSP library for demonstration.")
        _clean_and_export_single_library(
            Path(msp_files[0]),
            pipeline.clean_msp_library,
            pipeline.save_spectra_to_msp,
            output_dir,
        )
        pipeline.batch_clean_msp_libraries(
            data_dir,
            output_dir / "batch_msp",
            export_formats=("msp",),
        )
    else:
        logger.warning(f"No MSP files found in {data_dir}")


if __name__ == "__main__":
    main()
