# build_library_from_msp.py
# Description: Builds database from msp files containing tandem MS spectra.

import slab_processing_pipeline
from log import slab_logging

log_formatter, logger = slab_logging.main()


def library_settings():
    from os import path
    # define a list of library directories
    msp_libraries = path.abspath(r'.\database\msp databases')
    # TODO move MSP source to a separate folder
    return libs_pathlist


libs_pathlist = library_settings()

for library_path in libs_pathlist:
    print(f'Importing {library_path}...')
    msp_list_1 = slab_processing_pipeline.list_msp_libraries(
        library_path)  # all

    for file in libs_pathlist:
        msp_list_1 = slab_processing_pipeline.list_msp_libraries(
            file)  # list spectra in library
        print(f'Loaded {len(msp_list_1)} spectra from {file}')
        #* Export MSMS spectra in library as pickle for libraries
        for library in msp_list_1:
            try:
                cleaned_library = slab_processing_pipeline.clean_msp_library(
                    library)
                lib_name = library.split('\\')[-1].split('.')[0]
                #? As .pickle:
                slab_processing_pipeline.save_spectra_to_pickle(
                    cleaned_library, r'.\database', lib_name)
                logger.info(
                    f'Cleaned and exported to Pickle library: {lib_name}')
                #! Warning: Resource intensive.
                slab_processing_pipeline.save_spectra_to_msp(
                    cleaned_library, r'.\database', lib_name)
                logger.info(f'Cleaned and exported to MSP library: {lib_name}')
            except Exception as e:
                print(f'Error: {e}')
                continue