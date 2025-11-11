# build_library_from_msp.py
# Path: yogimass\build_library_from_msp.py
# Description: Builds database from msp files containing tandem MS spectra.
import yogimass_pipeline
from yogimass import yogimass_logging
log_formatter, logger = yogimass_logging.main()

def library_settings():
    # define a list of library directories
    small_msp_libraries = r'.\msp databases\small'
    small_msp_libraries_2 = r'.\msp databases\small_2'
    medium_msp_libraries = r'.\msp databases\medium'
    medium_msp_libraries_2 = r'.\msp databases\medium_2'
    large_msp_libraries = r'.\msp databases\large'
    very_large_msp_libraries = r'.\msp databases\large_2'

    libs_pathlist = [
        small_msp_libraries, small_msp_libraries_2, medium_msp_libraries,
        medium_msp_libraries_2, large_msp_libraries, very_large_msp_libraries
    ]
    return libs_pathlist
libs_pathlist = library_settings()

#* Deprecated
# def export_as_pickle(msp_library_dir = small_msp_libraries):
#     msp_list = yogimass_pipeline.list_msp_libraries(small_msp_libraries)

#     for library in msp_list:
#         library_name = library.split('\\')[-1].split('.')[0]
#         print(f'Importing {library_name}...')
#         try:
#             cleaned_library = yogimass_pipeline.clean_msp_library(library)
#         except Exception as e:
#             print(f'Error importing: {library_name}')
#             continue
#         yogimass_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database', library_name)
#         # yogimass_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', library_name)
#         return print(f'Cleaned and exported as Pickle: {library_name}')

'''
#* Deprecated - Export MSMS spectra in library as pickle for small lib 1
for library in msp_list_1:
    try:
        cleaned_library = yogimass_pipeline.clean_msp_library(library)
        lib_name = library.split('\\')[-1].split('.')[0]
        yogimass_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database',
                                             lib_name)
        #! as .msp:
        # yogimass_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', lib_name)
        print(f'Cleaned and exported library: {lib_name}')
    except Exception as e:
        print(f'Error / exception: {e}')
        continue

#* Repeat for small lib 2
msp_list_2 = yogimass_pipeline.list_msp_libraries(libs_pathlist[6])
for library in msp_list_2:
    try:
        cleaned_library = yogimass_pipeline.clean_msp_library(library)
        lib_name = library.split('\\')[-1].split('.')[0]
        yogimass_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database',
                                             lib_name)
        #! as .msp:
        # yogimass_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', lib_name)
        print(f'Cleaned and exported library: {lib_name}')
    except Exception as e:
        print(f'Error / exception: {e}')
        continue

# END
'''

for library_path in libs_pathlist:
    print(f'Importing {library_path}...')
    msp_list_1 = yogimass_pipeline.list_msp_libraries(library_path) # all

    for file in libs_pathlist:
        msp_list_1 = yogimass_pipeline.list_msp_libraries(file) # list spectra in library
        print(f'Loaded {len(msp_list_1)} spectra from {file}')
        #* Export MSMS spectra in library as pickle for libraries
        for library in msp_list_1:
            try:
                cleaned_library = yogimass_pipeline.clean_msp_library(library)
                lib_name = library.split('\\')[-1].split('.')[0]
                #? As .pickle:
                yogimass_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database',
                                                    lib_name)
                logger.info(f'Cleaned and exported to Pickle library: {lib_name}')
                #? as .msp: 
                # #! Warning: Resource intensive. Needs work!
                # yogimass_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', lib_name)
                # logger.info(f'Cleaned and exported to MSP library: {lib_name}')
            except Exception as e:
                print(f'Error / exception: {e}')
                continue