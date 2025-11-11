from numpy import e
import MSMS_pipeline
import matchms

small_msp_libraries = r'D:\Mass Spectrometry Libraries\msp Files\small'
small_msp_libraries_2 = r'D:\Mass Spectrometry Libraries\msp Files\small_2'
medium_msp_libraries = r'D:\Mass Spectrometry Libraries\msp Files\medium'
medium_msp_libraries_2 = r'D:\Mass Spectrometry Libraries\msp Files\medium_2'
large_msp_libraries = r'D:\Mass Spectrometry Libraries\msp Files\large'
very_large_msp_libraries = r'D:\Mass Spectrometry Libraries\msp Files\large_2'

libs_pathlist = [
    small_msp_libraries, small_msp_libraries_2, medium_msp_libraries,
    medium_msp_libraries_2, large_msp_libraries, very_large_msp_libraries
]

# def export_as_pickle(msp_library_dir = small_msp_libraries):
#     msp_list = MSMS_pipeline.list_msp_libraries(small_msp_libraries)

#     for library in msp_list:
#         library_name = library.split('\\')[-1].split('.')[0]
#         print(f'Importing {library_name}...')
#         try:
#             cleaned_library = MSMS_pipeline.clean_msp_library(library)
#         except Exception as e:
#             print(f'Error importing: {library_name}')
#             continue
#         MSMS_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database', library_name)
#         # MSMS_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', library_name)
#         return print(f'Cleaned and exported as Pickle: {library_name}')

'''
#* Export MSMS spectra in library as pickle for small lib 1
for library in msp_list_1:
    try:
        cleaned_library = MSMS_pipeline.clean_msp_library(library)
        lib_name = library.split('\\')[-1].split('.')[0]
        MSMS_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database',
                                             lib_name)
        #! as .msp:
        # MSMS_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', lib_name)
        print(f'Cleaned and exported library: {lib_name}')
    except Exception as e:
        print(f'Error / exception: {e}')
        continue

#* Repeat for small lib 2
msp_list_2 = MSMS_pipeline.list_msp_libraries(libs_pathlist[6])
for library in msp_list_2:
    try:
        cleaned_library = MSMS_pipeline.clean_msp_library(library)
        lib_name = library.split('\\')[-1].split('.')[0]
        MSMS_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database',
                                             lib_name)
        #! as .msp:
        # MSMS_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', lib_name)
        print(f'Cleaned and exported library: {lib_name}')
    except Exception as e:
        print(f'Error / exception: {e}')
        continue

# END
'''

for library_path in libs_pathlist:
    print(f'Importing {library_path}...')
    msp_list_1 = MSMS_pipeline.list_msp_libraries(library_path) # all

    for file in libs_pathlist:
        #! why is this listed again?
        msp_list_1 = MSMS_pipeline.list_msp_libraries(file) # list spectra in library
        print(f'Loaded {len(msp_list_1)} spectra from {file}')
        #* Export MSMS spectra in library as pickle for libraries
        for library in msp_list_1:
            try:
                cleaned_library = MSMS_pipeline.clean_msp_library(library)
                lib_name = library.split('\\')[-1].split('.')[0]
                #? As .pickle:
                # MSMS_pipeline.save_spectra_to_pickle(cleaned_library, r'.\database',
                #                                     lib_name)
                #? as .msp:
                MSMS_pipeline.save_spectra_to_msp(cleaned_library, r'.\database', lib_name)
                print(f'Cleaned and exported library: {lib_name}')
                input('Continue?')
            except Exception as e:
                print(f'Error / exception: {e}')
                continue
