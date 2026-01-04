from glob import glob as glob
from os import path as path
import pandas as pd
from datetime import datetime as dt
import os
timestamp = dt.now().strftime(r"%Y%m%d_%H%M%S")

# Intensity threshold
min_area= 10000
min_product_area = 4000

# data import and settings
# io paths
alignment_dir = path.abspath(r'./import/')
msdialtxt_list = glob(alignment_dir + '/*.txt')
output_dir = path.abspath(r'./out/')
if not path.exists(output_dir):
    os.makedirs(output_dir)

# fragment ranges
mz_range1 = list(range(191, 198)) # A, quinone-like
mz_range2 = list(range(132, 139)) # B, terpene 1
mz_range3 = list(range(152, 157)) # B, terpene 2
mz_range4 = list(range(168, 173)) # B, terpene 3

# fragment mz lists to screen for matching peaks
# TODO update to use mz_range_list
mz_ranges = mz_range1, mz_range2, mz_range3, mz_range4

# Integer m/z set from fragment ranges
t_q_check = (191, 192, 193, 194, 195, 196, 197, 152, 153, 154, 155, 156, 157, 167, 168, 169, 170, 171, 172, 173, 167, 168, 169, 170, 171, 172, 173)

def data_import_and_settings(ms_dial_txt, min_area) -> pd.DataFrame:
    # if the first line of the text file contains the word "Area" then read in the file with pandas
    file_name = path.basename(ms_dial_txt)
    msdial_df = pd.DataFrame()
    if 'MSMS spectrum' in open(ms_dial_txt).readline():
        msdial_df = pd.read_table(ms_dial_txt, sep='\t')
        ms_dialdf = msdial_df[msdial_df['Area'] > min_area]
        ms_dialdf = msdial_df[msdial_df['RT (min)'] > 0.5]
        ms_dialdf = msdial_df[msdial_df['RT (min)'] < 25]
        msdial_df = ms_dialdf[msdial_df['MSMS spectrum'].str.len() > 50]
        msdial_df['Filename'] = file_name
        # msdial_df.reset_index()
    return msdial_df

# get peak list from MSMS spectrum
def matching_peaksset(msdial_df, mz_range_list, min_product_area) -> set:
    peak_hits = set()
    # for mz_range in mz_range_list:
    #     print(f'Searching for peaks in range: {range(mz_range)}')
    for peak_number, spec in enumerate(msdial_df['MSMS spectrum']):
        peak_number -= 1 # subtract 1 to match with MS-Dial PeakID number
        if hasattr(spec, '__iter__'):
            # convert msms spectra and matching peaks from string to dicts
            peak_list = str(spec).split(' ')
            # get msms spectrum dict for plotting
            msms_spectra_dict = {}
            for peak in peak_list:
                peak_mz, peak_abundance = float(
                    peak.split(":")[0]), int(peak.split(":")[1])
                msms_spectra_dict[peak_mz] = peak_abundance
                # if peak m/z is in range, add to peak set, return list of peaks
                for unit_mz in mz_range_list:
                    if (int(peak_mz) == unit_mz) and (peak_abundance > min_product_area):
                        peak_hits.add(peak_number)
                        print(f'Found match at peakID: {peak_number}')
    return peak_hits

def combine_matching_peaks(msdial_txt_list, mz_range_list, min_product_area):
    from os import path
    combined_df = pd.DataFrame()
    for ms_dial_txt in msdial_txt_list:
        try:
            filename = path.basename(ms_dial_txt)
            og_df = data_import_and_settings(ms_dial_txt, min_area = min_area)
            peaks = matching_peaksset(og_df, mz_range_list, min_product_area)
            filtered_df = og_df.iloc[list(peaks), :]
            filtered_df['Filename'] = str(filename)
            combined_df = combined_df.append(filtered_df)
        except:
            print(f'Error with file: {ms_dial_txt}')
    combined_df.drop_duplicates(subset=['MSMS spectrum'], inplace=True)
    combined_df.to_csv(f'./out/Peak_match_{mz_range_list[0]}-{mz_range_list[-1]}_{timestamp}.csv', index=False)
    return combined_df

def main():
    results = pd.DataFrame()
    for range in mz_ranges:
        main_df = combine_matching_peaks(msdialtxt_list, range, min_product_area = min_product_area)
        results = results.append(main_df)
        results.drop_duplicates(subset=['MSMS spectrum'], inplace=True)
    results.sort_values(by=['Area', 'S/N'], inplace=True)
    results.to_csv(f'./out/Combined_peak_matches_{timestamp}.csv', index=False)
    return results

if __name__ == '__main__':
    results_df = main()
