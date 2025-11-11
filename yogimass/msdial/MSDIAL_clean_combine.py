# Functions for processing MS-DIAL exported data.
from glob import glob
import os
import pandas as pd
from datetime import datetime as dt
timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
script_path = os.path.dirname(os.path.realpath(__file__))

def set_io_dirs(def_in=r'import', def_out=r'export/msdial_processed_data'):
    in_path = os.path.abspath(os.path.join(script_path, def_in))
    out_path = os.path.abspath(os.path.join(script_path, def_out))
    if not os.path.exists(in_path):
        print(
            f'Input directory not found at: {in_path}. Creating empty input folder.')
        os.mkdir(out_path)
    if not os.path.exists(out_path):
        print(f'Creating output directory at: {out_path}')
        os.mkdir(out_path)
    return in_path, out_path

def load_msdial_data(tsv_path):
    '''
    Read MS-DIAL peak data (.tsv), adds filename information, return as a dataframe.
    '''
    data_df = pd.read_csv(tsv_path, sep='\t', header=0)
    # proc_data['Alignment ID'] = proc_data['Alignment ID'].astype(int)
    return data_df

def get_MSMS_peaks(processed_data):
    '''
    For a given alignment ID, return the raw MS/MS spectrum.
    '''
    spectrum_number = int(input('Enter alignment ID: '))
    processed_data_subset = processed_data[processed_data['Alignment ID'] == spectrum_number]
    select_msms = processed_data['MS/MS spectrum'].where(
        processed_data['Alignment ID'] == spectrum_number).dropna()
    spectrum_peaks = select_msms.iloc[0].split() # first entry is the tandem MS data

    peak_table = pd.DataFrame()
    for i in spectrum_peaks:
        peak_table = peak_table.append(
            pd.DataFrame([i.split(':')], columns=['m/z', 'intensity']))
        peak_table['m/z'] = peak_table['m/z'].astype(float)
        peak_table['intensity'] = peak_table['intensity'].astype(float)
    peak_table.reset_index(drop=True, inplace=True)
    peak_table.sort_values(by='m/z', inplace=True)
    # precursor_for_spectrum = processed_data['Average Mz'].where(processed_data['Alignment ID'] == spectrum_number) # FIXME
    # adduct_for_spectrum = processed_data['Adduct type'].where(processed_data['Alignment ID'] == spectrum_number) # FIXME
    # spectrum_precursor = msms_peaks[0].split() # first entry is the tandem MS data

    ''' Example Usage:
    # Copy peaks to clipboard for export of alignment ID
    msms_peaks = get_MSMS_peaks()

    # For sending to Excel, CCDB, and plotting
    print('Trimming top 10 peaks for clipboard')
    top_10_peaks = msms_peaks.nlargest(10, 'intensity')
    top_10_peaks.to_clipboard(index=False)
    print('MSMS spectrum copied to clipboard.')
    '''
    return peak_table

def plot_EI_spectrum(peaks_table_from_MSMS):
    import matplotlib.pyplot as plt
    barplot = plt.bar(peaks_table_from_MSMS['m/z'],
                      peaks_table_from_MSMS['intensity'],
                      width=0.5)
    plt.xlabel('m/z')
    plt.ylabel('Counts')

    # TODO annotations
    mz_annotations = peaks_table_from_MSMS['m/z'].round(1)
    # mz_top10_labels = mz_annotations.nlargest(10)

    # export = plt.bar_label(barplot,
    #           labels = mz_annotations,
    #           fmt = '%g',
    #           label_type = 'edge',
    #           padding = 0.2)
    return barplot

def msdial_spectra_to_dict(spectrum_path):
###############################################################################
    """
    Converts MS-Dial alignment results export files MSMS spectra to a list of dictionaries composed of m/z and intensity values. Dictionary list represents all MSMS spectra in the MS-Dial results.

    Example input:
    > .\import\qhunt\ms_dial_export\CBD_J_P1-D-1_01_9198.txt

    Args:
        spectrum_path (pathlike): Path to MS-Dial alignment results text file (.txt, tab-delimited)

    Returns:
        list: List of dictionaries containing m/z and intensity values for each MSMS spectrum.
    """
    msdial_df = pd.read_table(spectrum_path, sep="\t")
    msms_spectra_list = []
    for number, spectrum in enumerate(msdial_df["MSMS spectrum"]):
        for ms_peak in str(msdial_df["MSMS spectrum"][number]).split(" "):
            peak_dict = {}
            peak_mz = ms_peak.split(":")[0]
            peak_abundance = ms_peak.split(":")[1]
            if peak_mz not in peak_dict:
                peak_dict[peak_mz] = peak_abundance
            elif peak_mz.get() == 0:
                peak_dict[peak_mz] = peak_abundance
            else:
                pass
            msms_spectra_list.append(peak_dict)
    return msms_spectra_list

def list_msdial_txt(directory):
    txt_list = glob(os.path.join(directory, '*.txt'))
    msdial_txt_list = [item for item in txt_list if 'Scan' in open(item).readline()]
    return msdial_txt_list

def append_df_expname(df, exp_name):
    df['Experiment'] = exp_name
    return df

def lowercase_name(df):
    df['Name'] = df['Name'].str.lower()
    return df

def get_filename(path):
    return os.path.basename(path).split('.')[0]

def name_char_format(df):
    '''
    Replaces certain special characters in the name column.

    Args:
        df (dataframe): Dataframe containing the MS-DIAL assignments.

    Returns:
        dataframe: Original dataframe with special characters replaced.
    '''
    df['Name'] = df['Name'].str.replace('_', ' ', regex = False)
    df['Name'] = df['Name'].str.replace('+', '', regex = False)
    df['Name'] = df['Name'].str.replace('/', '', regex = False)
    df['Name'] = df['Name'].str.replace('"', '', regex = False) # Not working?
    return df

def process_msdial_list(msdial_txt_list):
    for experiment in msdial_txt_list:
        exp_df = load_msdial_data(experiment)
        exp_df = lowercase_name(exp_df)
        exp_df = name_char_format(exp_df)
        exp_df = append_df_expname(exp_df, get_filename(experiment))
        exp_df.to_csv(os.path.join(output_dir, get_filename(experiment) + '.csv'), index = False)
    return print(f'Processed {len(msdial_txt_list)} MS-Dial experiments.')

def combine_results(csv_dir):
    '''
    Combines MS-DIAL (.csv) file summaries into a single dataframe.
    Results are exported to output/combined_results/combined_MSDIAL_results_<timestamp>.

    Args:
        csv_dir (_type_): _description_

    Returns:
        dataframe: Dataframe with combined and filtered MS-Dial data from all experiments (.csv) in folder.
    '''
    csv_files = glob(csv_dir + '/*.csv')
    try:
        combined_df = pd.concat(map(pd.read_csv, csv_files), ignore_index = True)
    except os.error as err:
        return print(err)
    combined_dir = os.path.join(output_dir, f'combined_results')
    if not os.path.exists(combined_dir):
        os.mkdir(combined_dir)
    combined_df.to_excel(os.path.join(combined_dir, f'combined_MSDIAL_results_{timestamp}.xlsx'), index = False)
    combined_df.to_csv(os.path.join(combined_dir, f'combined_MSDIAL_results_{timestamp}.csv'), index = False)
    return combined_df

input_dir, output_dir = set_io_dirs()
msdial_txt_list = list_msdial_txt(input_dir)
process_msdial_list(msdial_txt_list)
combine_results(output_dir)