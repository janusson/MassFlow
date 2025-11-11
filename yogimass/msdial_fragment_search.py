# Description: Searches tandem MS peak data (MS-Dial) for CID-generated product ions.

import pandas as pd
from glob import glob as glob
from os import path as path
from datetime import datetime as dt

# data import and settings
timestamp = dt.now().strftime(r"%Y%m%d_%H%M%S")

# io paths
alignment_dir = path.abspath(r'./import/')
alignment_file_list = glob(alignment_dir + '/*.txt')
# ms_dial_txt = alignment_file_list[17]

# MSMS fragment m/z ranges
mz_range_1 = list(range(191, 198))  # quinone-like
mz_range_2 = list(range(204, 208))  # alternate quinone-like
mz_range_3 = list(range(152, 158))  # terpene
mz_range_4 = list(range(167, 174))  # terpene 2
mz_range_5 = list(range(167, 174))  # terpene 3

all_fragment_mzranges = [
    mz_range_1, mz_range_2, mz_range_3, mz_range_4, mz_range_5
]
t_q_check = [mz_range_1, mz_range_3, mz_range_4, mz_range_5]


def plot_mass_spectrum_with_labels(data_dictionary,
                                   filename,
                                   output_path=".\\out\\plots\\"):
    """
    Plots a mass spectrum dictionary {m/z:counts} with labels for each peak.

    Args:
        data_dictionary (dict): mass-to-charge and intensity key:value pair dictionary
        filename (str): name of the file to be plotted
        output_path (str): path to save the plot

    Returns:
        plot: Saves a plot to local output (.\\out) as {filename}.png
    """
    from os import path
    from matplotlib import pyplot as plt
    from decimal import Decimal
    import os

    # check if output directory exists, if not create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # plot font properties
    from matplotlib.font_manager import FontProperties

    font = FontProperties()
    font.set_family("sans-serif")
    font.set_name("Segoe UI")
    font.set_size(14)

    # round values in dictionary to 2 decimal places for new dictionary of plot labels
    rounded_dict = {}
    for mz, counts in data_dictionary.items():
        new_mz = float(Decimal(mz).quantize(Decimal("0.01")))
        rounded_dict[new_mz] = int(counts)
    # plot setup
    plot1 = plt.bar(
        data_dictionary.keys(),
        data_dictionary.values(),
        color="purple",
        edgecolor="black",
        linewidth=0,
        width=0.4,
        alpha=0.8,
    )
    # Plot appearance:
    plt.xlabel("m/z", fontproperties=font, style="italic")
    plt.ylabel("Counts", fontproperties=font)
    plt.title(f"{filename}", fontproperties=font)
    try:  #! TODO save plot error?
        plt.bar_label(plot1,
                      str(rounded_dict.keys()),
                      fontproperties=font,
                      fontsize=10)
        plot_name = path.join(output_path, filename)
        output_path = path.join(plot_name + ".png")
        plt.savefig(output_path)
    except:
        print("Error: plotting failed.")
    #  debugging log items:
    # logger.info(f"Saving plot to: {output_path}")
    # logger.info(f"Graphic saved as {filename}.png")
    return plot1


def match_nominal_products(ms_dial_txt,
                           mz_range_list,
                           min_area=0,
                           min_product_area=0):
    '''
    Converts MS-Dial MSMS peak data to a dictionary and filters by product ion m/z ranges. Returns a list of peak numbers with matching product ions m/z ranges. Also exports MSMS spectra (.png) for matching peaks.
    
    Args:
        ms_dial_txt (pathlike): Path to MS-Dial alignment results text file
        mz_range_list (list): List of integer m/z values to consider as a product ion match.
        min_area (int, optional): Minimum MSMS product ion peak area to be considered a match. Defaults to 0.

    Returns:
        str, set: MS-Dial text filename and set of peak numbers with matching product ion m/z ranges.
    '''
    # import ms-dial alignment results as a dataframe
    from os import path
    filename = path.basename(ms_dial_txt)
    peak_nums = set()
    results_dict = {}
    msdial_df = pd.read_table(ms_dial_txt, sep='\t')
    if 'MSMS spectrum' in msdial_df.columns:
        # filter by minimum precursor ion area
        filter_msdial_df = msdial_df[msdial_df['Area'] > min_area]
        for peak_number, spec in enumerate(filter_msdial_df['MSMS spectrum']):
            peak_number -= 1 # subtract 1 to match with MS-Dial peak number
            try:
                if len(spec) > 0 and hasattr(spec, '__iter__'):
                    # convert msms spectra and matching peaks from string to dicts
                    peak_list = str(spec).split(' ')
                    msms_spectra_dict = {}
                    for peak in peak_list:
                        peak_mz, peak_abundance = float(
                            peak.split(":")[0]), int(peak.split(":")[1])
                        msms_spectra_dict[peak_mz] = peak_abundance
                        for unit_mz in mz_range_list:
                            if (int(peak_mz) == unit_mz) and (
                                    peak_abundance > min_product_area):
                                peak_nums.add(peak_number)
                                results_dict[peak_number] = str(
                                    msms_spectra_dict)
                                #TODO: plotting
                                # plot filtered and unfiltered spectra matching the precursor range, use filtered peaks as labels
                                # plot_mass_spectrum_with_labels(msms_spectra_dict, f'{filename}_peak{peak_number}_MSMS_spectrum', output_path=".\\out\\plots\\")
                                # plot_mass_spectrum_with_labels(filtered_msms_dict, f'{filename}_peak{peak_number}_filtered_MSMS_peaks', output_path=".\\out\\plots\\")
                            else:
                                continue
            except TypeError as e:
                pass
    return filename, peak_nums, results_dict


results_df = pd.DataFrame()
for fragments in t_q_check:
    results_df['Fragment m/z range'] = str(f'{fragments[0]}-{fragments[1]}')
    for ms_dial_file in alignment_file_list:
        name, nums, df = match_nominal_products(ms_dial_file,
                                                mz_range_1,
                                                min_area=10000,
                                                min_product_area=5000)
        # tune min_area (precursor) and min_product_area (product) as needed.
        # Use a lower precursor ion threshold and a greater product ion threshold
        #  for enhanced selectivity.
        x = pd.DataFrame.from_dict(
            df, orient='index').rename(columns={0: 'MSMS Spectrum'})
        print(x.head())
        x['Filename'] = name
        print(f'Appending {name} peak numbers: {nums}')
        x['Peak Number'] = x.index
        results_df = results_df.append(x)
        x.reset_index().dropna()
results_df.dropna()

# make totals column which is sum of values in MSMS spectrum dictionary
# results_df['Totals'] = results_df['MSMS Spectrum'].apply(
#     lambda x: sum(dict(x).values()))

results_df.to_csv(f'{timestamp}_slabmsdial_results.csv', index=False)