# slab_plot_spectra.py
# plot msms spectrum dictionary

import pandas as pd
from log import slab_logging
log_formatter, logger = slab_logging.main()
logger.info('Starting plot_mass_spectrum_with_labels1.py')

combined_peak_matches = pd.read_csv(r'C:\Programming\SLAB\out\qh_selected_peaks.csv')

# convert string to xy pairs
def msms_string_to_xy(msms_string):
    '''
    Converts an MS-Dial peak: {mz:intensity} string to a list of xy pairs.

    Args:
        msms_string (string): An MS-Dial output MSMS spectrum string.

    Returns:
        dict: A dictionary of the MSMS spectrum in the format {mz:intensity}.
    '''
    from collections import OrderedDict
    peak_list = str(msms_string).split(' ')
    msms_spectra_dict = OrderedDict()
    for peak in peak_list:
        peak_mz, peak_abundance = float(
            peak.split(":")[0]), int(peak.split(":")[1])
        msms_spectra_dict[peak_mz] = peak_abundance
    return msms_spectra_dict

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
    import pandas as pd

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
        if int(counts) > 0:
            rounded_dict[new_mz] = int(counts)
        # rounded_dict[new_mz] = int(counts)
    # plot setup
    plot1 = plt.bar(
        data_dictionary.keys(),
        data_dictionary.values(),
        color="navy",
        edgecolor="navy",
        linewidth=None,
        width=0.01,
        alpha=0.8,
    )
    plt.xlim(10, max(rounded_dict.keys())+10)
    plt.ylim(0, max(rounded_dict.values())+50000) #? needs tuning to spectrum height
    # Plot labels:
    plt.xlabel("m/z", fontproperties=font, style="italic")
    plt.ylabel("Counts", fontproperties=font)
    plt.title(f"{filename.split('.')[0]}", fontproperties=font)
    # plt.bar_label(plot1,
    #                 rounded_dict.keys(),
    #                 fontproperties=font,
    #                 fontsize=10)

    plot_name = path.join(output_path, filename.split('.')[0])
    # ax = plt.gca()

    output_path = path.join(plot_name + "_MSMS_Spectrum.png")
    plt.savefig(output_path)
    #  debugging log:
    logger.info(f"Graphic saved to: {output_path}")
    return plot1

for i, spectrum in enumerate(combined_peak_matches['MSMS spectrum']):
    filename = combined_peak_matches['Filename'][i]
    spectrum = msms_string_to_xy(spectrum)
    plot_mass_spectrum_with_labels(spectrum, filename)