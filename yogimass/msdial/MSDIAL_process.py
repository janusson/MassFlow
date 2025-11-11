# Uses combined output from MSDIAL_clean_combine.py for data cleaning and analysis.

# Import modules
import os
import sys
import numpy as np
import pandas as pd
from glob import glob as glob
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

csv_list = glob(
    r"C:\Programming\yogimass\msdial\export\msdial_processed_data\combined_results"
    + r"/*.csv"
)

# for a single csv file:
summary_df = pd.read_csv(csv_list[0])  # 'csv_list[0]' to be replaced by 'i' in for loop

summary_df.head(1)

summary_df.sort_values(by=["Model ion area"], inplace=True)
drop_df = summary_df.dropna(subset=["Name"])

drop_df.head()

# experiment paramters (mill speed, mill time, batch number, etc.)
