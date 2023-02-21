import os
import pandas as pd


def read_result_csv(root_directory, dir_substring="") -> pd.DataFrame:
    dataframe = pd.DataFrame()
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if str(filename) == "results.csv" and dir_substring in str(dirpath):
                element = pd.read_csv(os.path.join(dirpath, filename))
                dataframe = pd.concat([dataframe, element])
    return dataframe
