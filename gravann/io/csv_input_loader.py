import os
import pandas as pd


def read_result_csv(root_directory) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if str(filename) == "results.csv" and str(dirpath).__contains__("noise"):
                element = pd.read_csv(os.path.join(dirpath, filename))
                dataframe = pd.concat([dataframe, element])
    return dataframe
