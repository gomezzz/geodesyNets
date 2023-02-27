import os
from typing import Union, List

import pandas as pd


def read_result_csv(
        root_directory: os.PathLike,
        include: Union[List[str], None] = None,
        exclude: Union[List[str], None] = None
) -> pd.DataFrame:
    """This method reads all csv files in a given root directory, where the sub-path contains a specific pattern.

    Args:
        root_directory: the root directory from which to recursively search for csv files
        include: the patterns the sub-path must contain
        exclude: the patterns the sub-paths is not allowed to contain

    Returns:
        pandas DataFrame of concatenated csv files

    """
    dataframe = pd.DataFrame()
    for dir_path, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if not filename.endswith(".csv"):
                continue
            path = os.path.join(dir_path, filename)
            if any(x in path for x in include) and not any(x in path for x in exclude):
                element = pd.read_csv(path)
                dataframe = pd.concat([dataframe, element])
    return dataframe
