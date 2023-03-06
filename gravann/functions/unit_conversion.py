import re
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy._typing import ArrayLike

CONVERSION_FACTORS = {
    "bennu": (352.1486930549145, 7.329e10 * 6.67430e-11 / 352.1486930549145 ** 2),
    "bennu_nu": (352.1486930549145, 7.329e10 * 6.67430e-11 / 352.1486930549145 ** 2),
    "churyumov-gerasimenko": (3126.6064453124995, 9.982e12 * 6.67430e-11 / 3126.6064453124995 ** 2),
    "eros": (20413.864850997925, 6.687e15 * 6.67430e-11 / 20413.864850997925 ** 2),
    "itokawa": (350.438691675663, 3.51e10 * 6.67430e-11 / 350.438691675663 ** 2),
    "itokawa_nu": (350.438691675663, 3.51e10 * 6.67430e-11 / 350.438691675663 ** 2),
    "torus": (3126.6064453124995, 9.982e12 * 6.67430e-11 / 3126.6064453124995 ** 2),
    "hollow": (3126.6064453124995, 9.982e12 * 6.67430e-11 / 3126.6064453124995 ** 2),
    "hollow_nu": (3126.6064453124995, 9.982e12 * 6.67430e-11 / 3126.6064453124995 ** 2),
    "hollow2": (3126.6064453124995, 9.982e12 * 6.67430e-11 / 3126.6064453124995 ** 2),
    "hollow2_nu": (3126.6064453124995, 9.982e12 * 6.67430e-11 / 3126.6064453124995 ** 2)
}


def convert_pandas_altitudes(
        result_df: pd.DataFrame, altitudes: List[float], convert_height: bool = False, only_height: bool = False,
) -> pd.DataFrame:
    """Converts the entries of the given pandas DataFrame. Operates in-place!

    Args:
        result_df: the result dataframe for the validation & training containing all information
        altitudes: the altitudes used in the configuration as extra-validation heights, e.g. [0.25, 0.5, 1.0]
        convert_height: if True, replace the normalized heights with the real heights, e.g. 0.05 --> 0.31224...
        only_height: if True, replace the names with only the height, e.g. relRMSE@0.05 --> only 0.05
            (can unambiguous in the end!)


    Returns:
        pandas DataFrame

    """
    # Legacy stuff
    if "Sample" in result_df:
        result_df.rename(columns={"Sample": "sample"}, inplace=True)
    if "Target Sampler Domain" in result_df:
        result_df.rename(columns={"Target Sampler Domain": "sample_domain"}, inplace=True)

    # Iterate over all possible samples
    for sample_name, (altitude2metric, value2metric) in CONVERSION_FACTORS.items():
        # First adapt validation results
        for column in [x for x in result_df.columns if x.lower().startswith(("Normalized L1 Loss", "RMSE", "relRMSE"))]:
            result_df.loc[result_df["sample"] == sample_name, column] *= value2metric
        # Second adapt the columns from the sampling domain
        factor = altitude2metric if convert_height else 1.0
        result_df.loc[result_df["sample"] == sample_name, "sample_domain"] = \
            result_df.loc[result_df["sample"] == sample_name, "sample_domain"].apply(
                lambda val: np.fromstring(val.replace('[', '').replace(']', ''), sep=',') * factor
            )
        # Rename columns of altitudes to real values
        for column in [x for x in result_df.columns if "@Altitude_" in x]:
            match = re.search(r"(.*@)Altitude_(\d+)", column) \
                if not only_height else re.search(r"().*@Altitude_(\d+)", column)
            if match is not None:
                altitude_prefix = match.group(1)
                altitude_id = int(match.group(2))
                result_df.rename(columns={
                    column: f"{altitude_prefix}{altitudes[altitude_id] * factor}"
                }, inplace=True)
    return result_df


def convert_altitude(sample: str, altitudes: ArrayLike, forward: bool = True) -> ArrayLike:
    """Converts the unitless altitudes of an ArrayLike to metric.

    Args:
        sample: the sample
        altitudes: the altitudes to convert
        forward: if true (default) does the above, otherwise the function converts from meter to unitless

    Returns:
        the converted altitudes in meter [m]

    """
    try:
        conversion_constant, _ = CONVERSION_FACTORS[sample]
        return altitudes * conversion_constant if forward else altitudes / conversion_constant
    except KeyError:
        raise NotImplementedError(f"The requested sample {sample} does not yet have a conversion factor!")


def convert_acceleration(sample: str, acceleration: torch.Tensor, forward: bool = True) -> torch.Tensor:
    """Converts the given unitless accelerations to SI units [m/s^2].

    Args:
        sample: the sample body's name
        acceleration: the accelerations to converts (cartesian vector)
        forward: if true (default) does the above, otherwise the function converts from meter to unitless

    Returns:
        the converted acceleration in [m/s^2]

    """
    try:
        _, conversion_constant = CONVERSION_FACTORS[sample]
        return acceleration * conversion_constant if forward else acceleration / conversion_constant
    except KeyError:
        raise NotImplementedError(f"The requested sample {sample} does not yet have a conversion factor!")
