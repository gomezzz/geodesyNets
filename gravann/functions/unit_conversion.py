import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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


def convert_pandas_altitudes(result_df: pd.DataFrame, altitudes) -> pd.DataFrame:
    # Legacy stuff
    if "Sample" in result_df:
        result_df.rename(columns={"Sample": "sample"}, inplace=True)
    if "Target Sampler Domain" in result_df:
        result_df.rename(columns={"Target Sampler Domain": "sample_domain"}, inplace=True)

    for sample_name, (altitude2metric, value2metric) in CONVERSION_FACTORS.items():
        # np.fromstring("[1.0, 5.0]".replace('[', '').replace(']', ''), sep=',')
        # First adapt validation results
        for column in [x for x in result_df.columns if x.lower().startswith(("Normalized L1 Loss", "RMSE", "relRMSE"))]:
            result_df.loc[result_df["sample"] == sample_name, column] *= value2metric
        # Second adapt the columns from the sampling domain
        result_df.loc[result_df["sample"] == sample_name, "sample_domain"] = \
            result_df.loc[result_df["sample"] == sample_name, "sample_domain"].apply(
                lambda val: np.fromstring(val.replace('[', '').replace(']', ''), sep=',') * altitude2metric
            )
    return result_df


def convert_altitude(sample: str, altitudes: ArrayLike) -> ArrayLike:
    """Converts the unitless altitudes of an ArrayLike to metric.

    Args:
        sample: the sample
        altitudes: the altitudes to convert

    Returns:
        the converted altitudes in meter [m]

    """
    try:
        conversion_constant, _ = CONVERSION_FACTORS[altitudes]
        return altitudes * conversion_constant
    except KeyError:
        raise NotImplementedError(f"The requested sample {sample} does not yet have a conversion factor!")
