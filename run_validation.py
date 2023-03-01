import os
import pathlib

from gravann import validator_runner

if __name__ == "__main__":
    input_directory = pathlib.Path("F:/tmp")
    output_directory = pathlib.Path("./results/re-validation")
    output_directory.mkdir(parents=True, exist_ok=True)

    validator_runner.run(
        input_directory,
        output_directory,
        N=1000,
        sampling_altitudes=[1.0],
        integration_points=50000
    )
