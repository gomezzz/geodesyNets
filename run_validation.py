import pathlib
import sys

from gravann import validator_runner

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IOError("No input provided!")
    name = sys.argv[1]
    input_directory = pathlib.Path(f"./results/{name}")
    output_directory = pathlib.Path(f"./results/re-validation/{name}")
    output_directory.mkdir(parents=True, exist_ok=True)

    validator_runner.run(
        input_directory,
        output_directory,
        N=1000,
        sampling_altitudes=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        integration_points=50000
    )
