import sys
import tomli

from gravann import training_runner

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IOError("No input provided!")
    with open(sys.argv[1], "rb") as config_file:
        cfg = tomli.load(config_file)
    print("INPUT LOADED")
    training_runner.run(cfg)
