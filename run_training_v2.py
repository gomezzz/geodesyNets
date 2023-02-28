import sys

import toml

from gravann import training_runner

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IOError("No input provided!")
    cfg = toml.load(sys.argv[1])
    print("INPUT LOADED")
    training_runner.run(cfg)
