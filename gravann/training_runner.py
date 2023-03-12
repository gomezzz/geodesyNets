import functools
import itertools
import os

import pandas as pd

from gravann.training import training as trainer
from gravann.util import get_asteroid_bounding_box, enableCUDA


def run(cfg: dict) -> None:
    """This function runs all the permutations of above settings
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    results_df = pd.DataFrame()

    enableCUDA()
    print("Using the following samples:", cfg["samples"])
    print("Will use device ", os.environ["TORCH_DEVICE"])

    iterable_parameters = [
        cfg["seed"],
        cfg["ground_truth"],
        cfg["noise"],
        cfg["training"]["loss"],
        cfg["training"]["batch_size"],
        cfg["training"]["learning_rate"],
        cfg["model"]["encoding"],
        cfg["model"]["activation"],
        cfg["model"]["hidden_layers"],
        cfg["model"]["n_neurons"],
        cfg["siren"]["omega"],
        cfg["target_sampling"]["method"],
        cfg["target_sampling"]["domain"]
    ]

    it: int = 0

    combi = functools.reduce(lambda x, y: x * y, map(lambda x: len(x), iterable_parameters)) * len(cfg["samples"])
    print(f"#### - TOTAL AMOUNT OF ITERATIONS {combi}")

    for sample in cfg["samples"]:
        print(f"###### - SAMPLE START {sample}")
        if cfg["integration"]["limit_domain"]:
            cfg["integration"]["domain"] = get_asteroid_bounding_box(asteroid_pk_path=f"3dmeshes/{sample}.pk")
        for (
                seed, ground_truth, noise,
                loss, batch_size, learning_rate,
                encoding, activation, hidden_layers, n_neurons,
                omega,
                sample_method, sample_domain,
        ) in itertools.product(*iterable_parameters):
            print("######## - SINGLE RUN START")
            run_results = trainer.run_training_configuration({
                ########################################################################################################
                # Name of the sample and other administrative stuff like the chosen seed
                ########################################################################################################
                "sample": sample,
                "output_folder": f"{os.path.join('results', cfg['name'])}",
                "run_id": it,
                "plotting_points": cfg["plotting_points"],
                "seed": seed,
                ########################################################################################################
                # Chosen Ground Truth
                ########################################################################################################
                "ground_truth": ground_truth,
                ########################################################################################################
                # Training configuration & Validation Configuration
                ########################################################################################################
                "loss": loss,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "iterations": cfg["training"]["iterations"],
                "validation_points": cfg["training"]["validation_points"],
                "validation_ground_truth": cfg["training"]["validation_ground_truth"],
                "use_acceleration": cfg["model"]["use_acceleration"],
                "validation_sampling_altitudes": cfg["training"]["validation_sampling_altitudes"],
                ########################################################################################################
                # Model Configuration
                ########################################################################################################
                "encoding": encoding,
                "activation": activation,
                "hidden_layers": hidden_layers,
                "n_neurons": n_neurons,
                "model_type": cfg["model"]["type"],
                ########################################################################################################
                # Sirene Configuration
                ########################################################################################################
                "omega": omega,
                ########################################################################################################
                # Target Point Sampling Configuration
                ########################################################################################################
                "sample_method": sample_method,
                "sample_domain": sample_domain,
                ########################################################################################################
                # Noise Configuration
                ########################################################################################################
                "noise": noise,
                ########################################################################################################
                # Integration Configuration
                ########################################################################################################
                "integrator": "trapezoid",
                "integration_points": cfg["integration"]["points"],
                "integration_domain": cfg["integration"]["domain"]
            })
            results_df = results_df.append(run_results, ignore_index=True)
            results_df.to_csv(os.path.join("results", cfg['name'], f"results_checkpoint_{it:04d}.csv"), index=False)
            it += 1
            print("######## - SINGLE RUN DONE")
        print(f"###### - SAMPLE {sample} DONE")
    print("#### - ALL ITERATIONS DONE")
    print(f"Writing results csv to {cfg['name']}")
    if os.path.isfile(os.path.join("results", cfg['name'], "results.csv")):
        previous_results = pd.read_csv(os.path.join("results", cfg['name'], "results.csv"))
        results_df = pd.concat([previous_results, results_df])
    results_df.to_csv(os.path.join("results", cfg['name'], "results.csv"), index=False)
    print("#### - EVERYTHING DONE")
