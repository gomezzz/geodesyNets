import itertools
import os
import sys

import pandas as pd
import toml
import torch

import gravann


def run(cfg: dict, results_df: pd.DataFrame) -> None:
    """This function runs all the permutations of above settings
    """
    print("Using the following samples:", cfg["samples"])
    gravann.enableCUDA()
    print("Will use device ", os.environ["TORCH_DEVICE"])

    print("#### - START WITH ITERATIONS")
    for sample in cfg["samples"]:
        print(f"###### - SAMPLE START {sample}")
        if cfg["integration"]["limit_domain"]:
            cfg["integration"]["domain"] = gravann.get_asteroid_bounding_box(asteroid_pk_path=f"3dmeshes/{sample}.pk")
        for (
                ground_truth,
                loss, batch_size, learning_rate,
                encoding, activation, hidden_layers, n_neurons,
                omega,
                sample_method, sample_domain,
        ) in itertools.product(
            cfg["ground_truth"],
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
        ):
            print("######## - SINGLE RUN START")
            run_results = gravann.run_training_v2({
                ########################################################################################################
                # Name of the sample and other administrative stuff
                ########################################################################################################
                "sample": sample,
                "output_folder": cfg["output_folder"],
                "plotting_points": cfg["plotting_points"],
                ########################################################################################################
                # Chosen Ground Truth
                ########################################################################################################
                "ground_truth": ground_truth,
                ########################################################################################################
                # Training configuration & Validation Configuration
                ########################################################################################################
                "loss_fn": loss,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "iterations": cfg["training"]["iterations"],
                "validation_points": cfg["training"]["validation_points"],
                "validation_ground_truth": cfg["training"]["validation_ground_truth"],
                "use_acceleration": cfg["model"]["use_acceleration"],
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
                "noise": None,
                ########################################################################################################
                # Integration Configuration
                ########################################################################################################
                "integrator": cfg["integrator"],
                "integration_points": cfg["integration"]["points"],
                "integration_domain": cfg["integration"]["domain"]
            })
            results_df = results_df.append(run_results, ignore_index=True)
            print("######## - SINGLE RUN DONE")
        print(f"###### - SAMPLE {sample} DONE")
    print("#### - ALL ITERATIONS DONE")
    print(f"Writing results csv to {cfg['output_folder']}")
    if os.path.isfile(f"{cfg['output_folder']}/results.csv"):
        previous_results = pd.read_csv(f"{cfg['output_folder']}/results.csv")
        results_df = pd.concat([previous_results, results_df])
    results_df.to_csv(f"{cfg['output_folder']}/results.csv", index=False)
    print("#### - EVERYTHING DONE")


def _init_env(cfg: dict) -> (dict, pd.DataFrame):
    # Select GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]

    # Select integrator and prepare folder path
    if cfg["model"]["use_acceleration"]:
        cfg["integrator"] = gravann.ACC_trap
        cfg["name"] = cfg["name"] + "_" + "ACC"
    else:
        cfg["integrator"] = gravann.U_trap_opt
        cfg["name"] = cfg["name"] + "_" + "U"

    cfg["name"] = cfg["name"] + "_" + cfg["model"]["type"]

    if cfg["integration"]["limit_domain"]:
        cfg["name"] = cfg["name"] + "_" + "limit_int"

    cfg["output_folder"] = "results/" + cfg["name"] + "/"

    # Init results dataframe
    results_df = pd.DataFrame(
        columns=["Sample", "Type", "Model", "Loss", "Encoding", "Integrator", "Activation", "Batch Size", "LR",
                 "Ground Truth", "Target Sampler", "Noise", "Integration Points", "Final Loss",
                 "Final WeightedAvg Loss"])

    return cfg, results_df


def _cfg_to_func(cfg: dict) -> dict:
    losses, encodings, activations = [], [], []

    for loss in cfg["training"]["loss"]:
        losses.append(getattr(gravann, loss))

    for encoding in cfg["model"]["encoding"]:
        encodings.append(getattr(gravann, encoding)())

    for activation in cfg["model"]["activation"]:
        if activation == "Abs":
            activations.append(gravann.AbsLayer())
        else:
            activations.append(getattr(torch.nn, activation)())

    cfg["training"]["loss"] = losses
    cfg["model"]["encoding"] = encodings
    cfg["model"]["activation"] = activations
    return cfg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IOError("No input provided!")
    cfg = toml.load(sys.argv[1])
    cfg = _cfg_to_func(cfg)
    cfg, results_df = _init_env(cfg)
    print(cfg)
    print("INPUT LOADED")
    run(cfg, results_df)
