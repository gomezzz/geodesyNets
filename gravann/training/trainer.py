import pickle as pk
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from gravann.output import plot_model_rejection
from gravann.output import plot_saver
from . import training_initializer, validator
from .losses import *


def train_on_batch_v2(points, prediction_fn, labels, loss_fn, optimizer, scheduler):
    """Trains the passed model on the passed batch

    Args:
        points (tensor): target points for training
        prediction_fn (func): prediction func of the model
        labels (tensor): labels at the target points
        loss_fn (func): loss function for training
        optimizer (torch optimizer): torch optimizer to use
        scheduler (torch LR scheduler): torch LR scheduler to use

    Returns:
        torch tensor: losses
    """
    predicted = prediction_fn(points)
    c = torch.sum(predicted * labels) / torch.sum(predicted * predicted)

    if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss]:
        loss = loss_fn(predicted, labels)
    else:
        loss = loss_fn(predicted.view(-1), labels.view(-1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step(loss.item())

    return loss, c


def run_training_v2(cfg: dict) -> pd.DataFrame:
    """Runs a specific parameter configuration.

    Args:
        cfg: dictionary with the configuration for the training run

    Returns:
        results
    """
    # Time Measurement
    start_time = time.time()

    # Initialize the environment and prepare the run folder
    run_folder = training_initializer.init_environment(cfg)
    # Initialize the model and associated torch training utility
    model, early_stopper, optimizer, scheduler = training_initializer.init_model_and_optimizer(
        run_folder=run_folder,
        encoding=cfg["encoding"],
        n_neurons=cfg["n_neurons"],
        activation=cfg["activation"],
        model_type=cfg["model_type"],
        omega=cfg["omega"],
        hidden_layers=cfg["hidden_layers"],
        learning_rate=cfg["learning_rate"]
    )
    # Initialize the target point sampler
    target_points_sampler = training_initializer.init_training_sampler(
        sample=cfg["sample"], target_sample_method=cfg["sample_method"],
        sample_domain=cfg["sample_domain"], batch_size=cfg["batch_size"]
    )
    # Initialize the prediction function by binding the model and defined configuration parameters
    prediction_fn = training_initializer.init_prediction_label(
        model=model, encoding=cfg["encoding"],
        integration_points=cfg["integration_points"], integration_domain=cfg["integration_domain"],
        use_acc=cfg["use_acceleration"]
    )
    # Initialize the input data
    input_data = training_initializer.init_input_data(
        sample=cfg["sample"]
    )
    # Initialize the label function by binding the sample data
    label_fn = training_initializer.init_ground_truth_labels(
        ground_truth=cfg["ground_truth"], input_data=input_data, use_acc=cfg["use_acceleration"]
    )
    # Add noise on top (if defined)
    label_fn = training_initializer.init_noise(
        label_fn, cfg["noise_method"], cfg["noise_params"]
    )

    # When a new network is created: init empty training logs and loss trend indicators
    loss_log, lr_log, weighted_average_log, n_inferences = [], [], [], []
    weighted_average = deque([], maxlen=20)
    target_points, labels = [], []

    t = tqdm(range(cfg["iterations"]), ncols=150)
    # At the beginning (first plots) we assume no learned c
    c = 1.
    for it in t:
        if it % 500 == 0:
            plot_model_rejection(
                model, encoding=cfg["encoding"],
                views_2d=True, bw=True, N=cfg["plotting_points"], alpha=0.1, s=50, c=c,
                save_path=f"{run_folder}rejection_plot_iter{it}.png"
            )
            plt.close('all')
        # Each ten epochs we resample the target points
        if it % 10 == 0:
            target_points = target_points_sampler()
            labels = label_fn(target_points)

        # Train
        loss, c = train_on_batch_v2(
            target_points, prediction_fn, labels,
            cfg["loss_fn"], optimizer, scheduler
        )

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        lr_log.append(optimizer.param_groups[0]['lr'])
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        n_inferences.append((cfg["integration_points"] * cfg["batch_size"]) // 1000)
        wa_out = np.mean(weighted_average)

        t.set_postfix_str(f"L={loss.item():.3e} | AvgL={wa_out:.3e} | c={c:.3e}")

        if early_stopper.early_stop(loss.item(), model):
            print(f"Early stopping at minimal loss {early_stopper.minimal_loss}")
            break

        torch.cuda.empty_cache()
    # Restore best checkpoint
    print("Restoring best checkpoint for validation...")
    model.load_state_dict(torch.load(run_folder + "best_model.mdl"))

    # Compute the validation
    print("Validating...")
    validation_kwargs = {
        "N": cfg["validation_points"],
        "progressbar": False,
        "sampling_altitudes": cfg["validation_sampling_altitudes"]
    }
    validation_kwargs.update(input_data)
    validation_results = validator.validation_v2(
        model,
        cfg["encoding"],
        cfg["sample"],
        cfg["validation_ground_truth"],
        cfg.get("use_acceleration", True),
        500000,
        **validation_kwargs
    )

    print("Saving...")
    plot_saver.save_results(loss_log, weighted_average_log, validation_results, model, run_folder)

    plot_saver.save_plots_v2(
        model, cfg["encoding"], cfg["sample"],
        lr_log, loss_log, weighted_average_log,
        n_inferences, run_folder, c, cfg["plotting_points"]
    )

    # store run config
    cfg_dict = {"Sample": cfg["sample"],
                "Seed": cfg["seed"],
                "Type": "ACC" if cfg["use_acceleration"] else "U",
                "Model": cfg["model_type"],
                "Loss": cfg["loss_fn"].__name__,
                "Encoding": cfg["encoding"].name,
                "Integrator": cfg["integrator"].__name__,
                "Activation": str(cfg["activation"])[:-2],
                "n_neurons": cfg["n_neurons"],
                "hidden_layers": cfg["hidden_layers"],
                "Batch Size": cfg["batch_size"],
                "LR": cfg["learning_rate"],
                "Ground Truth": cfg["ground_truth"],
                "Noise Method": cfg["noise_method"],
                "Noise Params": str(cfg["noise_params"]),
                "Target Sampler Method": cfg["sample_method"],
                "Target Sampler Domain": cfg["sample_domain"],
                "Integration Points": cfg["integration_points"],
                "c": c}

    with open(run_folder + 'config.pk', 'wb') as handle:
        pk.dump(cfg_dict, handle)

    # Compute validation results
    val_res = validator.validation_results_unpack_df(validation_results)

    # Time Measurements
    end_time = time.time()
    runtime = end_time - start_time

    result_dictionary = {"Sample": cfg["sample"],
                         "Seed": cfg["seed"],
                         "Type": "ACC" if cfg["use_acceleration"] else "U",
                         "Model": cfg["model_type"],
                         "Loss": cfg["loss_fn"].__name__,
                         "Encoding": cfg["encoding"].name,
                         "Integrator": cfg["integrator"].__name__,
                         "Activation": str(cfg["activation"])[:-2],
                         "Batch Size": cfg["batch_size"],
                         "LR": cfg["learning_rate"],
                         "Ground Truth": cfg["ground_truth"],
                         "Target Sampler Method": cfg["sample_method"],
                         "Target Sampler Domain": cfg["sample_domain"],
                         "Noise Method": cfg["noise_method"],
                         "Noise Params": str(cfg["noise_params"]),
                         "Integration Points": cfg["integration_points"],
                         "Runtime": runtime,
                         "Final Loss": loss_log[-1],
                         "Final WeightedAvg Loss": weighted_average_log[-1]}
    results_df = pd.concat([pd.DataFrame([result_dictionary]), val_res], axis=1)
    return results_df
