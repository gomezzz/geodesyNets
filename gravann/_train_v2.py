import pickle as pk
import time
from collections import deque

import pandas as pd
from tqdm import tqdm

from ._encodings import *
from ._io import save_results, save_plots_v2
from ._losses import contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss
# Required for loading runs
from ._train_v2_init import init_training_sampler, init_environment, init_model_and_optimizer, init_prediction_label, \
    init_ground_truth_labels, init_input_data
from ._validation import validation_results_unpack_df
from ._validation_v2 import validation_v2


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
    run_folder = init_environment(cfg)
    # Initialize the model and associated torch training utility
    model, early_stopper, optimizer, scheduler = init_model_and_optimizer(
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
    target_points_sampler = init_training_sampler(
        sample=cfg["sample"], target_sample_method=cfg["sample_method"],
        sample_domain=cfg["sample_domain"], batch_size=cfg["batch_size"]
    )
    # Initialize the prediction function by binding the model and defined configuration parameters
    prediction_fn = init_prediction_label(
        integrator=cfg["integrator"], model=model, encoding=cfg["encoding"],
        integration_points=cfg["integration_points"], integration_domain=cfg["integration_domain"]
    )
    # Initialize the input data
    input_data = init_input_data(
        sample=cfg["sample"]
    )
    # Initialize the label function by binding the sample data
    label_fn = init_ground_truth_labels(
        ground_truth=cfg["ground_truth"], input_data=input_data
    )

    # When a new network is created: init empty training logs and loss trend indicators
    loss_log, lr_log, weighted_average_log, n_inferences = [], [], [], []
    weighted_average = deque([], maxlen=20)
    target_points, labels = [], []

    t = tqdm(range(cfg["iterations"]), ncols=150)
    # At the beginning (first plots) we assume no learned c
    c = 1.
    for it in t:
        # TODO Each hundred epochs we could produce the plots
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
        "progressbar": False
    }
    validation_kwargs.update(input_data)
    validation_results = validation_v2(
        model,
        cfg["encoding"],
        cfg["sample"],
        cfg["validation_ground_truth"],
        cfg.get("use_acceleration", True),
        500000,
        **validation_kwargs
    )

    print("Saving...")
    save_results(loss_log, weighted_average_log, validation_results, model, run_folder)

    save_plots_v2(
        model, cfg["encoding"], cfg["sample"],
        lr_log, loss_log, weighted_average_log,
        n_inferences, run_folder, c, cfg["plotting_points"]
    )

    # store run config
    cfg_dict = {"Sample": cfg["sample"],
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
                "Noise": cfg["noise"],
                "Target Sampler": cfg["sample_method"],
                "Integration Points": cfg["integration_points"],
                "c": c}

    with open(run_folder + 'config.pk', 'wb') as handle:
        pk.dump(cfg_dict, handle)

    # Compute validation results
    val_res = validation_results_unpack_df(validation_results)

    # Time Measurements
    end_time = time.time()
    runtime = end_time - start_time

    result_dictionary = {"Sample": cfg["sample"],
                         "Type": "ACC" if cfg["use_acceleration"] else "U",
                         "Model": cfg["model_type"],
                         "Loss": cfg["loss_fn"].__name__,
                         "Encoding": cfg["encoding"].name,
                         "Integrator": cfg["integrator"].__name__,
                         "Activation": str(cfg["activation"])[:-2],
                         "Batch Size": cfg["batch_size"],
                         "LR": cfg["learning_rate"],
                         "Ground Truth": cfg["ground_truth"],
                         "Target Sampler": cfg["sample_method"],
                         "Noise": cfg["noise"],
                         "Integration Points": cfg["integration_points"],
                         "Runtime": runtime,
                         "Final Loss": loss_log[-1],
                         "Final WeightedAvg Loss": weighted_average_log[-1]}
    results_df = pd.concat(
        [pd.DataFrame([result_dictionary]), val_res], axis=1)
    return results_df
