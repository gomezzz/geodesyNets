import matplotlib.colors as colors
import numpy as np
import pyvista as pv
import torch
from matplotlib import pyplot as plt

from .._mascon_labels import ACC_L as MASCON_ACC_L
from .._io import load_polyhedral_mesh, load_mascon_data
from .._sample_observation_points import get_target_point_sampler
from ._polyhedral_labels import ACC_L as POLYHEDRAL_ACC_L
from ._polyhedral_utils import GRAVITY_CONSTANT_INVERSE, calculate_density

pv.set_plot_theme("night")


def plot_polyhedral_mascon_acceleration(sample, plane="XY", altitude=0.1, save_path=None, N=5000, logscale=False):
    """Plots the relative error of the computed acceleration between mascon model and neural network

    Args:
        sample (str): Path to sample mesh
        plane (str, optional): Either "XY","XZ" or "YZ". Defines cross-section. Defaults to "XY".
        altitude (float, optional): Altitude to compute error at. Defaults to 0.1.
        save_path (str, optional): Pass to store plot, if none will display. Defaults to None.
        N (int, optional): Number of points to sample. Defaults to 5000.
        logscale (bool, optional): Logscale errors. Defaults to False.
    Raises:
        ValueError: On wrong input

    Returns:
        plt.Figure: created plot
    """
    # Get the vertices and triangles
    mesh_vertices, mesh_triangles = load_polyhedral_mesh(sample)
    # Get the mascon data
    mascon_points, mascon_masses = load_mascon_data(sample)

    print("Sampling points at altitude")
    points = get_target_point_sampler(N, method="altitude", bounds=[
        altitude], limit_shape_to_asteroid=f"./3dmeshes/{sample}.pk", replace=False)()

    print("Got ", len(points), " points.")
    if plane == "XY":
        cut_dim = 2
        cut_dim_name = "z"
        x_dim = 0
        y_dim = 1
    elif plane == "XZ":
        cut_dim = 1
        cut_dim_name = "y"
        x_dim = 0
        y_dim = 2
    elif plane == "YZ":
        cut_dim = 0
        cut_dim_name = "x"
        x_dim = 1
        y_dim = 2
    else:
        raise ValueError("Plane has to be either XY, XZ or YZ")

    # Left and Right refer to values < 0 and > 0 in the non-crosssection dimension

    print("Splitting in left / right hemisphere")
    points_left = points[points[:, cut_dim] < 0]
    points_right = points[points[:, cut_dim] > 0]

    print("Left: ", len(points_left), " points.")
    print("Right: ", len(points_right), " points.")

    polyhedral_left, mascon_left, relative_error_left = [], [], []
    polyhedral_right, mascon_right, relative_error_right = [], [], []

    # Compute accelerations in left points, then right points
    # for both network and mascon model
    batch_size = 100
    mascon_label = MASCON_ACC_L
    polyhedral_label = POLYHEDRAL_ACC_L

    density = calculate_density(mesh_vertices, mesh_triangles)

    for idx in range((len(points_left) // batch_size) + 1):
        indices = list(range(idx * batch_size, np.minimum((idx + 1) * batch_size, len(points_left))))

        mascon_left.append(
            mascon_label(points_left[indices], mascon_points, mascon_masses).detach())
        polyhedral_left.append(
            polyhedral_label(points_left[indices], mesh_vertices, mesh_triangles, density).detach())

        torch.cuda.empty_cache()

    for idx in range((len(points_right) // batch_size) + 1):
        indices = list(range(idx * batch_size, np.minimum((idx + 1) * batch_size, len(points_right))))

        mascon_right.append(
            mascon_label(points_right[indices], mascon_points, mascon_masses).detach())
        polyhedral_right.append(
            polyhedral_label(points_right[indices], mesh_vertices, mesh_triangles, density).detach())

        torch.cuda.empty_cache()

    # Accumulate all results
    mascon_left = torch.cat(mascon_left)
    polyhedral_left = torch.cat(polyhedral_left)
    mascon_right = torch.cat(mascon_right)
    polyhedral_right = torch.cat(polyhedral_right)

    # Compute relative errors for each hemisphere (left, right)
    relative_error_left = (torch.sum(torch.abs(polyhedral_left - mascon_left), dim=1) /
                           torch.sum(torch.abs(mascon_left + 1e-8), dim=1)).cpu().numpy()
    relative_error_right = (torch.sum(torch.abs(polyhedral_right - mascon_right), dim=1) /
                            torch.sum(torch.abs(mascon_right + 1e-8), dim=1)).cpu().numpy()

    min_err = np.minimum(np.min(relative_error_left),
                         np.min(relative_error_right))
    max_err = np.maximum(np.max(relative_error_left),
                         np.max(relative_error_right))

    norm = colors.Normalize(vmin=min_err, vmax=max_err)

    if logscale:
        relative_error_left = np.log(relative_error_left)
        relative_error_right = np.log(relative_error_right)

    # Get X,Y coordinates of analyzed points
    X_left = points_left[:, x_dim].cpu().numpy()
    Y_left = points_left[:, y_dim].cpu().numpy()

    X_right = points_right[:, x_dim].cpu().numpy()
    Y_right = points_right[:, y_dim].cpu().numpy()

    # Plot left side stuff
    fig = plt.figure(figsize=(8, 4), dpi=100, facecolor='white')
    fig.suptitle("Relative acceleration error in " +
                 plane + " cross section", fontsize=12)
    ax = fig.add_subplot(121, facecolor="black")

    p = ax.scatter(X_left, Y_left, c=relative_error_left,
                   cmap="plasma", alpha=1.0, s=int(N * 0.0005 + 0.5), norm=norm)

    cb = plt.colorbar(p, ax=ax)
    cb.ax.tick_params(labelsize=7)
    if logscale:
        cb.set_label('Log(Relative Error)', rotation=270, labelpad=15)
    else:
        cb.set_label('Relative Error', rotation=270, labelpad=15)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel(plane[0], fontsize=8)
    ax.set_ylabel(plane[1], fontsize=8)
    ax.set_title(cut_dim_name + " < 0")
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', 'box')
    ax.annotate("Mascon Acc. Mag=" + str(torch.mean(torch.sum(torch.abs(mascon_left), dim=1)).cpu().numpy()) +
                "\n" + "Polyhedral Acc. Mag=" +
                str(torch.mean(
                    torch.sum(torch.abs(polyhedral_left), dim=1)).cpu().numpy()),
                (-0.95, 0.8), fontsize=8, color="white")

    # Plot right side stuff
    ax = fig.add_subplot(122, facecolor="black")

    p = ax.scatter(X_right, Y_right, c=relative_error_right,
                   cmap="plasma", alpha=1.0, s=int(N * 0.0005 + 0.5), norm=norm)

    cb = plt.colorbar(p, ax=ax)
    cb.ax.tick_params(labelsize=7)
    if logscale:
        cb.set_label('Log(Relative Error)', rotation=270, labelpad=15)
    else:
        cb.set_label('Relative Error', rotation=270, labelpad=15)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel(plane[0], fontsize=8)
    ax.set_ylabel(plane[1], fontsize=8)
    ax.set_title(cut_dim_name + " > 0")
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', 'box')
    ax.annotate("Mascon Acc. Mag=" + str(torch.mean(torch.sum(torch.abs(mascon_right), dim=1)).cpu().numpy()) +
                "\n" + "Polyhedral Acc. Mag=" +
                str(torch.mean(
                    torch.sum(torch.abs(polyhedral_right), dim=1)).cpu().numpy()),
                (-0.95, 0.8), fontsize=8, color="white")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()

    return ax, mascon_right
