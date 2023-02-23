"""
This module helps in the design and analysis of Artificial Neural Networks to represent the gravity field of celestial objects.
It was developed by the Advanced Conpcets team in the context of the project "ANNs for geodesy".
"""
import os

# Importing encodings for the spacial asteroid dimensions
from gravann.network._encodings import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates
# Importing misc methods for 3D graphics
from gravann.util._hulls import alpha_shape, ray_triangle_intersect, rays_triangle_intersect, is_outside, is_inside, \
    is_outside_torch
# Importing the method to integrate the density rho(x,y,z) output of an ANN in the unit cube
from gravann.functions._integration import compute_sobol_points, compute_integration_grid
# Methods to load mascons etc.
from gravann.input._io import load_sample, load_polyhedral_mesh, load_mascon_data
# Computation of the constant c (kappa in the paper)
from gravann.training._kappa import compute_c_for_model, compute_c_for_model_v2
# Importing the losses
from gravann.training._losses import normalized_loss, mse_loss, normalized_L1_loss, contrastive_loss, \
    normalized_sqrt_L1_loss, \
    normalized_relative_L2_loss, normalized_relative_component_loss
# Import the labeling functions the mascons
# Importing the mesh_conversion methods
from gravann.util._mesh_conversion import create_mesh_from_cloud, create_mesh_from_model
# Importing the plots
from gravann.output._plots import plot_mascon, plot_model_grid, plot_model_rejection, plot_model_contours, \
    plot_potential_contours
from gravann.output._plots import plot_mesh, plot_model_mesh, plot_point_cloud_mesh, plot_points, \
    plot_model_mascon_acceleration
from gravann.output._plots import plot_model_vs_cloud_mesh, plot_gradients_per_layer, plot_model_vs_mascon_rejection, \
    plot_model_vs_mascon_contours
# Importing methods to sample points around asteroid
from gravann.util._sample_observation_points import get_target_point_sampler
# Stokes coefficient utilities
from gravann.util._stokes import mascon2stokes, Clm, Slm, constant_factors, legendre_factory_torch
# Import training utility functions
from gravann.training._train_v2 import run_training_v2
# Import utility functions
from gravann.util._utils import max_min_distance, enableCUDA, fixRandomSeeds, print_torch_mem_footprint, \
    get_asteroid_bounding_box, \
    EarlyStopping, unpack_triangle_mesh
# Importing the validation method
from gravann.training._validation import validation, validation_results_unpack_df
from gravann.training._validation_v2 import validation_v2
# Custom layer for siren
from gravann.network._abs_layer import AbsLayer

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = 'cpu'
