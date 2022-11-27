import pytest
import pickle as pk
import torch
import numpy as np
import itertools
import gravann

# ====================== TEST PARAMETERS ======================
# The tested bodies
TEST_FILENAMES = ["bennu", "churyumov-gerasimenko", "eros", "hollow", "itokawa", "planetesimal", "torus"]
# The tested distances to the body center
TEST_DISTANCES = [1.0, 5.0, 1e2, 5e2, 1e3]
# The test epsilon needed for comparison between mascon & polyhedral model
TEST_EPSILON = 1e-1
# Exclude certain bodies with distance combinations from failing with the currently set TEST_EPSILON
TEST_EXCLUDE = {
}
# ====================== TEST PARAMETERS ======================


# Inverse of Gravity Constant 6.6743015×10−11
GRAVITY_CONSTANT_INVERSE = 1.49828e10


def get_mascon_data(filename):
    with open(filename, "rb") as file:
        mascon_points, mascon_masses_u, name = pk.load(file)
        mascon_points = torch.tensor(mascon_points)
        mascon_masses_u = torch.tensor(mascon_masses_u)
        return mascon_points, mascon_masses_u


def get_mesh_data(filename):
    with open(filename, "rb") as f:
        return pk.load(f)


def get_scaling_factor(mascon_points, mascon_masses, vertices, triangles):
    # Generate the input
    coordinates = np.array([-1.0, 1.0])
    cartesian_points = np.array(np.meshgrid(coordinates, coordinates, coordinates)).T.reshape(-1, 3)
    cartesian_points_tensor = torch.tensor(cartesian_points)
    # Evaluate the models
    mascon_potential = torch.squeeze(gravann.U_L(cartesian_points_tensor, mascon_points, mascon_masses))
    polyhedral_potential = torch.squeeze(
        gravann.polyhedral_U_L(cartesian_points_tensor, vertices, triangles, GRAVITY_CONSTANT_INVERSE))
    # Compute the scaling factor as average around our normed body
    return torch.mean(mascon_potential / polyhedral_potential)


def get_data(file_name):
    # Read the mascon & mesh data
    mascon_points, mascon_masses = get_mascon_data(f"./mascons/{file_name}.pk")
    vertices, triangles = get_mesh_data(f"./3dmeshes/{file_name}.pk")

    # Compute the scaling factor as average around our normed body
    scaling_factor = get_scaling_factor(mascon_points, mascon_masses, vertices, triangles)

    # Pack everything together
    return (mascon_points, mascon_masses), (vertices, triangles), scaling_factor, file_name


# The Data produced by get_data
TEST_DATA = [get_data(file_name) for file_name in TEST_FILENAMES]

# The parameters combined as test input data & the test distances
TEST_PARAMETERS = list(itertools.product(TEST_DATA, TEST_DISTANCES))

# More readable names for the parametrized tests
TEST_NAMES = [f"{name}-{distance}" for name, distance in itertools.product(TEST_FILENAMES, TEST_DISTANCES)]


@pytest.mark.parametrize("data, distance", TEST_PARAMETERS, ids=TEST_NAMES)
def test_compare_mascon_polyhedral_model(data, distance):
    mascon_data, mesh_data, scaling_factor, body_name = data
    # Set the print precision of torch for more reasonable messages
    torch.set_printoptions(precision=20)
    # Get the Mascon Parameters
    mascon_points, mascon_masses = mascon_data
    assert torch.sum(mascon_masses) == pytest.approx(1.0)
    # Get the Mesh Parameters
    vertices, triangles = mesh_data

    # Get a normed density for the polyhedral model (set G to 1 via its inverse * a scaling_factor)
    # The scaling factor is derived via the difference of mascon & polyhedral model
    density = GRAVITY_CONSTANT_INVERSE * scaling_factor

    coordinates = np.array([-distance, distance])
    cartesian_points = np.array(np.meshgrid(coordinates, coordinates, coordinates)).T.reshape(-1, 3)
    cartesian_points_tensor = torch.tensor(cartesian_points)

    # Compute the potential and the acceleration with the two model
    mascon_potential = torch.squeeze(gravann.U_L(cartesian_points_tensor, mascon_points, mascon_masses))
    polyhedral_potential = torch.squeeze(gravann.polyhedral_U_L(cartesian_points_tensor, vertices, triangles, density))
    mascon_acceleration = torch.squeeze(gravann.ACC_L(cartesian_points_tensor, mascon_points, mascon_masses))
    polyhedral_acceleration = torch.squeeze(
        gravann.polyhedral_ACC_L(cartesian_points_tensor, vertices, triangles, density))

    # Compare the results
    if not (body_name in TEST_EXCLUDE and distance in TEST_EXCLUDE[body_name]):
        assert mascon_potential == pytest.approx(polyhedral_potential, rel=TEST_EPSILON)
        assert mascon_acceleration == pytest.approx(polyhedral_acceleration, rel=TEST_EPSILON)
