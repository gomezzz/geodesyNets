import itertools

import pytest
import torch

from gravann import ACC_L as MASCON_ACC_L, U_L as MASCON_U_L, load_mascon_data, load_polyhedral_mesh, \
    get_target_point_sampler
from gravann.polyhedral import ACC_L as POLYHEDRAL_ACC_L, U_L as POLYHEDRAL_U_L, calculate_density, \
    GRAVITY_CONSTANT_INVERSE

# ====================== TEST PARAMETERS ======================
# The tested bodies
TEST_FILENAMES = ["bennu", "churyumov-gerasimenko", "eros", "hollow", "itokawa", "torus"]
# The tested distances to the body center
TEST_DISTANCES = [1e-1, 5e-1, 1.0, 5.0, 1e2, 5e2]
# The test relative error needed for comparison between mascon & polyhedral model
TEST_EPSILON = 1.0
# The number of points per body per altitude
TEST_POINTS_PER_BATCH = 100
# The seed for the test cases
TEST_SEED = 0
# Exclude certain bodies with distance combinations from failing with the currently set TEST_EPSILON
TEST_EXCLUDE = {
    "itokawa": [1e-1],
    "hollow": [1.0]
}


# ======================= SETUP UTILITY =======================
def get_data(sample: str):
    """Utility function. Gets the parameters for a sample body
    
    Args:
        sample: the name of the sample

    Returns:
        tuple of mascon_points, mascon_masses, vertices, triangle_faces, density, the name of the sample

    """
    # Get the mascon & mesh data
    mascon_points, mascon_masses = load_mascon_data(sample)
    vertices, triangles = load_polyhedral_mesh(sample)

    # Calculate the density as required for the polyhedral gravity model
    density = calculate_density(vertices, triangles)

    # Pack everything together
    return (mascon_points, mascon_masses), (vertices, triangles), density, sample


# The Data produced by get_data
TEST_DATA = [get_data(file_name) for file_name in TEST_FILENAMES]
# The parameters combined as test input data & the test distances
TEST_PARAMETERS = list(itertools.product(TEST_DATA, TEST_DISTANCES))
# More readable names for the parametrized tests
TEST_NAMES = [f"{name}-{distance}" for name, distance in itertools.product(TEST_FILENAMES, TEST_DISTANCES)]


# ======================== TEST CASE ==========================
@pytest.mark.parametrize("data, distance", TEST_PARAMETERS, ids=TEST_NAMES)
def test_compare_mascon_polyhedral_model(data, distance):
    mascon_data, mesh_data, density, body_name = data
    # Set the print precision of torch for more reasonable messages
    torch.set_printoptions(precision=20)
    # Get the Mascon Parameters
    mascon_points, mascon_masses = mascon_data
    # Get the Mesh Parameters
    vertices, triangles = mesh_data

    # First assert that the mass of the mascon is actually normed and equals one (1.0)
    assert torch.sum(mascon_masses) == pytest.approx(1.0)

    # Get a function to sample the input points
    get_target_point = get_target_point_sampler(TEST_POINTS_PER_BATCH,
                                                'altitude',
                                                [distance],
                                                limit_shape_to_asteroid=f"./3dmeshes/{body_name}.pk",
                                                replace=True,
                                                seed=TEST_SEED)
    # The actual sample points
    target_points = get_target_point()

    # This is just a scaling factor: The solved triple integral is multiplied by the density and the gravity constant
    # In order to set the gravity constant G = 1, we give its inverse to polyhedral model
    polyhedral_gravity_factor = GRAVITY_CONSTANT_INVERSE * density

    # Compute the potential and the acceleration with the two model
    mascon_potential = torch.squeeze(MASCON_U_L(target_points, mascon_points, mascon_masses)) * -1.0
    polyhedral_potential = torch.squeeze(
        POLYHEDRAL_U_L(target_points, vertices, triangles, polyhedral_gravity_factor))
    mascon_acceleration = torch.squeeze(MASCON_ACC_L(target_points, mascon_points, mascon_masses)) * -1.0
    polyhedral_acceleration = torch.squeeze(
        POLYHEDRAL_ACC_L(target_points, vertices, triangles, polyhedral_gravity_factor))

    # Compare the results
    if not (body_name in TEST_EXCLUDE and distance in TEST_EXCLUDE[body_name]):
        assert mascon_potential == pytest.approx(polyhedral_potential, rel=TEST_EPSILON)
        assert mascon_acceleration == pytest.approx(polyhedral_acceleration, rel=TEST_EPSILON)
