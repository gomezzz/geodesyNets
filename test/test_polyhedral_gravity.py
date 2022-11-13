import pytest
import pickle as pk
import torch
import numpy as np
import gravann._polyhedral_labels as polyhedral_labels
import gravann._mascon_labels as mascons_labels

EROS_LP_FILE_NAME = './3dmeshes/eros.pk'
EROS_MASCON_FILE_NAME = './mascons/eros.pk'


def mascon_data(filename):
    with open(filename, "rb") as file:
        mascon_points, mascon_masses_u, name = pk.load(file)
        mascon_points = torch.tensor(mascon_points)
        mascon_masses_u = torch.tensor(mascon_masses_u)
        return mascon_points, mascon_masses_u


def mesh_data(filename):
    with open(filename, "rb") as f:
        return pk.load(f)


@pytest.fixture()
def setup():
    torch.set_printoptions(precision=14)


def test_eros_far(setup):
    coordinates = np.array([-100.0, 100.0])
    target_points = np.array(np.meshgrid(coordinates, coordinates, coordinates)).T.reshape(-1, 3)
    target_points_tensor = torch.tensor(target_points)

    # Inverse of Gravity Constant 6.6743015×10−11
    density = 1.49828e10

    mascon_points, mascon_masses = mascon_data(EROS_MASCON_FILE_NAME)
    vertices, triangles = mesh_data(EROS_LP_FILE_NAME)

    mascon_potential = torch.squeeze(mascons_labels.U_L(target_points_tensor, mascon_points, mascon_masses))
    polyhedral_potential = torch.squeeze(polyhedral_labels.U_L(target_points, vertices, triangles, density))

    scaling_factor = mascon_potential[0] / polyhedral_potential[0]

    mascon_acceleration = torch.squeeze(mascons_labels.ACC_L(target_points_tensor, mascon_points, mascon_masses))
    polyhedral_acceleration = torch.squeeze(polyhedral_labels.ACC_L(target_points, vertices, triangles, density))

    assert mascon_potential == pytest.approx(polyhedral_potential * scaling_factor, abs=1e-8), \
        f"Mascon {mascon_potential}; Polyhedral {polyhedral_potential}"

    assert mascon_acceleration == pytest.approx(polyhedral_acceleration * scaling_factor, abs=1e-8), \
        f"Mascon {mascon_acceleration}; Polyhedral {polyhedral_acceleration}"


def test_eros_mid(setup):
    coordinates = np.array([-10.0, 10.0])
    target_points = np.array(np.meshgrid(coordinates, coordinates, coordinates)).T.reshape(-1, 3)
    target_points_tensor = torch.tensor(target_points)

    # Inverse of Gravity Constant 6.6743015×10−11
    density = 1.49828e10

    mascon_points, mascon_masses = mascon_data(EROS_MASCON_FILE_NAME)
    vertices, triangles = mesh_data(EROS_LP_FILE_NAME)

    mascon_potential = torch.squeeze(mascons_labels.U_L(target_points_tensor, mascon_points, mascon_masses))
    polyhedral_potential = torch.squeeze(polyhedral_labels.U_L(target_points, vertices, triangles, density))

    scaling_factor = mascon_potential[0] / polyhedral_potential[0]

    mascon_acceleration = torch.squeeze(mascons_labels.ACC_L(target_points_tensor, mascon_points, mascon_masses))
    polyhedral_acceleration = torch.squeeze(polyhedral_labels.ACC_L(target_points, vertices, triangles, density))

    assert mascon_potential == pytest.approx(polyhedral_potential * scaling_factor, abs=1e-8), \
        f"Mascon {mascon_potential}; Polyhedral {polyhedral_potential}"

    assert mascon_acceleration == pytest.approx(polyhedral_acceleration * scaling_factor, abs=1e-8), \
        f"Mascon {mascon_acceleration}; Polyhedral {polyhedral_acceleration}"

def test_eros_near(setup):
    coordinates = np.array([-1.0, 1.0])
    target_points = np.array(np.meshgrid(coordinates, coordinates, coordinates)).T.reshape(-1, 3)
    target_points_tensor = torch.tensor(target_points)

    # Inverse of Gravity Constant 6.6743015×10−11
    density = 1.49828e10

    mascon_points, mascon_masses = mascon_data(EROS_MASCON_FILE_NAME)
    vertices, triangles = mesh_data(EROS_LP_FILE_NAME)

    mascon_potential = torch.squeeze(mascons_labels.U_L(target_points_tensor, mascon_points, mascon_masses))
    polyhedral_potential = torch.squeeze(polyhedral_labels.U_L(target_points, vertices, triangles, density))

    scaling_factor = mascon_potential[0] / polyhedral_potential[0]

    mascon_acceleration = torch.squeeze(mascons_labels.ACC_L(target_points_tensor, mascon_points, mascon_masses))
    polyhedral_acceleration = torch.squeeze(polyhedral_labels.ACC_L(target_points, vertices, triangles, density))

    assert mascon_potential == pytest.approx(polyhedral_potential * scaling_factor, abs=1e-5), \
        f"Mascon {mascon_potential}; Polyhedral {polyhedral_potential}"

    assert mascon_acceleration == pytest.approx(polyhedral_acceleration * scaling_factor, abs=1e-5), \
        f"Mascon {mascon_acceleration}; Polyhedral {polyhedral_acceleration}"
