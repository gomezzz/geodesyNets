import pytest
import pickle as pk
import torch
import gravann._polyhedral_labels as polyhedral_labels
import gravann._mascon_labels as mascons_labels


EROS_LP_FILE_NAME = './3dmeshes/eros_lp.pk'
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


def test_eros_potential():
    target_points = [[0.0, 0.0, 0.0]]
    target_points_tensor = torch.tensor(target_points)

    # density = 1.49828e10

    mascon_points, mascon_masses = mascon_data(EROS_MASCON_FILE_NAME)
    vertices, triangles = mesh_data(EROS_LP_FILE_NAME)

    mascon_potential = mascons_labels.U_L(target_points_tensor, mascon_points, mascon_masses)
    polyhedral_potential = polyhedral_labels.U_L(target_points, vertices, triangles, 1.0)

    assert mascon_potential == polyhedral_potential, \
        f"Mascon {mascon_potential}; Polyhedral {polyhedral_potential}"


def test_eros_acceleration():
    target_points = [[0.0, 0.0, 0.0]]
    target_points_tensor = torch.tensor(target_points)

    # density = 1.49828e10

    mascon_points, mascon_masses = mascon_data(EROS_MASCON_FILE_NAME)
    vertices, triangles = mesh_data(EROS_LP_FILE_NAME)

    mascon_acceleration = mascons_labels.ACC_L(target_points_tensor, mascon_points, mascon_masses)
    polyhedral_acceleration = polyhedral_labels.ACC_L(target_points, vertices, triangles, 1.0)

    assert all([m == p for m, p in zip(mascon_acceleration[0], polyhedral_acceleration[0])]), \
        f"Mascon {mascon_acceleration[0]}; Polyhedral {polyhedral_acceleration[0]}"
