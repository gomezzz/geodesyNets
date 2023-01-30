import polyhedral_gravity
import torch

from ._polyhedral_utils import GRAVITY_CONSTANT_INVERSE, calculate_density


def U_L(target_points, mesh_vertices, mesh_edges):
    """Computes the gravity potential (G=1) created by a mascon in the target points.
    (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the potential is seeked.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_edges (2-D array-like): a (N,) array-like object containing the edges of the polyhedron

    Returns:
        1-D array-like: a (N, 1) torch tensor containing the gravity potential (G=1) at the target points
    """
    result = _evaluate_polyhedral_model(target_points, mesh_vertices, mesh_edges)
    try:
        return torch.tensor([potential for potential, acceleration, tensor in result])
    except TypeError:
        # If target_points was a 1D array-like, handle it
        potential, acceleration, tensor = result
        return torch.tensor([potential])


def ACC_L(target_points, mesh_vertices, mesh_edges):
    """Computes the acceleration due to the mascon at the target points.
    (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the  acceleration should be computed.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_edges (2-D array-like): a (N,) array-like object containing the edges of the polyhedron

    Returns:
        1-D array-like: a (N, 3) torch tensor containing the acceleration (G=1) at the target points
    """
    result = _evaluate_polyhedral_model(target_points, mesh_vertices, mesh_edges)
    try:
        return torch.tensor([acceleration for potential, acceleration, tensor in result])
    except TypeError:
        # If target_points was a 1D array-like, handle it
        potential, acceleration, tensor = result
        return torch.tensor([acceleration])


def _evaluate_polyhedral_model(target_points, mesh_vertices, mesh_edges):
    density = calculate_density(mesh_vertices, mesh_edges)
    density_gravity_factor = density * GRAVITY_CONSTANT_INVERSE * -1.0
    # Convert to numpy array, as the interface does not accept a torch tensor
    if torch.is_tensor(target_points):
        target_points = target_points.cpu().detach().numpy()
    return polyhedral_gravity.evaluate(mesh_vertices, mesh_edges, density_gravity_factor, target_points)
