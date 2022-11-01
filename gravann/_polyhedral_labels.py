import polyhedral_gravity

import torch


def U_L(target_points, mesh_vertices, mesh_edges, density=1.0):
    """
    Computes the gravity potential (G=1) created by a mascon in the target points. (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the potential is seeked.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_edges (2-D array-like): a (N,) array-like object containing the edges of the polyhedron
        density: scalar: the constant density of the polyhedron

    Returns:
        1-D array-like: a (N, 1) torch tensor containing the gravity potential (G=1) at the target points
    """
    result = polyhedral_gravity.evaluate(mesh_vertices, mesh_edges, density, target_points)
    return torch.tensor([potential for potential, acceleration, tensor in result])


def ACC_L(target_points, mesh_vertices, mesh_edges, density=1.0):
    """
    Computes the acceleration due to the mascon at the target points. (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the  acceleration should be computed.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_edges (2-D array-like): a (N,) array-like object containing the edges of the polyhedron
        density: scalar: the constant density of the polyhedron

    Returns:
        1-D array-like: a (N, 3) torch tensor containing the acceleration (G=1) at the target points
    """
    result = polyhedral_gravity.evaluate(mesh_vertices, mesh_edges, density, target_points)
    return torch.tensor([acceleration for potential, acceleration, tensor in result])
