import torch

from gravann.functions import integration
from gravann.input import sample_reader
from gravann.labels import mascon, polyhedral
from gravann.util import get_target_point_sampler


def compute_c_for_model_v2(model, encoding, method, use_acc=True, **kwargs):
    """Convenience function to calculate the current c constant for a model (based on mascon or polyhedral mesh data)

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        method (str): either 'mascon' or 'polyhedral'
        use_acc (bool): if acceleration should be used (otherwise potential)


    Keyword Args:
        sample (str): sample name (file name) instead of mascons or mesh data
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): asteroid mascon masses
        mesh_vertices ((N, 3) array-like): the vertices of a polyhedron
        mesh_faces ((N, 3) array-like): the triangular faces of a polyhedron
    """
    sample = kwargs.get("sample", None)
    if method == 'mascon':
        if sample is not None:
            mascon_points, mascon_masses, mascon_masses_nu = sample_reader.load_sample(sample, use_acc)
        else:
            mascon_points, mascon_masses, mascon_masses_nu = kwargs.get("mascon_points", None), kwargs.get(
                "mascon_masses", None), kwargs.get("mascon_masses_nu", None)
        return compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu, use_acc)
    elif method == 'polyhedral':
        if sample is not None:
            mesh_vertices, mesh_faces = sample_reader.load_polyhedral_mesh(sample)
        else:
            mesh_vertices, mesh_faces = kwargs.get("mesh_vertices", None), kwargs.get("mesh_faces", None)
        return _compute_c_for_model_polyhedral(model, encoding, mesh_vertices, mesh_faces, use_acc)
    else:
        raise NotImplemented(f"The method {method} is not implemented for compute_c!")


def compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu=None, use_acc=True):
    """Computes the current c constant for a model (given mascon points)

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): asteroid mascon masses
        use_acc (bool): if acceleration should be used (otherwise potential)
    """
    if use_acc:
        if mascon_masses_nu is None:
            return _compute_c_for_model(
                lambda x: mascon.acceleration(x, mascon_points, mascon_masses),
                lambda x: integration.acceleration_trapezoid(x, model, encoding, N=100000)
            )
        else:
            return _compute_c_for_model(
                lambda x: mascon.acceleration_differential(x, mascon_points, mascon_masses, mascon_masses_nu),
                lambda x: integration.acceleration_trapezoid(x, model, encoding, N=100000)
            )
    else:
        return _compute_c_for_model(
            lambda x: mascon.potential(x, mascon_points, mascon_masses),
            lambda x: integration.potential_trapezoid(x, model, encoding, N=100000)
        )


def _compute_c_for_model_polyhedral(model, encoding, mesh_vertices, mesh_faces, use_acc=True):
    """Computes the current c constant for a model.

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mesh_vertices ((N, 3) array): mesh vertices
        mesh_faces ((M, 3) array): mesh triangles
        use_acc (bool): if acceleration should be used (otherwise potential)
    """
    if use_acc:
        return _compute_c_for_model(
            lambda x: polyhedral.acceleration(x, mesh_vertices, mesh_faces),
            lambda x: integration.acceleration_trapezoid(x, model, encoding, N=100000)
        )
    else:
        return _compute_c_for_model(
            lambda x: polyhedral.potential(x, mesh_vertices, mesh_faces),
            lambda x: integration.potential_trapezoid(x, model, encoding, N=100000)
        )


def _compute_c_for_model(label, label_trap):
    """Common method computing the current c constant for a model.

    Args:
        label: mascon or polyhedral label (only takes target points as argument)
        label_trap : label of the trained model (only takes target points as argument)
    """
    targets_point_sampler = get_target_point_sampler(1000, method="spherical", bounds=[0.81, 1.0])
    target_points = targets_point_sampler()
    labels = label(target_points)
    predicted = label_trap(target_points)
    return (torch.sum(predicted * labels) / torch.sum(predicted * predicted)).item()
