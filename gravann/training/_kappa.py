import torch
from gravann.functions._integration import ACC_trap, U_trap_opt
from gravann.labels._mascon_labels import acceleration_mascon_differential as MASCON_ACC_L, \
    potential_mascon as MASCON_U_L, acceleration_mascon_differential as MASCON_ACC_L_differential
from gravann.polyhedral import ACC_L as POLYHEDRAL_ACC_L, U_L as POLYHEDRAL_U_L

from gravann.input._io import load_sample, load_polyhedral_mesh
from gravann.util._sample_observation_points import get_target_point_sampler


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
            mascon_points, mascon_masses, mascon_masses_nu = load_sample(sample, use_acc)
        else:
            mascon_points, mascon_masses, mascon_masses_nu = kwargs.get("mascon_points", None), kwargs.get(
                "mascon_masses", None), kwargs.get("mascon_masses_nu", None)
        return compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu, use_acc)
    elif method == 'polyhedral':
        if sample is not None:
            mesh_vertices, mesh_faces = load_polyhedral_mesh(sample)
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
                lambda x: MASCON_ACC_L(x, mascon_points, mascon_masses),
                lambda x: ACC_trap(x, model, encoding, N=100000)
            )
        else:
            return _compute_c_for_model(
                lambda x: MASCON_ACC_L_differential(x, mascon_points, mascon_masses, mascon_masses_nu),
                lambda x: ACC_trap(x, model, encoding, N=100000)
            )
    else:
        return _compute_c_for_model(
            lambda x: MASCON_U_L(x, mascon_points, mascon_masses),
            lambda x: U_trap_opt(x, model, encoding, N=100000)
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
            lambda x: POLYHEDRAL_ACC_L(x, mesh_vertices, mesh_faces),
            lambda x: ACC_trap(x, model, encoding, N=100000)
        )
    else:
        return _compute_c_for_model(
            lambda x: POLYHEDRAL_U_L(x, mesh_vertices, mesh_faces),
            lambda x: U_trap_opt(x, model, encoding, N=100000)
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
