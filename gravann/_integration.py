import numpy as np
import torch
import sobol_seq
from ._encodings import direct_encoding

# We generate 200000 low-discrepancy points in 3D upon import and store it as a global
# variable
sobol_points = sobol_seq.i4_sobol_generate(3, 200000)

# Naive Montecarlo method


def U_Pmc(target_points, model, encoding=direct_encoding(), N=3000):
    if model[0].in_features != encoding.dim:
        print("encoding is incompatible with the model")
        raise ValueError
    # We generate randomly points in the [-1,1]^3 bounds
    sample_points = torch.rand(N, 3) * 2 - 1
    nn_inputs = encoding(sample_points)
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 1)
    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval

# Low-discrepancy Montecarlo


def U_Pld(target_points, model, encoding=direct_encoding(), N=3000, noise=1e-5):
    if model[0].in_features != encoding.dim:
        print("encoding is incompatible with the model")
        raise ValueError
    if N > np.shape(sobol_points)[0]:
        print("Too many points the sobol sequence stored in a global variable only contains 200000.")
    # We generate randomly points in the [-1,1]^3 bounds
    sample_points = torch.tensor(
        sobol_points[:N, :] * 2 - 1) + torch.rand(N, 3) * noise
    nn_inputs = encoding(sample_points)
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 1)
    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval