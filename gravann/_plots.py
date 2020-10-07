from matplotlib import pyplot as plt
import torch

def plot_mascon(points, masses=None, elev=45, azim=125, alpha=0.1, s=None):
    """Plots a mascon model

    Args:
        points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points
        masses (1-D array-like): a (N,) array-like object containing the values for the point masses
        elev (float): elevation of the starting 3D view
        azim (float): azimuth for the starting 3D view
        alpha (float): alpha for the mass visualization
        s (int): scale for the visualized masses

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if s is None:
        s = 22000 / len(points)

    if masses is None:
        normalized_masses = s
    else:
        normalized_masses = masses / sum(masses)
        normalized_masses = normalized_masses * s * len(x)

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')

    # And visualize the masses
    ax.scatter(x, y, z, color='k', s=normalized_masses, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(222)
    ax2.scatter(x, y, color='k', s=normalized_masses, alpha=alpha)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])

    ax3 = fig.add_subplot(223)
    ax3.scatter(x, z, color='k', s=normalized_masses, alpha=alpha)
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])

    ax4 = fig.add_subplot(224)
    ax4.scatter(y, z, color='k', s=normalized_masses, alpha=alpha)
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])


def plot_model_grid(model, encoding, N=20, bw=False, alpha=0.2, views_2d=True):
    """Plots the neural model of the asteroid density in the [-1,1]**3 cube showing
    the density value on a grid.

    Args:
        model (callable (N,M)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): grid size (N**3 points will be plotted).
        bw (bool): when True considers zero density as white and transparent. The final effect is a black and white plot
        alpha (float): alpha for the visualization
        views_2d (bool): activates also the 2d projections

    """
    # We create the grid
    x = torch.linspace(-1, 1, N)
    y = torch.linspace(-1, 1, N)
    z = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid((x, y, z))

    # We compute the density on the grid points (no gradient as its only for plotting)
    nn_inputs = torch.cat(
        (X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), dim=1)
    nn_inputs = encoding(nn_inputs)
    RHO = model(nn_inputs).detach()

    # And we plot it
    fig = plt.figure()
    if views_2d:
        ax = fig.add_subplot(221, projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
    if bw:
        col = torch.cat((1-RHO, 1-RHO, 1-RHO, RHO), dim=1)
        alpha = None
    else:
        col = RHO

    ax.scatter(X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1),
               marker='.', c=col, s=100, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45., azim=125.)

    if views_2d:
        ax2 = fig.add_subplot(222)
        ax2.scatter(X.reshape(-1, 1)[:, 0], Y.reshape(-1, 1)
                    [:, 0], marker='.', c=col, s=100, alpha=alpha)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])

        ax3 = fig.add_subplot(223)
        ax3.scatter(X.reshape(-1, 1)[:, 0], Z.reshape(-1, 1)
                    [:, 0], marker='.', c=col, s=100, alpha=alpha)
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])

        ax4 = fig.add_subplot(224)
        ax4.scatter(Y.reshape(-1, 1)[:, 0], Z.reshape(-1, 1)
                    [:, 0], marker='.', c=col, s=100, alpha=alpha)
        ax4.set_xlim([-1, 1])
        ax4.set_ylim([-1, 1])



def plot_model_rejection(model, encoding, N=30**3, views_2d=False, bw=False, alpha=0.2, crop_p=1e-2, s=100):
    """Plots the neural model of the asteroid density in the [-1,1]**3 cube interpreting the density
    as a probability distribution and performing a rejection sampling approach

    Args:
        model (callable (N,M)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points to be considered.
        views_2d (bool): activates also the 2d projections
        bw (bool): results in a black and white plot
        alpha (float): alpha for the visualization
        crop_p (float): all points below this density are rejected
        s (int): size of the non rejected points visualization
    """
    points = torch.rand(N, 3) * 2 - 1
    nn_inputs = encoding(points)
    RHO = model(nn_inputs).detach()
    mask = RHO > (torch.rand(N, 1) + crop_p)
    RHO = RHO[mask]
    points = [[it[0].item(), it[1].item(), it[2].item()]
              for it, m in zip(points, mask) if m]
    points = torch.tensor(points)
    fig = plt.figure()
    if views_2d:
        ax = fig.add_subplot(221, projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
    if bw:
        col = 'k'
    else:
        col = RHO
    # And we plot it
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               marker='.', c=col, s=s, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45., azim=125.)

    if views_2d:
        ax2 = fig.add_subplot(222)
        ax2.scatter(points[:, 0], points[:, 1],
                    marker='.', c=col, s=s, alpha=alpha)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])

        ax3 = fig.add_subplot(223)
        ax3.scatter(points[:, 0], points[:, 2],
                    marker='.', c=col, s=s, alpha=alpha)
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])

        ax4 = fig.add_subplot(224)
        ax4.scatter(points[:, 1], points[:, 2],
                    marker='.', c=col, s=s, alpha=alpha)
        ax4.set_xlim([-1, 1])
        ax4.set_ylim([-1, 1])
