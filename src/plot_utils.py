import numpy as np
from pylab import cm
import matplotlib
import matplotlib.pyplot as plt

__all__ = ['load_inferno', 'make_onerow']

def load_inferno(n=10, colormap='inferno'):
    """ Returns a discrete colormap with n values.
    """
    cmap = cm.get_cmap(colormap, n)
    colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colors.append(matplotlib.colors.rgb2hex(rgb))
    colors = np.array(colors)[1:-1]
    return colors

def make_onerow():
    """
    Creates a broken axis plot for one visit.
    """
    fig, axes = plt.subplots(ncols=5, nrows=1,
                             figsize=(20,5))

    ax = axes.reshape(-1)

    d = 0.025

    for j in range(5):
        if j > 0 and j < 4:
            ax[j].spines['right'].set_visible(False)
            ax[j].spines['left'].set_visible(False)
            ax[j].set_yticks([])


            kwargs = dict(transform=ax[j].transAxes, color='k', clip_on=False)
            ax[j].plot((1-d,1+d), (-d,+d), **kwargs)
            ax[j].plot((1-d,1+d),(1-d,1+d), **kwargs)
            kwargs.update(transform=ax[j].transAxes)  # switch to the bottom axes
            ax[j].plot((-d,+d), (1-d,1+d), **kwargs)
            ax[j].plot((-d,+d), (-d,+d), **kwargs)

        elif j == 0:
            ax[j].spines['right'].set_visible(False)
            kwargs = dict(transform=ax[j].transAxes, color='k', clip_on=False)
            ax[j].plot((1-d,1+d), (-d,+d), **kwargs)
            ax[j].plot((1-d,1+d),(1-d,1+d), **kwargs)

        else:
            ax[j].spines['left'].set_visible(False)
            ax[j].set_yticks([])
            kwargs.update(transform=ax[j].transAxes)  # switch to the bottom axes
            ax[j].plot((-d,+d), (1-d,1+d), **kwargs)
            ax[j].plot((-d,+d), (-d,+d), **kwargs)

        ax[j].set_rasterized(True)

    return fig, ax


def make_tworow():
    """
    Creates a broken axis plot for one visit.
    """
    fig, axes = plt.subplots(ncols=5, nrows=2,
                             figsize=(20,10))

    ax = axes.reshape(-1)

    d = 0.025

    for j in range(len(ax)):

        if j!=0 and j !=4 and j!=5 and j !=9:
            ax[j].spines['right'].set_visible(False)
            ax[j].spines['left'].set_visible(False)
            ax[j].set_yticks([])


            kwargs = dict(transform=ax[j].transAxes, color='k', clip_on=False)
            ax[j].plot((1-d,1+d), (-d,+d), **kwargs)
            ax[j].plot((1-d,1+d),(1-d,1+d), **kwargs)
            kwargs.update(transform=ax[j].transAxes)

            ax[j].plot((-d,+d), (1-d,1+d), **kwargs)
            ax[j].plot((-d,+d), (-d,+d), **kwargs)

        elif j == 0 or j == 5:
            ax[j].spines['right'].set_visible(False)
            kwargs = dict(transform=ax[j].transAxes, color='k', clip_on=False)
            ax[j].plot((1-d,1+d), (-d,+d), **kwargs)
            ax[j].plot((1-d,1+d),(1-d,1+d), **kwargs)

        else:
            ax[j].spines['left'].set_visible(False)
            ax[j].set_yticks([])
            kwargs.update(transform=ax[j].transAxes)

            ax[j].plot((-d,+d), (1-d,1+d), **kwargs)
            ax[j].plot((-d,+d), (-d,+d), **kwargs)

        ax[j].set_rasterized(True)

    return fig, ax

def plot_combined_lines(table, lines, factor=1e14):
    """
    Creates an n x 3 grid of subplots comparing the combined line profiles,
    the difference between in- and out-of transit observations, and the ratio
    between in- and out-of transit observations (following the figures of Linsky
    et al. 2010).

    Parameters
    ----------
    table : astropy.table.Table
       Table outputs from `TransitsWithCos.combine_lines()`.
    lines : np.ndarray, list
       List of which lines were used in the analysis. This will be used as the
       subplot titles as well.
    """
    fig, axes = plt.subplots(nrows=3, ncols=len(lines)+1, figsize=(18,10),
                             sharex=True)
    axes = axes.reshape(-1)
    fig.set_facecolor('w')

    color = ['k', 'r']
    key = ['it', 'oot']

    for i in range(len(lines)):
        axes[i].set_title(lines[i])
        for j in range(2):
            axes[i].plot(table['velocity'],
                         table['line{0:02d}_{1}_flux'.format(i, key[j])]*factor,
                         color=color[j])
        axes[i+len(lines)+1].plot(table['velocity'],
                                (table['line{0:02d}_it_flux'.format(i)] -
                                    table['line{0:02d}_oot_flux'.format(i)])*factor,
                                color='k')
        axes[i+len(lines)*2+2].plot(table['velocity'],
                                  (table['line{0:02d}_it_flux'.format(i)] /
                                     table['line{0:02d}_oot_flux'.format(i)]),
                                  color='k')

        if i == 0:
            summed_it = table['line{0:02d}_it_flux'.format(i)]
            summed_oot = table['line{0:02d}_oot_flux'.format(i)]
        else:
            summed_it += table['line{0:02d}_it_flux'.format(i)]
            summed_oot += table['line{0:02d}_it_flux'.format(i)]

    axes[len(lines)].plot(table['velocity'], summed_it*factor, c=color[0])
    axes[len(lines)].plot(table['velocity'], summed_oot*factor, c=color[1])

    axes[len(lines)*2+1].plot(table['velocity'], (summed_it - summed_oot)*factor, 'k')
    axes[len(lines)*3+2].plot(table['velocity'], summed_it / summed_oot, 'k')

    axes[len(lines)].set_title('Combined')

    axes[0].set_ylabel('Flux Density\n[10$^{-14}$ erg s$^{-1}$ cm$^{-1} \AA^{-1}$]')
    axes[len(lines)+1].set_ylabel('Difference\n(black-red)')
    axes[(len(lines)+1)*2].set_ylabel('Ratio\n(black/red)')

    for i in np.arange(0,len(lines)+1,1)+len(lines)+1:
        axes[i].axhline(0, color='darkorange')
        axes[i+3].axhline(1, color='darkorange')
        axes[i+3].set_ylim(-1,3)

    axes[-2].set_xlabel(r'Velocity [km s$^{-1}$]')
    return fig
