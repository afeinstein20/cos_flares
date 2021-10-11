import matplotlib.pyplot as plt

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


