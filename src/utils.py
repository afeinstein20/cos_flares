import numpy as np
from astropy.table import Table
from astropy import units, constants

__all__ = ['load_data', 'load_binned_data', 'load_table', 'blackbody', 'load_inferno']


def load_data(fname='/Users/arcticfox/Documents/AUMic/reduced/data.npy'):
    """ Returns wavelength, flux, orbits.
    """
    dat = np.load(fname, allow_pickle=True)
    return dat[0] + 0.0, dat[1] + 0.0, dat[2] + 0.0

def load_binned_data(fname='/Users/arcticfox/Documents/AUMic/reduced/binned_data.npy'):
    dat = np.load(fname, allow_pickle=True)
    return dat[0]+0.0, dat[1]+0.0, dat[2]+0.0

def load_table(fname='/Users/arcticfox/Documents/AUMic/reduced/ew.tab'):
    """ Returns table with times and equivalent widths.
    """
    tab = Table.read(fname, format='ascii')
    return tab

def measure_linewidth(wavelength, flux, error, line):
    """ Measures the equivalent width of the line
    """

def load_inferno(n=10, colormap='inferno'):
    """ Returns a discrete colormap with n values.
    """
    from pylab import cm
    import matplotlib
    
    cmap = cm.get_cmap(colormap, n)
    colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colors.append(matplotlib.colors.rgb2hex(rgb))
    colors = np.array(colors)[1:-1]
    return colors

def blackbody(wavelength, T):
    frac = (2 * constants.h * constants.c**2)/wavelength**5
    e = (constants.h*constants.c)/(wavelength*constants.k_B*T)
    return frac * 1/(np.exp(e)-1)

              
