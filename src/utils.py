import numpy as np
from lmfit.models import Model
from astropy.table import Table
from astropy import units, constants

__all__ = ['load_data', 'load_binned_data', 'load_table', 
           'blackbody', 'load_inferno', 'build_lmfit']


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

def gaussian(x, mu, std, f, lsf):
    exp = -0.5 * (x-mu)**2 / std**2
    denom = std * np.sqrt(np.pi * 2.0)
    g = f / denom * np.exp(exp)
    return np.convolve(lsf, g, 'same')

def build_lmfit(x, lsf, params, std=0):
    """
    Builds a gaussian model with lmfit parameters table.

    Parameters
    ----------
    x : np.array
    params : lmfit.parameter.Parameters
    std : int, optional
       Key to build a +/- 1 std model. Default is 0 (returns
       best_fit). Other options = +1 (1 std model) or -1 (-1
       std model).

    Returns
    -------
    gmodel
    """
    nparams = 3
    ngauss = int(len(params)/nparams)


    for i in range(ngauss):
        if i == 0:
            gmodel = Model(gaussian, prefix='g{}_'.format(i))
        else:
            gmodel += Model(gaussian, prefix='g{}_'.format(i))

    pars = gmodel.make_params()

    for p in list(params.keys()):
        if std > -2:
            pars[p].set(value=params[p].value + params[p].stderr * std)
        else:
            pars[p].set(value=params[p].value)

    init = gmodel.eval(pars, x=x, lsf=lsf)
    return init
