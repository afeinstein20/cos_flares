import numpy as np
from scipy.special import erf
from lmfit.models import Model
from astropy.table import Table
from astropy import units, constants

__all__ = ['load_data', 'load_binned_data', 'load_table', 
           'blackbody', 'load_inferno', 'build_lmfit',
           'flare_model', 'gaussian', 'skewed_gaussian',
           'multi_peaks', 'convolved_model']


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

def gaussian(x, mu, std, f):#, lsf):
    """ A gaussian model. """
    exp = -0.5 * (x-mu)**2 / std**2
    denom = std * np.sqrt(np.pi * 2.0)
    g = f / denom * np.exp(exp)
    return g
#    return np.convolve(lsf, g, 'same')

def skewed_gaussian(x, eta, omega, alpha, offset, normalization):
    """ A skewed gaussian model.  """
    # alpha = skew (alpha > 0, skewed right; alpha < 0, skewed left)
    # omega = scale
    # eta = mean
    t = alpha * (x - eta) / omega
    Psi = 0.5 * (1 + erf(t / np.sqrt(2)))
    psi = 2.0 / (omega * np.sqrt(2 * np.pi)) * np.exp(- (x-eta)**2 / (2.0 * omega**2))
    return (psi * Psi)/normalization + offset

def convolved_model(x, eta, omega, alpha, normalization,
                    amp, t0, rise, decay, offset):
    """ Fits the flares with a convolution of the skewed Gaussian and
        traditional white-light Davenport flare model. """
    m1 = skewed_gaussian(x, eta, omega, alpha, 0, normalization)
    m2 = flare_model(x, amp, t0, rise, decay, offset_g=0, offset_e=0)

    return np.convolve(m1, m2, mode='same') + offset


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


def flare_model(x, amp, t0, rise, decay, offset_g, offset_e):
        """           
        Models the flare with an array of times (for double-peaked
        events). Uses the standard model of a Gaussian rise and   
        exponential decay.

        Parameters
        ----------
        x : np.array
           Time array to fit the data to.
        amp : float 
           Amplitude of flare.
        t0 : float  
           T0 of flare.           
        rise : float   
           Gaussian rise factor for flare.
        decay : float                    
           Exponential decay factor for flare.

        Returns
        -------
        model : np.array
           Flare model. 
        """
        gr = np.zeros(len(x))
        ed = np.zeros(len(x))
        flux = np.zeros(len(x))

        g = amp * np.exp( -(x[x<t0] - t0)**2.0 / (2.0*rise**2.0) )
        g += flux[x<t0]
        gr[x<t0] += g
        gr[x<t0] += offset_g

        e = amp * np.exp( -(x[x>=t0] - t0) / decay ) + flux[x>=t0]
        ed[x>=t0] += e
        ed[x>=t0] += offset_e

        return gr + ed

def build_lmfit_flare(x, params, std=0):
    """
    Builds the flare model using the output parameters from
    lmfit.
    
    Parameters
    ----------
    x : np.array
       X-axis (probably time).
    params : lmfit.parameter.Parameters
    nflares : 
    std : int, optional 
       Key to build a +/- 1 std model. Default is 0 (returns
       best_fit). Other options = +1 (1 std model) or -1 (-1  
       std model).

    Returns
    -------
    fmodel
    """
    nparams = 5
    nflares = int(len(params)/nparams)

    for i in range(ngauss):
        if i == 0:
            fmodel = Model(flare_model, prefix='f{0:02d}_'.format(i))
        else:
            fmodel += Model(flare_model, prefix='f{0:02d}_'.format(i))

    pars = fmodel.make_params()

    for p in list(params.keys()):
        if std > -2:
            pars[p].set(value=params[p].value + params[p].stderr * std)
        else:
            pars[p].set(value=params[p].value)

    init = fmodel.eval(pars, x=x)
    return init

def multi_peaks(ttest, test):
    """
    Used to identify multiple peaks in the same flare.

    test = time
    ttest = flux
    """

    p1 = np.argmax(test)
    
    try:
        arg = np.where((ttest.value>ttest[p1].value+100))[0]
        p2 = np.argmax(test[arg])
    except:
        arg=np.arange(0,len(ttest),1,dtype=int)
        p2=0
        
    try:
        arg3 = np.where((ttest.value>ttest[arg][p2].value+100))[0]
        p3 = np.argmax(test[arg3])

    except:
        arg3=np.arange(0,len(ttest),1,dtype=int)
        p3=0
        
    t0 = np.array([ttest[p1].value, ttest[arg][p2].value, ttest[arg3][p3].value])
    amp = np.array([test[p1], test[arg][p2], test[arg3][p3]])#/1e-13

    return t0, amp
