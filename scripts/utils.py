import numpy as np
from astropy import units
from astropy.table import Table

__all__ = ['load_data', 'load_binned_data', 'load_table', 'load_lines', 'load_inferno',
           'to_velocity']


def load_data(fname='/Users/arcticfox/Documents/AUMic/data.npy'):
    """ Returns wavelength, flux, orbits.
    """
    dat = np.load(fname, allow_pickle=True)
    return dat[0] + 0.0, dat[1] + 0.0, dat[2] + 0.0

def load_binned_data(fname='/Users/arcticfox/Documents/AUMic/binned_data.npy'):
    dat = np.load(fname, allow_pickle=True)
    return dat[0]+0.0, dat[1]+0.0, dat[2]+0.0

def load_table(fname='/Users/arcticfox/Documents/AUMic/ew.tab'):
    """ Returns table with times and equivalent widths.
    """
    tab = Table.read(fname, format='ascii')
    return tab

def to_velocity(wave, mid=None):
    if mid == None:
        mid = int(len(wave)/2)
    else:
        mid = np.where(wave>=mid)[0][0]
    lambda0 = wave[mid] + 0.0
    rv_m_s = ((wave - lambda0)/lambda0 * 3e8)*units.m/units.s
    rv_km_s = rv_m_s.to(units.km/units.s)
    return rv_km_s, mid

def load_lines():
    """ Returns dictionary of lines.
    Array = central wavelength (AA), and integrated
    region (km / s).
    """
    lines = {'CIII':[1175.59, -240, 230],
             'SiII':[1264.738, -50, 100],
             'SiIII':[1294.5480, -100, 100], ## NEW
             'SiIV':[1393.7570, -100, 100],           
             'NV_1':[1238.831, -80, 80],
             'NV_2':[1242.804, -70, 70],
             'CII_1':[1334.532, -50, 70], # edited
             'CII_2':[1335.708, -80, 60], # edited
             'OI':[1302.1689, -100, 100], ## NEW
             'OI_1':[1304.8580, -100,100], ## NEW
             'OI_2':[1306.0291, -100,100], ## NEW
             'SiIII_2':[1194.0490, -100,100], ## NEW
             'SiII_2':[1194.5000, -100, 100], ## NEW
             'NI':[1200.2260,-100,100], ## NEW
             'SiIII_3':[1200.9611, -100,100], ## NEW
             'SiII_30':[1259.5210, -50,50], ## NEW
             'SiII_31':[1260.4220, -50, 50], ## NEW
             'NeIV':[1277.7045, -50, 50], ## NEW
             'SiIII_3':[1298.8940, -100,100], ## NEW
             'SiII_4':[1309.2760, -50,50], ## NEW
             'FeXIX':[1328.9061, -50,50], ## NEW
             'OI_30':[1355.5980, -100,100], ## NEW
             'OI_31':[1358.5120, -50,50]
             }
    return lines

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
