import batman
import numpy as np
from astropy import units
import matplotlib.pyplot as plt

__all__ = ['limb_darkening', 'limb_brightening', 'K', 'E',
           'Theta', 'Pi']

def limb_darkening(t, p):
    """
    Limb darkened transit model
    """
    params = batman.TransitParams()
    params.t0 = p['t0']            #time of inferior conjunction
    params.per = p['per']          #orbital period
    params.rp = p['rp']            #planet radius (in units of stellar radii)
    params.a = p['arstar']         #semi-major axis (in units of stellar radii)
    params.inc = p['inc']          #orbital inclination (in degrees)
    params.ecc = p['ecc']          #eccentricity
    params.w = 92.                 #longitude of periastron (in degrees)
    params.u = p['u']              #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"      #limb darkening model

    m = batman.TransitModel(params, t)  #initializes model
    flux = m.light_curve(params)        #calculates light curve
    return flux

def Theta(x):
    arr = np.zeros(len(x))
    arr[x > 0] = 1.0
    return arr

def K(k):
    """
    Complete Legendre elliptical integral of the first kind using
    Hasting's approximation.
    """
    m1=1.0-k**2
    a0=1.38629436112
    a1=0.09666344259
    a2=0.03590092383
    a3=0.03742563713
    a4=0.01451196212
    b0=0.5
    b1=0.12498593597
    b2=0.06880248576
    b3=0.03328355346
    b4=0.00441787012
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*np.log(m1)
    return ek1 - ek2

def E(k):
    """
    Complete Legendre elliptical integral of the second kind
    using Hasting's approximation.
    """
    m1=1.0-k**2
    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639
    ee1=1.0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*np.log(1.0/m1)
    return ee1 + ee2

def Pi(n, k):
    """
    Complete Legendre elliptical integral of the third kind using
    the algorithm from Bulirsch (1965)
    """
    kc = np.sqrt(1.0 - k**2.0)
    p = n + 1.0
    if np.min(p) < 0.0:
        print("Negative p")
    m0 = 1.0
    c = 1.0
    p = np.sqrt(p)
    d = 1.0 / p
    e = kc
    loop = True
    while loop:
        f = c
        c = d / p + f
        g = e / p
        d = (f * g + d) * 2.0
        p = g + p
        g = m0
        m0 = kc + m0
        if np.max(np.abs(1.0 - kc / g)) > 1e-13:
            kc = 2.0 * np.sqrt(e)
            e = kc * m0
        else:
            loop = False
    return 0.5 * np.pi * (c * m0 + d) / (m0 * (m0 + p))

def limb_brightening(b, p):
    a = np.zeros(len(b))

    indx = np.where(b+p < 1.0)[0]

    if(len(indx) > 0):
        k=np.sqrt(4.0*b[indx]*p/(1.0-(b[indx]-p)**2))

        a[indx]=4.0/np.sqrt(1.0-(b[indx]-p)**2)*(((b[indx]-p)**2-1.0)*E(k) \
                 -(b[indx]**2-p**2)*K(k)+(b[indx]+p)/(b[indx]-p) \
                 *Pi(4.0*b[indx]*p/(b[indx]-p)**2,k))

    indx = np.where((b+p > 1.0) & (b-p < 1.0))[0]

    if(len(indx) > 0):
        k=np.sqrt((1.0-(b[indx]-p)**2)/4.0/b[indx]/p)

        a[indx]=2.0/(b[indx]-p)/np.sqrt(b[indx]*p)*(4.0*b[indx]*p*(p-b[indx])*E(k) \
               +(-b[indx]+2.0*b[indx]**2*p+p-2.0*p**3)*K(k) \
               +(b[indx]+p)*Pi(-1.0+1.0/(b[indx]-p)**2,k))

    lc = 1.0-(4.0*np.pi*(p > b)+a)/4.0/np.pi
    return -(-lc+1)+1
