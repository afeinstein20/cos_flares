import copy
import pickle
import numpy as np
import scipy as sci
from tqdm import tqdm
import matplotlib as mpl
import astropy.units as u
import ChiantiPy.core as ch
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.constants as const
from dynesty import utils as dyfunc
from dynesty.plotting import _quantile
from scipy.interpolate import interp1d
from dynesty import DynamicNestedSampler


__all__ = ['ChiantiSetup', 'DEMModeling']


class ChiantiSetup(object):
    """
    Implementing the DEM fitting routing of Duvvuri et al. (2021).
    Original routines here were written by Hannah Diamond-Lowe and
       were implemented in Diamond-Lowe et al. (2022).
    """

    def __init__(self, linelist, logT_range, wave_range, 
                 eDensity=1e8,
                 abundance='sun_coronal_2012_schmelz'):
        """
        Initializes the class.

        Parameters
        ----------
        linelist : dictionary
        logT_range : np.ndarray
           An array or list of beginning and ending log(temperatures)
           to run the Chianti function over. Should be a list/array  
           of length = 2.      
        wave_range : np.ndarray
           An array or list of beginning and ending wavelengths (in
           angstroms) to run the Chianti function over. Should be a
           list/array of length = 2.
        eDensity : float, optional 
           The electron density to evaluate the Chianti lines over.
           Default is 1e8.     
        abundance : str, optional
           The reference file from the Chianti database to evalute
           the library with respect to. Default is `sun_coronal_2012_schmelz`. 
        """
        self.linelist   = linelist
        self.eDensity   = eDensity
        self.logT_range = logT_range
        self.wave_range = wave_range
        self.abundance  = abundance

        print("Setting up Chianti spectrum. This may take a few minutes . . .")
        self.library_setup()


    def library_setup(self):
        """
        Uses the Chianti database to pull all ion species
        available for a given temperature and wavelength range.
        This function may take a few minutes.

        Attributes
        ----------
        bunch : ChiantiPy.core.Spectrum.bunch
        """
        tempRange = np.logspace(self.logT_range[0], self.logT_range[1], 5)
        bunch = ch.bunch(temperature=tempRange, eDensity=self.eDensity,
                         wvlRange=self.wave_range,
                         abundance=self.abundance, 
                         elementList=['al','ar','c','ca','co','cr','fe','h',
                                      'he','k','mg','mn','n','na','ne','ni','o',
                                      'p','s','si','ti','zn'])
        self.bunch = bunch


    def create_chiname(self, ion):
        """
        Takes an ion name with roman numerals and makes it 
        Chianti readable, ex: CIII --> c_3. This will break
        if any elements have 'I', 'V' or 'X' in their names
        or if the ion species has any roman numberals > 'X'.

        Parameters
        ----------
        ion : str

        Return
        ------
        chiname : str
        """
        roman = {'I':1,'V':5,'X':10,'IV':4,'IX':9}

        i,num = 0,0
        element,numerals='',''

        # separates element from numerals
        for char in ion:
            if char in roman:
                numerals+=char
            else:
                element+=char

        # converts roman numberals to digits
        while i < len(numerals):
    
            if i+1<len(numerals) and numerals[i:i+2] in roman:
                num+=roman[numerals[i:i+2]]
                i+=2
            else:
                num+=roman[numerals[i]]
                i+=1

        if num > 0:
            return element.lower() + '_' + str(num)
        else:
            return element#.lower()


    def emissivity_functions(self):
        """
        Retrieves the emissivity functions from Chianti.

        Attributes
        ----------
        G_T : dictionary
           Dictionary of emissivity functions for each of the
           ions given in the initial line list.
        tempRangeLines : np.ndarray
           Array temperature ranges used to create the emissivity
           functions.
        """

        G_T = {}
        tempRangeLines = np.logspace(self.logT_range[0], self.logT_range[1], 100)

        G_T['lineTemp'] = tempRangeLines

        for line in list(self.linelist.keys()):
            if line == 'Xray': continue
            if line == 'EUV': continue

            CHIname = self.create_chiname(line)
            self.linelist[line]['CHIname'] = CHIname

            G_T[CHIname] = {}
            try:
                ChiantiIon = ch.ion(CHIname, 
                                    temperature=G_T['lineTemp'], 
                                    eDensity=self.eDensity, 
                                    abundance=self.abundance)
                
                ChiantiIon.intensity()
                
                for w in self.linelist[line]['centers']:
                    if line == 'CIII':
                        wvlRange = [w-3, w+3]
                    else: 
                        wvlRange = [w-0.1, w+0.1]
                        
                    dist = np.abs(np.asarray(ChiantiIon.Intensity['wvl']) - w)
                    idx = np.argmin(dist)
                    
                    G_T[CHIname][w] = ChiantiIon.Intensity['intensity'][:,idx]
            except:
                print("Couldn't get: ", line)
        G_T['ionTemp'] = tempRangeLines
        self.G_T = G_T


    def get_all_ions(self, top=10):
        """
        
        Parameters
        ----------
        top : int or list, optional
           The number of transition lines for each ion in
           the wavelength range requested. Default is 10.
           If top is a list, it should of length linelist.
        """

        allIons = self.bunch.IonsCalculated

        for i,ion in enumerate(tqdm(allIons)):

            if ion not in self.G_T.keys(): 
                self.G_T[ion] = {}

            ChiantiIon = ch.ion(ion, 
                                temperature=self.G_T['ionTemp'], 
                                eDensity=self.eDensity, 
                                abundance=self.abundance)

        # get the top 10 (?) transition lines for each ion in the part of the spectrum we can't see
        #    can go back and do this for *every* transition if needed            
            if type(top) == np.ndarray or type(top) == list:
                ChiantiIon.intensityList(wvlRange=self.wave_range, top=top[i])
            else:
                ChiantiIon.intensityList(wvlRange=self.wave_range, top=top)

            try:
                # 'intensityTop' comes in order of increasing wavelength, but I want it in decreasing intensity, from highest to lowest
                topArray = np.array([ChiantiIon.Intensity['intensityTop'], 
                                     ChiantiIon.Intensity['wvlTop']]).T
                topArray = topArray[topArray[:,0].argsort()].T
                wvls = topArray[1][::-1] # go in order of highest intensity wavelength to lowest
            except(KeyError): continue
            
            for w in wvls:
                dist = np.abs(np.asarray(ChiantiIon.Intensity['wvl']) - w)
                idx = np.argmin(dist)
                try:
                    GofT = ChiantiIon.Intensity['intensity'][:,idx]
                    self.G_T[ion][w] = GofT
                except(AttributeError): continue
                
        return


class DEMModeling(object):
    """
    Does all the DEM modeling routines.
    """

    def __init__(self, linelist, G_T):
        """
        Initializes the class.

        Parameters
        ----------
        linelist : dictionary
        """
        self.linelist = linelist
        self.G_T = G_T


    def fit_DEM(self, peakFormT, DEM, DEMerr):
        """
        Fits the DEM using a Chebyshev polynomial.
        """
        
        def lnlike(p):
            c_n = p[:5]
            s = 10**(p[-1])
            cheb = np.polynomial.chebyshev.Chebyshev(c_n, domain=[4, 8])
        #ensure that the first derivative at log10(T) = 4 the derivative is negative
            if cheb.deriv(m=1)(4) > 0: 
                return -np.inf
            if cheb.deriv(m=1)(8) > 0: 
                return -np.inf
        #ensure that the base-10 log of the polynomial is positive log10(6) to prevent really small DEMs
            if cheb(6) < 0: 
                return -np.inf
            modelDEM = np.array(cheb(peakFormT))
            
            logl = np.log(1 / np.sqrt(2*np.pi*(DEMerr**2 + (s * modelDEM)**2))) - ((DEM - modelDEM)/(2*np.sqrt(DEMerr**2 + (s * modelDEM)**2)))**2
            return np.sum(logl)

        def ptform(p):
            xcopy = np.array(p)
            c0 = xcopy[0] * 6 + 20            # sample c0, which sets the mean location of DEM, on U[20, 26]
            c1_5 = xcopy[1:5] * 20 + (-10)        # sample c1 through c5 on U[-10, 10] (can increase this)  
            s = np.log10(xcopy[-1]) * 4 + -2     # sample log10(s) on U[-2, 2]               
            xnew = np.hstack([c0, c1_5, s])
            return xnew
        
        ndim = 6
        dsampler = DynamicNestedSampler(lnlike, ptform, ndim)
        dsampler.run_nested(wt_kwargs={'pfrac': 1.0})

        return dsampler.results

    def estimate_EUV_from_DEM(self):
        """
        Estimates the EUV flux from the DEM models.

        Attributes
        ----------
        dem_ions_lo : dictionary
           Lower error to the EUV calculated per ion.
        dem_ions : dictionary
           Median of the EUV calculated per ion.
        dem_ions_hi : dictionary
           Upper error to the EUV calculated per ion.
        """        
        
        DEM_function_Trange = 10**self.linelist['EUV']['DEMUV']['Chebyshev']['Trange']

        for i in range(3):
            localdict = {}
            
            DEM_function = 10**self.linelist['EUV']['DEMUV']['Chebyshev']['Fit'][i]

            for ion in self.G_T.keys():
                if ion=='lineTemp': continue
                if ion=='ionTemp': continue
                for c, center in enumerate(self.G_T[ion].keys()):

                    if self.G_T[ion][center].shape == self.G_T['ionTemp'].shape: 
                        ionTrange = G_T['ionTemp']
                    elif self.G_T[ion][center].shape == self.G_T['lineTemp'].shape:
                        ionTrange = self.G_T['lineTemp']

                    G_T_interp = np.interp(DEM_function_Trange, ionTrange, self.G_T[ion][center])
                    integrand = G_T_interp * DEM_function
                    I_ul = sci.integrate.simps(integrand, DEM_function_Trange)
                    
                    if ion not in localdict.keys():
                        localdict[ion] = {}
                        localdict[ion]['centers'] = []
                        localdict[ion]['log10SFline'] = []
                        localdict[ion]['centers'].append(center)
                        localdict[ion]['log10SFline'].append(np.log10(I_ul*np.pi))
                    else:
                        localdict[ion]['centers'].append(center)
                        localdict[ion]['log10SFline'].append(np.log10(I_ul*np.pi))

            if i==0:
                self.linelist['EUV']['DEMUV']['ions_lo'] = localdict
                self.dem_ions_lo = localdict
            elif i==1:
                self.linelist['EUV']['DEMUV']['ions'] = localdict
                self.dem_ions = localdict
            elif i==2:
                self.linelist['EUV']['DEMUV']['ions_hi'] = localdict
                self.dem_ions_hi = localdict
