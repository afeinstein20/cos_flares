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
#        self.emissivity_functions()


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
                print(line)
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
                #if w in self.G_T[ion].keys():
                #    print('already have {} in {} G(T)'.format(w, ion))
                #    continue
                dist = np.abs(np.asarray(ChiantiIon.Intensity['wvl']) - w)
                idx = np.argmin(dist)
                try:
                    GofT = ChiantiIon.Intensity['intensity'][:,idx]
                    self.G_T[ion][w] = GofT
                except(AttributeError): continue
                
        return
