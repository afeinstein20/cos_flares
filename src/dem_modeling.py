"""
This DEM modeling follows the methods of Duvvuri et al. (2021).
The bulk of the code was written by Hannah Diamond-Lowe as part of 
    Diamond-Lowe et al. (2021).
Documentation and restructuring was done by Adina Feinstein.
"""

import copy
import pickle
import numpy as np
import scipy as sci
from tqdm import tqdm
import matplotlib as mpl
from astropy import units
import ChiantiPy.core as ch
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.constants as const
from dynesty import utils as dyfunc
from dynesty.plotting import _quantile
from scipy.interpolate import interp1d
from dynesty import DynamicNestedSampler

from lmfit.models import Model
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import line_flux, equivalent_width
from astropy.nddata import StdDevUncertainty, NDUncertainty

from utils import gaussian


__all__ = ['init_dict', 'setup_linelist', 'ChiantiSetup', 'DEMModeling']


def init_dict(ions):
    """
    Populates the linelist dictionary with subdictionaries
    and lists as appropriate.

    Parameters
    ----------
    ions : list, np.ndarray
       List of ion keys to use.
    
    Returns
    -------
    linelist : dictionary
       The beginning of the linelist dictionary.
    subkeys : list
       The keys for each ion dictionary.
    """
    subkeys = ['centers', 'lineX', 'lineY', 'lineYerr',
               'lineFit', 'amp', 'ampErr', 'fittedCenter',
               'sig', 'sigErr', 'Fline', 'Ffit', 'FlineErr', 
               'FlineErr', 'FfitErr', 'EW', 'log10SFline',
               'log10SFlineErr']

    all_keys = np.zeros(len(ions), dtype='U32')
    for i in range(len(ions)):
        if ' ' in ions[i]:
            all_keys[i] = ions[i].replace(' ', '') # Removes spaces if necessary
        else:
            all_keys[i] = ions[i]
    all_keys = np.unique(all_keys) # Populates for unique ions only

    linelist = {}

    for i in range(len(all_keys)):
        linelist[all_keys[i]] = {}
    
        for s in range(len(subkeys)):
            linelist[all_keys[i]][subkeys[s]] = []

    return linelist, subkeys


def setup_linelist(wavelength, flux, flux_err, line_table,
                   distance, distanceErr, radius, radiusErr, scaling=1e-14,
                   flux_units=units.erg/units.s/units.cm**2/units.AA,
                   xray=False):
    """
    Creates the dictionary format needed to run the DEM modeling.

    Parameters
    ----------
    wavelength : np.array
       1D wavelength array.
    flux : np.array
       1D flux array.
    flux_err : np.array
       1D flux error array.
    line_table : astropy.table.Table
       Table with emission lines to evaluate over. The table
       should include the line name and center wavelength at 
       a minimum. Could also include `vmin` and `vmax`, or 
       the range to include in each feature.
    distance : float, astropy.units.Unit
       Distance to the star. Given in cm or with astropy units.
    distanceErr : float, astropy.units.Unit
       Error on the distance. Given in cm or with astropy units.   
    radius : float, astropy.units.Unit
       Radius of the star. Given in cm or with astropy units.
    radiusErr : float, astropy.units.Unit
       Error on the radius. Given in cm or with astropy units.
    scaling : float, optional
       Used to scale the spectrum for line-fitting purposes.
       Default is 1e-14.
    flux_units : astropy.units.Unit, optional
       The flux units of the spectrum. Default is erg/s/cm^2/AA.
    xray : bool, optional
       A key to switch to X-ray specific lines or not. Default
       is False.
    """
    if xray==False:
        line_table = line_table[line_table['X-ray'] == 0]
    else:
        line_table = line_table[line_table['X-ray'] == 1]

    linelist, subkeys = init_dict(line_table['Ion'])
    surface_scaling = ( (distance/radius)**2 ).value

    for i in range(len(line_table)):

        if line_table['quality'][i] == 0: # quality control on lines
            
            if ' ' in line_table['Ion'][i]:
                main_key = line_table['Ion'][i].replace(' ', '')
            else:
                main_key = line_table['Ion'][i]

            # gets the appropriate wavelength region
            q = ( (wavelength >= line_table['wmin'][i]) &
                  (wavelength <= line_table['wmax'][i]) )

            # Fits a Gaussian profile to the line
            gmodel = Model(gaussian, prefix='g_')
            pars = gmodel.make_params()
            pars['g_mu'].set(value=line_table['wave_obs'][i], 
                             min=wavelength[q][0],
                             max=wavelength[q][-1])
            pars['g_std'].set(value=0.1, min=0.01, max=20)
            pars['g_f'].set(value=0.5, min=0.01, max=40)
            init = gmodel.eval(pars, x=wavelength[q])
            out  = gmodel.fit(flux[q]/scaling, 
                              pars,
                              x=wavelength[q])
            mini = out.minimize(max_nfev=2000)

            # Gets errors on the amplitude
            try:
                upp = gaussian(wavelength[q], mini.params['g_mu'].value, mini.params['g_std'].value,
                               mini.params['g_f'].value+mini.params['g_f'].stderr)
                low = gaussian(wavelength[q], mini.params['g_mu'].value, mini.params['g_std'].value,
                               mini.params['g_f'].value-mini.params['g_f'].stderr)
                amp_std = np.nanmedian([np.nanmax(upp)-np.nanmax(out.best_fit),
                                        np.nanmax(out.best_fit)-np.nanmax(low)])
            except:
                print("Couldn't get amplitude error for: ", line_table['Ion'][i], " at ", line_table['wave_obs'][i])
                amp_std = 0.0

            # Sets up Spectrum1D objects for measuring line fluxes
            #    Spectrum1D object for the data
            s1d = Spectrum1D(spectral_axis=wavelength[q]*units.AA, 
                             flux=flux[q]*flux_units,
                             uncertainty=StdDevUncertainty(flux_err[q]*flux_units))
            #    Spectrum1D object for the best fit
            s2d = Spectrum1D(spectral_axis=wavelength[q]*units.AA, 
                             flux=out.best_fit*scaling*flux_units,
                             uncertainty=StdDevUncertainty(flux_err[q]*flux_units))

            if main_key in list(linelist.keys()): # double checks the line is in the dict
                linelist[main_key]['centers'].append(line_table['wave_obs'][i])          # center wavelength
                
                linelist[main_key]['lineX'].append(wavelength[q])                        # wavelength array
                linelist[main_key]['lineY'].append(flux[q])                              # flux array
                linelist[main_key]['lineYerr'].append(flux_err[q])                       # flux error array
                
                linelist[main_key]['lineFit'].append(out.best_fit*scaling)               # best-fit Gaussian
                linelist[main_key]['amp'].append(np.nanmax(out.best_fit)*scaling)        # amplitude of the fit
                linelist[main_key]['ampErr'].append(amp_std*scaling)                     # error on the amp fit
                linelist[main_key]['fittedCenter'].append(mini.params['g_mu'].value)     # fitted line center
                linelist[main_key]['sig'].append(mini.params['g_std'].value)             # std of the Gaussian
                linelist[main_key]['sigErr'].append(mini.params['g_std'].stderr)         # error on the std
                
                linelist[main_key]['Fline'].append(line_flux(s1d).value/(4*np.pi)**2)    # line flux of data
                linelist[main_key]['Ffit'].append(line_flux(s2d).value/(4*np.pi)**2)     # line flux of model
                linelist[main_key]['FlineErr'].append(line_flux(s1d).uncertainty.value)  # error on line flux of data
                linelist[main_key]['FfitErr'].append(line_flux(s2d).uncertainty.value)   # error on line flux of model
                
                linelist[main_key]['EW'].append(equivalent_width(s1d).value)             # line equivalent width
                
                sf = np.log10(surface_scaling*line_flux(s1d).value)                      # surface flux of line
                linelist[main_key]['log10SFline'].append(sf + 0.0)                       #   surface flux is log10(sf)
                
                sfErr = np.sqrt((line_flux(s1d).uncertainty/line_flux(s1d))**2 + 
                                (radiusErr/radius)**2 + 
                                (distanceErr/distance)**2)                               # error on surface flux
                linelist[main_key]['log10SFlineErr'].append(sfErr.value/np.log(10))      #    error is log10(sfErr)

            
    # just some necessary cleaning/reformatting for later
    for ak in list(linelist.keys()):
        for s in subkeys:
            linelist[ak][s] = np.array(linelist[ak][s])

    return linelist
    


class ChiantiSetup(object):
    """
    Implementing the DEM fitting routing of Duvvuri et al. (2021).
    Original routines here were written by Hannah Diamond-Lowe and
       were implemented in Diamond-Lowe et al. (2022).
    """

    def __init__(self, linelist, logT_range, wave_range, 
                 eDensity=1e8,
                 abundance='sun_coronal_2012_schmelz', setup=False):
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

        if setup:
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
        tempRange = np.logspace(self.logT_range[0], self.logT_range[1], 10)
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

    def __init__(self, linelist, G_T, logT_range):
        """
        Initializes the class.

        Parameters
        ----------
        linelist : dictionary
        G_T : dictionary
           Dictionary of emissivity functions for each ion.
        logT_range : list
           An array or list of beginning and ending log(temperatures)
           to run the Chianti function over. Should be a list/array
           of length = 2. Should be the same as what was input in
           `ChiantiSetup`.
        """
        self.linelist = linelist
        self.G_T = G_T
        self.logT_range = logT_range


    def create_DEM(self, specified_lines=[], resample=False, 
                   nsamples=5000, quick_plot=False,
                   results_filename='DEMresults',
                   grid_filename='DEMgrid'):
        """
        Calculates the differential emission measurements (DEMs). This function
        requires that the linelist dictionary has all transitions separated.

        Parameters
        ----------
        specified_lines : list, optional
           Default is None.
        resample : bool, optional
           Default is False.
        nsamples : int, optional
           Number of resamples to create. Default is 5000.
        quick_plot : bool, optional
           A quick look at the DEM fitting. Default is False.
        results_filename : str, optional
           The filename prefix for the `.pkl` file with the
           DEM fitted results. Default is `DEMresults`.
        grid_filename : str, optional
           The filename prefix for the `.pkl` file with the
           DEMgrid outputs. Default is `DEMgrid`.

        Attributes
        ----------
        T : np.array
           Temperature array for the DEM
        DEM : np.array
           Median DEM fit.
        DEM_low : np.array
           Lower estimate to the DEM fit.
        DEM_upp : np.array
           Upper estimate to the DEM fit.
        """
        
        lines = self.linelist

        if len(specified_lines) == 0:
            specified_lines = list(lines.keys())


        peakFormTemps, avgDEMs, avgDEMErrs, weights = [], [], [], []
        for line in lines.keys():

            if line not in specified_lines:
                lines[line]['DEMEstimate'] = False
                continue
            else: 
                lines[line]['DEMEstimate'] = True

            I_ul = 10**lines[line]['log10SFline'] / np.pi  #[erg/cm^2/s/sr]  surface flux
            I_ulErr = lines[line]['log10SFlineErr'] * I_ul * np.log(10) / np.pi   #[erg/cm^2/s/sr]  surface flux error

            lines[line]['I_ul'] = I_ul
            lines[line]['I_ulErr'] = I_ulErr
            lines[line]['peakFormTemp'] = []
            lines[line]['integratedGT'] = []
            lines[line]['avgDEM'] = []
            lines[line]['avgDEMErr'] = []

            for i, w in enumerate(lines[line]['centers']):
                try:
                    peakInd = np.argwhere(self.G_T[lines[line]['CHIname']][w] == self.G_T[lines[line]['CHIname']][w].max())[0]
                    lines[line]['peakFormTemp'].append(self.G_T['lineTemp'][peakInd])
                    
                    integratedGT = sci.integrate.simps(self.G_T[lines[line]['CHIname']][w], self.G_T['lineTemp'])
                    lines[line]['integratedGT'].append(integratedGT)
                    
                    avgDEM = lines[line]['I_ul'][i]/integratedGT
                    avgDEMErr = lines[line]['I_ulErr'][i]/integratedGT
                    lines[line]['avgDEM'].append(avgDEM)
                    lines[line]['avgDEMErr'].append(avgDEMErr)
                    
                    peakFormTemps.append(np.log10(lines[line]['peakFormTemp'][i]))
                    avgDEMs.append(np.log10(lines[line]['avgDEM'][i]))
                    avgDEMErrs.append(avgDEMErr / (avgDEM * np.log(10)))
                    
                    weights.append(10**(lines[line]['log10SFlineErr'][i]))
                    
                except:
                    print(lines[line]['CHIname'], w)
                #print(line, i, w, 10**(lines[line]['log10SFlineErr'][i]))


        peakFormTemps = np.hstack(peakFormTemps)
        avgDEMs = np.hstack(avgDEMs)
        avgDEMErrs = np.hstack(avgDEMErrs)
        
        DEMarray = np.array([peakFormTemps, avgDEMs, weights]).T
        DEMarray = DEMarray[DEMarray[:,0].argsort()].T

        if resample:
            results = self.fit_DEM(peakFormTemps, avgDEMs, avgDEMErrs)
            pickle.dump(results, open('{}.pkl'.format(results_filename), 'wb'))
        else:
            try: 
                results = pickle.load(open('{}.pkl'.format(results_filename), 'rb'))
            except FileNotFoundError :
                results = self.fit_DEM(peakFormTemps, avgDEMs, avgDEMErrs)
                pickle.dump(results, open('{}.pkl'.format(results_filename), 'wb'))

        samples = results.samples
        quantiles = [dyfunc.quantile(samps, [.16, .5, .84], weights=np.exp(results['logwt']-results['logwt'][-1])) for samps in samples.T]
        params = np.array(list(map(lambda v: (v[0], v[1], v[2]), quantiles)))
        values = params[:,1]
        values_lo = params[:,0]
        values_hi = params[:,2]
        
        tempRangeLines = self.G_T['lineTemp']
        T = np.log10(tempRangeLines)

        ## PRETTY SURE RIGHT NOW THIS DOESN'T HANDLE NO RESAMPLING ##
        if resample:
            DEMgrid = self.dem_resample(nsamples, results, T, samples)
            pickle.dump(DEMgrid, open('{}.pkl'.format(grid_filename), 'wb'))
        else:
            try:
                DEMgrid = pickle.load(open('{}.pkl'.format(grid_filename), 'rb'))
            except FileNotFoundError:
                DEMgrid = self.dem_resample(nsamples, results, T, samples)
                pickle.dump(DEMgrid, open('{}.pkl'.format(grid_filename), 'wb'))


        DEM = np.percentile(DEMgrid, 50, axis=0)
        DEMlo = np.percentile(DEMgrid, 16, axis=0)
        DEMhi = np.percentile(DEMgrid, 84, axis=0)

        if quick_plot:
            for dg in DEMgrid:
                plt.plot(T, dg, color='k', lw=1, alpha=0.01)
            plt.plot(T, DEM, color='darkred', alpha=0.8, lw=3)
            plt.plot(T, DEMlo, color='deepskyblue', alpha=0.8, lw=3)
            plt.plot(T, DEMhi, color='deepskyblue', alpha=0.8, lw=3)
            
            plt.xlabel('Temperature [K]')
            plt.ylabel('DEM [cm$^{-5}$ K$^{-1}$]')
            plt.ylim(18,27)
            plt.xlim(4,8)
            plt.show()

        if 'EUV' not in lines.keys(): 
            lines['EUV'] = {}
        if 'DEMUV' not in lines['EUV'].keys(): 
            lines['EUV']['DEMUV'] = {}

        lines['EUV']['DEMUV']['Chebyshev'] = {}
        lines['EUV']['DEMUV']['Chebyshev']['Trange'] = np.log10(tempRangeLines)
        lines['EUV']['DEMUV']['Chebyshev']['Fit'] = [DEMlo, DEM, DEMhi]

        self.linelist = lines
        self.T = T
        self.DEM = DEM
        self.DEM_low = DEMlo
        self.DEM_upp = DEMhi
        
        

    def dem_resample(self, nsamples, results, T, samples):
        """
        Creates the DEM grid output with resampling.

        Parameters
        ----------
        nsamples : int, optional
           The number of samples for the resampling
           routine.
        """
        wgts = np.exp(results['logwt']-results['logwt'][-1])
        DEMgrid = np.zeros((nsamples, len(T)))

        for i, c in enumerate(np.random.choice(len(wgts), nsamples, replace=False, p=wgts/np.sum(wgts))):
            params = samples.T[:, c]
            c_n = params[0:5]
            cheb = np.polynomial.chebyshev.Chebyshev(c_n, domain=[self.logT_range[0], 
                                                                  self.logT_range[1]])
            modelDEM = np.array(cheb(T))
            DEMgrid[i] += modelDEM

        return DEMgrid


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
                        ionTrange = self.G_T['ionTemp']
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
