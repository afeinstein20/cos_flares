import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ['plot_hdl_dem', 'plot_binned_spectrum',
           'compare_with_data']

def plot_hdl_dem(path, ax1, c='k', label='', ax2=None, alpha=0.5):
    """
    Plots Diamond-Lowe+2021 DEM model and indvidual points.

    Parameters
    ----------
    path : str
       Where the pickle output files are stored.
    ax1 : matplotlib.axes._subplots.AxesSubplot
       Subplot for the quiescent DEM outputs.
    c : str
       Color to plot the data with. Default is black.
    label : str
       Label for the data.
    ax2 : matplotlib.axes._subplots.AxesSubplot, optional
       A secondary axis to plot the flare DEM outputs. Default
       is None.

    Returns
    -------
    None
    """
    quies = pd.read_pickle(os.path.join(path, r'aumic_q_dem_models.pkl'))
    flare = pd.read_pickle(os.path.join(path, r'aumic_f_dem_models.pkl'))

    for line in list(flare.keys())[:-1]:
        for i, w in enumerate(flare[line]['centers']):
            try:
                if ax2 is not None:
                    ax2.errorbar(np.log10(flare[line]['peakFormTemp'][i]), 
                                 np.log10(flare[line]['avgDEM'][i]), 
                                 fmt='o', color=c)
                ax1.errorbar(np.log10(quies[line]['peakFormTemp'][i]), 
                             np.log10(quies[line]['avgDEM'][i]), 
                             fmt='o', color=c, markeredgecolor='k', ms=7)
                
                if ((np.log10(flare[line]['peakFormTemp'][i])>6) and 
                    (np.log10(flare[line]['avgDEM'][i])>28)):
                    print(line, quies[line]['centers'][i])
                if np.log10(flare[line]['avgDEM'][i])>35:
                    print(line, quies[line]['centers'][i])

            except:
                print(line)
                pass
    
    if ax2 is not None:
        ax2.plot(flare['EUV']['DEMUV']['Chebyshev']['Trange'], 
                 flare['EUV']['DEMUV']['Chebyshev']['Fit'][1], 
                 lw=3, color=c, zorder=3,
                 alpha=alpha)
    
    ax1.plot(quies['EUV']['DEMUV']['Chebyshev']['Trange'], 
             quies['EUV']['DEMUV']['Chebyshev']['Fit'][1], 
             lw=3, color=c,  zorder=3, label=label,
             alpha=alpha)
    
    return


def plot_binned_spectrum(w, f, e, distance, ax, label='',
                         bins=np.arange(0,4000,1), c='k'):
    """
    Plots the binned DEM spectrum to comapre to data.

    Parameters
    ----------
    w : np.array
       Wavelength array.
    f : np.array
       Flux density array.
    conversion : float
       Value to convert flux density to flux. Should be consistent
       with the units of flux density arrary.
    ax : matplotlib.axes._subplots.AxesSubplot
       Axis to plot the data.
    label : str, optional
       Label for the data. Default is None.
    bins : np.array, optional
       The bins to sum the flux density over.
       Default is [0,4000] in 1 AA binsizes.
    c : str, optional
       Color to plot in. Default is black.
    
    Returns
    -------
    None
    """
    flux = np.array([])
    err  = np.array([])
    centers = np.array([])
    for i in range(len(bins)-1):
        if bins[i+1]-bins[i] < 30:
            q = np.where( (w>bins[i]) & (w<=bins[i+1]))[0]

            if len(q)>0:
                flux = np.append(flux, np.trapz(f[q], w[q]))
                err  = np.append(err , np.sqrt(np.nansum(e[q]**2))/(len(q)-1))
                centers = np.append(centers, (bins[i]+bins[i+1])/2.0)
    # Conversion from flux density to flux
    flux = flux*4*np.pi*distance**2
    q = flux > 0
    # plot errorbar function
    ax.errorbar(centers[q], 
                flux[q], yerr=err[q]*4*np.pi*distance**2,
                marker='o', c=c, ms=10, label=label, linestyle='')
    return


def compare_with_data(dem_wave, dem_flux, dem_err,
                      data_wave, data_flux, data_err,
                      lines,
                      distance, ax, c='k', label='', binsize=None):
    """
    Compares the line flux from the DEM to data.

    Parameters
    ----------
    dem_wave : np.array
       Array of DEM wavelengths.
    dem_flux : np.array 
       Array of DEM fluxes.
    data_wave : np.array 
       Array of observed wavelengths.
    data_flux : np.array 
       Array of observed fluxes.
    lines : np.array
       Array of centers of lines to evaluate over.
    distance : float
       Distance to star. Used to convert flux density to flux.
    ax : matplotlib.axes._subplots.AxesSubplot 
       Axis to plot the data on.
    c : str, optional
       Color to plot the data with. Default is black.
    label : str, optional
       Label for the data. Default is None.
    binsize : np.array, optional
       Wavelength width around each line to sum over. Should
       be of length equal to lines. Default is width of 1AA.

    Returns
    -------
    None
    """
    if binsize is None:
        binsize = np.full(len(lines), 1)

    for i in range(len(lines)):
        dem_q = ( (dem_wave > lines[i]-binsize[i]) &
                  (dem_wave < lines[i]+binsize[i]) )

        data_q = ( (data_wave > lines[i]-binsize[i]) &
                   (data_wave < lines[i]+binsize[i]) )

        flux_dem  = np.trapz(dem_flux[dem_q], dem_wave[dem_q]) * 4*np.pi*distance**2
        err_dem   = np.trapz(dem_err[dem_q], dem_wave[dem_q])  * 4*np.pi*distance**2

        flux_data = np.trapz(data_flux[data_q], data_wave[data_q]) * 4*np.pi*distance**2
        err_data  = np.trapz(data_err[data_q], data_wave[data_q]) * 4*np.pi*distance**2

        ax.errorbar(np.log10(flux_data), 
                    np.log10(flux_dem), 
                    #xerr=np.abs(np.log10(flux_data) - np.log10(err_data)),
                    #yerr=np.abs(np.log10(flux_dem) - np.log10(err_dem)),
                    marker='o', c=c, ms=8, linestyle='')

    return
