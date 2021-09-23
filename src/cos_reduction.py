import calcos
import os, sys
import numpy as np
from astropy import units
from astropy.io import fits
from costools import splittag
from astropy.table import Table
from scipy.interpolate import interp1d
from lightkurve.lightcurve import LightCurve

__all__ = ['cosReduce']

class cosReduce(object):
    """
    A class to take your Hubble/COS observations
    and split them via time-tag markers (and do
    other things).
    """
    
    def __init__(self, rootname, input_path):
        """
        Initializes the class to reduce your COS data.

        Parameters
        ----------
        rootname : np.array
           A list or array of the rootnames for the series
           of observations.
        input_path : str
           The path to where the raw data is stored.
        output_path : str
           The path to where the output data will be stored.
           This will create a subdirectory `a` and `b` when
           necessary.

        Attributes
        ----------
        rootname : np.array
        input_path : str
        output_path : str
        """
        self.rootname = rootname
        self.input_path = input_path

    def split_corrtag(self, output_path, increment=30, 
                      starttime=None, endtime=None, time_list=None):
        """
        Splits the Hubble COS observations based on
        time the photons hit the detector. Files will be saved
        as FITS files with the naming convention of 
        `split_[rootname]_[n].fits`.

        Parameters
        ----------
        output_path : str
           Where the splittag data will be stored.
        increment : int, optional
           The length of each time interval in seconds. Default
           is 30s.
        starttime : float, optional
           The time at the beginning of the first interval.
           Default is None.
        endtime : float, optional
           The time at the end of the last interval. Default is
           None.
        time_list : str, optional
           Comma- or blank-separated string of times for intervals
           if you wish the intervals to be uneven. Default is None.
        """
        
        for i in range(len(self.rootname)):
            corrtag_a = os.path.join(self.input_path,
                                     '{}_corrtag_a.fits'.format(self.rootname[i]))
            corrtag_b = os.path.join(self.input_path,
                                     '{}_corrtag_b.fits'.format(self.rootname[i]))
        
            splittag.splittag(corrtag_a, os.path.join(output_path,
                                                      'split_{}'.format(self.rootname[i])),
                              increment=increment, starttime=starttime,
                              endtime=endtime, time_list=time_list)
            
            splittag.splittag(corrtag_b, os.path.join(output_path,
                                                      'split_{}'.format(self.rootname[i])),
                              increment=increment, starttime=starttime,
                              endtime=endtime, time_list=time_list)
                              
        return

    def check_ref_files(self, path=None):
        """
        Checks to make sure all lref files that are needed for the data reduction
        have been downloaded. Files can be in their own directory. In addition to
        checking if the files are downloaded, you will need to add the following
        to your .bashrc file:
        `export lref="$/HOME/[path]"`
        for calcos to work properly.

        Parameters
        ----------
        path : str, optional
           The path where the reference files are stored. If no path is passed,
           the default is `self.input_path`.

        Returns
        -------
        missing : np.array
           A list of missing reference files.
        """
        filetypes = ['FLATFILE', 'DEADTAB', 'BPIXTAB', 'SPOTTAB', 'GSAGTAB',
                     'HVTAB', 'BRFTAB', 'GEOFILE', 'DGEOFILE', 'TRACETAB'
                     'PROFTAB', 'TWOZXTAB', 'XWLKFILE', 'YWLKFILE', 'PHATAB',
                     'PHAFILE', 'BADTTAB', 'XTRACTAB', 'LAMPTAB', 'DISPTAB',
                     'IMPHTTAB', 'FLUXTAB', 'WCPTAB', 'BRSTTAB', 'TDSTAB',
                     'SPWCSTAB']

        all_files = []
        needed_files = []

        for i in range(len(self.rootname[i])):
            for letter in ['a', 'b']:
                fn = os.path.join(self.input_path, self.rootname[i] + '_corrtag_{}.fits'.format(letter))
                hdu = fits.open(fn)

                for ft in filetypes:
                    if hdu[0].header[ft] not in needed_files:
                        all_files.append(hdu[0].header[ft])
                        
                hdu.close()

        if path is None:
            path = self.input_path

        downloaded = os.listdir(path)

        for fn in all_files:
            if fn[0].split('$')[-1] not in downloaded and l[0]!='N/A':
                needed_files.append(fn)

        if len(needed_files) > 0:
            return needed_files
        else:
            return("You have all reference files needed!")
            
