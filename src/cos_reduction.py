import calcos
import os, sys
import numpy as np
from tqdm import tqdm
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

        Attributes
        ----------
        rootname : np.array
        input_path : str
        """
        self.rootname = rootname
        self.input_path = input_path

        self.splittag_path = None
        self.reduced_path = None
        self.path_a = None
        self.path_b = None

        self.orbit = None
        self.times = None


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

        Attributes
        ----------
        splittag_path : str
           The path to where the splittag data is saved.
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
                              
        self.splittag_path = output_path


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

        for i in range(len(self.rootname)):
            for letter in ['a', 'b']:
                fn = os.path.join(self.input_path, self.rootname[i] + '_corrtag_{}.fits'.format(letter))
                hdu = fits.open(fn)

                for ft in filetypes:
                    try:
                        if hdu[0].header[ft] not in needed_files:
                            all_files.append(hdu[0].header[ft])
                    except:
                        pass
                        
                hdu.close()

        if path is None:
            path = self.input_path

        downloaded = os.listdir(path)

        for fn in all_files:
            if fn.split('$')[-1] not in downloaded and fn!='N/A':
                needed_files.append(fn.split('$')[-1])

        if len(needed_files) > 0:
            return needed_files
        else:
            return("You have all reference files needed!")
            

    def reduce_data(self, output_path):
        """
        Uses the calcos package to reduce new data.

        Parameters
        ----------
        output_path : str
           Where the output reduced data should be stored.
           This will create a subdirectory `a` and `b` when
           necessary.

        Attributes
        ----------
        reduced_path : str
           Where the reduced data is stored.
        path_a : str
           Where the reduced corrtag_a output files are stored.
        path_b : str
           Where the reduced corrtag_b output files are stored.
        """
        path_a = os.path.join(output_path, 'a')
        path_b = os.path.join(output_path, 'b')

        if os.path.exists(output_path):
            if not os.path.exists(path_a):
                os.mkdir(path_a)
            if not os.path.exists(path_b):
                os.mkdir(path_b)

        else:
            try:
                os.mkdir(output_path)
                os.mkdir(path_a)
                os.mkdir(path_b)
            except OSError:
                return("Couldn't find or create the output_path.")
                         
        for letter in ['a', 'b']:
            if letter == 'a':
                outdir = path_a
            else:
                outdir = path_b
            split_files = np.unique(np.sort([os.path.join(self.splittag_path, i)
                                             for i in os.listdir(self.splittag_path)
                                             if i.endswith('{}.fits'.format(letter))]))
            for f in tqdm(range(len(split_files))):
                calcos.calcos(split_files[f], verbosity=0,
                              outdir=outdir)

        self.path_a = path_a
        self.path_b = path_b
        self.reduced_path = output_path


    def bookkeeping(self):
        """
        Does minor tasks like renaming the files such that they are ordered in
        time properly, creates an array of corresponding orbit numbers, and 
        creates an array of times.

        Attributes
        ----------
        orbit : np.array
           Array of orbit values, indexing at 0. 
        times : np.array
           Array of times extracted from FITS headers.
        """

        self.fix_naming(self.path_a)
        self.fix_naming(self.path_b)

        self.add_orbits(os.listdir(self.path_a))
        self.add_times(os.listdir(self.path_a))


    def fix_naming(self, path):
        """ 
        Fixes the naming convention of the calcos files.

        Parameters
        ----------
        path : str
           The path where the output calcos files are stored. This function
           only renames the `x1d.fits` files.
        """
        files = [os.path.join(path, i) for i in 
                 os.listdir(path) if i.endswith('x1d.fits')]
        new, cp = [], []
        deflen = len(files[int(files/2)])

        for fn in files:
            if len(fn) < deflen:
                cp.append(fn)
                r = fn.split('_')
                r[-2] = '0'+r[-2]
                new.append('_'.join(e for e in r))

        if len(new) > 0:
            for i in range(len(new)):
                os.replace(cp[i], new[i])
        return
                  
        
    def add_orbits(self, files):
        """ 
        Creates an array of affiliated orbit values.
        
        Parameters
        ----------
        files : np.array
           Array of file names with the rootname included.

        Attributes
        ----------
        orbit : np.array
           Array of orbit values, indexing at 0.
        """
        orbits = np.zeros(len(files))

        for i,fn in enumerate(files):
            r = fn.split('_')[-3] # grabs the rootname
            orbits[i] = np.where(np.array(self.rootname)==r)[0]

        self.orbit = orbit
        return


    def add_times(self, files):
        """
        Creates an array of times as extracted from the FITS
        headers.

        Parameters
        ----------
        files : np.array
           Array of file names.
           
        Attributes
        ----------
        times : np.array
           Array of time values, given in seconds.
        """
        times = np.zeros((len(files),2))

        for i in range(len(files)):
            hdu = fits.open(files[i])
            start = (hdu[1].header['EXPSTART']*units.day).to(units.s)
            t = [float(i) for i in str(hdu[0].header['HISTORY']).split('was')[-1][1:].split(' to ')]
            times[i][0] = t[0] + start.value
            times[i][1] = t[1] + start.value
            hdu.close()
        self.times = np.nanmedian(times, axis=1)
        return
