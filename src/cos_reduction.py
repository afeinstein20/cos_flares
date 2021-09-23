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
    
    def __init__(self, rootname, input_path, output_path):
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

        """
