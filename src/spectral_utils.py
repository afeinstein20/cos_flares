import numpy as np
from astropy import units
from astropy.table import Table, Column

__all__ = ['to_velocity', 'measure_ew']


def to_velocity(wave, mid):
    """
    Converts wavelength to velocity space given some
    reference wavelength.

    Parameters
    ----------
    wave : np.array
       Wavelength array
    mid : float, optional
       The middle wavelength to center 0 km/s on.
       Default is none and will take the center of
       the spectrum as the 0 point.

    Returns
    -------
    rv_km_s : np.array
       The velocity in km/s.
    mid : float
       Where the velocity = 0.
    """

    if mid == None:
        mid = int(len(wave)/2)
    else:
        mid = np.where(wave>=mid)[0][0]

    lambda0 = wave[mid] + 0.0
    rv_m_s = ((wave - lambda0)/lambda0 * 3e8)*units.m/units.s
    rv_km_s = rv_m_s.to(units.km/units.s)

    return rv_km_s, mid


def measure_ew(time, wavelength, flux, flux_err, line_table,
               ion=None, line=None, vmin=None, vmax=None,
               orbit_num='all', binsize=3, width_table=None, error_table=None):
    """
    Measures the equivalent width of a given line. Either the
    ion name (from line_table) can be passed in or the
    center wavelength, vmin (minimum velocity of the line), and
    the vmax (maximum velocity of the line) can be passed in.

    Parameters
    ----------
    ion : str, optional
       The ion name in line_table to use
       for the analysis.
    line : float, optional
       The center of the line to measure.
    vmin : float, optional
       The lower velocity to incorporate into the
       line measurement.
    vmax : float, optional
       The upper velocity to incorporate into the
       line measurement.
    orbit : int, optional
       Applies a mask to a specified orbit. Default
       is all orbits.
    binsize : int, optional
       Accounts for binning in the error calculation.
       If the data is not binned, use binsize=1. Default
       is 3.

    Returns
    ----------
    width_table : astropy.table.Table
       Adds a column of measured equivalent widths to the
       attribute width_table. Columns are replaced when
       lines are re-measured.
    error_table : astropy.table.Table
       Adds a column of equivalent width errors to the
       attribute error_table. Columns are replaced when
       lines are re-measured.

    """
    if width_table is None:
        width_table = Table()
        error_table = Table()

    if ion is not None and line_table is not None:
        line = line_table[line_table['ion']==ion]['wave_c']+0.0
        vmin = line_table[line_table['ion']==ion]['vmin']+0.0
        vmax = line_table[line_table['ion']==ion]['vmax']+0.0
    elif ion is not None and line_table is None:
        return('No table found. Please load the line table first with \
                load_line_table().')

    widths = np.zeros(len(time))
    errors = np.zeros(len(time))

    for i in range(len(time)):
        v, _ = to_velocity(wavelength[i], mid=line)
        reg = np.where( (v.value >= vmin) & (v.value <= vmax) )[0]

        widths[i] = np.nansum(flux[i][reg])
        errors[i] = np.sqrt(np.nansum(flux_err[i][reg]**2))

    try:
        if ion is not None:
            width_table.add_column(Column(widths, ion))
        else:
            width_table.add_column(Column(widths, str(np.round(line,3))))
    except ValueError:
        if ion is not None:
            width_table.replace_column(ion, widths)
        else:
            width_table.replace_column(str(np.round(line,3)), widths)

    try:
        if ion is not None:
            error_table.add_column(Column(errors, ion))
        else:
            error_table.add_column(Column(errors, str(np.round(line,3))))
    except ValueError:
        if ion is not None:
            error_table.replace_column(ion, errors)
        else:
            error_table.replace_column(str(np.round(line,3)), errors)

    return width_table, error_table
