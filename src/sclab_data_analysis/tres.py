import numpy as np
from scipy.interpolate import interp1d
import importlib

def apodization(y, x, apodization_width):
    """
    Applies apodization to the input data using a Gaussian window.

    Parameters:
        y: 1D NumPy array
            The input data to be apodized (e.g., interferogram).
        x: 1D NumPy array
            The corresponding position axis.
        apodization_width: float
            Normalized standard deviation of the Gaussian window with respect to the length of the position axis.

    Returns:
        apodized_y: 1D NumPy array
            The apodized version of the input data.
        gaussian_window: 1D NumPy array
            The Gaussian apodization window applied to the data.
    """
    # Normalize the position axis
    x_min = np.min(x)
    x_max = np.max(x)
    x_normalized = (x - x_min) / (x_max - x_min)

    # Create a Gaussian window
    center = 0.5  # Center of the Gaussian window in normalized units
    std_dev = apodization_width  # Standard deviation of the Gaussian window
    gaussian_window = np.exp(-0.5 * ((x_normalized - center) / std_dev) ** 2)

    # Apply the Gaussian window to the input data
    apodized_y = y * gaussian_window

    return apodized_y, gaussian_window

def dft_2DMAP(map, pos, start_wave, end_wave, samples, apodization_width):
    """
    Compute the Time-Resolved Emission Map from the interferometric map.

    Parameters:
        map: 2D NumPy array
            Interferometric map where each row corresponds to a histogram for a specific delay specified by `pos`.
        pos: 1D NumPy array
            Position axis.
        start_wave: float
            Start wavelength for spectra calculation.
        end_wave: float
            End wavelength for spectra calculation.
        samples: int
            Number of spectral samples to calculate.
        apodization_width: float
            Normalized standard deviation with respect to the position axis length for the apodization function.

    Returns:
        MAP: 2D NumPy array
            Time-resolved emission map.
        wave: 1D NumPy array
            Corresponding wavelength axis.
    """
    # Load spectral calibration
    with importlib.resources.path('sclab_data_analysis', 'data') as data_path:
        cal = np.loadtxt(data_path / "Initialize_parameters_cal.txt")

    # Compute frequency boundaries
    start_freq = interp1d(1.0 / cal[0, :], cal[1, :], kind='linear')(1.0 / end_wave)
    end_freq = interp1d(1.0 / cal[0, :], cal[1, :], kind='linear')(1.0 / start_wave)

    # Ensure `pos` is a row vector
    if pos.ndim > 1 and pos.shape[0] > pos.shape[1]:
        pos = pos.flatten()

    # Create differential axis
    dpos = np.append(np.diff(pos), 0)

    # Create frequency axis
    freq = np.linspace(start_freq, end_freq, samples)

    # Compute DFT
    exp_map = np.exp(-1j * 2 * np.pi * np.outer(pos, freq))
    MAP = np.zeros((samples, map.shape[1]), dtype=np.float64)

    for k in range(map.shape[1]):
        interf, _ = apodization(map[:, k] - np.mean(map[:, k]), pos, apodization_width)
        MAP[:, k] = np.abs(np.dot(dpos * interf, exp_map))

    # Compute wavelength axis
    unique_indices = np.unique(cal[1, :], return_index=True)[1]
    cal_unique = cal[:, unique_indices]
    wave = 1.0 / interp1d(cal_unique[1, :], 1.0 / cal_unique[0, :], kind='linear', fill_value='extrapolate')(freq)

    return MAP, wave

def interf_calibration2D(interf2d, pos_axis):
    """
    Calibrate the measurement map and the position axis to be ready for Fourier transform.

    Parameters:
        interf2d: 2D NumPy array
            Interference map where each column is an interferogram.
        pos_axis: 1D NumPy array
            Position axis corresponding to the measured map.

    Returns:
        calib_interf2d: 2D NumPy array
            Calibrated interference map.
        calib_pos_axis: 1D NumPy array
            Calibrated position axis.
    """
    # Get calibrated position axis and indices for slicing
    calib_pos_axis, left_index, right_index = pos_axis, 0, pos_axis.size #Get_Calibrated_Position_Axis_new(pos_axis)

    # Slice the interferometric map based on the calibrated indices
    calib_interf2d = interf2d[left_index:right_index, :]

    return calib_interf2d, calib_pos_axis



