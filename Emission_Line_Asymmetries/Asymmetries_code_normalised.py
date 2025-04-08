################################################################################
################################################################################
#     _                                       _        _          
#    / \   ___ _   _ _ __ ___  _ __ ___   ___| |_ _ __(_) ___ ___ 
#   / _ \ / __| | | | '_ ` _ \| '_ ` _ \ / _ | __| '__| |/ _ / __|
#  / ___ \\__ | |_| | | | | | | | | | | |  __| |_| |  | |  __\__ \
# /_/   \_|___/\__, |_| |_| |_|_| |_| |_|\___|\__|_|  |_|\___|___/
#   ____       |___/                                              
#  / ___|___   __| | ___                                          
# | |   / _ \ / _` |/ _ \                                         
# | |__| (_) | (_| |  __/                                         
#  \____\___/ \__,_|\___|                                         
################################################################################
################################################################################
# This script is researching if an asymmetric line profile in H-alpha's  
# line is connected to the presence of outflows in cataclysmic variables.
# The script is designed to be run in a Jupyter Notebook environment.
# The script is divided into 9 steps. We load PYTHON simulation grids and fit 
# the data with a Gaussian with a linear continuum underneath. We then calculate
# the equivalent width excesses for each spectrum in the grid. We then plot the
# equivalent width excesses for all runs against Teo's data. Finally, we 
# investigate whether than is a trend in the connections between excess 
# equivalent widths and parameter combinations in the PYTHON grid space.

# This code script is or inputing only normalised spectra with a continuum of 1.
# As this is a ripped copy from unnormalised. The continuum functions/code is still
# implimented. However, for the excess calculations, the value 1 is just used 
# instead. The continuum is probably doing mad fits here, but it's not used.
################################################################################
################################################################################


# %% 1
################################################################################
print('STEP 1: IMPORTING MODULES')
################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import fnmatch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from prettytable import PrettyTable
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import itertools
import mplcyberpunk # mplcyberpunk.add_glow_effects()
import matplotx
import scienceplots
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from IPython.display import display, Math, Latex, Markdown
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import datetime
import os
import scipy.signal as signal

plt.style.use('Solarize_Light2')
np.random.seed(5025)
# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
#plt.style.use('science')
script_ran_once = False

# %% 2
################################################################################
print('STEP 2: LOADING DATA AND BUILDING GRID')
################################################################################

path_to_grid = '../Release_Ha_grid_spec_files/'
files = os.listdir(path_to_grid) # list of files in directory

# -------INPUTS-------- #
wavelength_range = (6435, 6685) # set desired wavelength range, start with the narrowest range
inclination_column = 10 # 10-14 = 20,45,60,72.5,85
grid_mixed = True # if grid is mixed, set to True to rerun the script at difference wavelength ranges
                  # you can change the wavelength range in the second input below.
# Be sensible with intervals or you'll break code. (near peak first, far peak second)
blue_peak_mask = (24,55) # number of angstroms to cut around the peak, blue minus.
red_peak_mask = (24,55) # number of angstroms to cut around the peak, red plus.
#spectra_normalised = False # if the spectra are normalised, set to True
# --------------------- #
H_alpha = 6562.819
grid_length = np.arange(0,729)
if not grid_mixed:
    run_number = grid_length # run number ranges range(0,729)
    print('Ran normal grid')
    
if grid_mixed and not script_ran_once: 
    mixed_runs = [77,79,158,160,240,241,320,322,414,425,468,477,478,479,480,481,482,483,563,565,567,569,580,639,640,641,642,643,644,645,646,667,668,669,712,713,720,721,722,723,724,725,726,727]
    run_number = np.setdiff1d(grid_length, mixed_runs) # remove mixed runs from run_number

sorted_files = []
for num in run_number: # run number ranges range(0,729)
    run = f'rerun{num}.spec'
    file = fnmatch.filter(files, run) # matching file with loop run number
    sorted_files.append(path_to_grid+str(file[0])) # files sorted in order of run number
    
if grid_mixed and script_ran_once:
    wavelength_range = (6260, 6860) # new wavelength range INPUT HERE!!!
    sorted_files = []
    for num in mixed_runs: # run number ranges range(0,729)
        run = f'rerun{num}.spec'
        file = fnmatch.filter(files, run) # matching file with loop run number
        sorted_files.append(path_to_grid+str(file[0])) # files sorted in order of run number
    print('Ran mixed grid')

# wavelengths in angstroms ordered by reversing list 
wavelengths = np.loadtxt(sorted_files[0], usecols=1, skiprows=81)[::-1]
wave_mask = (wavelengths > wavelength_range[0]) & (wavelengths < wavelength_range[1]) # mask for wavelength range
wavelengths = wavelengths[wave_mask] # updating to new wavelength range
grid = np.empty((len(sorted_files), len(wavelengths))) # empty array to store flux values
wavelength_grid = np.empty((len(sorted_files), len(wavelengths))) # empty array to store wavlength values

for i in tqdm(range(len(sorted_files))):
    flux = np.loadtxt(sorted_files[i], usecols=inclination_column, skiprows=81)[::-1] # flux values for each file
    grid[i] = flux[wave_mask] # adding fluxes into the grid
    wave = np.loadtxt(sorted_files[i], usecols=1, skiprows=81)[::-1] # wavelength values for each file
    wavelength_grid[i] = wave[wave_mask] # adding wavelengths into the grid
    
# %% 2 (MOLLY GRID)
################################################################################
print('STEP 2 (MOLLY GRID): LOADING DATA AND BUILDING GRID')
################################################################################
# load npy file : np.save('molly_spectra.npy', {'run_numbers': run_numbers, 'wavelength_grid': wavelength_grid, 'flux_grid': flux_grid, 'times': times, 'systems': systems})
index_limits = (0,62) # 0,17
grid_length = np.arange(index_limits[0],index_limits[1])
molly_data = np.load('molly_spectra.npy', allow_pickle=True)
run_numbers = molly_data.item().get('run_numbers')[index_limits[0]:index_limits[1]]
wavelength_grid = molly_data.item().get('wavelength_grid')[index_limits[0]:index_limits[1]]
grid = molly_data.item().get('flux_grid')[index_limits[0]:index_limits[1]]
times = molly_data.item().get('times')[index_limits[0]:index_limits[1]]
systems = molly_data.item().get('systems')[index_limits[0]:index_limits[1]]

    
wavelengths = wavelength_grid[0]
# -------INPUTS-------- #
wavelength_range = (6435, 6660) # set desired wavelength range, start with the narrowest range
inclination_column = 10 # 10-14 = 20,45,60,72.5,85
grid_mixed = False # if grid is mixed, set to True to rerun the script at difference wavelength ranges
                  # you can change the wavelength range in the second input below.
# Be sensible with intervals or you'll break code. (near peak first, far peak second)
blue_peak_mask = (22,55) # number of angstroms to cut around the peak, blue minus.
red_peak_mask = (22,55) # number of angstroms to cut around the peak, red plus.
# --------------------- #


# %% 3
################################################################################
print('STEP 3: SKLEARN FITTING THE CONTINUUM ONLY, NOISE ERROR CALCULATED')
################################################################################

def continuum(wavelengths, spectrum, intervals=(6220, 6480, 6600, 6900)):
    """ Function to determine the underlying continuum and noise of a spectrum.

    Args:
        wavelengths (ndarray): wavelengths of the spectrum
        spectrum (ndarray): a single spectrum from the grid

    Returns:
        continuum (ndarray): A continuum array of fluxes the same length as the spectrum
        reg.coef_ (float): The gradient of the linear fit
        reg.intercept_ (float): The y-intercept of the linear fit
        noise (float): The standard deviation of the conitnuum (noise)
    """

    # Continuum portion of the spectrum. Removing the emission lines
    #intervals = (6220, 6480, 6600, 6900) # wavelength intervals for continuum
    con_wave_mask_low = (wavelengths > intervals[0]) & (wavelengths < intervals[1])
    con_wave_mask_high = ((wavelengths > intervals[2]) & (wavelengths < intervals[3]))
    con_mask = con_wave_mask_low | con_wave_mask_high # mask only for continuum wavelength range

    # Fitting linear regression model to the continuum to subtract continuum
    reg = LinearRegression()
    reg.fit(wavelengths[con_mask].reshape(-1,1), spectrum[con_mask]) # fitting linear regression model
    continuum = reg.predict(wavelengths.reshape(-1,1)) # predicting continuum values
    
    line_spectrum = spectrum - continuum # subtracting continuum from grid
    noise = np.std(line_spectrum[con_mask]) # standard deviation of the noise
    
    return continuum, reg.coef_, reg.intercept_, noise

sk_con_data = np.array([]) # a continuum fit excluding peak features
sk_slopes = np.array([]) # the gradient of the linear fit
sk_intercepts = np.array([]) # the y-intercept of the linear fit
sk_noise_error = np.array([]) # the standard deviation of the noise excluding the peak features
if not script_ran_once:
    intervals = (6450,6475,6625,6650) # wavelength intervals for continuum
elif script_ran_once:
    intervals = (6300, 6325, 6775, 6800) # wavelength intervals for continuum
for i in range(len(grid)): # Iterating through grid
    con, m, c, noise = continuum(wavelengths, grid[i], intervals)
    sk_con_data = np.append(sk_con_data, con)
    sk_slopes = np.append(sk_slopes, m)
    sk_intercepts = np.append(sk_intercepts, c)
    sk_noise_error = np.append(sk_noise_error, noise)
    
sk_con_data = sk_con_data.reshape(len(grid), len(wavelengths)) # append method back to grid shape

# %% 4
################################################################################
print('STEP 4: RESAMPLING THE GRID DATA FROM NOISE ERROR FUNCTION')
################################################################################

def resample_data(wavelengths, grid_data, noise_error):
    """ Resampling the data from the noise error. 
    The noise error is used to generate a normal distribution of errors for each pixel.
    A random number is chosen from the normal distribution and added to the original flux value.
    This is done for each pixel in the spectrum.

    Args:
        wavelengths (ndarray): wavelengths of the spectrum
        grid_data (ndarray): all spectra from the grid
        noise_error (ndarray): the standard deviation of the noise excluding the peak features

    Returns:
        resampled_grid (ndarray): the resampled grid
    """
    resampled = np.array([])
    for i in range(len(grid_data)):
        noise = np.random.normal(0, noise_error[i], len(wavelengths)) # normal distribution of errors
        resampled = np.append(resampled, grid_data[i] + noise) # adding noise to original fluxes
    resampled = resampled.reshape(len(grid_data), len(wavelengths)) # append method back to grid shape
    return resampled

# %% 5
################################################################################
print('STEP 5: FITTING THE GAUSSIAN WITH CONTINUUM TO THE DATA FUNCTION')
################################################################################

def gaussian_with_continuum(w, amp, mu, sigma, m, b, run):
    """ Fitting a Gaussian function with a linear continuum underneath
    for the H_alpha emission line.

    Args:
        w (np.array): A 1D array of wavelengths
        amp (float): Amplitude of the Gaussian
        mu (float): Centre of the Gaussian peak
        sigma (float): Width of the Gaussian peak
        m (float): Slope of the continuum
        b (float): Intercept of the continuum at H-alpha.
        run (int): The run number of the spectrum. Stored with initial estimates.
                    as a last resort if you need to identify a run.

    Returns:
        fit (np.array): The gaussian and continuum theoretical fit
    """
    #mu = H_alpha # fixing Halpha
    
    
    const = amp * (1.0 / (np.fabs(sigma)*(2*np.pi)**0.5))
    fline = const * np.exp(-0.5*((w - mu)/np.fabs(sigma))**2.0)
    if sigma < 0: # if trialling negative sigma, give it a super poor fit
        fline = -1000
    if amp < 0:
        fline = -1000
    # if spectra_normalised:
    #     return fline + 1
    #else:
        #m = sk_slopes[int(run)] # fixing the slope for the minute
        #b = sk_intercepts[int(run)] # fixing the intercept for the minute
        #fcont = m * (w - H_alpha) + b
    return fline + 1

def fit_data(wavelengths, grid):
    """ Relies on function gaussian_with_continuum., Curve_fitting the gaussian with 
    continuum function to the data iterating spectra over the whole grid.

    Args:
        wavelengths (array): wavelength ranges
        grid (2D array): individual spectrum is a row entry with the columns containing the fluxes
        initials (list): initial estimates for the parameters of the gaussian with continuum function

    Returns:
        2D array: optimal parameters for each spectrum in the grid
    """
    
    parameters_all = np.array([])   # storing all parameters
    initials_all = np.array([])     # storing all initial estimates
    pcov_all = np.array([])         # storing all covariance matrices
    infodict_all = np.array([])     # storing all infodict
    
    for run in range(len(grid)):
        spectrum = grid[run, :] 
        
        # estimate for initial slope
        initial_m = 0
        
        # estimate for initial intercept
        initial_c = 1
        
        # estimate for the initial Gaussian peak (mean) location
        gaussian_only = spectrum - 1
        initial_mu = wavelengths[np.argmax(gaussian_only)]
        #initial_mu = H_alpha # fixing Halpha for the minute
        # estimate for initial sigma is 2.355 times the FWHM
        try:
            mask = (gaussian_only > np.max(gaussian_only)/ 3) # mask for fluxes greater than half peak
            #print(mask)
            for index, mask_bool in enumerate(mask):          # get the first true index in the mask list
                if mask_bool == True: 
                    index1 = index
                    break
            for index, mask_bool in enumerate(mask[::-1]):    # get the last true index in the mask list
                if mask_bool == True:
                    index2 = len(mask) - index
                    break
            initial_sigma = (wavelengths[index2] - wavelengths[index1])/2 # estimate initial sigma (FWHM/2 Due to double peaks)
        except:
            initial_sigma = 5
        
        # estimate for initial amplitude
        initial_amp = np.max(gaussian_only) * initial_sigma * np.sqrt(2*np.pi) # estimate initial amplitude
        
        # sorting initial estimates for parameters into a list
        initials = [initial_amp, initial_mu, initial_sigma, initial_m, initial_c, run]
        initials_all = np.append(initials_all, initials) # storing all initial estimates
        # limit wavelength to fit only 22Å either side of H_alpha (core of line)
        # spectrum = spectrum[(wavelengths > H_alpha - 22) & (wavelengths < H_alpha + 22)]
        # #sk_continuum = sk_continuum[run][(wavelengths > H_alpha - 22) & (wavelengths < H_alpha + 22)]
        # wavelengths_cut = wavelengths[(wavelengths > H_alpha - 22) & (wavelengths < H_alpha + 22)]
        parameters, pcov, infodict, _, _= curve_fit(gaussian_with_continuum, 
                                           wavelengths, 
                                           spectrum, 
                                           p0=initials, 
                                           maxfev=1_000_000,
                                           full_output=True
                                           )
        # adding info to arrays
        parameters_all = np.append(parameters_all, parameters)
        pcov_all = np.append(pcov_all, pcov)
        infodict_all = np.append(infodict_all, infodict)
    
    # amending to grid shape    
    parameters_all = parameters_all.reshape(len(grid), len(initials))
    initials_all = initials_all.reshape(len(grid), len(initials))
    pcov_all = pcov_all.reshape(len(grid), len(initials), len(initials))
    infodict_all = infodict_all.reshape(len(grid), 1)
    return parameters_all, initials_all, pcov_all, infodict_all

# Initialising arrays/variables
parameter_names = ['Amplitude', 'Mean', 'Sigma', 'Slope', 'Intercept']
H_alpha = 6562.819 #4861.333	 # 6562.819 # Emission line we are fitting

# %% 6
################################################################################
print('STEP 6: CALCULATING THE EQUIVALENT WIDTH EXCESSES')
################################################################################
fwhm_bounds = np.array([])
blue_window = np.array([])
red_window = np.array([])

def equivalent_width_excess(wavelengths, data, gaussian_fit, continuum_fit, shift='blue', peak_mask=(0,1000)):
    """ Returning the equivalent width excesses for each spectrum in the grid.
    You can choose to return the blue or red wing excesses by setting the shift.
    
    Formula: EW_excess = sum((data - gaussian_fit) / continuum_fit) * delta_wavelength
    
    PYTHON fluxes are left aligned to the wavelengths when binned. So code here
    reflects the delta_wavelength is associated with the left side flux value 
    of a bin. The peak_mask selected by the user may not exactly align with a 
    wavelength value. Hence, the code will find the nearest neighbour to the 
    wavelength within the range of the mask. This is also to exclude partial bins
    where large assumptions about the flux behaviour would have to be made. 
    TO BE CHANGED LATER IF PYTHON CHANGES BINNING STRUCTURE.
    
    Parameters:
        wavelengths (ndarray): wavelength array
        data (ndarray): the original data
        gaussian_fit (ndarray): the gaussian fit to the data
        continuum_fit (ndarray): the continuum fit to the data
        shift (str): 'blue' or 'red' for the blue or red wing excesses
        peak_mask (tuple): tuple wavelengths to cut around the peak. Default is 0,1000 doesn't cut anything.
            The value is the number of angstroms to cut around the peak, blue minus, red plus. First value is 
            the value closer to H_alpha.
    Returns:
        ew (ndarray): equivalent width excesses for each spectrum in the grid
    """
    
    H_alpha = 6562.819 #4861.333	#6562.819 # Å
    
    if shift == 'blue':
        
        # cutting to only have wavelengths the blue side of H_alpha
        index = np.where(wavelengths < H_alpha - peak_mask[0])[0][-1] # index of the last wavelength less than H_alpha
        slice_index_high_lambda = index + 1 # index to slice the data
        wavelengths = wavelengths[:slice_index_high_lambda] # slicing the wavelengths
        data = data[:, :slice_index_high_lambda] # slicing the data
        gaussian_fit = gaussian_fit[:, :slice_index_high_lambda] # slicing the gaussian fit
        continuum_fit = continuum_fit[:, :slice_index_high_lambda] # slicing the continuum fit
        
        index = np.where(wavelengths > H_alpha - peak_mask[1])[0][0] # index of the first wavelength greater than mask limits
        slice_index_low_lambda = index # index to slice the data
        wavelengths = wavelengths[slice_index_low_lambda:] # slicing the wavelengths
        data = data[:, slice_index_low_lambda:] # slicing the data
        gaussian_fit = gaussian_fit[:, slice_index_low_lambda:] # slicing the gaussian fit
        continuum_fit = continuum_fit[:, slice_index_low_lambda:] # slicing the continuum fit       
        
    if shift == 'red':
        
        # cutting to only have wavelengths the red side of H_alpha
        index = np.where(wavelengths > H_alpha + peak_mask[0])[0][0] # index of the first wavelength greater than H_alpha
        slice_index_low_lambda = index # index to slice the data
        wavelengths = wavelengths[slice_index_low_lambda:] # slicing the wavelengths
        data = data[:, slice_index_low_lambda:] # slicing the data
        gaussian_fit = gaussian_fit[:, slice_index_low_lambda:] # slicing the gaussian fit
        continuum_fit = continuum_fit[:, slice_index_low_lambda:] # slicing the continuum fit
        
        index = np.where(wavelengths < H_alpha + peak_mask[1])[0][-1] # index of the last wavelength less than mask limits
        slice_index_high_lambda = index + 1 # index to slice the data
        wavelengths = wavelengths[:slice_index_high_lambda] # slicing the wavelengths
        data = data[:, :slice_index_high_lambda] # slicing the data
        gaussian_fit = gaussian_fit[:, :slice_index_high_lambda] # slicing the gaussian fit
        continuum_fit = continuum_fit[:, :slice_index_high_lambda] # slicing the continuum fit
        
    # calculating the delta wavelengths for left aligned
    wave_diff = np.diff(wavelengths) 
    #cut the last element off the flux lists to match the len(delta wavelength)
    wavelengths = wavelengths[:-1]
    data = data[:, :-1]
    gaussian_fit = gaussian_fit[:, :-1]
    continuum_fit = continuum_fit[:, :-1]
    
    # Equivalent width excess calculation
    fraction = (data - gaussian_fit) / 1 # continuum_fit removed
    ew = [np.sum(fraction[i] * wave_diff) for i in range(len(fraction))]
        
    return ew

def fit_procedure(wavelengths, grid):
    """This function is to simplify the procedure of fitting the data. We take
    the PYTHON grid and inital """
    # Fitting the grid's emission peak AND continuum
    fit_parameters, initials_all, pcov_all, infodict_all= fit_data(wavelengths, grid) # optimal parameters for a spectrum
    fitted_grid = np.array([])
    for spectrum in range(len(grid)):
        fitted_grid = np.append(fitted_grid, gaussian_with_continuum(wavelengths, *fit_parameters[spectrum])) # simulating that optimal fit
    fitted_grid = fitted_grid.reshape(len(grid), len(wavelengths))

    # Fitting the grid's continuum ONLY
    fit_con = np.array([])
    for spectrum in range(len(grid)):
        fit_con = np.append(fit_con, fit_parameters[spectrum][3]*(wavelengths-H_alpha) + fit_parameters[spectrum][4])
    fit_con = fit_con.reshape(len(grid), len(wavelengths))
    return fitted_grid, fit_con, fit_parameters, initials_all, pcov_all, infodict_all


# Displaying the blue and red wavelengths either side of H_alpha
blue_wavelengths = wavelengths[wavelengths < H_alpha]
red_wavelengths = wavelengths[wavelengths > H_alpha]
print(f'Blue: {blue_wavelengths[0]}-{blue_wavelengths[-1]}Å  Red: {red_wavelengths[0]}-{red_wavelengths[-1]}Å  H_alpha: {H_alpha}Å')

# Calculating the equivalent width excess data points for the PYTHON Grid
fitted_grid, fit_con, fit_parameters, initials_all, pcov_all, infodict_all = fit_procedure(wavelengths, grid) #TODO  this causes the covariance issue
blue_ew_excess = equivalent_width_excess(wavelengths, grid, fitted_grid, fit_con, shift='blue', peak_mask=blue_peak_mask)
red_ew_excess = equivalent_width_excess(wavelengths, grid, fitted_grid, fit_con, shift='red', peak_mask=red_peak_mask)

# Calculating the equivalent width excess errors from resampling grid
resampled_blue_ew_excess = np.array([])
resampled_red_ew_excess = np.array([])
resampled_grids = np.array([]) # to store grid for EW calculations later
samples = 150
for sample in tqdm(range(samples)):
    resampled_grid = resample_data(wavelengths, grid, sk_noise_error) # resampled grid
    resampled_grids = np.append(resampled_grids, resampled_grid)
    fitted_grid, fit_con, fit_parameters, initials_all, _, _ = fit_procedure(wavelengths, resampled_grid)
    blue = equivalent_width_excess(wavelengths, resampled_grid, fitted_grid, fit_con, shift='blue', peak_mask=blue_peak_mask)
    red = equivalent_width_excess(wavelengths, resampled_grid, fitted_grid, fit_con, shift='red', peak_mask=red_peak_mask)
    resampled_blue_ew_excess = np.append(resampled_blue_ew_excess, blue)
    resampled_red_ew_excess = np.append(resampled_red_ew_excess, red)
#resampled grid shape is 150 * (run_number x number of wavelengths)
resampled_grids = resampled_grids.reshape(samples, len(grid), len(wavelengths))
resampled_blue_ew_excess = resampled_blue_ew_excess.reshape(samples, len(grid))
resampled_red_ew_excess = resampled_red_ew_excess.reshape(samples, len(grid))
blue_ew_excess_error = np.std(resampled_blue_ew_excess, axis=0)
red_ew_excess_error = np.std(resampled_red_ew_excess, axis=0)

if not grid_mixed:
    print("You aren't running a mixed grid. Run Step 7 and skip Step 8")
    
if grid_mixed and not script_ran_once:
    print('You are running a mixed grid. Be sure to run Step 7.')
    
if grid_mixed and script_ran_once:
    print("You've ran the mixed grid, run step 8 to mix the data.")

# %% 6 (FWHM)
################################################################################
print('STEP 6 (FWHM): CALCULATING THE EQUIVALENT WIDTH EXCESSES')
################################################################################
%matplotlib inline
def fwhm_equivalent_width_excess(wavelengths, 
                                 data, 
                                 gaussian_fit, 
                                 continuum_fit,
                                 peak_mean, 
                                 fwhm_bounds,
                                 shift='blue', 
                                 inner=1.0, 
                                 outer=5.0):
    """ Returning the equivalent width excesses for each spectrum in the grid.
    You can choose to return the blue or red wing excesses by setting the shift.
    
    Formula: EW_excess = sum((data - gaussian_fit) / continuum_fit) * delta_wavelength
    
    The mask limits for this formula are set as multiples of FWHM.
    
    SIROCCO fluxes are left aligned to the wavelengths when binned. So code here
    reflects the delta_wavelength is associated with the left side flux value 
    of a bin. The peak_mask selected by the user may not exactly align with a 
    wavelength value. Hence, the code will find the nearest neighbour to the 
    wavelength within the range of the mask. This is also to exclude partial bins
    where large assumptions about the flux behaviour would have to be made. 
    TO BE CHANGED LATER IF SIROCCO CHANGES BINNING STRUCTURE.
    
    Parameters:
        wavelengths (ndarray): wavelength array
        data (ndarray): the original data
        gaussian_fit (ndarray): the gaussian fit to the data
        continuum_fit (ndarray): the continuum fit to the data
        peak_mean (ndarray): the fitted mean of the gaussian
        fwhm_bounds (ndarray): the blue and red wavelengths for the fwhm bounds
        shift (str): 'blue' or 'red' for the blue or red wing excesses
        inner (float): the inner limit fwhm multiple for the mask
        outer (float): the outer limit fwhm multiple for the mask
    Returns:
        ew (ndarray): equivalent width excesses for each spectrum in the grid
    """
    
    H_alpha = 6562.819  # Å
    
    # Adjust the FWHM bounds to the inner and outer limits
    blue_fwhm = fwhm_bounds[:, 0]
    inner_blue = peak_mean - ((peak_mean - blue_fwhm) * inner)
    outer_blue = peak_mean - ((peak_mean - blue_fwhm) * outer)
    b_peak_mask = np.array([inner_blue, outer_blue]).T  # Blue peak mask for each spectrum
    
    red_fwhm = fwhm_bounds[:, 1]
    inner_red = peak_mean + ((red_fwhm - peak_mean) * inner)
    outer_red = peak_mean + ((red_fwhm - peak_mean) * outer)
    r_peak_mask = np.array([inner_red, outer_red]).T  # Red peak mask for each spectrum
    
    ew = []  # List to store equivalent widths
    
    for i in range(data.shape[0]):  # Iterate over each spectrum
        #print(f'Iteration: {i}')
        if shift == 'blue':
            # Create a mask for the wavelengths in the blue shift for this spectrum
            wave_mask = (wavelengths >= outer_blue[i]) & (wavelengths <= inner_blue[i])
        elif shift == 'red':
            # Create a mask for the wavelengths in the red shift for this spectrum
            wave_mask = (wavelengths >= inner_red[i]) & (wavelengths <= outer_red[i])
        else:
            raise ValueError("Invalid shift value. Must be 'blue' or 'red'.")
        
        # Slice the wavelength and data arrays using the mask
        wl = wavelengths[wave_mask]
        d = data[i, wave_mask]
        gf = gaussian_fit[i, wave_mask]
        #cf = continuum_fit[i, wave_mask]
        
        # Calculate the delta wavelengths
        wave_diff = np.diff(wl)
        # Adjust arrays to match the length of wave_diff
        wl = wl[:-1]
        d = d[:-1]
        gf = gf[:-1]
        #cf = cf[:-1]
        
        # Compute the equivalent width for this spectrum
        fraction = (d - gf) / 1 #cf
        #print(f'Fraction shape: {np.shape(fraction)}')
        ew_value = np.sum(fraction * wave_diff)
        #print(f'EW value: {ew_value}')
        ew.append(ew_value)
        
        # plotting bounds on the spectra blue
        # if i > 650:
        #     print(f'Spectra{run_number[i]}, {fwhm_bounds[i]}')
        #     plt.plot(wavelengths, data[i], label='Data')
        #     plt.vlines([inner_blue[i], outer_blue[i]], np.min(data[i]), np.max(data[i]), colors='r', linestyles='dashed', label='Blue FWHM')
        #     plt.vlines(H_alpha, np.min(data[i]), np.max(data[i]), colors='g', linestyles='dashed', label='H_alpha')
        #     plt.plot(wavelengths, continuum_fit[i], label='Continuum')
        #     plt.plot(wavelengths, gaussian_fit[i], label='Gaussian')
        #     plt.legend()
        #     plt.show()
        #     for i in range(len(wave_mask)):
        #         if wave_mask[i] == True:
        #             print(wave_mask[i], wavelengths[i])
        
        # plotting bounds on the spectra red
        # if i==590:
        #     print(f'Spectra{run_number[i]}, {fwhm_bounds[i]}, {shift}, {ew_value}')
        #     plt.plot(wavelengths, data[i], label='Data')
        #     plt.vlines([inner_red[i], outer_red[i]], np.min(data[i]), np.max(data[i]), colors='r', linestyles='dashed', label='Red FWHM')
        #     plt.vlines(H_alpha, np.min(data[i]), np.max(data[i]), colors='g', linestyles='dashed', label='H_alpha')
        #     plt.plot(wavelengths, continuum_fit[i], label='Continuum')
        #     plt.plot(wavelengths, gaussian_fit[i], label='Gaussian')
        #     plt.legend()
        #     plt.show()
        #     for i in range(len(wave_mask)):
        #         if wave_mask[i] == True:
        #             print(wave_mask[i], wavelengths[i])
        
    # calculating the delta wavelengths for left aligned
    # wave_diff = np.diff(wavelengths) 
    # #cut the last element off the flux lists to match the len(delta wavelength)
    # wavelengths = wavelengths[:-1]
    # data = data[:, :-1]
    # gaussian_fit = gaussian_fit[:, :-1]
    # continuum_fit = continuum_fit[:, :-1]
    
    # # Equivalent width excess calculation
    # fraction = (data - gaussian_fit) / continuum_fit
    # ew = [np.sum(fraction[i] * wave_diff) for i in range(len(fraction))]
        
    return ew, b_peak_mask, r_peak_mask

def fwhm_calculations(wavelengths, data, peak_mean):
    """ A procedure to find the FWHM of a spectral line and report the 
    blue and red wavelengths either side of the peak.

    Args:
        wavelengths (ndarray): wavelength array
        data (ndarray): the original data
        peak_mean (ndarray): the fitted mean of the gaussian peak
    """
    H_alpha = 6562.819 # Å
    # Adjusting for the trend in continuum.
    data = data - 1 # continuum_fit
    
    # Finding the greatest flux value in the spectral line
    max_flux = np.array([np.max(data[i]) for i in range(len(data))])
    
    
    # Searching for the blue FWHM bound
    blue_fwhms = np.array([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] >= max_flux[i]/2:
                blue_bound = wavelengths[j]
                if blue_bound > peak_mean[i]: # don't search the wrong side of H_alpha
                    blue_bound = peak_mean[i]
                break
            
        blue_fwhms = np.append(blue_fwhms, blue_bound)
    
    # Searching for the red FWHM bound
    red_fwhms = np.array([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][::-1][j] >= max_flux[i]/2:
                red_bound = wavelengths[::-1][j]
                if red_bound < peak_mean[i]: # don't search the wrong side of H_alpha
                    red_bound = peak_mean[i]
                break
        red_fwhms = np.append(red_fwhms, red_bound)
        
    fwhm_bounds = np.array((blue_fwhms, red_fwhms)).T
    
    return fwhm_bounds

def fit_procedure(wavelengths, grid):
    """This function is to simplify the procedure of fitting the data. We take
    the PYTHON grid and inital """
    # Fitting the grid's emission peak AND continuum
    fit_parameters, initials_all, pcov_all, infodict_all = fit_data(wavelengths, grid) # optimal parameters for a spectrum
    fitted_grid = np.array([])
    for spectrum in range(len(grid)):
        fitted_grid = np.append(fitted_grid, gaussian_with_continuum(wavelengths, *fit_parameters[spectrum])) # simulating that optimal fit
    fitted_grid = fitted_grid.reshape(len(grid), len(wavelengths))

    # Fitting the grid's continuum ONLY
    fit_con = np.array([])
    for spectrum in range(len(grid)):
        fit_con = np.append(fit_con, fit_parameters[spectrum][3]*(wavelengths-H_alpha) + fit_parameters[spectrum][4])
    fit_con = fit_con.reshape(len(grid), len(wavelengths))
    return fitted_grid, fit_con, fit_parameters, initials_all, pcov_all, infodict_all


# Displaying the blue and red wavelengths either side of H_alpha
blue_wavelengths = wavelengths[wavelengths < H_alpha]
red_wavelengths = wavelengths[wavelengths > H_alpha]
print(f'Blue: {blue_wavelengths[0]}-{blue_wavelengths[-1]}Å  Red: {red_wavelengths[0]}-{red_wavelengths[-1]}Å  H_alpha: {H_alpha}Å')

# Calculating the equivalent width excess data points for the PYTHON Grid
fitted_grid, fit_con, fit_parameters, initials_all, _, _ = fit_procedure(wavelengths, grid) #TODO  this causes the covariance issue
fwhm_bounds = fwhm_calculations(wavelengths, grid, fit_parameters[:, 1]) # the mean of gaussian
blue_ew_excess, blue_window, _ = fwhm_equivalent_width_excess(wavelengths,
                                                grid,
                                                fitted_grid,
                                                fit_con,
                                                fit_parameters[:, 1],
                                                fwhm_bounds,
                                                shift='blue'
                                                )
red_ew_excess, _, red_window = fwhm_equivalent_width_excess(wavelengths,
                                             grid,
                                             fitted_grid,
                                             fit_con,
                                             fit_parameters[:, 1],
                                             fwhm_bounds,
                                             shift='red'
                                             )

# Calculating the equivalent width excess errors from resampling grid
resampled_blue_ew_excess = np.array([])
resampled_red_ew_excess = np.array([])
resampled_grids = np.array([]) # to store grid for EW calculations later
samples = 150
for sample in tqdm(range(samples)):
    resampled_grid = resample_data(wavelengths, grid, sk_noise_error) # resampled grid
    resampled_grids = np.append(resampled_grids, resampled_grid)
    
    resampled_fitted_grid, resampled_fit_con, resampled_fit_parameters, resampled_initials_all, _, _ = fit_procedure(wavelengths, resampled_grid)
    
    fwhm_bounds_resample = fwhm_calculations(wavelengths, resampled_grid, resampled_fit_parameters[:, 1])
    
    blue, _, _= fwhm_equivalent_width_excess(wavelengths,
                                        resampled_grid,
                                        resampled_fitted_grid,
                                        resampled_fit_con,
                                        resampled_fit_parameters[:, 1],
                                        fwhm_bounds_resample,
                                        shift='blue'
                                        )
    red, _, _= fwhm_equivalent_width_excess(wavelengths,
                                       resampled_grid,
                                       resampled_fitted_grid,
                                       resampled_fit_con,
                                       resampled_fit_parameters[:, 1],
                                       fwhm_bounds_resample,
                                       shift='red'
                                       )
    
    resampled_blue_ew_excess = np.append(resampled_blue_ew_excess, blue)
    resampled_red_ew_excess = np.append(resampled_red_ew_excess, red)
    
#resampled grid shape is 150 * (run_number x number of wavelengths)
resampled_grids = resampled_grids.reshape(samples, len(grid), len(wavelengths))
resampled_blue_ew_excess = resampled_blue_ew_excess.reshape(samples, len(grid))
resampled_red_ew_excess = resampled_red_ew_excess.reshape(samples, len(grid))
blue_ew_excess_error = np.std(resampled_blue_ew_excess, axis=0)
red_ew_excess_error = np.std(resampled_red_ew_excess, axis=0)

if not grid_mixed:
    print("You aren't running a mixed grid. Run Step 7 and skip Step 8")
    
if grid_mixed and not script_ran_once:
    print('You are running a mixed grid. Be sure to run Step 7.')
    
if grid_mixed and script_ran_once:
    print("You've ran the mixed grid, run step 8 to mix the data.")
    
# %% TOOL
################################################################################
print('TOOL: ANIMATED PLOTTING')
################################################################################

%matplotlib qt

# Plotting a grid of data as an animated plot. 
i_want_to_view_the_fits = True # right now, not later! You impatient mortal!

if i_want_to_view_the_fits:
    def slider_update(val):
        ax.clear()
        ax.plot(wavelengths, grid[int(val)], label='Original Data', color='black') # y=dictionary
        ax.plot(wavelengths, fitted_grid[int(val)], label='Optimal Gaussian with Continuum', color='red')
        ax.plot(wavelengths, fit_con[int(val)], label='Fitted Continuum', color='blue')
        # ax.plot(wavelengths, 
        #         gaussian_with_continuum(wavelengths, *initials_all[int(val)]), 
        #         label='Initial Gaussian with Continuum', 
        #         color='grey'
        #         )
        ax.plot(wavelengths, sk_con_data[int(val)], label='Sklearn Fitted Continuum', color='green')
        sum_squared = sum((grid[int(val)]-fitted_grid[int(val)])**2)
        ax.scatter(wavelengths, fitted_grid[int(val)], color = 'red')
        ax.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha +22, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha -22, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Wavelength ($Å$)')
        ax.set_ylabel('Flux')
        ax.set_xlim(6385, 6735)
        ax.set_title(f'Run {run_number[int(val)]}: Gaussian and Continuum \n Sum Squared = {sum_squared:.2e}')
        ax.legend()
        fig.canvas.draw_idle()

    def animation_setting_new_slider_value(frame):
        if anim.running:
            if grid_slider.val == len(grid)-1:
                grid_slider.set_val(0)
            else:
                grid_slider.set_val(grid_slider.val + 1)
                
    def play_pause(event):
        if anim.running:
            anim.running = False
            slider_update(grid_slider.val)
        else:
            anim.running = True

    def left_button_func(_) -> None:
        anim.running = False
        grid_slider.set_val(grid_slider.val - 1)
        slider_update(grid_slider.val)

    def right_button_func(_) -> None:
        anim.running = False
        grid_slider.set_val(grid_slider.val + 1)
        slider_update(grid_slider.val)
        
    fig, ax = plt.subplots(figsize=(12, 8)) # Creating Figure
    plt.subplots_adjust(bottom=0.2)

    ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03]) # Run Slider
    grid_slider = Slider(ax_slider, 'Run', 0, len(grid), valinit=0, valstep=1) 
    grid_slider.on_changed(slider_update)

    ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05]) # Play/Pause Button
    play_pause_button = Button(ax_play_pause, '>||')
    play_pause_button.on_clicked(play_pause)

    ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05]) # Left Button
    left_button = Button(ax_left_button, '<')
    left_button.on_clicked(left_button_func)

    ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05]) # Right Button
    right_button = Button(ax_right_button, '>')
    right_button.on_clicked(right_button_func)

    anim = FuncAnimation(fig, 
                        animation_setting_new_slider_value,
                        frames=len(grid),
                        interval=400
                        ) # setting up animation
    anim.running = True # setting off animation
else:
    print('We skipped using this tool, set to True if incorrect.')

# %% TOOL (FWHM): Animated Plotting
################################################################################
print('TOOL: ANIMATED PLOTTING')
################################################################################

%matplotlib qt

# Plotting a grid of data as an animated plot. 
i_want_to_view_the_fits = True # right now, not later! You impatient mortal!

if i_want_to_view_the_fits:
    def slider_update(val):
        ax.clear()
        ax.plot(wavelengths, grid[int(val)], label='Original Data', color='black') # y=dictionary
        ax.plot(wavelengths, fitted_grid[int(val)], label='Optimal Gaussian with Continuum', color='red')
        ax.plot(wavelengths, fit_con[int(val)], label='Fitted Continuum', color='blue')
        # ax.plot(wavelengths, 
        #         gaussian_with_continuum(wavelengths, *initials_all[int(val)]), 
        #         label='Initial Gaussian with Continuum', 
        #         color='grey'
        #         )
        #ax.plot(wavelengths, sk_con_data[int(val)], label='Sklearn Fitted Continuum', color='green')
        sum_squared = sum((grid[int(val)]-fitted_grid[int(val)])**2)
        ax.scatter(wavelengths, fitted_grid[int(val)], color = 'red')
        #ax.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=blue_window[int(val), 0], color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=blue_window[int(val), 1], color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=red_window[int(val), 0], color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=red_window[int(val), 1], color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
        #ax.axvline(x=fwhm_bounds[int(val), 0], color='navy', linestyle='--', alpha=0.5)
        #ax.axvline(x=fwhm_bounds[int(val), 1], color='crimson', linestyle='--', alpha=0.5)
        ax.set_xlabel('Wavelength ($Å$)')
        ax.set_ylabel('Flux')
        ax.set_xlim(6520, 6600)
        ax.set_title(f'Run {run_numbers[int(val)]}: Gaussian and Continuum \n Sum Squared = {sum_squared:.2e}')
        ax.text(0.54, 0.82,f'blue_ew_excess: {blue_ew_excess[int(val)]:.2f} , red_ew_excess: {red_ew_excess[int(val)]:.2f}',transform=ax.transAxes,
            fontsize=15, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
            ha='right', 
            va='bottom')
        ax.legend()
        fig.canvas.draw_idle()

    def animation_setting_new_slider_value(frame):
        if anim.running:
            if grid_slider.val == len(grid)-1:
                grid_slider.set_val(0)
            else:
                grid_slider.set_val(grid_slider.val + 1)
                
    def play_pause(event):
        if anim.running:
            anim.running = False
            slider_update(grid_slider.val)
        else:
            anim.running = True

    def left_button_func(_) -> None:
        anim.running = False
        grid_slider.set_val(grid_slider.val - 1)
        slider_update(grid_slider.val)

    def right_button_func(_) -> None:
        anim.running = False
        grid_slider.set_val(grid_slider.val + 1)
        slider_update(grid_slider.val)
        
    fig, ax = plt.subplots(figsize=(12, 8)) # Creating Figure
    plt.subplots_adjust(bottom=0.2)

    ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03]) # Run Slider
    grid_slider = Slider(ax_slider, 'Run', 0, len(grid), valinit=0, valstep=1) 
    grid_slider.on_changed(slider_update)

    ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05]) # Play/Pause Button
    play_pause_button = Button(ax_play_pause, '>||')
    play_pause_button.on_clicked(play_pause)

    ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05]) # Left Button
    left_button = Button(ax_left_button, '<')
    left_button.on_clicked(left_button_func)

    ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05]) # Right Button
    right_button = Button(ax_right_button, '>')
    right_button.on_clicked(right_button_func)

    anim = FuncAnimation(fig, 
                        animation_setting_new_slider_value,
                        frames=len(grid),
                        interval=400
                        ) # setting up animation
    anim.running = True # setting off animation
else:
    print('We skipped using this tool, set to True if incorrect.')
    
# %% 7
################################################################################
print("STEP 7: ARE YOU RUNNING A MIXED WAVELENGTH RANGED GRID?")
################################################################################

# If the grid has different wavelength ranges from re-running particular PYTHON
# runs, then you need to run the script again with the new wavelength range.
# This code stores your first run's results. Only built for two iterations atm.
# Could be adapted for more iterations if needed. Not currently however.
# The tolist() is due to numpy arrays not liking being uneven in shape.

if grid_mixed and not script_ran_once:
    print('Now repeat Steps 2,3 and 6')
    stored_results = {'wavelength_grid': wavelength_grid.tolist(),
                        'grid': grid.tolist(),
                        'fitted_grid': fitted_grid.tolist(),
                        'fit_con': fit_con.tolist(),
                        'fit_parameters': fit_parameters.tolist(), 
                        'initials_all': initials_all.tolist(), 
                        #'pcov_all': pcov_all.tolist(), 
                        #'infodict_all': infodict_all.tolist(), 
                        'blue_ew_excess': blue_ew_excess, 
                        'red_ew_excess': red_ew_excess, 
                        'blue_ew_excess_error': blue_ew_excess_error.tolist(), 
                        'red_ew_excess_error': red_ew_excess_error.tolist(),
                        'sk_con_data': sk_con_data.tolist(),
                        'sk_slopes': sk_slopes.tolist(),
                        'sk_intercepts': sk_intercepts.tolist(),
                        'sk_noise_error': sk_noise_error.tolist(),
                        'run_number': run_number.tolist(),
                        'resampled_grids': resampled_grids.tolist(),
                        # 'fwhm_bounds': fwhm_bounds.tolist(), # FWHM Comment in/out
                        # 'blue_window': blue_window.tolist(),
                        # 'red_window': red_window.tolist(),
                        }

    script_ran_once = True
    
if not grid_mixed:
    final_results = {'wavelength_grid': wavelength_grid.tolist(),
                        'grid': grid.tolist(),
                        'fitted_grid': fitted_grid.tolist(),
                        'fit_con': fit_con.tolist(),
                        'fit_parameters': fit_parameters.tolist(),
                        'initials_all': initials_all.tolist(),
                        #'pcov_all': pcov_all.tolist(),
                        #'infodict_all': infodict_all.tolist(),
                        'blue_ew_excess': blue_ew_excess,
                        'red_ew_excess': red_ew_excess,
                        'blue_ew_excess_error': blue_ew_excess_error.tolist(),
                        'red_ew_excess_error': red_ew_excess_error.tolist(),
                        'sk_con_data': sk_con_data.tolist(),
                        'sk_slopes': sk_slopes.tolist(),
                        'sk_intercepts': sk_intercepts.tolist(),
                        'sk_noise_error': sk_noise_error.tolist(),
                        'run_number': grid_length.tolist(),
                        'resampled_grids': resampled_grids.tolist(),
                        # 'fwhm_bounds': fwhm_bounds.tolist(), # FWHM Comment in/out
                        # 'blue_window': blue_window.tolist(),
                        # 'red_window': red_window.tolist(),
                        }
    print("You can skip Step 8 and move onto Step 9")
 
# %% 8
################################################################################
print("STEP 8: MERGE MIXED GRID WITH THE REST OF THE GRID")
################################################################################  

# adding the stored mixed grid to the rest of the grid with the run_numbers 
# at the correct indexing location

if grid_mixed and script_ran_once:
    mixed_results = {'wavelength_grid': wavelength_grid.tolist(),
                        'grid': grid.tolist(),
                        'fitted_grid': fitted_grid.tolist(),
                        'fit_con': fit_con.tolist(),
                        'fit_parameters': fit_parameters.tolist(),
                        'initials_all': initials_all.tolist(),
                        #'pcov_all': pcov_all.tolist(),
                        #'infodict_all': infodict_all.tolist(),
                        'blue_ew_excess': blue_ew_excess,
                        'red_ew_excess': red_ew_excess,
                        'blue_ew_excess_error': blue_ew_excess_error.tolist(),
                        'red_ew_excess_error': red_ew_excess_error.tolist(),
                        'sk_con_data': sk_con_data.tolist(),
                        'sk_slopes': sk_slopes.tolist(),
                        'sk_intercepts': sk_intercepts.tolist(),
                        'sk_noise_error': sk_noise_error.tolist(),
                        'run_number': mixed_runs,
                        'resampled_grids': resampled_grids.tolist(), #(150, (685, 500))
                        # 'fwhm_bounds': fwhm_bounds.tolist(),
                        # 'blue_window': blue_window.tolist(),
                        # 'red_window': red_window.tolist(),
                        }
    
    # Combining any previous results together, use final_results dictionary
    # for code from here on out.
    final_results = {key : [] for key in mixed_results.keys()}                 

    for run in grid_length:
        if run in stored_results['run_number']:
            index = stored_results['run_number'].index(run)
            for key in final_results:
                if key != 'resampled_grids':
                    final_results[key].append(stored_results[key][index])
        elif run in mixed_runs:
            index = mixed_results['run_number'].index(run)
            for key in final_results:
                if key != 'resampled_grids':                
                    final_results[key].append(mixed_results[key][index])
            
# different structure for the resampled grids (samples, (run_number, wavelengths))
# it's horrendous to read/code, im sorry, but it works.
final_results['resampled_grids'] = [[] for _ in range(samples)]
for sample in range(samples):
    for run in grid_length:
        if run in stored_results['run_number']:
            index = stored_results['run_number'].index(run)
            final_results['resampled_grids'][sample].append(stored_results['resampled_grids'][sample][index])
        elif run in mixed_runs:
            index = mixed_results['run_number'].index(run)
            final_results['resampled_grids'][sample].append(mixed_results['resampled_grids'][sample][index])
            
# %% 9
################################################################################
print("STEP 9: PLOTTING THE EW EXCESSES FOR ALL RUNS AGAINST TEO'S DATA")
################################################################################

# EVERY RUN IS PLOTTED HERE FOR THE EQUIVALENT WIDTH EXCESSES, NO BAD FITS REMOVED
# YOU CAN POTENTIALLY SKIP THIS STEP IF YOU WANT TO REMOVE BAD FITS STRAIGHT AWAY.

%matplotlib inline
plt.style.use('science')

# loading Teo's data from csv files
bz_cam = np.loadtxt('Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Cuneo_2023_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Cuneo_2023_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

system_labels = ['MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr',
       'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr',
       'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'MV_Lyr', 'V425_Cas', 'V425_Cas',
       'V425_Cas', 'V425_Cas', 'V425_Cas', 'V425_Cas', 'V425_Cas',
       'V425_Cas', 'V425_Cas', 'V425_Cas', 'V425_Cas', 'V425_Cas',
       'V751_Cyg', 'V751_Cyg', 'V751_Cyg', 'V751_Cyg', 'V751_Cyg',
       'V751_Cyg', 'V751_Cyg', 'V751_Cyg', 'V751_Cyg', 'V751_Cyg',
       'V751_Cyg', 'V751_Cyg', 'V751_Cyg', 'V751_Cyg', 'V751_Cyg',
       'V751_Cyg', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam',
       'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam',
       'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam', 'BZ_Cam']
# Define a mapping from system name to colour
mapping = {
    'MV_Lyr': 'orange',
    'V425_Cas': 'green',
    'V751_Cyg': 'pink',
    'BZ_Cam': 'cyan'
}
# Create a new array of colours matching the system_labels
system_colours_extended = [mapping[label] for label in system_labels]

# plot concentric circles of a given radius
def circle(radius):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    return (a, b)

plt.figure(figsize=(7,7))
ranges = slice(0,62) 

plt.errorbar(
    final_results['red_ew_excess'][ranges],
    final_results['blue_ew_excess'][ranges], 
    xerr=final_results['red_ew_excess_error'][ranges], 
    yerr=final_results['blue_ew_excess_error'][ranges], 
    fmt='none', 
    ecolor='grey', 
    alpha=0.3,
    zorder=-1
)

target = plt.scatter(
    final_results['red_ew_excess'][ranges],
    final_results['blue_ew_excess'][ranges],
    color=system_colours_extended, #final_results['run_number'][ranges],
    s=15,
    label='Grid Data',
    #cmap='rainbow'
)

# Label final_results points
labels_final = [
    '1a','1b','1c','1d',
    '2a','2b','2c','2d',
    '3a','3b','3c','3d',
    '4a','4b','4c','4d'
]
for i, (x, y) in enumerate(zip(final_results['red_ew_excess'][ranges],
                               final_results['blue_ew_excess'][ranges])):
    if i < len(labels_final):
        plt.annotate(labels_final[i], (x, y), xytext=(3,3),
                     textcoords='offset points', fontsize=8)

# KDE
kde = KernelDensity(bandwidth=0.9, kernel='gaussian')
kde.fit(np.vstack([final_results['red_ew_excess'][ranges],
                   final_results['blue_ew_excess'][ranges]]).T)
xcenters = np.linspace(
    min(final_results['red_ew_excess'][ranges]) - 2,
    max(final_results['red_ew_excess'][ranges]) + 2,
    100
)
ycenters = np.linspace(
    min(final_results['blue_ew_excess'][ranges]) - 2,
    max(final_results['blue_ew_excess'][ranges]) + 2,
    100
)
X, Y = np.meshgrid(xcenters, ycenters)
xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

# Plotting mv_lyr data
plt.scatter(mv_lyr[:,0], mv_lyr[:,1], color='brown', s=20, marker='o', label='Cuneo mv_lyr', edgecolor='orange')
plt.scatter(bz_cam[:,0], bz_cam[:,1], color='navy', s=20, marker='o', label='Cueno bz_cam', edgecolor='cyan')
plt.scatter(v425_cas[:,0], v425_cas[:,1], color='lime', s=20, marker='o', label='Cuneo v425_cas', edgecolor='green')
plt.scatter(v751_cyg[:,0], v751_cyg[:,1], color='purple', s=20, marker='o', label='Cuneo v751_cyg', edgecolor='pink')
# mapping = {
#     'MV_Lyr': 'orange',
#     'V425_Cas': 'yellow',
#     'V751_Cyg': 'green',
#     'BZ_Cam': 'cyan'
# }
# Label mv_lyr points
labels_mv = [
    "1c", "1b", "1a", "1d",
    "2a", "2b", "2c", "2d",
    "3a", "3b", "3c", "3d",
    "4a", "4b", "4c", "4d"
]
labels_mv = [
    "1c", "1b", "1a", "1d",
    "4b", "4d", "2a", "2c",
    "4c", "2d", "2b", "4a",
    "3d", "3b", "3c", "3a"
]
for i, (mx, my) in enumerate(mv_lyr):
    if i < len(labels_mv):
        plt.annotate(labels_mv[i], (mx, my), xytext=(3,3),
                     textcoords='offset points', fontsize=8)

plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
linear_thrs = 0.1
plt.xscale('symlog', linthresh=linear_thrs)
plt.yscale('symlog', linthresh=linear_thrs)
plt.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1, label='symlog = 0.1')
plt.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
plt.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
plt.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
plt.xlabel('Red Wing EW Excess ($Å$)')
plt.ylabel('Blue Wing EW Excess ($Å$)')
plt.title('Red vs Blue Wing Excess')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.legend()
plt.show()

# %% 10
################################################################################
print('STEP 10: CUTTING POOR DATA FITS LIKE BIMODAL EMISSION LINES OR NO VISIBLE EMISSION LINES')
################################################################################

# if final results file exists
if os.path.exists(f'final_results_inc_col_{inclination_column}.npy'):
    final_results = np.load(f'final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()
else:
    print('not exist')

%matplotlib inline

# Finding the flat spectra with no emission lines
removed_runs = []
for num in grid_length: # run number ranges range(0,729)
    run = f'rerun{num}.spec'
    file = fnmatch.filter(files, run) # matching file with loop run number
    removed_runs.append(path_to_grid+str(file[0])) # files sorted in order of run number

# initialising lists to store flat and high error runs
cut_runs = []
cut_runs_2 = []

# finding the flat spectra with no emission lines
for run in range(len(removed_runs)):
    flux = np.loadtxt(removed_runs[run], usecols=inclination_column, skiprows=81)[::-1] # flux values for each file
    wavelength = np.loadtxt(removed_runs[run], usecols=1 , skiprows=81)[::-1] # wavelength values for each file
    gradient = final_results['sk_slopes'][run] # continuum gradient
    intercept = final_results['sk_intercepts'][run] # continuum intercept
    # converting continuum function to flux values at the right wavelengths
    rebased_con_fluxes = gradient*wavelength + intercept
    flux_without_continuum = flux - rebased_con_fluxes # flux without continuum
    cut_flux = flux_without_continuum[100:-100] # cutting the edges of the flux as unphysical
    cut_wavelength = wavelength[100:-100] # cutting the edges of the wavelength as unphysical
    if np.max(cut_flux) < 4e-15: # flux limit to determine if flat
        cut_runs.append(run)

# finding the high error runs
# creating histrograms of the FRMS's and RMS's from the original data,gaussian fits and continuum data
fig, ax = plt.subplots(1,5, figsize=(15,5))

def frms(data, fit, continuum) -> list:
    """Fractional RMS of the data, fit and continuum."""
    frms = []
    for run in range(len(data)):
        frac = [((data[run][i] - fit[run][i]) / continuum[run][i])**2 for i in range(len(data[run]))]
        summation = np.sum(frac)
        frms_value = np.sqrt(summation / len(data[0]))
        frms.append(frms_value)
    return frms

def rms(data, fit, continuum) -> list:
    """RMS of the data, fit and continuum."""
    rms = []
    for run in range(len(data)):
        frac = [(data[run][i] - fit[run][i])**2 for i in range(len(data[run]))]
        summation = np.sum(frac)
        term = np.sqrt(summation / len(data[0]))
        rms_value = term / np.mean(continuum[run])
        rms.append(rms_value)
    return rms

def chi_2(data, fit, noise) -> list:
    """Chi squared of the data, fit and noise."""
    chi_2 = []
    for run in range(len(data)):
        frac = [(data[run][i] - fit[run][i])**2 for i in range(len(data[run]))]
        summation = np.sum(frac)
        chi_2.append(summation / noise[run])
    return chi_2

def rss(data, fit) -> list:
    """Residual sum of squares of the data and fit."""
    rss = []
    for run in range(len(data)):
        frac = [(data[run][i] - fit[run][i])**2 for i in range(len(data[run]))]
        summation = np.sum(frac)
        rss.append(summation)
    return rss

def continuum_differences(gaussian_continuum, sklearn_continuum) -> list:
    """A check on the difference between fitted continuums. If gaussian quite
    different to the sklearn continuum, it'll likely be a bad fit which will
    lead to inaccurate EWs.

    Args:
        gaussian_continuum (list): Fitted Gaussian continuum component. 
        sklearn_continuum (list): Sklearn fitted continuum from the spectra wings.

    Returns:
        list: Aboslute differences between the two continuums.
    """
    diff = []
    for run in range(len(gaussian_continuum)):
        gauss = gaussian_continuum[run]
        sk = sklearn_continuum[run]

        #abs_diff = [np.abs(gauss[i] - sk[i]) for i in range(len(gauss))]
        #diff.append(np.sum(abs_diff))
        rel_diff = [np.abs((gauss[i] - sk[i]) / sk[i]) for i in range(len(gauss))]
        diff.append(np.sum(rel_diff))
    
    return diff

def double_peak_detection(fluxes, continuum) -> list:
    """A systematic method to identify the number of peaks with a spectrum. 
    Primarily important in detecting the bimodal emission lines.

    Args:
        fluxes (list): A spectrums flux values.
        continuum (list): The corressponding continuum fit values. 

    Returns:
        tuple and int: Scipy peak indexes and information, the number of peaks.
    """
    
    fluxes = np.array(fluxes)
    continuum = np.array(continuum)
    
    fluxes -= continuum
    peak = signal.find_peaks(fluxes, height=0.3*np.max(fluxes), distance=10, prominence=0.1*np.max(fluxes))
    number = len(peak[0])
    
    return peak, number

# test_runs = np.arange(0,729,1)
# #numbers = []
# for test_run in test_runs:
#     if test_run not in cut_runs:
#         test_peak, number_of_peaks = double_peak_detection(
#             final_results['grid'][test_run],
#             final_results['sk_con_data'][test_run])
#         print(f'Number of Peaks {test_run} : {number_of_peaks}')
#         numbers.append(number_of_peaks)
#         if number_of_peaks == 1:
#             test = np.array(final_results['grid'][test_run]) - np.array(final_results['sk_con_data'][test_run])
            

#             plt.plot(final_results['wavelength_grid'][test_run],
#                     test, 
#                     label='Original Data',
#                     color='black'
#                     )
#             for i in range(number_of_peaks):
#                 plt.vlines(x=final_results['wavelength_grid'][test_run][test_peak[0][i]],
#                         ymin=min(test),
#                         ymax=max(test),
#                         color='red',
#                         linestyle='--'
#                         )
#             plt.show()

# plt.hist(numbers, bins=50)
# plt.ylabel('Number of Occurences')
# plt.xlabel('Number of Detected Peaks')
# #plt.title(f'Inclination {incs[inclination_column]}°')
# plt.show()


# working out error data for each spectra and fit
frms_data = frms(final_results['grid'], final_results['fitted_grid'], final_results['fit_con'])
rms_data = rms(final_results['grid'], final_results['fitted_grid'], final_results['fit_con'])
chi_2_data = chi_2(final_results['grid'], final_results['fitted_grid'], final_results['sk_noise_error'])
rss_data = rss(final_results['grid'], final_results['fitted_grid'])
diff = continuum_differences(final_results['fit_con'], final_results['sk_con_data'])

cut_runs_3 = []
peak_colour_map = []
test_runs = np.arange(0,729,1)
for test_run in test_runs:
    test_peak, number_of_peaks = double_peak_detection(
        final_results['grid'][test_run],
        final_results['sk_con_data'][test_run])
    if number_of_peaks > 2: 
        cut_runs_3.append(test_run)
        peak_colour_map.append('pink')
    elif number_of_peaks == 2: 
        peak_colour_map.append('red')
    elif number_of_peaks == 1:
        peak_colour_map.append('black')

# Plotting each method and their histograms (error size vs frequency)
ax[0].hist(frms_data, bins=50, label='FRMS')
ax[0].set_xlabel('Fractional RMS')
ax[0].set_ylabel('Frequency')
ax[0].set_title(r'$FRMS = \sqrt{\frac{1}{N} \sum_i^N\left(\frac{y_{Data,i}-y_{fit,i}}{y_{con, i}}\right)^2}$')

ax[1].hist(rms_data, bins=50, label='RMS')
ax[1].set_xlabel('RMS')
ax[1].set_ylabel('Frequency')
ax[1].set_title(r'$RMS = \frac{\sqrt{\frac{1}{N} \sum_i^N\left(y_{Data,i}-y_{fit,i}\right)^2}}{\bar{y}_{con}}$')

ax[2].hist(chi_2_data, bins=50, label=r'$\chi^2$')
ax[2].set_xlabel(r'$\chi^2$')
ax[2].set_ylabel('Frequency')
ax[2].set_title(r'$\chi^2 = \frac{\sum_i^N\left(y_{Data,i}-y_{fit,i}\right)^2}{\sigma_i^2}$')

ax[3].hist(rss_data, bins=50, label='RSS')
ax[3].set_xlabel('RSS')
ax[3].set_ylabel('Frequency')
ax[3].set_title(r'$RSS = \sum_i^N\left(y_{Data,i}-y_{fit,i}\right)^2$') 

ax[4].hist(diff, bins=50, label='Continuum Differences')
ax[4].vlines(20, min(diff), max(diff), color='red', linestyle='--')
ax[4].set_xlabel('Continuum Differences')
ax[4].set_ylabel('Frequency')
ax[4].set_title('Continuum Differences')
plt.show()

# Plotting the error data as a function of run number (i.e which run is bad)
fig, ax = plt.subplots(1,5, figsize=(15,5))
ax[0].plot(frms_data, label='FRMS')
ax[0].set_xlabel('Run Number')
ax[0].set_ylabel('Fractional RMS')
ax[0].set_title(r'$FRMS = \sqrt{\frac{1}{N} \sum_i^N\left(\frac{y_{Data,i}-y_{fit,i}}{y_{con, i}}\right)^2}$')

ax[1].set_xlabel('Run Number')
ax[1].set_ylabel('RMS')
ax[1].plot(rms_data, label='RMS')
ax[1].set_title(r'$RMS = \frac{\sqrt{\frac{1}{N} \sum_i^N\left(y_{Data,i}-y_{fit,i}\right)^2}}{\bar{y}_{con}}$')

ax[2].plot(chi_2_data, label=r'$\chi^2$')
ax[2].set_xlabel('Run Number')
ax[2].set_ylabel(r'$\chi^2$')
ax[2].set_title(r'$\chi^2 = \frac{\sum_i^N\left(y_{Data,i}-y_{fit,i}\right)^2}{\sigma_i^2}$')

ax[3].plot(rss_data, label='RSS')
ax[3].set_xlabel('Run Number')
ax[3].set_ylabel('RSS')
ax[3].set_title(r'$RSS = \sum_i^N\left(y_{Data,i}-y_{fit,i}\right)^2$')

ax[4].plot(diff, label='Continuum Differences')
ax[4].set_xlabel('Run Number')
ax[4].set_ylabel('Continuum Differences')
ax[4].set_title('Continuum Differences')
plt.show()

# Plotting the frequency of the blue and red ew excess errors 
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].hist(final_results['red_ew_excess_error'], bins=50, label='Red EW Excess Error')
ax[0].set_xlabel('Red EW Excess Error')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Red EW Excess Error')

ax[1].hist(final_results['blue_ew_excess_error'], bins=50, label='Blue EW Excess Error')
ax[1].set_xlabel('Blue EW Excess Error')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Blue EW Excess Error')
plt.show()

# Determined if blue and red EW excess error greater than 0.5, too uncertain
for i in range(len(removed_runs)):
    if final_results['red_ew_excess_error'][i] > 0.5 or final_results['blue_ew_excess_error'][i] > 0.5:
        cut_runs_2.append(i)
    elif frms_data[i] > 5e-1 or rms_data[i] > 5e-1:
        cut_runs_2.append(i)

# Removing large differences in continuum fits
cut_runs_4 = []
for i in range(len(removed_runs)):
    if diff[i] >= 20:
        cut_runs_4.append(i)
        
cut_runs = np.append(cut_runs, cut_runs_2) # adding high error runs to flat runs
cut_runs = np.append(cut_runs, cut_runs_3) # adding bimodal emission lines to high error runs
cut_runs = np.append(cut_runs, cut_runs_4) # adding large continuum differences to bimodal emission lines
cut_runs = np.unique(cut_runs) # removing duplicates

# add to final results dictionary
final_results['cut_runs'] = cut_runs
final_results['peak_colour_map'] = peak_colour_map

# TODO Cut out runs where gaussian continuum very different to sklearn continuum



# %%
# store the results into an npy file
np.save(f'final_results_inc_col_{inclination_column}.npy', final_results)

# %%
################################################################################
print("STEP 10.5: INDIVIDUAL PLOTTING THE EW EXCESSES FOR ALL RUNS AGAINST TEO'S DATA")
################################################################################

# EVERY RUN IS PLOTTED HERE FOR THE EQUIVALENT WIDTH EXCESSES, NO BAD FITS REMOVED
# YOU CAN POTENTIALLY SKIP THIS STEP IF YOU WANT TO REMOVE BAD FITS STRAIGHT AWAY.

%matplotlib inline
cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#cut_sk_con_data = np.delete(final_results['sk_con_data'], cut_runs)
cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
cut_grid_length = np.delete(grid_length, cut_runs)


# TODO FIGURE 4
# loading Teo's data from csv files
bz_cam = np.loadtxt('Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Cuneo_2023_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Cuneo_2023_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# plot concentric circles of a given radius
def circle(radius):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    return (a, b)

# Create masks for single-peaked and double-peaked spectra
cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
single_peak_mask = cut_peak_colour_map == 'black'
double_peak_mask = cut_peak_colour_map == 'red'

plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 15})
ranges = slice(0,-1) # to select particular portions of the grid if desired
plt.errorbar(cut_red_ew_excess,
             cut_blue_ew_excess, 
             xerr=cut_red_ew_excess_error, 
             yerr=cut_blue_ew_excess_error, 
             fmt='none', 
             ecolor = 'grey', 
             alpha=0.5,
             zorder=-1
             ) # error bars for scatterplot below

# Plot double-peaked spectra
plt.scatter(cut_red_ew_excess[double_peak_mask],
            cut_blue_ew_excess[double_peak_mask],
            c='red',
            s=10,
            label='Double-peaked Spectra')

# Plot single-peaked spectra
plt.scatter(cut_red_ew_excess[single_peak_mask],
            cut_blue_ew_excess[single_peak_mask],
            c='black',
            s=10,
            label='Single-peaked Spectra')

kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
kde.fit(np.vstack([cut_red_ew_excess[double_peak_mask], cut_blue_ew_excess[double_peak_mask]]).T)

xcenters = np.linspace(min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2, 500)
ycenters = np.linspace(min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2, 500)
X, Y = np.meshgrid(xcenters, ycenters)
xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

contour_list = np.exp(np.arange(-3.0, 2.5, 0.5))
# plt.contourf(X, Y, Z, levels=contour_list, cmap='Grays', alpha=0.6)
# plt.contour(X, Y, Z, levels=contour_list, colors='red')

# plot all teos data as the same scatter plot
# plt.scatter(bz_cam[:,0], bz_cam[:,1], label='Cúneo et al. (2023)', color='blue', s=10, marker='o')
# plt.scatter(mv_lyr[:,0], mv_lyr[:,1], color='blue', s=10, marker='o')
# plt.scatter(v425_cas[:,0], v425_cas[:,1], color='blue', s=10, marker='o')
# plt.scatter(v751_cyg[:,0], v751_cyg[:,1], color='blue', s=10, marker='o')

#TODO Verify references WE have 20 or 45 degrees Binary.period(hr) 3.2
    # 1.	BZ Cam: 12°–40° (References: Ringwald & Naylor 1998; Honeycutt et al. 2013) Binary.period(hr) 3.685
    # 2.	V751 Cyg: <50° (References: Greiner et al. 1999; Patterson et al. 2001) Binary.period(hr) 3.467
    # 3.	MV Lyr: 10°–13° (Skillman et al. 1995); 7° ± 1° (Linnell et al. 2005) Binary.period(hr) 3.2
    # 4.	V425 Cas: 25° ± 9° (Shafter & Ulrich 1982; Ritter & Kolb 2003) Binary.period(hr) 3.6

# vertical and horizontal lines at 0 i.e axes
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)

# plot formatting
incs = [0 for i in range(10)] # to indent incs to the same column index as files
[incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

plt.xlabel('Red Wing EW Excess ($Å$)')
plt.ylabel('Blue Wing EW Excess ($Å$)')
plt.title(f'Red vs Blue Wing Excess at {incs[inclination_column]}° inclination')
# sigma clip the data to remove outliers
max_red = np.mean(cut_red_ew_excess) + 3*np.std(cut_red_ew_excess)
min_red = np.mean(cut_red_ew_excess) - 3*np.std(cut_red_ew_excess)
max_blue = np.mean(cut_blue_ew_excess) + 3*np.std(cut_blue_ew_excess)
min_blue = np.mean(cut_blue_ew_excess) - 3*np.std(cut_blue_ew_excess)

linear_thrs = 0.1
plt.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)
plt.xscale('symlog', linthresh=linear_thrs)
plt.yscale('symlog', linthresh=linear_thrs)
plt.xlim(-50,50)
plt.ylim(-50,50)

plt.legend(loc='upper left')
plt.show()

# %% 11
################################################################################
print('STEP 11: REPLOTTING THE EW EXCESSES WITHOUT THE CUT RUNS')
################################################################################

%matplotlib qt
#import sns

incs = [0 for i in range(10)] # to indent incs to the same column index as files
[incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

path_to_table = '../Grids/Wider_Ha_grid_spec_files/Grid_runs_full_table.txt'
ascii_table = np.genfromtxt(f'{path_to_table}',
                    delimiter='|',
                    skip_header=3,
                    skip_footer=1,
                    dtype=float
                    )

# removing nan column due to pretty table
ascii_table = np.delete(ascii_table, 0, 1) # array, index position, axis
parameter_table = np.delete(ascii_table, -1, 1)

sim_parameters = [r'$\dot{M}_{disk}$',
        r'$\dot{M}_{wind}$',
        r'$KWD.d$',
        r'$r_{exp}$',
        r'$acc_{length}$',
        r'$acc_{exp}$'
        ] # PYTHON Parameters
# remove the first column of parameter_table
parameter_table = np.delete(parameter_table, 0, 1)

# removing bad and flat fits from the data for the final plot of the EW excesses
# cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
# cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
# cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
# cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
# #cut_sk_con_data = np.delete(final_results['sk_con_data'], cut_runs)
# cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
# cut_grid_length = np.delete(grid_length, cut_runs)

# Loading Teo's data from csv files
bz_cam = np.loadtxt('Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Cuneo_2023_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Cuneo_2023_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# Animation for the EW excess plots showing the data for particular runs and 
# the fit to data to produce than run's EW excess.
def slider_update(val):
    """When the slide updates, this function is called which replots the graph."""
    # ax[1] is plotting the Gaussian and Continuum fit to the data
    ax[1].clear() # clear the axis to update the plot
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['grid'][int(val)], 
               label='Original Data', 
               color='black'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['fitted_grid'][int(val)],
               label='Optimal Gaussian with Continuum', 
               color='red'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['fit_con'][int(val)], 
               label='Fitted Continuum', 
               color='blue'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)],
               final_results['sk_con_data'][int(val)],
                label='Sklearn Fitted Continuum',
                color='green'
                )
    # plotting the mask lines where the equalivalent width is calculated
    ax[1].axvline(x=H_alpha-blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha-blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha+red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha+red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
    ax[1].set_xlabel('Wavelength ($Å$)') # plot formatting
    ax[1].set_ylabel('Flux')
    ax[1].legend()
    # Add a text box for fit stats\
    textstr = '\n'.join((
        f'FRMS = {frms_data[int(val)]:.2e}',
        f'RMS = {rms_data[int(val)]:.2e}',
        f'$chi^2$ = {chi_2_data[int(val)]:.2e}',
        f'RSS = {rss_data[int(val)]:.2e}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.9, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    paramstr = '\n'.join([f'{sim_parameters[i]} = {parameter_table[int(val),i]:.2e}' for i in range(6)])
    ax[1].text(0.9, 0.8, paramstr, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    # ax2 is plotting an individual EW excess point for a particular run
    # This changes with the animation
    if val in cut_grid_length: # the whole grid isn't plotted, so check first
        index = cut_grid_length.tolist().index(val) # run index as skipping missed plots
        ax2.clear() # clear the axis to update the plot
        ax2.set_ylim(ax[0].get_ylim()) # aligning the axes with original plot
        ax2.errorbar(cut_red_ew_excess[index],
                     cut_blue_ew_excess[index],
                     xerr=cut_red_ew_excess_error[index],
                     yerr=cut_blue_ew_excess_error[index],
                     fmt='o',
                     ecolor='black',
                     zorder=5,
                     c='black'
                     )
        ax2.legend([f'Run {val}'], loc='upper left')
        ax[1].set_title(f"✔︎ Run {int(val)} ✔︎ \n Gaussian and Continuum Fit to Data")
    else: 
        ax2.clear() # clear the axis to update the plot
        ax2.set_ylim(ax[0].get_ylim())
        ax[1].set_title(f'X Run {int(val)} X \n Gaussian and Continuum Fit to Data')
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(_):
    """Animation works by the frame updating the slider value which then 
    calls the slider_update function, which replots the graphs."""
    if anim.running:
        if grid_slider.val == len(final_results['grid'])-1: # if at the end of the grid
            grid_slider.set_val(0) # go back to the start
        else:
            grid_slider.set_val(grid_slider.val + 1) # else go to the next run
            
def play_pause(_):
    """A play/pause button to control the animation."""
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_):
    """A left button to go back one run."""
    anim.running = False
    if grid_slider.val == 0:
        grid_slider.set_val(len(final_results['grid'])-1)
    else:
        grid_slider.set_val(grid_slider.val - 1)
    slider_update(grid_slider.val)

def right_button_func(_):
    """A right button to go forward one run."""
    anim.running = False
    if grid_slider.val == len(final_results['grid'])-1:
        grid_slider.set_val(0)
    else:
        grid_slider.set_val(grid_slider.val + 1)
    slider_update(grid_slider.val)
    
fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Creating Figure
plt.subplots_adjust(left=0.1, bottom=0.25, wspace=0.2) # organising figure
ax2 = ax[0].twinx() # creating another axis (ax2) to highlight a particular run 

def init_plot():
    """Initialising the animation plot with the first run."""
    #ax[0] is plotting the EW excesses. This graph is static.
    ax[0].errorbar(cut_red_ew_excess,
                  cut_blue_ew_excess,
                  xerr=cut_red_ew_excess_error,
                  yerr=cut_blue_ew_excess_error, 
                  fmt='none', 
                  ecolor='grey', 
                  alpha=0.5, 
                  zorder=2
                   ) # error bars for scatterplot below
    target = ax[0].scatter(cut_red_ew_excess,
                        cut_blue_ew_excess, 
                        #c=cut_grid_length,
                        s=15,
                        label='Grid Data',
                        color='blue',
                        #cmap='rainbow', 
                        zorder=3
                        ) # scatter plot assigned as target for colour bar
    #plt.colorbar(target, label='Run Number', ax=ax[0], pad=0.1) # colour bar
    ax[0].scatter(bz_cam[:,0], bz_cam[:,1], label='BZ Cam', color='black', s=10, marker='v')
    ax[0].scatter(mv_lyr[:,0], mv_lyr[:,1], label='MV Lyr', color='black', s=10, marker='^')
    ax[0].scatter(v425_cas[:,0], v425_cas[:,1], label='V425 Cas', color='black', s=10, marker='<')
    ax[0].scatter(v751_cyg[:,0], v751_cyg[:,1], label='V751 Cyg', color='black', s=10, marker='>')
    #contour plot a kernel density estimate of the data
    # # Create a 2D histogram with a density plot using matplotlib
    # kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    # kde.fit(np.vstack([cut_red_ew_excess, cut_blue_ew_excess]).T)
    
    # xcenters = np.linspace(min(cut_red_ew_excess), max(cut_red_ew_excess), 100)
    # ycenters = np.linspace(min(cut_blue_ew_excess), max(cut_blue_ew_excess), 100)
    # X, Y = np.meshgrid(xcenters, ycenters)
    # xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
    # Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)
    
    #ax[0].contourf(X, Y, Z, cmap='Blues', alpha=0.6)
    #ax[0].contour(X, Y, Z, colors='Blue')
    # ax[0].contourf(X, Y, Z, levels=contour_list, cmap='Grays', alpha=0.6)
    # ax[0].contour(X, Y, Z, levels=contour_list, colors='black')
    # hist, xedges, yedges = np.histogram2d(cut_red_ew_excess, cut_blue_ew_excess, bins=20, density=True)
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # X, Y = np.meshgrid(xcenters, ycenters)
    
    # Plot the density plot
    # ax[0].contourf(X, Y, hist.T, cmap='Blues', alpha=0.6)
    # ax[0].contour(X, Y, hist.T, colors='blue')
    # plot formatting
    ax[0].set_xlabel('Red Wing EW Excess ($Å$)')
    ax[0].set_ylabel('Blue Wing EW Excess ($Å$)')
    #plot y=x line
    ax[0].plot(np.linspace(-10,10,100), np.linspace(-10,10,100), color='black', linestyle='--', alpha=0.5)
    #plot y=-x line
    ax[0].plot(np.linspace(-10,10,100), -np.linspace(-10,10,100), color='black', linestyle='--', alpha=0.5)
    
    
    ax[0].set_title(f'Inc {incs[inclination_column]}° \n Red vs Blue Wing EW Excesses')
    
    ax[0].axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
    ax[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
    
    xlim = min(cut_red_ew_excess)-2,max(cut_red_ew_excess)+2
    ylim = min(cut_blue_ew_excess)-2,max(cut_blue_ew_excess)+2
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.37, 1))

    # ax[1] is plotting the Gaussian and Continuum fit to the data
    ax[1].plot(final_results['wavelength_grid'][0],
               final_results['grid'][0], 
               label='Original Data',
               color='black'
               )
    ax[1].plot(final_results['wavelength_grid'][0], 
               final_results['fitted_grid'][0],
               label='Optimal Gaussian with Continuum',
               color='red'
               )
    ax[1].plot(final_results['wavelength_grid'][0], 
               final_results['fit_con'][0], 
               label='Fitted Continuum', 
               color='blue'
               )
    ax[1].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
    
    ax[1].set_xlabel('Wavelength ($Å$)')
    ax[1].set_ylabel('Flux')
    ax[1].set_title(f'Run 0: \n Gaussian and Continuum Fit to Data')
    ax[1].legend()
    
    # Add text to the top left of the ax[0] plot
    ax[0].text(xlim[0]-2.5, ylim[1]+0.8, 'Inverse\nP-Cygni?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[0]-2.5, ylim[0]-1.5, 'Broad Absorption\nWings?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[1]-1, ylim[1]+0.8, 'Broad Emission\nWings?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[1]-1, ylim[0]-1.5, 'P-Cygni?', fontsize=12, color='black', fontweight='bold')

# Adding Run Slider axis
ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03])
grid_slider = Slider(ax_slider, 'Run', 0, 728, valinit=0, valstep=1)
grid_slider.on_changed(slider_update)

# Adding Play/Pause Button axis
ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05])
play_pause_button = Button(ax_play_pause, '>||')
play_pause_button.on_clicked(play_pause)

# Adding Left Button axis
ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05])
left_button = Button(ax_left_button, '<')
left_button.on_clicked(left_button_func)

# Adding Right Button axis
ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05])
right_button = Button(ax_right_button, '>')
right_button.on_clicked(right_button_func)

# Setting up the animation
init_plot() # initialising the plot
anim = FuncAnimation(fig, 
                     animation_setting_new_slider_value,
                     frames=len(final_results['grid']),
                     interval=400
                     ) # setting up animation
anim.running = True # setting off animation

# %% 11 (FWHM)
################################################################################
print('STEP 11: REPLOTTING THE EW EXCESSES WITHOUT THE CUT RUNS FWHM MASK')
################################################################################

%matplotlib qt
#import sns

incs = [0 for i in range(10)] # to indent incs to the same column index as files
[incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

path_to_table = '../Grids/Wider_Ha_grid_spec_files/Grid_runs_full_table.txt'
ascii_table = np.genfromtxt(f'{path_to_table}',
                    delimiter='|',
                    skip_header=3,
                    skip_footer=1,
                    dtype=float
                    )

# removing nan column due to pretty table
ascii_table = np.delete(ascii_table, 0, 1) # array, index position, axis
parameter_table = np.delete(ascii_table, -1, 1)

sim_parameters = [r'$\dot{M}_{disk}$',
        r'$\dot{M}_{wind}$',
        r'$KWD.d$',
        r'$r_{exp}$',
        r'$acc_{length}$',
        r'$acc_{exp}$'
        ] # PYTHON Parameters
# remove the first column of parameter_table
parameter_table = np.delete(parameter_table, 0, 1)

# removing bad and flat fits from the data for the final plot of the EW excesses
# cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
# cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
# cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
# cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
# #cut_sk_con_data = np.delete(final_results['sk_con_data'], cut_runs)
# cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
# cut_grid_length = np.delete(grid_length, cut_runs)

# Loading Teo's data from csv files
bz_cam = np.loadtxt('Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Cuneo_2023_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Cuneo_2023_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# Animation for the EW excess plots showing the data for particular runs and 
# the fit to data to produce than run's EW excess.
def slider_update(val):
    """When the slide updates, this function is called which replots the graph."""
    # ax[1] is plotting the Gaussian and Continuum fit to the data
    ax[1].clear() # clear the axis to update the plot
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['grid'][int(val)], 
               label='Original Data', 
               color='black'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['fitted_grid'][int(val)],
               label='Optimal Gaussian with Continuum', 
               color='red'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['fit_con'][int(val)], 
               label='Fitted Continuum', 
               color='blue'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)],
               final_results['sk_con_data'][int(val)],
                label='Sklearn Fitted Continuum',
                color='green'
                )
    # plotting the mask lines where the equalivalent width is calculated
    ax[1].axvline(x=final_results['blue_window'][int(val)][0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['blue_window'][int(val)][1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][int(val)][0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][int(val)][1], color='red', linestyle='--', alpha=0.5)
    ax[1].set_xlabel('Wavelength ($Å$)') # plot formatting
    ax[1].set_ylabel('Flux')
    ax[1].legend()
    # Add a text box for fit stats\
    textstr = '\n'.join((
        f'FRMS = {frms_data[int(val)]:.2e}',
        f'RMS = {rms_data[int(val)]:.2e}',
        f'$chi^2$ = {chi_2_data[int(val)]:.2e}',
        f'RSS = {rss_data[int(val)]:.2e}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.9, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    paramstr = '\n'.join([f'{sim_parameters[i]} = {parameter_table[int(val),i]:.2e}' for i in range(6)])
    ax[1].text(0.9, 0.8, paramstr, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    # ax2 is plotting an individual EW excess point for a particular run
    # This changes with the animation
    if val in cut_grid_length: # the whole grid isn't plotted, so check first
        index = cut_grid_length.tolist().index(val) # run index as skipping missed plots
        ax2.clear() # clear the axis to update the plot
        #ax2.set_ylim(ax[0].get_ylim()) # aligning the axes with original plot
        ax2.set_xlim(-50,50)
        ax2.set_ylim(-50,50)
        ax2.set_xscale('symlog', linthresh=linear_thrs)
        ax2.set_yscale('symlog', linthresh=linear_thrs)
        ax2.errorbar(cut_red_ew_excess[index],
                     cut_blue_ew_excess[index],
                     xerr=cut_red_ew_excess_error[index],
                     yerr=cut_blue_ew_excess_error[index],
                     fmt='o',
                     ecolor='black',
                     zorder=5,
                     c='black'
                     )
        ax2.legend([f'Run {val}'], loc='upper left')
        ax[1].set_title(f"✔︎ Run {int(val)} ✔︎ \n Gaussian and Continuum Fit to Data")
    else: 
        ax2.clear() # clear the axis to update the plot
        #ax2.set_ylim(ax[0].get_ylim())
        ax2.set_xlim(-50,50)
        ax2.set_ylim(-50,50)
        ax2.set_xscale('symlog', linthresh=linear_thrs)
        ax2.set_yscale('symlog', linthresh=linear_thrs)
        ax[1].set_title(f'X Run {int(val)} X \n Gaussian and Continuum Fit to Data')
        # Adding a red faded background to the plot
        ax[1].axvspan(6400, 6700, color='red', alpha=0.3)
        
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(_):
    """Animation works by the frame updating the slider value which then 
    calls the slider_update function, which replots the graphs."""
    if anim.running:
        if grid_slider.val == len(final_results['grid'])-1: # if at the end of the grid
            grid_slider.set_val(0) # go back to the start
        else:
            grid_slider.set_val(grid_slider.val + 1) # else go to the next run
            
def play_pause(_):
    """A play/pause button to control the animation."""
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_):
    """A left button to go back one run."""
    anim.running = False
    if grid_slider.val == 0:
        grid_slider.set_val(len(final_results['grid'])-1)
    else:
        grid_slider.set_val(grid_slider.val - 1)
    slider_update(grid_slider.val)

def right_button_func(_):
    """A right button to go forward one run."""
    anim.running = False
    if grid_slider.val == len(final_results['grid'])-1:
        grid_slider.set_val(0)
    else:
        grid_slider.set_val(grid_slider.val + 1)
    slider_update(grid_slider.val)
    
fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Creating Figure
plt.subplots_adjust(left=0.1, bottom=0.25, wspace=0.2) # organising figure
ax2 = ax[0].twinx() # creating another axis (ax2) to highlight a particular run 

def init_plot():
    """Initialising the animation plot with the first run."""
    #ax[0] is plotting the EW excesses. This graph is static.
    ax[0].errorbar(cut_red_ew_excess,
                  cut_blue_ew_excess,
                  xerr=cut_red_ew_excess_error,
                  yerr=cut_blue_ew_excess_error, 
                  fmt='none', 
                  ecolor='grey', 
                  alpha=0.5, 
                  zorder=2
                   ) # error bars for scatterplot below
    target = ax[0].scatter(cut_red_ew_excess,
                        cut_blue_ew_excess, 
                        #c=cut_grid_length,
                        s=15,
                        label='Grid Data',
                        color='blue',
                        #cmap='rainbow', 
                        zorder=3
                        ) # scatter plot assigned as target for colour bar
    #plt.colorbar(target, label='Run Number', ax=ax[0], pad=0.1) # colour bar
    ax[0].scatter(bz_cam[:,0], bz_cam[:,1], label='BZ Cam', color='black', s=10, marker='v')
    ax[0].scatter(mv_lyr[:,0], mv_lyr[:,1], label='MV Lyr', color='black', s=10, marker='^')
    ax[0].scatter(v425_cas[:,0], v425_cas[:,1], label='V425 Cas', color='black', s=10, marker='<')
    ax[0].scatter(v751_cyg[:,0], v751_cyg[:,1], label='V751 Cyg', color='black', s=10, marker='>')
    #contour plot a kernel density estimate of the data
    # # Create a 2D histogram with a density plot using matplotlib
    # kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    # kde.fit(np.vstack([cut_red_ew_excess, cut_blue_ew_excess]).T)
    
    # xcenters = np.linspace(min(cut_red_ew_excess), max(cut_red_ew_excess), 100)
    # ycenters = np.linspace(min(cut_blue_ew_excess), max(cut_blue_ew_excess), 100)
    # X, Y = np.meshgrid(xcenters, ycenters)
    # xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
    # Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)
    
    #ax[0].contourf(X, Y, Z, cmap='Blues', alpha=0.6)
    #ax[0].contour(X, Y, Z, colors='Blue')
    # ax[0].contourf(X, Y, Z, levels=contour_list, cmap='Grays', alpha=0.6)
    # ax[0].contour(X, Y, Z, levels=contour_list, colors='black')
    # hist, xedges, yedges = np.histogram2d(cut_red_ew_excess, cut_blue_ew_excess, bins=20, density=True)
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # X, Y = np.meshgrid(xcenters, ycenters)
    
    # Plot the density plot
    # ax[0].contourf(X, Y, hist.T, cmap='Blues', alpha=0.6)
    # ax[0].contour(X, Y, hist.T, colors='blue')
    # plot formatting
    ax[0].set_xlabel('Red Wing EW Excess ($Å$)')
    ax[0].set_ylabel('Blue Wing EW Excess ($Å$)')
    #plot y=x line
    ax[0].plot(np.linspace(-10,10,100), np.linspace(-10,10,100), color='black', linestyle='--', alpha=0.5)
    #plot y=-x line
    ax[0].plot(np.linspace(-10,10,100), -np.linspace(-10,10,100), color='black', linestyle='--', alpha=0.5)
    
    
    ax[0].set_title(f'Inc {incs[inclination_column]}° \n Red vs Blue Wing EW Excesses')
    
    ax[0].axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
    ax[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
    
    xlim = min(cut_red_ew_excess)-2,max(cut_red_ew_excess)+2
    ylim = min(cut_blue_ew_excess)-2,max(cut_blue_ew_excess)+2
    ax[0].set_xlim(-50,50)
    ax[0].set_ylim(-50,50)

    linear_thrs = 0.1
    ax[0].plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)
    ax[0].set_xscale('symlog', linthresh=linear_thrs)
    ax[0].set_yscale('symlog', linthresh=linear_thrs)
    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.37, 1))

    # ax[1] is plotting the Gaussian and Continuum fit to the data
    ax[1].plot(final_results['wavelength_grid'][0],
               final_results['grid'][0], 
               label='Original Data',
               color='black'
               )
    ax[1].plot(final_results['wavelength_grid'][0], 
               final_results['fitted_grid'][0],
               label='Optimal Gaussian with Continuum',
               color='red'
               )
    ax[1].plot(final_results['wavelength_grid'][0], 
               final_results['fit_con'][0], 
               label='Fitted Continuum', 
               color='blue'
               )
    ax[1].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['blue_window'][0][0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['blue_window'][0][1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][0][0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][0][1], color='red', linestyle='--', alpha=0.5)
    
    ax[1].set_xlabel('Wavelength ($Å$)')
    ax[1].set_ylabel('Flux')
    ax[1].set_title(f'Run 0: \n Gaussian and Continuum Fit to Data')
    ax[1].legend()
    
    # Add text to the top left of the ax[0] plot
    ax[0].text(xlim[0]-2.5, ylim[1]+0.8, 'Inverse\nP-Cygni?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[0]-2.5, ylim[0]-1.5, 'Broad Absorption\nWings?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[1]-1, ylim[1]+0.8, 'Broad Emission\nWings?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[1]-1, ylim[0]-1.5, 'P-Cygni?', fontsize=12, color='black', fontweight='bold')

# Adding Run Slider axis
ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03])
grid_slider = Slider(ax_slider, 'Run', 0, 728, valinit=0, valstep=1)
grid_slider.on_changed(slider_update)

# Adding Play/Pause Button axis
ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05])
play_pause_button = Button(ax_play_pause, '>||')
play_pause_button.on_clicked(play_pause)

# Adding Left Button axis
ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05])
left_button = Button(ax_left_button, '<')
left_button.on_clicked(left_button_func)

# Adding Right Button axis
ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05])
right_button = Button(ax_right_button, '>')
right_button.on_clicked(right_button_func)

# Setting up the animation
init_plot() # initialising the plot
anim = FuncAnimation(fig, 
                     animation_setting_new_slider_value,
                     frames=len(final_results['grid']),
                     interval=400
                     ) # setting up animation
anim.running = True # setting off animation


# %% 11 (FWHM) Molly Grid
################################################################################
print('STEP 11: REPLOTTING THE EW EXCESSES WITHOUT THE CUT RUNS FWHM MASK')
################################################################################

%matplotlib qt
#import sns

# incs = [0 for i in range(10)] # to indent incs to the same column index as files
# [incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

#path_to_table = '../Grids/Wider_Ha_grid_spec_files/Grid_runs_full_table.txt'
# ascii_table = np.genfromtxt(f'{path_to_table}',
#                     delimiter='|',
#                     skip_header=3,
#                     skip_footer=1,
#                     dtype=float
#                     )

# removing nan column due to pretty table
# ascii_table = np.delete(ascii_table, 0, 1) # array, index position, axis
# parameter_table = np.delete(ascii_table, -1, 1)

# sim_parameters = [r'$\dot{M}_{disk}$',
#         r'$\dot{M}_{wind}$',
#         r'$KWD.d$',
#         r'$r_{exp}$',
#         r'$acc_{length}$',
#         r'$acc_{exp}$'
#         ] # PYTHON Parameters
# # remove the first column of parameter_table
# parameter_table = np.delete(parameter_table, 0, 1)

# removing bad and flat fits from the data for the final plot of the EW excesses
# cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
# cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
# cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
# cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
# #cut_sk_con_data = np.delete(final_results['sk_con_data'], cut_runs)
# cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
# cut_grid_length = np.delete(grid_length, cut_runs)

# Loading Teo's data from csv files
# bz_cam = np.loadtxt('Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
# mv_lyr = np.loadtxt('Cuneo_2023_data/MV Lyr.csv', delimiter=',')
# v425_cas = np.loadtxt('Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# Animation for the EW excess plots showing the data for particular runs and 
# the fit to data to produce than run's EW excess.
def slider_update(val):
    """When the slide updates, this function is called which replots the graph."""
    # ax[1] is plotting the Gaussian and Continuum fit to the data
    ax[1].clear() # clear the axis to update the plot
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['grid'][int(val)], 
               label='Original Data', 
               color='black'
               )
    ax[1].plot(final_results['wavelength_grid'][int(val)], 
               final_results['fitted_grid'][int(val)],
               label='Optimal Gaussian with Continuum', 
               color='red'
               )
    # ax[1].plot(final_results['wavelength_grid'][int(val)], 
    #            final_results['fit_con'][int(val)], 
    #            label='Fitted Continuum', 
    #            color='blue'
    #            )
    # ax[1].plot(final_results['wavelength_grid'][int(val)],
    #            final_results['sk_con_data'][int(val)],
    #             label='Sklearn Fitted Continuum',
    #             color='green'
    #             )
    # plotting the mask lines where the equalivalent width is calculated
    ax[1].axvline(x=final_results['blue_window'][int(val)][0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['blue_window'][int(val)][1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][int(val)][0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][int(val)][1], color='red', linestyle='--', alpha=0.5)
    ax[1].set_xlabel('Wavelength ($Å$)') # plot formatting
    ax[1].set_ylabel('Flux')
    ax[1].legend(loc='upper left')
    # Add a text box for fit stats\
    # textstr = '\n'.join((
    #     f'FRMS = {frms_data[int(val)]:.2e}',
    #     f'RMS = {rms_data[int(val)]:.2e}',
    #     f'$chi^2$ = {chi_2_data[int(val)]:.2e}',
    #     f'RSS = {rss_data[int(val)]:.2e}'))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax[1].text(0.9, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,
    #     verticalalignment='top', bbox=props)
    # paramstr = '\n'.join([f'{sim_parameters[i]} = {parameter_table[int(val),i]:.2e}' for i in range(6)])
    # ax[1].text(0.9, 0.8, paramstr, transform=ax[1].transAxes, fontsize=14,
    #     verticalalignment='top', bbox=props)
    
    # ax2 is plotting an individual EW excess point for a particular run
    # This changes with the animation
    if val in cut_grid_length: # the whole grid isn't plotted, so check first
        index = cut_grid_length.tolist().index(val) # run index as skipping missed plots
        ax2.clear() # clear the axis to update the plot
        ax2.set_ylim(ax[0].get_ylim()) # aligning the axes with original plot
        ax2.set_xlim(-50,50)
        ax2.set_ylim(-50,50)
        ax2.set_xscale('symlog', linthresh=linear_thrs)
        ax2.set_yscale('symlog', linthresh=linear_thrs)
        ax2.errorbar(cut_red_ew_excess[index],
                     cut_blue_ew_excess[index],
                     xerr=cut_red_ew_excess_error[index],
                     yerr=cut_blue_ew_excess_error[index],
                     fmt='o',
                     ecolor='black',
                     zorder=5,
                     c='black'
                     )
        ax2.legend([f'Run {val}'], loc='upper left')
        ax[1].set_title(f"✔︎ Run {int(val)} ✔︎ \n Gaussian and Continuum Fit to Data")
    else: 
        ax2.clear() # clear the axis to update the plot
        #ax2.set_ylim(ax[0].get_ylim())
        ax2.set_xlim(-50,50)
        ax2.set_ylim(-50,50)
        ax2.set_xscale('symlog', linthresh=linear_thrs)
        ax2.set_yscale('symlog', linthresh=linear_thrs)
        ax[1].set_title(f'X Run {int(val)} X \n Gaussian and Continuum Fit to Data')
        # Adding a red faded background to the plot
        ax[1].axvspan(6400, 6700, color='red', alpha=0.3)
        
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(_):
    """Animation works by the frame updating the slider value which then 
    calls the slider_update function, which replots the graphs."""
    if anim.running:
        if grid_slider.val == len(final_results['grid'])-1: # if at the end of the grid
            grid_slider.set_val(0) # go back to the start
        else:
            grid_slider.set_val(grid_slider.val + 1) # else go to the next run
            
def play_pause(_):
    """A play/pause button to control the animation."""
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_):
    """A left button to go back one run."""
    anim.running = False
    if grid_slider.val == 0:
        grid_slider.set_val(len(final_results['grid'])-1)
    else:
        grid_slider.set_val(grid_slider.val - 1)
    slider_update(grid_slider.val)

def right_button_func(_):
    """A right button to go forward one run."""
    anim.running = False
    if grid_slider.val == len(final_results['grid'])-1:
        grid_slider.set_val(0)
    else:
        grid_slider.set_val(grid_slider.val + 1)
    slider_update(grid_slider.val)
    
fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Creating Figure
plt.subplots_adjust(left=0.1, bottom=0.25, wspace=0.2) # organising figure
ax2 = ax[0].twinx() # creating another axis (ax2) to highlight a particular run 

def init_plot():
    """Initialising the animation plot with the first run."""
    #ax[0] is plotting the EW excesses. This graph is static.
    ax[0].errorbar(cut_red_ew_excess,
                  cut_blue_ew_excess,
                  xerr=cut_red_ew_excess_error,
                  yerr=cut_blue_ew_excess_error, 
                  fmt='none', 
                  ecolor='grey', 
                  alpha=0.5, 
                  zorder=2
                   ) # error bars for scatterplot below
    target = ax[0].scatter(cut_red_ew_excess,
                        cut_blue_ew_excess, 
                        #c=cut_grid_length,
                        s=15,
                        label='Grid Data',
                        color='blue',
                        #cmap='rainbow', 
                        zorder=3
                        ) # scatter plot assigned as target for colour bar
    #plt.colorbar(target, label='Run Number', ax=ax[0], pad=0.1) # colour bar
    ax[0].scatter(bz_cam[:,0], bz_cam[:,1], label='BZ Cam', color='black', s=10, marker='v')
    ax[0].scatter(mv_lyr[:,0], mv_lyr[:,1], label='MV Lyr', color='black', s=10, marker='^')
    ax[0].scatter(v425_cas[:,0], v425_cas[:,1], label='V425 Cas', color='black', s=10, marker='<')
    ax[0].scatter(v751_cyg[:,0], v751_cyg[:,1], label='V751 Cyg', color='black', s=10, marker='>')
    #contour plot a kernel density estimate of the data
    # # Create a 2D histogram with a density plot using matplotlib
    # kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    # kde.fit(np.vstack([cut_red_ew_excess, cut_blue_ew_excess]).T)
    
    # xcenters = np.linspace(min(cut_red_ew_excess), max(cut_red_ew_excess), 100)
    # ycenters = np.linspace(min(cut_blue_ew_excess), max(cut_blue_ew_excess), 100)
    # X, Y = np.meshgrid(xcenters, ycenters)
    # xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
    # Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)
    
    #ax[0].contourf(X, Y, Z, cmap='Blues', alpha=0.6)
    #ax[0].contour(X, Y, Z, colors='Blue')
    # ax[0].contourf(X, Y, Z, levels=contour_list, cmap='Grays', alpha=0.6)
    # ax[0].contour(X, Y, Z, levels=contour_list, colors='black')
    # hist, xedges, yedges = np.histogram2d(cut_red_ew_excess, cut_blue_ew_excess, bins=20, density=True)
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # X, Y = np.meshgrid(xcenters, ycenters)
    
    # Plot the density plot
    # ax[0].contourf(X, Y, hist.T, cmap='Blues', alpha=0.6)
    # ax[0].contour(X, Y, hist.T, colors='blue')
    # plot formatting
    ax[0].set_xlabel('Red Wing EW Excess ($Å$)')
    ax[0].set_ylabel('Blue Wing EW Excess ($Å$)')
    #plot y=x line
    ax[0].plot(np.linspace(-10,10,100), np.linspace(-10,10,100), color='black', linestyle='--', alpha=0.5)
    #plot y=-x line
    ax[0].plot(np.linspace(-10,10,100), -np.linspace(-10,10,100), color='black', linestyle='--', alpha=0.5)
    
    
    ax[0].set_title(f'Red vs Blue Wing EW Excesses')
    
    ax[0].axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
    ax[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
    
    xlim = min(cut_red_ew_excess)-2,max(cut_red_ew_excess)+2
    ylim = min(cut_blue_ew_excess)-2,max(cut_blue_ew_excess)+2
    ax[0].set_xlim(-50,50)
    ax[0].set_ylim(-50,50)

    linear_thrs = 0.1
    ax[0].plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)
    ax[0].set_xscale('symlog', linthresh=linear_thrs)
    ax[0].set_yscale('symlog', linthresh=linear_thrs)
    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.37, 1))

    # ax[1] is plotting the Gaussian and Continuum fit to the data
    ax[1].plot(final_results['wavelength_grid'][0],
               final_results['grid'][0], 
               label='Original Data',
               color='black'
               )
    ax[1].plot(final_results['wavelength_grid'][0], 
               final_results['fitted_grid'][0],
               label='Optimal Gaussian with Continuum',
               color='red'
               )
    # ax[1].plot(final_results['wavelength_grid'][0], 
    #            final_results['fit_con'][0], 
    #            label='Fitted Continuum', 
    #            color='blue'
    #            )
    ax[1].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['blue_window'][0][0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['blue_window'][0][1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][0][0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=final_results['red_window'][0][1], color='red', linestyle='--', alpha=0.5)
    
    ax[1].set_xlabel('Wavelength ($Å$)')
    ax[1].set_ylabel('Flux')
    ax[1].set_title(f'Run 0: \n Gaussian and Continuum Fit to Data')
    ax[1].legend(loc='upper left')
    
    # Add text to the top left of the ax[0] plot
    ax[0].text(xlim[0]-2.5, ylim[1]+0.8, 'Inverse\nP-Cygni?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[0]-2.5, ylim[0]-1.5, 'Broad Absorption\nWings?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[1]-1, ylim[1]+0.8, 'Broad Emission\nWings?', fontsize=12, color='black', fontweight='bold')
    ax[0].text(xlim[1]-1, ylim[0]-1.5, 'P-Cygni?', fontsize=12, color='black', fontweight='bold')

# Adding Run Slider axis
ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03])
grid_slider = Slider(ax_slider, 'Run', 0, len(final_results['grid']), valinit=0, valstep=1)
grid_slider.on_changed(slider_update)

# Adding Play/Pause Button axis
ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05])
play_pause_button = Button(ax_play_pause, '>||')
play_pause_button.on_clicked(play_pause)

# Adding Left Button axis
ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05])
left_button = Button(ax_left_button, '<')
left_button.on_clicked(left_button_func)

# Adding Right Button axis
ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05])
right_button = Button(ax_right_button, '>')
right_button.on_clicked(right_button_func)

# Setting up the animation
init_plot() # initialising the plot
anim = FuncAnimation(fig, 
                     animation_setting_new_slider_value,
                     frames=len(final_results['grid']),
                     interval=400
                     ) # setting up animation
anim.running = True # setting off animation

# %%
import pandas as pd
single_peaks = pd.read_csv('Sorted_Wind_Parameters_Table.csv', delimiter=',')

# %% 12 TRENDS
################################################################################
print('STEP 12: FINDING TRENDS IN THE DATA')
################################################################################

# first row is column names
single_peaks = pd.read_csv('Sorted_Wind_Parameters_Table.csv', delimiter=',')
# Make a plots folder with todaydate and inc in the name
today = datetime.date.today()
today = today.strftime("%d-%m-%Y")
inc = incs[inclination_column]
folder_name = f'Plots/{today}_inc_{inc}'
scatterplots = f'{folder_name}/scatter_plots'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    os.makedirs(scatterplots)
    
# TODO Remove parameter dependencies. Something in trends section
# is changing grid_length which affects the PYTHON plotting tool
# %% Excess Cornerplot
################################################################################
print('TRENDS: EW EXCESSES VS SIMULATION PARAMETERS CORNERPLOT')
################################################################################
%matplotlib inline
# TODO plot scatter plots on a cornerplot figure

# importing the run parameter combinations
# Loading run files parameter combinations from pretty table file
path_to_table = '../Grids/Wider_Ha_grid_spec_files/Grid_runs_full_table.txt'
ascii_table = np.genfromtxt(f'{path_to_table}',
                    delimiter='|',
                    skip_header=3,
                    skip_footer=1,
                    dtype=float
                    )

# removing nan column due to pretty table
ascii_table = np.delete(ascii_table, 0, 1) # array, index position, axis
parameter_table = np.delete(ascii_table, -1, 1)

path2_to_table = '../Grids/Wider_Ha_grid_spec_files/Grid_runs_combinations.txt'
combination_table = np.genfromtxt(f'{path2_to_table}',
                    delimiter='|',
                    skip_header=3,
                    skip_footer=1,
                    dtype=float
                    )
combination_table = np.delete(combination_table, 0, 1) # array, index position, axis
combination_table = np.delete(combination_table, 0, 1) # array, index position, axis
combination_table = np.delete(combination_table, -1, 1) # array, index position, axis

sim_parameters = [r'$\dot{M}_{disk}$',
        r'$\dot{M}_{wind}$',
        r'$KWD.d$',
        r'$r_{exp}$',
        r'$acc_{length}$',
        r'$acc_{exp}$'
        ] # PYTHON Parameters

cut_parameter_table = np.delete(parameter_table, cut_runs, 0)
# plotting the blue excess widths as a cornerplot style plot comparing two 
# parameters at a time to see if there are any trends in the data. The larger the
# excess the higher alpha the scatter point.
cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)

def blue_3d():
    print('Generating Blue EW Excess 3D plot')
    fig, ax = plt.subplots(6, 6, figsize=(14, 14), subplot_kw=dict(projection="3d"))

    for run in range(len(cut_grid_length)):
        for i in range(6):
            for j in range(6):
                if i >= j:
                    if cut_blue_ew_excess[run] <= 0:
                        color = 'green'
                    else:
                        color = 'blue'
                    ax[i, j].scatter([cut_parameter_table[run][j+1]], 
                                    [cut_parameter_table[run][i+1]],
                                    np.abs(cut_blue_ew_excess[run]),
                                    color=color,
                                    #s=[np.abs(cut_blue_ew_excess[run])*100]
                                    alpha=1-(cut_blue_ew_excess_error[run]/0.5)
                                    )
                    
    #remove the top right quadrant of the plot
    for i in range(6):
        for j in range(6):
            if i < j:
                fig.delaxes(ax[i,j]) #ax[i, j].cla()# axis('off')
                
    # Remove axis labels that aren't on the outer edges of the corner plot
    for i in range(6):
        if i != 5:
            ax[i, 0].set_xticklabels([])
            ax[i, 0].set_xlabel('')
        ax[i, 0].set_ylabel(f'{sim_parameters[i]}')

    for j in range(6):
        if j != 0:
            ax[5, j].set_yticklabels([])
            ax[5, j].set_ylabel('')
        ax[5, j].set_xlabel(f'{sim_parameters[j]}')
        
    for i in range(1,5):
        for j in range(1,6):
                #ax[i, j].set_xticklabels([])
                #ax[i, j].set_yticklabels([])
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')
                
    for i in range(6):
            ax[i, i].set_zlabel('Excess ($Å$)')
                
    # Place text in the top right quadrant
    fig.text(0.5, 0.95, 'Blue Wing EW Excess vs Simulation Parameters', ha='center', fontsize=30)
    fig.text(0.8, 0.85, 'Blue = positive EW excess', ha='center', fontsize=30, color='blue')
    fig.text(0.8, 0.80, 'Green = negative EW excess', ha='center', fontsize=30, color='green')
    fig.text(0.8, 0.75, 'Z axis is magnitude of EW excess', ha='center', fontsize=30)
    fig.text(0.8, 0.65, 'The more faded the datapoint, \n the less confidence in the value', ha='center', fontsize=30)
    plt.show()

def red_2d():
    print('Generating Red EW Excess 2D plot')
    fig, ax = plt.subplots(6, 6, figsize=(15, 15))

    for run in range(len(cut_grid_length)):
        for i in range(6):
            for j in range(6):
                if i >= j:
                    if cut_red_ew_excess[run] <= 0:
                        color = 'green'
                    else:
                        color = 'red'
                    ax[i, j].scatter([cut_parameter_table[run][j+1]], 
                                    [cut_parameter_table[run][i+1]],
                                    color=color,
                                    s=[np.abs(cut_red_ew_excess[run])*100]
                                    #  alpha=0.5
                                    )
                    ax[i, j].set_xlabel(f'{sim_parameters[j]}')
                    ax[i, j].set_ylabel(f'{sim_parameters[i]}')

    for i in range(6):
        for j in range(6):
            if i < j:
                ax[i, j].axis('off')
                
    # Remove axis labels that aren't on the outer edges of the corner plot
    for i in range(6):
        if i != 5:
            ax[i, 0].set_xticklabels([])
            ax[i, 0].set_xlabel('')
        ax[i, 0].set_ylabel(f'{sim_parameters[i]}')

    for j in range(6):
        if j != 0:
            ax[5, j].set_yticklabels([])
            ax[5, j].set_ylabel('')
        ax[5, j].set_xlabel(f'{sim_parameters[j]}')
        
    for i in range(1,5):
        for j in range(1,6):
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')
            
    fig.text(0.5, 0.95, 'Red Wing EW Excess vs Simulation Parameters', ha='center', fontsize=30)
    fig.text(0.8, 0.85, 'Red = positive EW excess', ha='center', fontsize=30, color='red')
    fig.text(0.8, 0.80, 'Green = negative EW excess', ha='center', fontsize=30, color='green')
    fig.text(0.8, 0.75, 'Size of point = magnitude of EW excess', ha='center', fontsize=30)
    plt.show()

# use functions to show plots
blue_3d()
red_2d()

# %% Excess Corr Matrix
################################################################################
print('TRENDS: EW EXCESSES VS SIMULATION PARAMETERS CORRELATION MATRIX')
################################################################################
%matplotlib inline

# Add cut_blue and cut_red_ew_excesses to the parameter table
parameter_and_excess_data = np.column_stack((np.delete(cut_parameter_table, 0, 1),
                        cut_blue_ew_excess, 
                        cut_red_ew_excess
                        ))
feature_names = sim_parameters + ['Blue EW Excess', 'Red EW Excess']

# standardising the data and converting to a pandas dataframe
scaler = StandardScaler()
X = scaler.fit_transform(parameter_and_excess_data)
X = pd.DataFrame(X, columns=feature_names)

# only the red and blue EW excesses rows
corr = X.corr(method='spearman')
corr = corr.loc[['Blue EW Excess', 'Red EW Excess'], :] 

# plot the correlation matrix
fig, ax = plt.subplots()
im = ax.imshow(corr, cmap="bwr", vmin=-1, vmax=1) # matrix (-1/1 for colorbar)

# Set the colorbar horizontally above the matrix
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.3)
cbar.ax.set_xlabel("$r$", rotation=0)

ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7]) # column parameters
ax.set_xticklabels(list(feature_names), rotation=90)
ax.set_yticks([0, 1]) # only red, blue excess rows
ax.set_yticklabels(['Blue EW Excess', 'Red EW Excess'])
ax.grid(False) # hide the grid lines
ax.set_title(f"Spearman Correlation Matrix at {incs[inclination_column]}° inclination")

# Adding the correlation values to the squares
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig(f'{folder_name}/correlation_matrix.png', dpi=300)
plt.show()

display(corr) # display correlation matrix values in a table

# %% PCA
################################################################################
print('TRENDS: PCA ON THE DATA? (CAN SKIP)')
################################################################################
# TODO: Can I use PCA to see if there are any trends

# performing PCA with scaled parameter and excess data
pca = PCA()
pca.fit(X)

# creating a CDF like plot summing the variance explained to 1 with increasing number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of PCA')
plt.savefig(f'{folder_name}/PCA_explained_variance_ratio_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# %% Scatterplots
################################################################################
print('TRENDS: SCATTERPLOTS DISTRIBUTION OF THE DATA WITH A GIVEN PARAMETER')
################################################################################
%matplotlib inline

# For plt.savefig
saving_plot_names = ['Mdot_disk', 'Mdot_wind', 'KWD.d', 'r_exp', 'acc_length', 'acc_exp']

# Hard coded Wind mass loss rate figures to simplify the code (multiple*M_acc rate)
windlist = [[9e-11, 3e-10, 9e-10], [3e-10, 1e-9, 3e-9], [9e-10, 3e-9, 9e-9]]
windvalues = [0.03, 0.1, 0.3] # The wind mass loss rate multiples

for j in range(0,6):
    for i in range(0,3):
        plt.errorbar(cut_red_ew_excess,
                        cut_blue_ew_excess,
                    xerr=cut_red_ew_excess_error,
                    yerr=cut_blue_ew_excess_error,
                    fmt='yo', 
                    ecolor = 'grey', 
                    alpha=0.2,
                    zorder = 0
                    ) # error bars for scatterplot below
        
        #plotting red scatters of individual parameters
        chosen_parameter_row = i # changes value
        chosen_parameter_column = j # changes parameter 1 is dodgy atm 
        value = combination_table[chosen_parameter_row][chosen_parameter_column]
        for index, run in enumerate(cut_parameter_table):
            if chosen_parameter_column != 1:
                if f'{run[chosen_parameter_column+1]:.3e}' == f'{value:.3e}':
                    plt.scatter(cut_red_ew_excess[index], 
                                cut_blue_ew_excess[index], 
                                c='green', 
                                s=15, 
                                zorder = 1
                                )
                    
            elif chosen_parameter_column == 1: # TODO not plotting all available points yet
                # index where value is in windvalues
                wind_multiple = cut_parameter_table[index][1+1] / cut_parameter_table[index][0+1]
                if f'{wind_multiple:.3e}' == f'{float(value):.3e}':
                    plt.scatter(cut_red_ew_excess[index], 
                                cut_blue_ew_excess[index], 
                                c='green', 
                                s=15, 
                                zorder = 1
                                )

        # vertical and horizontal lines at 0 i.e axes
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)

        # plot formatting
        plt.xlabel('Red Wing EW Excess ($Å$)')
        plt.ylabel('Blue Wing EW Excess ($Å$)')
        plt.title('Red vs Blue Wing Excess')
        plt.xlim(min(cut_red_ew_excess)-2,max(cut_red_ew_excess)+2)
        plt.ylim(min(cut_blue_ew_excess)-2,max(cut_blue_ew_excess)+2)
        plt.title(f'highlighting {feature_names[chosen_parameter_column]} = {value}, inc = {incs[inclination_column]}°')
        plt.savefig(f'{folder_name}/scatter_plots/highlighting_{saving_plot_names[chosen_parameter_column]}_{value}_inc_{incs[inclination_column]}.png', dpi=300)
        plt.show()


# %% Excess 'weights' plot
################################################################################
print('TRENDS: VIEWING THE EXCESS VALUES IN A 3D PLOT OF 3 CHOSEN PARAMETERS')
################################################################################
%matplotlib qt

# !Code originally pulled from emulator code. Adapted for excess values instead 
# of PCA weights with the emulator. 

##### USER INPUT BELOW #####
desired_parameters = [1,2,4] # 0-5, pick 3, corresponds to parameter names
############################

latex_symbols = (f'Chosen parameters: {sim_parameters[desired_parameters[0]]}, {sim_parameters[desired_parameters[1]]}, {sim_parameters[desired_parameters[2]]}')
display(Latex(latex_symbols))

# Generating the data for the 3D plot
unique_points = combination_table.T # np.shape(6,3)
unique_combinations = np.delete(cut_parameter_table, 0, 1) # np.shape(268,6)
point_sizes = cut_blue_ew_excess # YOU CAN CHANGE THIS TO RED

# changing windmdot to be a multiple from combination data
for i in range(len(unique_combinations)):
    unique_combinations[i][1] = unique_combinations[i][1]/unique_combinations[i][0]

# finding which parameters weren't chosen by the user
fixed_parameters = [i for i in range(6) if i not in desired_parameters]

# finding all combinations which have the specified fixed parameter values
fixed_combinations = []
fixed_combination = [list(i) for i in unique_points[fixed_parameters]]
for i in itertools.product(*fixed_combination):
    fixed_combinations.append(list(i))
for i in range(len(fixed_combinations)):
    fixed_combinations[i] = [f'{fixed_combinations[i][j]:.3e}' for j in range(len(fixed_combinations[i]))]

# Animation for the EW excess plots showing the data for particular runs
def function_for_data(val,fixed_combinations, unique_combinations, fixed_parameters):
    # fitting the data where we plot 3 parameters with excess equilivalent 
    # widths as the sizes of the points. The fixed parameter values are set at 
    # by the slider value 0-27, iterating over the fixed_combinations list.
    fixed = fixed_combinations[val]
    x = []
    y = []
    z = []
    w = []
    c = []
    w2 = []
    for run in range(len(unique_combinations)):
        if float(unique_combinations[run][fixed_parameters[0]]) == float(fixed[0]) and float(unique_combinations[run][fixed_parameters[1]]) == float(fixed[1]) and float(unique_combinations[run][fixed_parameters[2]]) == float(fixed[2]):
            x.append(unique_combinations[run][desired_parameters[0]])
            y.append(unique_combinations[run][desired_parameters[1]])
            z.append(unique_combinations[run][desired_parameters[2]])
            w.append(point_sizes[run])
    
    for t in w:
        if t < 0:
            w2.append(abs(t)*500)
            c.append('red')
        else:
            w2.append(t*500)
            c.append('blue')
            
    return x, y, z, w, c, w2
        
    
def slider_update(val):
    """When the slide updates, this function is called which replots the graph."""
    ax.clear() # clear the axis to update the plot
    x,y,z,_,c,w2 = function_for_data(val,fixed_combinations, unique_combinations, fixed_parameters)
    ax.scatter(x, y, z, marker='o', color=c, s=w2)
    axis_info(val,w2)
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(_):
    """Animation works by the frame updating the slider value which then 
    calls the slider_update function, which replots the graphs."""
    if anim.running:
        if grid_slider.val == 26: # if at the end of the grid
            grid_slider.set_val(0) # go back to the start
        else:
            grid_slider.set_val(grid_slider.val + 1) # else go to the next run
            
def play_pause(_):
    """A play/pause button to control the animation."""
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_):
    """A left button to go back one run."""
    anim.running = False
    if grid_slider.val == 0:
        grid_slider.set_val(26)
    else:
        grid_slider.set_val(grid_slider.val - 1)
    slider_update(grid_slider.val)

def right_button_func(_):
    """A right button to go forward one run."""
    anim.running = False
    if grid_slider.val == 26:
        grid_slider.set_val(0)
    else:
        grid_slider.set_val(grid_slider.val + 1)
    slider_update(grid_slider.val)
    
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(projection='3d')

def axis_info(val,w2):
    """Setting up the axis for the plot."""
    ax.set_title('Blue Wing EW Excess vs Simulation Parameters')
    ax.set_xlabel(f'{feature_names[desired_parameters[0]]}')
    ax.set_ylabel(f'{feature_names[desired_parameters[1]]}')
    ax.set_zlabel(f'{feature_names[desired_parameters[2]]}')
    ax.set_xlim3d(unique_points[desired_parameters[0]][0], unique_points[desired_parameters[0]][-1])
    ax.set_ylim3d(unique_points[desired_parameters[1]][0], unique_points[desired_parameters[1]][-1])
    ax.set_zlim3d(unique_points[desired_parameters[2]][0], unique_points[desired_parameters[2]][-1])
    point = [Line2D([0], [0], label='manual point', marker='o', markersize=5, 
    linestyle='', color = colour) for colour in ['red', 'blue']]
    ax.legend(point, ['Negative', 'Positive'])
    textstr = '\n'.join((
        f'Fixed Parameters:',
        f'{feature_names[fixed_parameters[0]]} = {fixed_combinations[val][0]}',
        f'{feature_names[fixed_parameters[1]]} = {fixed_combinations[val][1]}',
        f'{feature_names[fixed_parameters[2]]} = {fixed_combinations[val][2]}',
        f'Number of data points: {len(w2)}'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax.text2D(1.05, 0.9, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
def init_plot():
    """Initialising the animation plot with the first run."""
    x,y,z,_,c,w2 = function_for_data(0,fixed_combinations, unique_combinations, fixed_parameters)
    ax.scatter(x, y, z, marker='o', color=c, s=w2)
    axis_info(0,w2)
    return # plot info here


# Adding Run Slider axis
ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03])
grid_slider = Slider(ax_slider, 'Combination', 0, 26, valinit=0, valstep=1)
grid_slider.on_changed(slider_update)

# Adding Play/Pause Button axis
ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05])
play_pause_button = Button(ax_play_pause, '>||')
play_pause_button.on_clicked(play_pause)

# Adding Left Button axis
ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05])
left_button = Button(ax_left_button, '<')
left_button.on_clicked(left_button_func)

# Adding Right Button axis
ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05])
right_button = Button(ax_right_button, '>')
right_button.on_clicked(right_button_func)

# Setting up the animation
init_plot() # initialising the plot
anim = FuncAnimation(fig, 
                     animation_setting_new_slider_value,
                     frames=27,
                     interval=700
                     ) # setting up animation
anim.running = True # setting off animation

# %% Excess B/R LR
################################################################################
print('TRENDS: LINEAR REGRESSION OF THE EW EXCESSES VS SIMULATION PARAMETERS')
################################################################################
%matplotlib inline
# Simple linear regression

#y = ax_1 + bx_2 + cx_3 + dx_4 + ex_5 + fx_6 + g
display(Math(r'y = ax_1 + bx_2 + cx_3 + dx_4 + ex_5 + fx_6 + g'))

# Redo-ing the label formatting for LaTeX equation outputs
sim_parameters = [r'\dot{M}_{disk}',
        r'\dot{M}_{wind}',
        r'KWD.d',
        r'r_{exp}',
        r'acc_{length}',
        r'acc_{exp}'
        ] # r for raw string to ignore escape sequences

# setting up the data cutting the first column 
X = cut_parameter_table[:,1:]

# only log10 the 0th, 1st, 2nd, and 4th columns. keep the rest the same
log_X = np.column_stack((np.log10(X[:,0]),
                         np.log10(X[:,1]),
                         np.log10(X[:,2]),
                         X[:,3],
                         np.log10(X[:,4]),
                         X[:,5]
                         ))

Y = np.column_stack((cut_blue_ew_excess, cut_red_ew_excess))

# setting up the linear regression model
model = LinearRegression()
model.fit(log_X, Y)

# printing the coefficients
print(f'R^2 score: {model.score(log_X, Y)}') # R^2 value

print('Linear Regression model for Blue_ew_excess') # in LaTeX style

# input variables into the string
eqn_blue = f'y = log({model.coef_[0][0]:.3f}{sim_parameters[0]}) + log({model.coef_[0][1]:.3f}{sim_parameters[1]}) + log({model.coef_[0][2]:.3f}{sim_parameters[2]}) + {model.coef_[0][3]:.3f}{sim_parameters[3]} + log({model.coef_[0][4]:.3f}{sim_parameters[4]}) + {model.coef_[0][5]:.3f}{sim_parameters[5]} + {model.intercept_[0]:.3f}$'
display(Math(f'{eqn_blue}')) # Latex formatted string to LaTeX output

print('Linear Regression model for Red_ew_excess') # in LaTeX style

# input variables into the string
eqn_red = f'y = log({model.coef_[1][0]:.3f}{sim_parameters[0]}) + log({model.coef_[1][1]:.3f}{sim_parameters[1]}) + log({model.coef_[1][2]:.3f}{sim_parameters[2]}) + {model.coef_[1][3]:.3f}{sim_parameters[3]} + log({model.coef_[1][4]:.3f}{sim_parameters[4]}) + {model.coef_[1][5]:.3f}{sim_parameters[5]} + {model.intercept_[1]:.3f}$'
display(Math(f'{eqn_red}')) # Latex formatted string to LaTeX output

y_pred_blue_excess = model.predict(log_X)[:,0]
y_pred_red_excess = model.predict(log_X)[:,1]

# Ordinary Least Squares (OLS) model blue
ols_X = sm.add_constant(log_X) # add an intercept value to the X parameters
ols = sm.OLS(Y[:,0], ols_X) # statsmodels OLS model
ols_result = ols.fit() # fitting the model
print(ols_result.summary()) # printing the summary of the model

# Ordinary Least Squares (OLS) model red
ols_X = sm.add_constant(log_X) # add an intercept value to the X parameters
ols = sm.OLS(Y[:,1], ols_X) # statsmodels OLS model
ols_result = ols.fit() # fitting the model
print(ols_result.summary()) # printing the summary of the model

# plotting the linear regression model
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(y_pred_blue_excess, Y[:,0], color='blue')
ax[0].plot([min(Y[:,0]), max(Y[:,0])], [min(Y[:,0]), max(Y[:,0])], color='black', linestyle='--')
ax[0].set_xlabel('Predicted Blue EW Excess')
ax[0].set_ylabel('Actual Blue EW Excess')
ax[0].set_title('Blue EW Excess Linear Regression Model')

ax[1].scatter(y_pred_red_excess, Y[:,1], color='red')
ax[1].plot([min(Y[:,1]), max(Y[:,1])], [min(Y[:,1]), max(Y[:,1])], color='black', linestyle='--')
ax[1].set_xlabel('Predicted Red EW Excess')
ax[1].set_ylabel('Actual Red EW Excess')
ax[1].set_title('Red EW Excess Linear Regression Model')

plt.tight_layout()
plt.savefig(f'{folder_name}/linear_regression_model_excesses.png', dpi=300)
plt.show()


# %% Calc EWs
################################################################################
print('TRENDS: CALCULATING EQUIVALENT WIDTHS/FWHM OF THE LINES')
################################################################################
%matplotlib inline


# At this point, excesses seem to be a waste of time. There isn't strong 
# regression relations, red/blue wing correlations and the diagnostic plots
# do not indicate any similarity to Teo's work. It's governed by noise and the 
# choice of masking profile over the emission line. We move onto working with
# simply the equivalent widths of a line and see if there are any trends.
# The equivalent widths are more robust as they don't require a fit to the 
# profile of the line with a Gaussian + continuum function. 

def equivalent_width(wavelengths, fluxes, continuum, colour) -> list:
    """Calculates the equivalent width of an emission line given the wavelengths,
    fluxes and continuum for a single spectrum
    Args:
        wavelengths (list): uneven array of wavelengths
        fluxes (list): uneven array of fluxes
        continuum (list): uneven array of continuum values
        colour (str): blue or red side of H_alpha line, or all
    Returns:
        equivalent_width (float): equivalent width of the line"""
    
    H_alpha = 6562.819
    
    # lists must be arrays for np.where
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    continuum = np.array(continuum)
    
    if colour == 'blue':
        peak_mask = (wavelengths < H_alpha)
    elif colour == 'red':
        peak_mask = (wavelengths > H_alpha)
    elif colour == 'all':
        peak_mask = np.array([True] * len(wavelengths))
    else:
        print('Please enter a valid colour: blue or red')
    
    #only use true values of wavelengths with the peak mask
    wavelengths = np.array(wavelengths[peak_mask])
    fluxes = np.array(fluxes[peak_mask])
    continuum = np.array(continuum[peak_mask])
        
    equivalent_width = np.trapz((fluxes/1)-1, wavelengths)
    return equivalent_width

# %%
def full_width_half_maximum(wavelengths, fluxes, continuum, double_peak=False) -> list:
    """Measuring the full width half maximum of emission lines. As there are double
    peaks, the outer half maximum is measured and the internal dips ignored. 
    TODO Implimented above, to be removed
    """
    H_alpha = 6562.819
    fwhm = 0
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    continuum = np.array(continuum)
    
    # remove trend from the continuum
    fluxes -= continuum
    peak = np.max(fluxes)
    
    # two different regimes if line indicates double peak or single peak
    if double_peak:
        pass
    else:
        pass
    
    return fwhm # angstroms


test_run = 470
test_data_blue = equivalent_width(final_results['wavelength_grid'][test_run],
                             final_results['grid'][test_run],
                             final_results['sk_con_data'][test_run], 
                             'blue')
test_fit_blue = equivalent_width(final_results['wavelength_grid'][test_run], 
                            final_results['fitted_grid'][test_run], 
                            final_results['fit_con'][test_run],
                            'blue')
print(f'EW_Data_blue = {test_data_blue:.3f}, EW_Fit_blue = {test_fit_blue:.3f}')
test_data_red = equivalent_width(final_results['wavelength_grid'][test_run],
                                final_results['grid'][test_run],
                                final_results['sk_con_data'][test_run], 
                                'red')
test_fit_red = equivalent_width(final_results['wavelength_grid'][test_run],
                                final_results['fitted_grid'][test_run],
                                final_results['fit_con'][test_run],
                                'red')
print(f'EW_Data_red = {test_data_red:.3f}, EW_Fit_red = {test_fit_red:.3f}')

plt.plot(final_results['wavelength_grid'][test_run],
         final_results['grid'][test_run], 
         label='Original Data',
         color='black'
         )
plt.plot(final_results['wavelength_grid'][test_run], 
         final_results['sk_con_data'][test_run],
         label='Sklearn Continuum',
         color='green'
         )
plt.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=(H_alpha-test_data_blue), color='blue', linestyle='--', alpha=0.5)
plt.axvline(x=(H_alpha+test_data_red), color='red', linestyle='--', alpha=0.5)
plt.xlabel('Wavelength ($Å$)')
plt.ylabel('Flux ($erg/s/cm^2/Å$) at 100.0 parsecs')
plt.title(f'Plotting a test example of the EW calculation for a PYTHON run \n run {test_run}')
plt.show()

plt.plot(final_results['wavelength_grid'][test_run],
         final_results['fitted_grid'][test_run],
         label='Optimal Gaussian with Continuum', 
         color='red'
         )
plt.plot(final_results['wavelength_grid'][test_run],
         final_results['fit_con'][test_run], 
         label='Fitted Continuum',
         color='blue'
         )
plt.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=(H_alpha-test_fit_blue), color='blue', linestyle='--', alpha=0.5)
plt.axvline(x=(H_alpha+test_fit_red), color='red', linestyle='--', alpha=0.5)
plt.title(f'Plotting a test example of the EW calculation for a model fit \n run {test_run}')
plt.xlabel('Wavelength ($Å$)')
plt.ylabel('Flux ($erg/s/cm^2/Å$) at 100.0 parsecs')
plt.show()

# collecting EWs throughout the grid
ew_data_blue = [equivalent_width(final_results['wavelength_grid'][run],
                                final_results['grid'][run],
                                final_results['sk_con_data'][run], 
                                'blue') for run in range(len(final_results['grid']))]
ew_data_red = [equivalent_width(final_results['wavelength_grid'][run],
                                final_results['grid'][run],
                                final_results['sk_con_data'][run], 
                                'red') for run in range(len(final_results['grid']))]
ew_fit_blue = [equivalent_width(final_results['wavelength_grid'][run],
                                final_results['fitted_grid'][run],
                                final_results['fit_con'][run], 
                                'blue') for run in range(len(final_results['grid']))]
ew_fit_red = [equivalent_width(final_results['wavelength_grid'][run],
                                final_results['fitted_grid'][run],
                                final_results['fit_con'][run], 
                                'red') for run in range(len(final_results['grid']))]

grid_length_copy = grid_length.copy()
cut_ew_data_blue = np.delete(ew_data_blue, cut_runs)
cut_ew_fit_blue = np.delete(ew_fit_blue, cut_runs)
cut_ew_data_red = np.delete(ew_data_red, cut_runs)
cut_ew_fit_red = np.delete(ew_fit_red, cut_runs)
cut_grid_length = np.delete(grid_length_copy, cut_runs)

# cutting out the negative ew values as it'll prompt warming everywhere when log
negative_runs_data = []
for i, _ in enumerate(ew_data_blue):
    if ew_data_blue[i] <= 0 or ew_data_red[i] <= 0:
        negative_runs_data.append(i)

negative_runs_fit = []
for i, _ in enumerate(ew_fit_blue):
    if ew_fit_blue[i] <= 0 or ew_fit_red[i] <= 0:
        negative_runs_fit.append(i)

negative_runs = sorted(list(set(negative_runs_data + negative_runs_fit)))

ew_data_blue = np.delete(ew_data_blue, negative_runs)
ew_fit_blue = np.delete(ew_fit_blue, negative_runs)
ew_data_red = np.delete(ew_data_red, negative_runs)
ew_fit_red = np.delete(ew_fit_red, negative_runs)
grid_length = np.delete(grid_length, negative_runs)

# Checking the cut run data is cutting poor EW fits. 
# plotting the blue EWs of the data and against the fit EW
x = np.linspace(min(np.log10(ew_fit_blue)), max(np.log10(ew_fit_blue)), 200)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
target = axs[0].scatter(np.log10(ew_fit_blue),
                        np.log10(ew_data_blue),
                        label='Blue EWs',
                        c=grid_length,
                        cmap='rainbow'
                        )
axs[0].plot(x, x, color='black', linestyle='--', alpha=0.5)
axs[0].set_title('All Spectral Runs Blue EWs')
axs[0].set_xlabel('Log(Fit EW)')
axs[0].set_ylabel('Log(Obs EW)')

# plotting the blue EWs of the cut data and against the fit EW
target2 = axs[1].scatter(np.log10(cut_ew_fit_blue),
                         np.log10(cut_ew_data_blue),
                         label='Blue EWs',
                         c=cut_grid_length,
                         cmap='rainbow'
                         )
axs[1].plot(x, x, color='black', linestyle='--', alpha=0.5)
axs[1].set_title('Only Spectral Run After Cut Blue EWs')
axs[1].set_xlabel('Log10(Fit EW)')
axs[1].set_ylabel('Log10(Obs EW)')
plt.colorbar(target, ax=axs[0], label='Grid Length')
plt.colorbar(target2, ax=axs[1], label='Grid Length')
plt.tight_layout()
plt.savefig(f'{folder_name}/blue_EWs_model_vs_data_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# Now for red EWs
# plotting the red EWs of the data and against the fit EW
x = np.linspace(min(np.log10(ew_fit_red)), max(np.log10(ew_fit_red)), 200)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
target3 = axs[0].scatter(np.log10(ew_fit_red),
                         np.log10(ew_data_red),
                         label='Red EWs',
                         c=grid_length,
                         cmap='rainbow'
                         )
axs[0].plot(x, x, color='black', linestyle='--', alpha=0.3)
axs[0].set_title('All Spectral Runs Red EWs')
axs[0].set_xlabel('Log(Fit EW)')
axs[0].set_ylabel('Log(Obs EW)')

# plotting the red EWs of the cut data and against the fit EW
target4 = axs[1].scatter(np.log10(cut_ew_fit_red),
                         np.log10(cut_ew_data_red),
                         label='Red EWs',
                         c=cut_grid_length,
                         cmap='rainbow'
                         )
axs[1].plot(x, x, color='black', linestyle='--', alpha=0.3)
axs[1].set_title('Only Spectral Run After Cut Red EWs')
axs[1].set_xlabel('Log10(Fit EW)')
axs[1].set_ylabel('Log10(Obs EW)')
plt.colorbar(target3, ax=axs[0], label='Grid Length')
plt.colorbar(target4, ax=axs[1], label='Grid Length')
plt.tight_layout()
plt.savefig(f'{folder_name}/red_EWs_model_vs_data_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# %% EW B vs R 
################################################################################
print('TRENDS: PLOTTING RED EW VS BLUE EW FOR A GIVEN INCLINATION')
################################################################################
%matplotlib inline

# This is to see if there is a strong correlation between the red and blue EWs
# As there is, there is unlikely a strong difference in the asymmetries. 

# for the y=x line
x = np.linspace(min(cut_ew_data_blue), max(cut_ew_data_blue), 200)

# Plot a diagnostic plot of the blue vs red EW in linear space
target = plt.scatter(cut_ew_data_red,
                     cut_ew_data_blue,
                     label='EWs',
                     c=cut_grid_length,
                     cmap='rainbow'
                     )
plt.colorbar(target, label='Grid Length')
plt.plot(x,x, color='black', linestyle='--', alpha=0.3, label='y=x')
plt.xlabel('Red EW')
plt.ylabel('Blue EW')
plt.title(f'Red vs Blue EWs for inclination {incs[inclination_column]}°')
plt.savefig(f'{folder_name}/red_vs_blue_EWs_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# Plot a diagnostic plot of the blue vs red EW in log space
target2 = plt.scatter(np.log10(ew_data_red),
                      np.log10(ew_data_blue),
                      label='EWs',
                      c=grid_length,
                      cmap='rainbow'
                      )
log_x = np.linspace(min(np.log10(ew_data_blue)), max(np.log10(ew_data_blue)), 200)
plt.colorbar(target2, label='Grid Length')
plt.plot(log_x,log_x, color='black', linestyle='--', alpha=0.3, label='y=x')
plt.xlabel('Log10(Red EW)')
plt.ylabel('Log10(Blue EW)')
plt.title(f'Log10 Red vs Blue EWs for inclination {incs[inclination_column]}°')
plt.savefig(f'{folder_name}/log_red_vs_blue_EWs_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# %% EW B/R LR
################################################################################
print('TRENDS: LINEAR REGRESSION OF THE EW PYTHON DATA VS SIMULATION PARAMETERS')
################################################################################
# TODO Resample to get errors upon coefficients
# Will need to refit the Sk_continuum for each for accurate fits. 

# Simple linear regression of EW vs simulation parameters

# setting up the data cutting the first column 
X = parameter_table[:,1:] # cut_parameter_table
X = np.delete(X, negative_runs, axis=0)

# if any Y element are <0 then set to 1e-50
log_Y = np.column_stack((np.log10(ew_data_blue), np.log10(ew_data_red)))

#find the NaN values in log_Y and store the index of the location
nan_values = np.argwhere(np.isnan(log_Y))
log_Y = np.delete(log_Y, nan_values, axis=0)

# only log10 the 0th, 1st, 2nd, and 4th columns. keep the rest the same
log_X = np.column_stack((np.log10(X[:,0]),
                         np.log10(X[:,1]),
                         np.log10(X[:,2]), 
                         X[:,3],
                         np.log10(X[:,4]),
                         X[:,5]
                         ))
log_X = np.delete(log_X, nan_values, axis=0)

# setting up the linear regression model
model = LinearRegression()
#model = Ridge(alpha=0.1)
model.fit(log_X, log_Y)

print(f'R^2 Score: \n {model.score(log_X, log_Y)}') # R^2 value

# print out my model equation from the coefficient and intercept 
display(Math(r'log(EW) = alog(p1) + blog(p2) + clog(p3) + d * p4 + elog(p5) + f * p6 + g'))

#log_combination_table = np.where(combination_table == 0, 1e-50, combination_table)
log_combination_table = np.column_stack((np.log10(combination_table[:,0]), 
                                         np.log10(combination_table[:,1]), 
                                         np.log10(combination_table[:,2]), 
                                         combination_table[:,3], 
                                         np.log10(combination_table[:,4]),
                                         combination_table[:,5]))
# print parameter names and combinations
print('-- The Combination Table of PYTHON parameters --')
table_labels = ['log10(mdot_disk)',
                'log10(mdot_wind)',
                'log10(kwd.d)',
                'r_exp',
                'log10(acc_length)',
                'acc_exp'
                ]
combi_table_df = pd.DataFrame(log_combination_table, columns=table_labels)
display(combi_table_df)

print('Linear Regression model for Blue EW python data') # in LaTeX style

# input variables into the string
eqn_blue = f'log(EW_blue) = {model.coef_[0][0]:.3f}log({sim_parameters[0]}) + {model.coef_[0][1]:.3f}log({sim_parameters[1]}) + {model.coef_[0][2]:.3f}log({sim_parameters[2]}) + {model.coef_[0][3]:.3f}{sim_parameters[3]} + {model.coef_[0][4]:.3f}log({sim_parameters[4]}) + {model.coef_[0][5]:.3f}{sim_parameters[5]} + {model.intercept_[0]:.3f}$'
display(Math(f'{eqn_blue}')) # Latex formatted string to LaTeX output

print('Linear Regression model for Red EW python data') # in LaTeX style

# input variables into the string
eqn_red = f'log(EW_red) = {model.coef_[1][0]:.3f}log({sim_parameters[0]}) + {model.coef_[1][1]:.3f}log({sim_parameters[1]}) + {model.coef_[1][2]:.3f}log({sim_parameters[2]}) + {model.coef_[1][3]:.3f}{sim_parameters[3]} + {model.coef_[1][4]:.3f}log({sim_parameters[4]}) + {model.coef_[1][5]:.3f}{sim_parameters[5]} + {model.intercept_[1]:.3f}$'
display(Math(f'{eqn_red}')) # Latex formatted string to LaTeX output


predicted_log_Y = model.predict(log_X) # predicting the log_Y values from model

# scatter plot comparing the predicted log_Y values to the actual log_Y values
# The closer to a y=y_pred line, the better the model. Indicates R^2 value

# plotting the linear regression model with colour map
fig,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(predicted_log_Y[:,0], log_Y[:,0], color='blue') # log_Y[:,0] <- column stack
ax[0].plot(log_Y, log_Y, label='y=y_pred') # prefect regression prediction line
ax[0].set_ylabel('Log10(EW Blue_Obs)')
ax[0].set_xlabel('Predicted Log10(EW Blue_Obs)')
ax[0].set_title('Linear regression model for blue EW python data')
ax[0].legend()

# same plot but colour maps to run numbers
temp = np.delete(grid_length, nan_values) # for colour bar
target = ax[1].scatter(predicted_log_Y[:,0], log_Y[:,0], c=temp, cmap='rainbow')
plt.colorbar(target, ax=ax[1], label='Grid Length')
ax[1].plot(log_Y, log_Y, label='y=y_pred') # prefect regression prediction line
ax[1].set_ylabel('Log10(EW Blue_Obs)')
ax[1].set_xlabel('Predicted Log10(EW Blue_Obs)')
ax[1].legend()
plt.savefig(f'{folder_name}/blue_EW_data_vs_predicted_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# if column stack blue and red seperate
fig2, ax2 = plt.subplots(1,2, figsize=(10,5))
ax2[0].scatter(predicted_log_Y[:,1], log_Y[:,1], color='red')
ax2[0].plot(log_Y, log_Y, label='y=y_pred')
ax2[0].set_ylabel('Log10(EW Red_Obs)')
ax2[0].set_xlabel('Predicted Log10(EW Red_Obs)')
ax2[0].set_title('Linear regression model for red EW python data')
ax2[0].legend()
# colour maps to run numbers
temp2 = np.delete(grid_length, nan_values)
target2 = ax2[1].scatter(predicted_log_Y[:,1], log_Y[:,1], c=temp2, cmap='rainbow')
plt.colorbar(target2, ax=ax2[1], label='Grid Length')
ax2[1].plot(log_Y, log_Y, label='y=y_pred')
ax2[1].set_ylabel('Log10(EW Red_Obs)')
ax2[1].set_xlabel('Predicted Log10(EW Red_Obs)')
ax2[1].legend()
plt.savefig(f'{folder_name}/red_EW_data_vs_predicted_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# %% Calc EW Errors
################################################################################
print('TRENDS: USING THE RESAMPLED DATA TO DETERMINE ERRORS ON THE EW FITS')
################################################################################

#log_Y_test = [ew_data_blue[i]+ew_data_red[i] for i,_ in enumerate(ew_data_red)] # differs from re-intergrating all wavelengths EW 
# use this below ew_data_all
ew_data_all = [equivalent_width(final_results['wavelength_grid'][run],
                                final_results['grid'][run],
                                final_results['sk_con_data'][run], 
                                'all') for run in range(len(final_results['grid']))]
ew_data_all_copy = ew_data_all.copy() # for use later in script
negative_runs_all = [i for i, val in enumerate(ew_data_all) if val <= 0]
ew_data_all = np.delete(ew_data_all, negative_runs_all)
grid_length_without_negative_runs = np.delete(np.arange(0,729), negative_runs_all)

# Resampling the data to determine errors on the equivalent widths
resampled_ew_data_all = []
for sample in tqdm(range(samples)):
    resampled_ew = [equivalent_width(final_results['wavelength_grid'][run],
                                              final_results['resampled_grids'][sample][run],
                                              final_results['sk_con_data'][run],
                                              'all') for run in range(len(final_results['grid']))
                                              ]
    resampled_ew_data_all.append(resampled_ew)

resampled_ew_data_all = np.array(resampled_ew_data_all)
ew_data_all_error = np.std(resampled_ew_data_all, axis=0)
ew_data_all_error_copy = ew_data_all_error.copy()
ew_data_all_error = np.delete(ew_data_all_error, negative_runs_all)
#grid_length_all = np.delete(grid_length, np.arange(0,728))

log_Y_all = np.log10(ew_data_all)
log_Y_all_error = (1/np.log(10))*(ew_data_all_error/ew_data_all)

# removing potentially negative EWs
# nan_values_all = np.argwhere(np.isnan(log_Y_all))
# log_Y_all = np.delete(log_Y_all, nan_values_all, axis=0)

ew = {'ew_data_all': ew_data_all_copy,
      'ew_data_all_error': ew_data_all_error_copy}
# save ew
np.save(f'cueno_ew_data_all_inc{inclination_column}.npy', ew)
# %% EW ALL LR
################################################################################
print('TRENDS: LINEAR REGRESSION OF THE WHOLE EW LINE VS SIMULATION PARAMETERS')
################################################################################

%matplotlib qt
# only log10 the 0th, 1st, 2nd, and 4th columns. keep the rest the same
X = parameter_table[:,1:] # cut_parameter_table
X_copy = X.copy() # for use later in script
X = np.delete(X, negative_runs_all, axis=0)
log_X_all = np.column_stack((np.log10(X[:,0]),
                             np.log10(X[:,1]),
                             np.log10(X[:,2]),
                             X[:,3],
                             np.log10(X[:,4]), 
                             X[:,5]
                             ))
# log_X_all = np.delete(log_X_all, nan_values_all, axis=0)

model_all = LinearRegression()
model_all.fit(log_X_all, log_Y_all)#, 1/log_Y_all_error) # 1/unc for weights from errors

print(f'R^2 Score: {model_all.score(log_X_all, log_Y_all)}')#, 1/log_Y_all_error)}')
print('Model fitting equation:')
display(Math(r'log(EW) = alog(p1) + blog(p2) + clog(p3) + d * p4 + elog(p5) + f * p6 + g'))

print('Linear Regression model for the entire EW line') # in LaTeX style

eqn_all = f'log(EW) = {model_all.coef_[0]:.3f}log({sim_parameters[0]}) + {model_all.coef_[1]:.3f}log({sim_parameters[1]}) + {model_all.coef_[2]:.3f}log({sim_parameters[2]}) + {model_all.coef_[3]:.3f}{sim_parameters[3]} + {model_all.coef_[4]:.3f}log({sim_parameters[4]}) + {model_all.coef_[5]:.3f}{sim_parameters[5]} + {model_all.intercept_:.3f}$'
display(Math(f'{eqn_all}')) # Latex formatted string to LaTeX output

predicted_log_Y_all = model_all.predict(log_X_all)

# Ordinary Least Squares (OLS) model
ols_X = sm.add_constant(log_X_all) # add an intercept value to the X parameters
ols = sm.OLS(log_Y_all, ols_X) # statsmodels OLS model
ols_result = ols.fit() # fitting the model
print(ols_result.summary()) # summary of the model (errors, coefficients, etc)

# Weighted Least Squares (WLS) model
wls = sm.WLS(log_Y_all, ols_X, weights=1/log_Y_all_error) # including y errors effect
wls_result = wls.fit() # fitting the model
print(wls_result.summary()) # summary of the model (errors, coefficients, etc)

# print the intercept and coefficient values ols
print(f'Intercept: {ols_result.params[0]}')
print(f'Coefficients: {ols_result.params[1:]}')
print(f'std_err: {ols_result.bse}')

predicted_log_Y_all_ols = ols_result.predict(ols_X)

# OLS error calculation using standard error of the coefficients
predicted_log_Y_all_ols_err = []
for i, _ in enumerate(ols_X):
    x_err_plus_c = [ols_result.bse[j]*abs(ols_X[i][j]) for j in range(len(ols_X[i]))]
    y_err = np.sqrt(np.sum(np.array(x_err_plus_c)**2))
    predicted_log_Y_all_ols_err.append(y_err)

predicted_log_Y_all_wls = wls_result.predict(ols_X)

# WLS error calculation using standard error of the coefficients
predicted_log_Y_all_wls_err = []
for i, _ in enumerate(ols_X):
    x_err_plus_c = [wls_result.bse[j]*abs(ols_X[i][j]) for j in range(len(ols_X[i]))]
    y_err = np.sqrt(np.sum(np.array(x_err_plus_c)**2))
    predicted_log_Y_all_wls_err.append(y_err)

fig, ax = plt.subplots(1,3, figsize=(15,5))
# SKlearn wls matches statsmodel wls
#ax[0].scatter(predicted_log_Y_all, log_Y_all, color='green', label = 'sklearn wls', alpha=0.5)
ax[0].errorbar(predicted_log_Y_all_ols, 
               log_Y_all,
               xerr=predicted_log_Y_all_ols_err,
               yerr=log_Y_all_error,
               fmt='none', 
               ecolor='grey',
               alpha=0.2,
               zorder=-1
)
ax[0].scatter(predicted_log_Y_all_ols, log_Y_all, color='red', label = 'statsmodels ols', alpha = 0.5)
ax[0].plot(log_Y_all, log_Y_all, label='y=y_pred')
ax[0].plot(log_Y_all+0.8, log_Y_all, label='y=y_pred-1', linestyle='--')
ax[0].plot(log_Y_all-0.8, log_Y_all, label='y=y_pred+1', linestyle='--')
ax[0].set_ylabel('Log10(EW All_Obs) -- TRUTH')
ax[0].set_xlabel('Predicted Log10(EW All_Obs)')
ax[0].set_title('OLS Linear regression model for \n entire line EW python data')
ax[0].set_xlim(-2,4)
ax[0].set_ylim(-4,3)
ax[0].legend()

ax[2].errorbar(predicted_log_Y_all_wls, 
               log_Y_all,
               xerr=predicted_log_Y_all_wls_err,
               yerr=log_Y_all_error,
               fmt='none', 
               ecolor='grey',
               alpha=0.2,
               zorder=-1
)
ax[2].scatter(predicted_log_Y_all_wls, log_Y_all, color='blue', label = 'statsmodels wls', alpha = 0.5)
ax[2].plot(log_Y_all, log_Y_all, label='y=y_pred')
ax[2].set_ylabel('Log10(EW All_Obs) -- TRUTH')
ax[2].set_xlabel('Predicted Log10(EW All_Obs)')
ax[2].set_title('WLS Linear regression model for \n entire line EW python data')
ax[2].set_xlim(-2,4)
ax[2].set_ylim(-4,3)
ax[2].legend()
# colour maps to run numbers
temp = np.delete(np.arange(0,729), negative_runs_all)
target = ax[1].scatter(predicted_log_Y_all, log_Y_all, c=temp, cmap='rainbow')
plt.colorbar(target, ax=ax[1], label='Grid Length')
ax[1].plot(log_Y_all, log_Y_all, label='y=y_pred')
ax[1].set_ylabel('Log10(EW All_Obs)')
ax[1].set_xlabel('Predicted Log10(EW All_Obs)')
ax[1].legend()
plt.savefig(f'{folder_name}/all_EW_data_vs_predicted_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

#cross validation
scores = cross_val_score(model_all, log_X_all, log_Y_all, cv=5, n_jobs=5)
print(f'Cross validated R^2 score: {scores}')

# TODO Difference between predicted and actual values correlates to EW_Excess?
#finding outliers from model
# parameter_table_copy = parameter_table.copy()
# parameter_table_without_negative_runs = np.delete(parameter_table_copy, negative_runs_all, axis=0)
count = []
for i in range(len(predicted_log_Y_all_ols)):
    if predicted_log_Y_all_ols[i] > log_Y_all[i]+0.8 or predicted_log_Y_all_ols[i] < log_Y_all[i]-0.8:
        lower_line = predicted_log_Y_all_ols[i] - 0.8
        upper_line = predicted_log_Y_all_ols[i] + 0.8
        if log_Y_all[i] > upper_line:
            count.append(grid_length_without_negative_runs[i])
        # if log_Y_all[i] < lower_line:
        #     #print(f'{log_Y_all[i]:.3f}, {predicted_log_Y_all_ols[i]:.3f}, {grid_length_without_negative_runs[i]}, {i}')
        #     count.append(grid_length_without_negative_runs[i]) # poor indexes

combination_list = [parameter_table[i] for i in count] # index to combination
combination_list = np.delete(combination_list, 0, axis=1) # remove first column
print(len(combination_list))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
%matplotlib qt

def slider_update(val):
    for i in range(6):
        ax[i//2, i%2].clear()
        ax[i//2, i%2].hist(combination_list[:,i], bins=50, color='orange')
        xlabel = sim_parameters[i]
        ax[i//2, i%2].set_xlabel(xlabel)
        ax[i//2, i%2].set_ylabel('Frequency')
        # highlight the bar on the plot for the specific combination val index
        index = count.index(int(val))
        ax[i//2, i%2].axvline(x=combination_list[int(index)][i], color='black', linestyle='--', alpha=0.5)
        # set ylimit to 50
        ax[i//2, i%2].set_ylim(0,len(count))

    fig.suptitle(f'Parameter Histograms for poor EW fits \n Combination {int(val)}')
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(frame):
    if anim.running:
        if grid_slider.val == max(count): # TODO size of grid
            grid_slider.set_val(count[0])
        else:
            next_index = count.index(int(grid_slider.val)) + 1
            grid_slider.set_val(count[next_index])
            
def play_pause(event):
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_) -> None:
    anim.running = False
    back_one = count.index(int(grid_slider.val)) - 1
    grid_slider.set_val(count[back_one])
    slider_update(grid_slider.val)

def right_button_func(_) -> None:
    anim.running = False
    forward_one = count.index(int(grid_slider.val)) + 1
    grid_slider.set_val(count[forward_one])
    slider_update(grid_slider.val)
    
fig, ax = plt.subplots(3,2, figsize=(10,10))
plt.subplots_adjust(bottom=0.2)
#fig.canvas.mpl_connect('button_press_event', slider_update)

ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03]) # Run Slider
grid_slider = Slider(ax_slider, 'Run', 0, max(count), valinit=count[0], valstep=count)
grid_slider.on_changed(slider_update)

ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05]) # Play/Pause Button
play_pause_button = Button(ax_play_pause, '>||')
play_pause_button.on_clicked(play_pause)

ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05]) # Left Button
left_button = Button(ax_left_button, '<')
left_button.on_clicked(left_button_func)

ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05]) # Right Button
right_button = Button(ax_right_button, '>')
right_button.on_clicked(right_button_func)

anim = FuncAnimation(fig, 
                     animation_setting_new_slider_value,
                     frames=len(combination_list),
                     interval=400
                     ) # setting up animation
anim.running = True # setting off animation

# %%
diff_pred_to_actual_logew = []
for i in range(len(predicted_log_Y_all_ols)):
    diff_pred_to_actual_logew.append(predicted_log_Y_all_ols[i] - log_Y_all[i])

temp2 = [final_results['red_ew_excess_error'][i]+final_results['blue_ew_excess_error'][i] for i in range(len(final_results['red_ew_excess_error']))]

temp = np.delete(temp2, negative_runs_all, axis=0)

# correlation matrix between temp and diff_pred_to_actual_logew
corr = np.corrcoef(temp, diff_pred_to_actual_logew)
print(f'Correlation between EW Excess and difference between predicted and actual EWs: {corr[0][1]}')


# %% NN LR
################################################################################
print('TRENDS: NEURAL NETWORK REGRESSION VS SIMULATION PARAMETERS')
################################################################################
%matplotlib inline
# To double check, the linear regression is work as expected and there is some
# improvements we could do. 

log_X_train, log_X_test, log_Y_train, log_Y_test = train_test_split(log_X_all, log_Y_all, test_size=0.2)

regr = MLPRegressor(hidden_layer_sizes=(50,50,50),
                    max_iter=5000,
                    activation='tanh',
                    ).fit(log_X_train, log_Y_train)
y_pred = regr.predict(log_X_test)

#calculating an R^2 score
print(f'R^2 Score: {regr.score(log_X_test, log_Y_test)}')
print(f'Number of Layers = {regr.n_layers_}, Number of Outputs = {regr.n_outputs_}')
print('Neural Network Settings:')
print(regr.get_params())

# plot the loss curve
plt.plot(regr.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve for Neural Network')
plt.savefig(f'{folder_name}/NN_loss_curve_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# plotting the predicted log_Y values against the actual log_Y values
plt.scatter(y_pred, log_Y_test, color='black')
plt.plot(log_Y_all, log_Y_all, label='y=y_pred')
plt.ylabel('Log10(EW All_Obs)')
plt.xlabel('Predicted Log10(EW All_Obs)')
plt.title('Neural Network regression model for entire line EW python data')
plt.legend(title=f'data_points = {len(log_Y_test)}')
plt.savefig(f'{folder_name}/all_EW_data_vs_predicted_NN_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

# cross validate the R^2 score
scores = cross_val_score(regr, log_X_all, log_Y_all, cv=5, n_jobs=5)
print(f'Cross validated R^2 score: {scores}')

# cross_val_predict - performs the regression train 5 times for the 5 different
# cross validation splits. This allows every data point to be a predicted EW
# and plots the whole data set minus negative EWs. 
predicted = cross_val_predict(regr, log_X_all, log_Y_all, cv=5, n_jobs=5)
plt.scatter(predicted, log_Y_all, color='black')
plt.plot(log_Y_all, log_Y_all, label='y=y_pred')
plt.ylabel('Log10(EW All_Obs)')
plt.xlabel('Predicted Log10(EW All_Obs)')
plt.title('Neural Network regression model for entire line EW python data')
plt.legend(title=f'data_points = {len(log_Y_all)}, cross validated')
plt.savefig(f'{folder_name}/all_EW_data_vs_predicted_NN_cross_val_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()


# %% 
seed = 1
model = KAN(width=[6,1,1], grid=3, k=3, seed=seed)

log_X_train, log_X_test, log_Y_train, log_Y_test = train_test_split(log_X_all, log_Y_all, test_size=0.2, random_state=42)

# covert to a torch object
log_X_train = torch.tensor(log_X_train, dtype=torch.float32)
log_Y_train = torch.tensor(log_Y_train, dtype=torch.float32)
log_X_test = torch.tensor(log_X_test, dtype=torch.float32)
log_Y_test = torch.tensor(log_Y_test, dtype=torch.float32)

print(log_X_train.shape, log_Y_train.shape)

dataset = {}
dataset['train_input'] = log_X_train
dataset['test_input'] = log_X_test
dataset['train_label'] = log_Y_train
dataset['test_label'] = log_Y_test

model(dataset['train_input'])
model.plot(beta=1)

model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
model.plot()

# %% EW ALL Corr Matrix
################################################################################
print('TRENDS: EW VS SIMULATION PARAMETERS CORRELATION MATRIX')
################################################################################
# Scaled or not seems to make no difference to the end result

X_cut_plus = np.delete(X_copy, negative_runs, axis=0)
ew_data_all_cut_plus = np.delete(ew_data_all_copy, negative_runs)
parameter_and_ew_data = np.column_stack((X_cut_plus, 
                                         ew_data_blue, 
                                         ew_data_red, 
                                         ew_data_all_cut_plus
                                         ))
feature_names = sim_parameters + ['Blue EW', 'Red EW', 'All EW']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(parameter_and_ew_data)
matrix_for_corr = pd.DataFrame(scaled_data, columns=feature_names)

corr = matrix_for_corr.corr(method='spearman')
corr = corr.loc[['Blue EW', 'Red EW', 'All EW'], :]

ax = plt.axes()
im = ax.imshow(corr, cmap='bwr', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(['Blue EW', 'Red EW', 'All EW'])
ax.grid(False)
plt.colorbar(im)
plt.title(f'Spearman Correlation Matrix of EWs at {incs[inclination_column]}°')
plt.savefig(f'{folder_name}/EW_corr_matrix_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

display(corr)

# %% LR Feature Ranking
################################################################################
print('TRENDS: FEATURE RANKING THE PARAMETERS FROM THE LINEAR REGRESSION MODEL')
################################################################################
# Feature ranking using Recursive Feature Elimination (RFE) to determine
# the most important features.This is done by recursively removing the least
# important features until the desired number of features is reached.

model = LinearRegression()
rfe = RFE(model, n_features_to_select=1)
rfe.fit(log_X_all, log_Y_all)
rfe_predicted = rfe.predict(log_X_all)

# only the selected features that are true
ranking = rfe.ranking_

# print the features in order of their ranking
print('Feature ranking for the parameters:')
for rank, feature in zip(ranking, sim_parameters):
    latex_style = f'Rank {rank}: {feature}'
    display(Math(latex_style))  # Latex formatted string to LaTeX output


# %% Tool Å to km/s
################################################################################
print('TOOL: Angstrom to km/s conversion and vice versa')
################################################################################
def angstrom_to_kms(wavelength):
    """Converts wavelength in angstroms from central h_alpha line to velocity in km/s.
    Args:
        wavelength (float): wavelength in angstroms"""
    kms = (wavelength - H_alpha) * 299792.458 / H_alpha
    return kms, print(f'{wavelength}Å = {kms}km/s')
    
def kms_to_angstrom(velocity):
    """Converts velocity in km/s to wavelength in angstroms from central h_alpha line.
    Args:
        velocity (float): velocity in km/s"""  
    angstrom = H_alpha * (velocity / 299792.458) + H_alpha
    return angstrom, print(f'{velocity}km/s = {angstrom}Å')

print(H_alpha)
a = kms_to_angstrom(1000) #22Å
kms_to_angstrom(4100) #54Å(2500) 88Å(4000)

print(H_alpha - kms_to_angstrom(1100)[0])
################################################################################
print('END OF CODE')
################################################################################
# %%
# %% FEATURE ENGINEERING EW ALL LR
################################################################################
print('TRENDS: LINEAR REGRESSION OF THE WHOLE EW LINE VS SIMULATION PARAMETERS')
################################################################################
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV

%matplotlib inline
# only log10 the 0th, 1st, 2nd, and 4th columns. keep the rest the same
X = parameter_table[:,1:] # cut_parameter_table
X_copy = X.copy() # for use later in script
# TODO fix the copies to not delete wrong things
log_Y_all_copy = log_Y_all.copy()
log_Y_all = log_Y_all_copy.copy()
log_Y_all_error_copy = log_Y_all_error.copy()
log_Y_all_error = log_Y_all_error_copy.copy()
X = np.delete(X, negative_runs_all, axis=0)
log_X_all = np.column_stack((#np.log10(X[:,0]), # can help reduce error
                             np.log10(X[:,1]),
                             np.log10(X[:,2]),
                             #X[:,3],
                             np.log10(X[:,4]),
                             #(X[:,5])
                             ))
# log_X_all = np.delete(log_X_all, nan_values_all, axis=0)

model_all = LinearRegression()
fixed_acc_exp_list = [] #TODO try fixing values to see is some are linear
for i, val in enumerate(X):
    if val[5] == 4.5:
        fixed_acc_exp_list.append(i)
log_X_all = np.delete(log_X_all, fixed_acc_exp_list, axis=0)
log_Y_all = np.delete(log_Y_all, fixed_acc_exp_list)
log_Y_all_error = np.delete(log_Y_all_error, fixed_acc_exp_list)

model_all.fit(log_X_all, log_Y_all)#, 1/log_Y_all_error) # 1/unc for weights from errors

print(f'R^2 Score: {model_all.score(log_X_all, log_Y_all)}')#, 1/log_Y_all_error)}')
print('Model fitting equation:')
display(Math(r'log(EW) = alog(p1) + blog(p2) + clog(p3) + d * p4 + elog(p5) + f * p6 + g'))
eqn = f'log(EW) = a{sim_parameters[0]} + b{sim_parameters[1]} + c{sim_parameters[2]} + d{sim_parameters[3]} + e{sim_parameters[4]} + f{sim_parameters[5]} + g'
display(Math(f'{eqn}'))

print('Linear Regression model for the entire EW line') # in LaTeX style

# eqn_all = f'log(EW) = {model_all.coef_[0]:.3f}log({sim_parameters[0]}) + {model_all.coef_[1]:.3f}log({sim_parameters[1]}) + {model_all.coef_[2]:.3f}log({sim_parameters[2]}) + {model_all.coef_[3]:.3f}{sim_parameters[3]} + {model_all.coef_[4]:.3f}log({sim_parameters[4]})+ {model_all.coef_[5]:.3f}{sim_parameters[5]} + {model_all.intercept_:.3f}$'
# display(Math(f'{eqn_all}')) # Latex formatted string to LaTeX output

print(f'Coefficents: {model_all.coef_}')
print(f'Intercept: {model_all.intercept_}')
predicted_log_Y_all = model_all.predict(log_X_all)

# Ordinary Least Squares (OLS) model
ols_X = sm.add_constant(log_X_all) # add an intercept value to the X parameters
ols = sm.OLS(log_Y_all, ols_X) # statsmodels OLS model
ols_result = ols.fit() # fitting the model
print(ols_result.summary()) # summary of the model (errors, coefficients, etc)

# Weighted Least Squares (WLS) model
wls = sm.WLS(log_Y_all, ols_X, weights=1/log_Y_all_error) # including y errors effect
wls_result = wls.fit() # fitting the model
print(wls_result.summary()) # summary of the model (errors, coefficients, etc)

# print the intercept and coefficient values ols
# print(f'Intercept: {ols_result.params[0]}')
# print(f'Coefficients: {ols_result.params[1:]}')
# print(f'std_err: {ols_result.bse}')

predicted_log_Y_all_ols = ols_result.predict(ols_X)

# OLS error calculation using standard error of the coefficients
predicted_log_Y_all_ols_err = []
for i, _ in enumerate(ols_X):
    x_err_plus_c = [ols_result.bse[j]*abs(ols_X[i][j]) for j in range(len(ols_X[i]))]
    y_err = np.sqrt(np.sum(np.array(x_err_plus_c)**2))
    predicted_log_Y_all_ols_err.append(y_err)

predicted_log_Y_all_wls = wls_result.predict(ols_X)

# WLS error calculation using standard error of the coefficients
predicted_log_Y_all_wls_err = []
for i, _ in enumerate(ols_X):
    x_err_plus_c = [wls_result.bse[j]*abs(ols_X[i][j]) for j in range(len(ols_X[i]))]
    y_err = np.sqrt(np.sum(np.array(x_err_plus_c)**2))
    predicted_log_Y_all_wls_err.append(y_err)

fig, ax = plt.subplots(1,3, figsize=(15,5))
# SKlearn wls matches statsmodel wls
#ax[0].scatter(predicted_log_Y_all, log_Y_all, color='green', label = 'sklearn wls', alpha=0.5)
ax[0].errorbar(predicted_log_Y_all_ols, 
               log_Y_all,
               xerr=predicted_log_Y_all_ols_err,
               yerr=log_Y_all_error,
               fmt='none', 
               ecolor='grey',
               alpha=0.2,
               zorder=-1
)
ax[0].scatter(predicted_log_Y_all_ols, log_Y_all, color='red', label = 'statsmodels ols', alpha = 0.5)
ax[0].plot(log_Y_all, log_Y_all, label='y=y_pred')
ax[0].plot(log_Y_all+1, log_Y_all, label='y=y_pred-1', linestyle='--')
ax[0].plot(log_Y_all-1, log_Y_all, label='y=y_pred+1', linestyle='--')
ax[0].set_ylabel('Log10(EW All_Obs) -- TRUTH')
ax[0].set_xlabel('Predicted Log10(EW All_Obs)')
ax[0].set_title('OLS Linear regression model for \n entire line EW python data')
ax[0].set_xlim(-2,4)
ax[0].set_ylim(-4,3)
ax[0].legend()

ax[2].errorbar(predicted_log_Y_all_wls, 
               log_Y_all,
               xerr=predicted_log_Y_all_wls_err,
               yerr=log_Y_all_error,
               fmt='none', 
               ecolor='grey',
               alpha=0.2,
               zorder=-1
)
ax[2].scatter(predicted_log_Y_all_wls, log_Y_all, color='blue', label = 'statsmodels wls', alpha = 0.5)
ax[2].plot(log_Y_all, log_Y_all, label='y=y_pred')
ax[2].set_ylabel('Log10(EW All_Obs) -- TRUTH')
ax[2].set_xlabel('Predicted Log10(EW All_Obs)')
ax[2].set_title('WLS Linear regression model for \n entire line EW python data')
ax[2].set_xlim(-2,4)
ax[2].set_ylim(-4,3)
ax[2].legend()
# colour maps to run numbers
temp = np.delete(np.arange(0,729), negative_runs_all)
target = ax[1].scatter(predicted_log_Y_all, log_Y_all, c=temp, cmap='rainbow')
plt.colorbar(target, ax=ax[1], label='Grid Length')
ax[1].plot(log_Y_all, log_Y_all, label='y=y_pred')
ax[1].set_ylabel('Log10(EW All_Obs)')
ax[1].set_xlabel('Predicted Log10(EW All_Obs)')
ax[1].set_title('Customising regression model for \n entire line EW python data')
ax[1].set_xlim(-2,4)
ax[1].set_ylim(-4,3)
ax[1].legend()
plt.savefig(f'{folder_name}/all_EW_data_vs_predicted_inc_{incs[inclination_column]}.png', dpi=300)
plt.show()

#cross validation
scores = cross_val_score(model_all, log_X_all, log_Y_all, cv=5, n_jobs=5)
print(f'Cross validated R^2 score: {scores}')

# TODO Difference between predicted and actual values correlates to EW_Excess?
#finding outliers from model
# parameter_table_copy = parameter_table.copy()
# parameter_table_without_negative_runs = np.delete(parameter_table_copy, negative_runs_all, axis=0)
count = []
for i in range(len(predicted_log_Y_all_ols)):
    if predicted_log_Y_all_ols[i] > log_Y_all[i]+1 or predicted_log_Y_all_ols[i] < log_Y_all[i]-1:
        lower_line = predicted_log_Y_all_ols[i] - 1
        if log_Y_all[i] < lower_line:
            #print(f'{log_Y_all[i]:.3f}, {predicted_log_Y_all_ols[i]:.3f}, {grid_length_without_negative_runs[i]}, {i}')
            count.append(grid_length_without_negative_runs[i]) # poor indexes

combination_list = [parameter_table[i] for i in count] # index to combination
combination_list = np.delete(combination_list, 0, axis=1) # remove first column

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
%matplotlib qt

def slider_update(val):
    for i in range(6):
        ax[i//2, i%2].clear()
        ax[i//2, i%2].hist(combination_list[:,i], bins=50, color='orange')
        xlabel = sim_parameters[i]
        ax[i//2, i%2].set_xlabel(xlabel)
        ax[i//2, i%2].set_ylabel('Frequency')
        # highlight the bar on the plot for the specific combination val index
        index = count.index(int(val))
        ax[i//2, i%2].axvline(x=combination_list[int(index)][i], color='black', linestyle='--', alpha=0.5)
        # set ylimit to 50
        ax[i//2, i%2].set_ylim(0,len(count))

    fig.suptitle(f'Parameter Histograms for poor EW fits \n Combination {int(val)}')
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(frame):
    if anim.running:
        if grid_slider.val == max(count): # TODO size of grid
            grid_slider.set_val(count[0])
        else:
            next_index = count.index(int(grid_slider.val)) + 1
            grid_slider.set_val(count[next_index])
            
def play_pause(event):
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_) -> None:
    anim.running = False
    back_one = count.index(int(grid_slider.val)) - 1
    grid_slider.set_val(count[back_one])
    slider_update(grid_slider.val)

def right_button_func(_) -> None:
    anim.running = False
    forward_one = count.index(int(grid_slider.val)) + 1
    grid_slider.set_val(count[forward_one])
    slider_update(grid_slider.val)
    
fig, ax = plt.subplots(3,2, figsize=(10,10))
plt.subplots_adjust(bottom=0.2)
#fig.canvas.mpl_connect('button_press_event', slider_update)

ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03]) # Run Slider
grid_slider = Slider(ax_slider, 'Run', 0, max(count), valinit=count[0], valstep=count)
grid_slider.on_changed(slider_update)

ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05]) # Play/Pause Button
play_pause_button = Button(ax_play_pause, '>||')
play_pause_button.on_clicked(play_pause)

ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05]) # Left Button
left_button = Button(ax_left_button, '<')
left_button.on_clicked(left_button_func)

ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05]) # Right Button
right_button = Button(ax_right_button, '>')
right_button.on_clicked(right_button_func)

anim = FuncAnimation(fig, 
                     animation_setting_new_slider_value,
                     frames=len(combination_list),
                     interval=400
                     ) # setting up animation
anim.running = True # setting off animation




































































# %%

################################################################################
# OLD CODE I DON'T HAVE THE HEART TO DELETE INCASE I NEED IT LATER FOR SOMETHING
################################################################################

#for grid_index in range(0,len(grid),1):  # iterating through each spectrum in the grid
    #initials = [1.0000e-11, 6.5628e+03, 1.2000e+01, 0, 3e-13]
    # Fitting the Gaussian with continuum function
    #parameters, cov_matrix = curve_fit(gaussian_with_continuum, 
                                       #wavelengths, 
                                       #with_con_grid[grid_index], 
                                       #p0=initials, 
                                       #maxfev=1000000
                                       #)
    
    #slope_parameters = np.append(slope_parameters, parameters[3]) # saving linear fit parameters
    #intercept_parameters = np.append(intercept_parameters, parameters[4])
    
    # print(f'------Run {run_number[grid_index]}------')
    # for index in range(len(parameter_names)):
    #     print(parameter_names[index], f'{parameters[index]:.5f}') # printing optimal parameters
    
    # Retrieving the fit for optimal and initial parameters. optimal added to array
    #gaussian_with_continuum_fit = gaussian_with_continuum(wavelengths, *parameters)
    # inital_gaussian_with_continuum = gaussian_with_continuum(wavelengths, *initials)
    #gaussian_fitted_grid = np.append(gaussian_fitted_grid, gaussian_with_continuum_fit)

    # plt.plot(wavelengths[peak_mask], with_con_grid[grid_index][peak_mask], label='Original Data', color='black')
    # plt.plot(wavelengths[peak_mask], inital_gaussian_with_continuum[peak_mask], label='Initial Gaussian with Continuum', color='grey', alpha=0.5)
    # plt.plot(wavelengths[peak_mask], gaussian_with_continuum_fit[peak_mask], label='Optimal Gaussian with Continuum', color='red')
    # plt.axvline(x=H_alpha, color='blue', linestyle='--', alpha=0.5)
    # plt.xlabel('Wavelength ($\AA$)')
    # plt.ylabel('Flux')
    # plt.title(f'Run {run_number[grid_index]}: Gaussian and Continuum')
    # plt.legend()
    # plt.show()
    
    # plt.plot(wavelengths, with_con_grid[grid_index], label='Original Data', color='black')
    # plt.plot(wavelengths, inital_gaussian_with_continuum, label='Initial Gaussian with Continuum', color='grey', alpha=0.5)
    # plt.plot(wavelengths, gaussian_with_continuum_fit, label='Optimal Gaussian with Continuum', color='red')
    # plt.axvline(x=H_alpha, color='blue', linestyle='--', alpha=0.5)
    # plt.xlabel('Wavelength ($\AA$)')
    # plt.ylabel('Flux')
    # plt.title(f'Run {run_number[grid_index]}: Gaussian and Continuum')
    # plt.legend()
    # plt.show()

# Reshaping optimal gaussian array to be the same shape as the grid
#gaussian_fitted_grid = gaussian_fitted_grid.reshape(len(grid), len(wavelengths))


# %%
# Step two: Fitting a ONLY the Gaussian function without underlying continuum
# from scipy.optimize import curve_fit

# def a_gaussian(x, a, mu, sigma):
#     return a*np.exp(-(x-mu)**2/(2*sigma**2))

# wavelengths[peak_mask] = (6475, 6650)
# peak_mask = (wavelengths > wavelengths[peak_mask][0]) & (wavelengths < wavelengths[peak_mask][1])
# wavelengths[peak_mask] = wavelengths[peak_mask]

# gaussian_fitted_grid = np.array([])
# parameter_names = ['Amplitude', 'Mean', 'Sigma']

# subtracted_con_grid = grid - sk_con_data
# #subtracted_con_grid = subtracted_con_grid / np.max(subtracted_con_grid, axis=1)[:,None] # normalised 
# for grid_index in range(0,len(subtracted_con_grid),1):
#     bounds = ([1e-18, 6400, 0.1], [5e-12, 6700, 200])
#     #bounds = ([1e-2, 6400, 0.1], [100, 6700, 200]) # for normalised grid
#     amplitude_max = max(subtracted_con_grid[grid_index][peak_mask]) - (min(subtracted_con_grid[grid_index][peak_mask])/2)
#     parameters, cov_matrix = curve_fit(a_gaussian, 
#                                        wavelengths[peak_mask], 
#                                        subtracted_con_grid[grid_index][peak_mask], 
#                                        p0=[amplitude_max, H_alpha, 12], #p0=[0.01, 6560, 12]
#                                        maxfev=1000000,
#                                        ftol=1e-14,
#                                        xtol=1e-18,
#                                        gtol=1e-18, # <- This major effects fitting ablility.
#                                        bounds=bounds,
#                                        method = 'dogbox')
#     for index in range(len(parameters)):
#         print(parameters[index], parameter_names[index])
#     inital_gaussian = a_gaussian(wavelengths[peak_mask], amplitude_max, H_alpha, 10)
#     optimal_gaussian = a_gaussian(wavelengths[peak_mask], *parameters)
#     gaussian_fitted_grid = np.append(gaussian_fitted_grid, optimal_gaussian)

#     plt.plot(wavelengths[peak_mask], subtracted_con_grid[grid_index][peak_mask], label='Subtracted Continuum Data', color='black')
#     plt.plot(wavelengths[peak_mask], inital_gaussian, label='Initial Gaussian', color='grey', alpha=0.5)
#     plt.plot(wavelengths[peak_mask], optimal_gaussian, label='Optimal Gaussian', color='red')
#     plt.axvline(x=H_alpha, color='blue', linestyle='--', alpha=0.5)
#     plt.xlabel('Wavelength ($\AA$)')
#     plt.ylabel('Flux')
#     plt.title(f'Run {grid_index+1}: Gaussian only')
#     plt.legend()
#     plt.show()

# gaussian_fitted_grid = gaussian_fitted_grid.reshape(len(grid), len(wavelengths[peak_mask]))

# -------------------------------------------------------------------
# %%
# Equivalent Width Function 
# def equivalent_width_excess(wavelengths, fluxes, continuum):
#     residual = 1 - (fluxes / continuum) # normalizing the fluxes
#     dispersion = np.mean(np.diff(wavelengths)) # dispersion of the wavelength
#     print(dispersion)
#     equi_w = dispersion * sum(residual)
#     return equi_w
    
# # Continuum function
# def continuum(wavelengths, spectrum):
#     """Function to determine the underlying continuum of the spectrum.

#     Args:
#         wavelengths (ndarray): wavelengths of the spectrum
#         spectrum (ndarray): a single spectrum from the grid

#     Returns:
#         ndarray: A continuum array of fluxes the same length as the spectrum
#     """
#     from sklearn.linear_model import LinearRegression

#     # Continuum portion of the spectrum. Removing the emission lines
#     intervals = (6000, 6400, 6800, 7000) # wavelength intervals for continuum
#     con_wave_mask_low = (wavelengths > intervals[0]) & (wavelengths < intervals[1])
#     con_wave_mask_high = ((wavelengths > intervals[2]) & (wavelengths < intervals[3]))
#     con_mask = con_wave_mask_low | con_wave_mask_high # mask for continuum wavelength range
#     intervals = wavelengths[con_mask] # updating to new continuum wavelength range

#     # Fitting linear regression model to the continuum to subtract continuum
#     reg = LinearRegression()
#     reg.fit(intervals.reshape(-1,1), spectrum[con_mask]) # fitting linear regression model
#     continuum = reg.predict(wavelengths.reshape(-1,1)) # predicting continuum values
    
#     return continuum


# grid_index = 0
# con = continuum(wavelengths, grid[grid_index])
# equi_w = equivalent_width_excess(wavelengths, grid[grid_index], con)
# print(equi_w)


# plt.plot(wavelengths, grid[grid_index])
# plt.plot(wavelengths, grid[grid_index] - continuum(wavelengths, grid[grid_index]))
# plt.plot(wavelengths, continuum(wavelengths, grid[grid_index]))
# plt.show()
# # -------------------------------------------------------------------
# # %%
# # NORMALIZING THE FLUX GRID. Don't use
# from sklearn.preprocessing import normalize
# grid = normalize(grid, axis=1, norm='max') # normalizing the flux grid
# plt.plot(wavelengths, grid[0])
# plt.plot(wavelengths, grid[122])
# plt.show()

# # %%


# # %%
# # CONTINUUM SUBTRACTION
# from sklearn.linear_model import LinearRegression

# continuum_results = np.array([])

# # Continuum portion of the spectrum. Removing the emission lines
# intervals = (6000, 6400, 6800, 7000) # wavelength intervals for continuum
# con_wave_mask_low = (wavelengths > intervals[0]) & (wavelengths < intervals[1])
# con_wave_mask_high = ((wavelengths > intervals[2]) & (wavelengths < intervals[3]))
# con_mask = con_wave_mask_low | con_wave_mask_high # mask for continuum wavelength range
# intervals = wavelengths[con_mask] # updating to new continuum wavelength range
# plt.plot(wavelengths, grid[0])
# # Fitting linear regression model to the continuum to subtract continuum
# reg = LinearRegression()
# for spectrum in range(len(grid)):
#     reg.fit(intervals.reshape(-1,1), grid[spectrum][con_mask]) # fitting linear regression model
#     continuum = reg.predict(wavelengths.reshape(-1,1)) # predicting continuum values
#     continuum_results = np.append(continuum_results, continuum)
#     grid[spectrum] = grid[spectrum] - continuum # subtracting continuum from grid

# plt.plot(wavelengths, grid[0])


# # %%
# print(len(continuum_results))
# # -------------------------------------------------------------------

# # %%
# # FITTING GAUSSIAN TO EMISSION LINES
# from scipy.optimize import curve_fit

# def gaussian_function(x, a, x0, sigma):
#     return a*np.exp(-(x-x0)**2/(2*sigma**2))# + offset

# peak_wave = (6475, 6700)#(6475,6650) # Wave mask for peak emission line
# peak_wave_mask = (wavelengths > peak_wave[0]) & (wavelengths < peak_wave[1])
# peak_wave = wavelengths[peak_wave_mask]

# parameter_results = np.array([])
# uncertainty_results = np.array([])
# residuals = np.array([])

# for spectrum in range(len(grid)):
#     #param_bounds=([1e-16, 6300, 1e-16, 1e-16], [np.inf, 6800, np.inf, np.inf])
#     parameters, cov_matrix = curve_fit(gaussian_function, peak_wave, grid[spectrum][peak_wave_mask], maxfev=1000000) # bounds=param_bounds p0=[5e-13, 6500, 20]
#     parameter_results = np.append(parameter_results, parameters)
#     uncertainty_results = np.append(uncertainty_results, np.sqrt(np.diag(cov_matrix)))
#     residual = grid[spectrum][peak_wave_mask] - gaussian_function(peak_wave, *parameters)
#     residuals = np.append(residuals, grid[spectrum][peak_wave_mask] - gaussian_function(peak_wave, *parameters))
#     # plotting the gaussian fit and residual on two subplots
#     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
#     ax1.plot(peak_wave, grid[spectrum][peak_wave_mask])
#     ax1.plot(peak_wave, gaussian_function(peak_wave, *parameters))
#     #ax1.set_ylim(-1e-13, 2e-12)
#     ax1.set_title(f'Run {spectrum+1}')
#     ax1.set_xlabel('Wavelength (Angstroms)')
#     ax1.set_ylabel('Flux')
#     ax2.plot(peak_wave, residual)
#     ax2.plot(peak_wave, np.zeros(len(peak_wave)), 'k--')
#     ax2.set_title('Residual')
#     ax2.set_xlabel('Wavelength (Angstroms)')
#     ax2.set_ylabel('Flux - Gaussian Fit')
#     plt.show()

# %%
# For a single spectrum
# parameters, cov_matrix = curve_fit(gaussian_function, peak_wave, grid[150][peak_wave_mask], p0=[1e-12, 6500, 10])
# print(parameters, 'parameters')
# print(np.sqrt(np.diag(cov_matrix)), 'uncertainties')
# plt.plot(peak_wave, grid[151][peak_wave_mask])
# plt.plot(peak_wave, gaussian_function(peak_wave, *parameters))
# plt.ylim(-1e-13, 2e-12)
# plt.show()
# %%

# plt.plot(wavelengths, grid[148])



# %%
# renaming test files to integrate with the original grid. 
# os.getcwd()
# for file in os.listdir("broad_short_spec_cv_grid_test_data"):
#     directory = "broad_short_spec_cv_grid_test_data/" # where located
#     current_name = directory + file # full file path
#     curr_run_num = int(file.split("_")[0][-2:]) # get run number
#     new_run_num = curr_run_num + 125 # add 125 to append the other directory
#     new_base = file[5:] # add the parameter names back again
#     new_name = directory + "run" + str(new_run_num) + new_base # form new name

#     os.rename(current_name, new_name) # Rename the file

# %%
# files = os.listdir('broad_short_spec_cv_grid/') # list of files in directory

# sorted_files = []
# for num in range(1,154): # run number ranges
#     if len(str(num)) == 1: # adding 0 to single digit run numbers
#         run = f'run0{num}_'
#     else:
#         run = f'run{num}_'
#     file = fnmatch.filter(files, run+'*') # matching file with loop run number
#     sorted_files.append('broad_short_spec_cv_grid/'+str(file[0])) # files sorted in order of run number

# %%
# Exporting python data to a text file
# Make a table with wavelengths and with_con_grid as columns
# test1 = np.array([wavelengths, grid[141]])
# test1 = test1.T
# np.savetxt('test141.txt', test1, delimiter=',', fmt='%s')
# plt.plot(wavelengths, grid[143])
# plt.show()

# %%
# Step2 Cut Code:
#removing the first 25 runs and runs 126-134 as useless data
# run_number = np.arange(1, len(grid)+1, 1) # run number array
# run_number = np.delete(run_number, np.s_[0:25], axis=0)
# run_number = np.delete(run_number, np.s_[100:110], axis=0)
# grid = np.delete(grid, np.s_[0:25], axis=0)
# grid = np.delete(grid, np.s_[100:110], axis=0)
# sorted_files = np.delete(sorted_files, np.s_[0:25], axis=0)
# sorted_files = np.delete(sorted_files, np.s_[100:110], axis=0)

#print('Run files in the grid', run_number)
# %%
# Step6 Cut code :
        # data_cut = [] # list initalisations old code
        # gaussian_fit_cut = []
        # continuum_fit_cut = []
        # for i in range(len(data)): # cutting the data to the same length as the wavelengths
        #     data_cut.append(data[i][:cut])
        #     gaussian_fit_cut.append(gaussian_fit[i][:cut])
        #     continuum_fit_cut.append(continuum_fit[i][:cut])
        
        # data_cut = [] # list initalisations
        # gaussian_fit_cut = []
        # continuum_fit_cut = []
        # for i in range(len(data)): # cutting the data to the same length as the wavelengths
        #     data_cut.append(data[i][-cut:])
        #     gaussian_fit_cut.append(gaussian_fit[i][-cut:])
        #     continuum_fit_cut.append(continuum_fit[i][-cut:])

        # data_cut = [] # list initalisations
        # gaussian_fit_cut = []
        # continuum_fit_cut = []
        # for i in range(len(data)): # cutting the data to the same length as the wavelengths
        #     data_cut.append(data[i][-cut:])
        #     gaussian_fit_cut.append(gaussian_fit[i][-cut:])
        #     continuum_fit_cut.append(continuum_fit[i][-cut:])
        
        # data_cut = [] # list initalisations
        # gaussian_fit_cut = []
        # continuum_fit_cut = []
        # for i in range(len(data)): # cutting the data to the same length as the wavelengths
        #     data_cut.append(data[i][:cut])
        #     gaussian_fit_cut.append(gaussian_fit[i][:cut])
        #     continuum_fit_cut.append(continuum_fit[i][:cut])
        
# %%
# Step 8 Cut code:
# For colour bar
#run_number = np.arange(1, len(grid)+1, 1)

#for i in range(0,25,5): # concentric circles
    #plt.plot(circle(i)[0], circle(i)[1], label=f'radius = {i}', alpha=0.2) 
# my data

# %%
# list of sum squares for each run
# %matplotlib qt
# sum_squared = np.array([])
# for i in range(len(grid)):
#     sum_squared = np.append(sum_squared, sum((grid[i][50:350] - fit_con[i][50:350])**2))
# plt.plot(run_number, sum_squared)
# plt.ylim(0, 1e-21)
# plt.show()
# # %%

# plt.hist(sum_squared, bins=10000)
# plt.xlim(0, 1e-20)
# #plt.ylim(0,30)
# plt.show()
# # %%
# ################################################################################
# print('STEP 11: REPLOTTING THE EW EXCESSES WITHOUT THE CUT RUNS')
# ################################################################################

# cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
# cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
# cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
# cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
# cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
# cut_grid_length = np.delete(grid_length, cut_runs)

# bz_cam = np.loadtxt('Teo_data/BZ Cam.csv', delimiter=',') 
# mv_lyr = np.loadtxt('Teo_data/MV Lyr.csv', delimiter=',')
# v425_cas = np.loadtxt('Teo_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Teo_data/V751 Cyg.csv', delimiter=',')
# %matplotlib qt

# def slider_update(val):
#     if val in cut_grid_length:
#         index = np.where(cut_grid_length == val)[0][0]
#         ax2.clear()
#         ax2.set_ylim(ax.get_ylim())
#         ax2.errorbar(cut_red_ew_excess[index],
#                      cut_blue_ew_excess[index],
#                         xerr=cut_red_ew_excess_error[index],
#                         yerr=cut_blue_ew_excess_error[index],
#                         fmt='o',
#                         ecolor = 'black',
#                         zorder=5, 
#                         markersize=5, 
#                         c='black'
#                         )
#         ax2.legend([f'Run {val}'], loc='upper left')
#     else: 
#         ax2.clear()
#         ax2.set_ylim(ax.get_ylim())
#     fig.canvas.draw_idle()

# def animation_setting_new_slider_value(frame):
#     if anim.running:
#         if grid_slider.val == 728: 
#             grid_slider.set_val(0)
#         else:
#             grid_slider.set_val(grid_slider.val + 1)
            
# def play_pause(event):
#     if anim.running:
#         anim.running = False
#         slider_update(grid_slider.val)
#     else:
#         anim.running = True

# def left_button_func(_) -> None:
#     anim.running = False
#     grid_slider.set_val(grid_slider.val - 1)
#     slider_update(grid_slider.val)

# def right_button_func(_) -> None:
#     anim.running = False
#     grid_slider.set_val(grid_slider.val + 1)
#     slider_update(grid_slider.val)
    
# fig, ax = plt.subplots(figsize=(10, 10)) # Creating Figure
# plt.subplots_adjust(bottom=0.2)
# # create another axis ax2 the same size as ax
# ax2 = ax.twinx()  # Move the initialization of ax2 inside the init_plot() function

# def init_plot():
#     ax.errorbar(cut_red_ew_excess,
#                 cut_blue_ew_excess,
#                 xerr=cut_red_ew_excess_error,
#                 yerr=cut_blue_ew_excess_error, 
#                 fmt='none', 
#                 ecolor='grey', 
#                 alpha=0.5, 
#                 zorder=2
#                 )
#     target = ax.scatter(cut_red_ew_excess,
#                         cut_blue_ew_excess, 
#                         c=cut_grid_length,
#                         s=15,
#                         label='Grid Data',
#                         cmap='rainbow', 
#                         zorder=3
#                         )
#     plt.colorbar(target, label='Run Number', ax=ax)
#     ax.scatter(bz_cam[:,0], bz_cam[:,1], label='BZ Cam', color='black', s=10, marker='v')
#     ax.scatter(mv_lyr[:,0], mv_lyr[:,1], label='MV Lyr', color='black', s=10, marker='^')
#     ax.scatter(v425_cas[:,0], v425_cas[:,1], label='V425 Cas', color='black', s=10, marker='<')
#     ax.scatter(v751_cyg[:,0], v751_cyg[:,1], label='V751 Cyg', color='black', s=10, marker='>')
#     # plot formatting
#     ax.set_xlabel('Red Wing EW Excess ($Å$)')
#     ax.set_ylabel('Blue Wing EW Excess ($Å$)')
#     ax.set_title('Red vs Blue Wing Excess')
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     xlim = min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2
#     ylim = min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.legend()  # bbox_to_anchor=(1.45, 0.25),loc='lower center'

# ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Run Slider
# grid_slider = Slider(ax_slider, 'Run', 0, 728, valinit=0, valstep=1)
# grid_slider.on_changed(slider_update)

# ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05])  # Play/Pause Button
# play_pause_button = Button(ax_play_pause, '>||')
# play_pause_button.on_clicked(play_pause)

# ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05])  # Left Button
# left_button = Button(ax_left_button, '<')
# left_button.on_clicked(left_button_func)

# ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05])  # Right Button
# right_button = Button(ax_right_button, '>')
# right_button.on_clicked(right_button_func)

# init_plot()

# anim = FuncAnimation(fig, 
#                      animation_setting_new_slider_value,
#                      frames=729,
#                      interval=400
#                      )  # setting up animation
# anim.running = True  # setting off animation


# %%
# ################################################################################
# print('TOOL: ANIMATED PLOTTING')
# ################################################################################
# # Plotting a grid of data as an animated plot. 
# # This is particularly useful for PYTHON grid data.

# %matplotlib qt

# def slider_update(val):
#     ax.clear()
#     ax.plot(final_results['grid'][int(val)], label='Original Data', color='black') # y=dictionary
#     ax.plot(final_results['fitted_grid'][int(val)], label='Optimal Gaussian with Continuum', color='red')
#     ax.plot(final_results['fit_con'][int(val)], label='Fitted Continuum', color='blue')
#     # ax.plot(
#     #         gaussian_with_continuum(wavelengths, *initials_all[int(val)]), 
#     #         label='Initial Gaussian with Continuum', 
#     #         color='grey'
#     #         )
#     ax.plot(final_results['sk_con_data'][int(val)], label='Sklearn Fitted Continuum', color='green')
#     #sum_squared = sum((final_results['grid'][int(val)]-final_results['fitted_grid'][int(val)])**2)
#     #ax.scatter(final_results['fitted_grid'][int(val)], color = 'red')
#     # ax.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
#     # ax.axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
#     # ax.axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
#     # ax.axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
#     # ax.axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
#     ax.set_xlabel('Wavelength ($Å$)')
#     ax.set_ylabel('Flux')
#     #ax.set_xlim(6385, 6735)
#     ax.set_title(f'Run {int(val)}: Gaussian and Continuum')
#     ax.legend()
#     fig.canvas.draw_idle()

# def animation_setting_new_slider_value(frame, i=0):
#     if anim.running:
#         slider_update(grid_slider.val)
            
# def play_pause(event):
#     if anim.running:
#         anim.running = False
#         slider_update(grid_slider.val)
#     else:
#         anim.running = True

# def left_button_func(_) -> None:
#     anim.running = False
#     grid_slider.set_val(cut_grid_length[index])
#     slider_update(grid_slider.val)

# def right_button_func(_) -> None:
#     anim.running = False
#     grid_slider.set_val(cut_grid_length[index])
#     slider_update(grid_slider.val)
    
# fig, ax = plt.subplots(figsize=(12, 8)) # Creating Figure
# plt.subplots_adjust(bottom=0.2)

# ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03]) # Run Slider
# grid_slider = Slider(ax_slider, 'Run', 0, 728, valinit=cut_grid_length[0], valstep=cut_grid_length)
# grid_slider.on_changed(slider_update)

# ax_play_pause = fig.add_axes([0.15, 0.1, 0.05, 0.05]) # Play/Pause Button
# play_pause_button = Button(ax_play_pause, '>||')
# play_pause_button.on_clicked(play_pause)

# ax_left_button = fig.add_axes([0.1, 0.1, 0.05, 0.05]) # Left Button
# left_button = Button(ax_left_button, '<')
# left_button.on_clicked(left_button_func)

# ax_right_button = fig.add_axes([0.2, 0.1, 0.05, 0.05]) # Right Button
# right_button = Button(ax_right_button, '>')
# right_button.on_clicked(right_button_func)

# index = 0
# anim = FuncAnimation(fig, 
#                      animation_setting_new_slider_value,
#                      frames=len(cut_grid_length),
#                      interval=400
#                      ) # setting up animation
# anim.running = True # setting off animation

# # %%
# ################################################################################
# print('TOOL: PLOTTING THE EW EXCESSES FOR A PARTICULAR RUN')
# ################################################################################

# #Plotting the excess mask over the gaussian fit an example spectrum
# %matplotlib inline
# spectrum_example = 0
# plt.plot(wavelengths, grid[spectrum_example], label='Original Data', color='black')
# plt.plot(wavelengths, fitted_grid[spectrum_example], label='Optimal Gaussian with Continuum', color='red')
# plt.plot(wavelengths, fit_con[spectrum_example], label='Fitted Continuum', color='blue')
# plt.plot(wavelengths, sk_con_data[spectrum_example], label='Sklearn Fitted Continuum', color='green')
# plt.scatter(wavelengths, fitted_grid[spectrum_example], color = 'red')
# #initials = [1.0000e-11, 6.5628e+03, 1.5000e+01, 0, 3e-13]
# plt.plot(wavelengths, gaussian_with_continuum(wavelengths, *initials_all[spectrum_example]), label='Initial Gaussian with Continuum', color='grey', alpha=0.5)
# plt.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
# plt.axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
# plt.axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
# plt.axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
# plt.axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
# plt.xlabel('Wavelength ($Å$)')
# plt.ylabel('Flux')
# #plt.xlim((H_alpha - blue_peak_mask[1]-20), (H_alpha + red_peak_mask[1]+20))
# plt.xlim(6385, 6735)
# #plt.xlim(wavelength_range[0], wavelength_range[1])
# #plt.ylim(0,0.3e-12)
# plt.title(f'Run {run_number[spectrum_example]}: Gaussian and Continuum')
# plt.legend()
# plt.show()

# # %%
# %matplotlib inline
# store_cuts = []

# a = [727, 726, 725, 724, 723, 722, 721, 720, 714, 713, 712, 711, 703, 702, 699, 684, 675, 671, 669, 668, 667, 664, 663, 662, 661, 660, 659, 658, 657, 655, 654, 653, 652, 651, 650, 649, 648, 646, 645, 644, 643, 642, 641, 640, 639, 580, 571, 569, 568, 567, 565, 564, 563, 562, 560, 552, 550, 540, 534, 533, 532, 531, 523, 522, 514, 513, 504, 486, 483, 482, 481, 480, 479, 478, 477, 469, 468, 425, 420, 418, 416, 415, 414, 411, 409, 408, 407, 406, 405, 402, 400, 370, 344, 343, 322, 320, 309, 307, 306, 303, 300, 297, 291, 289, 288, 285, 282, 280, 279, 271, 261, 255, 249, 247, 246, 243, 241, 240, 237, 235, 234, 207, 173, 160, 159, 158, 157, 147, 145, 138, 135, 111, 93, 79, 77, 64, 57, 33, 18]
# new_a = [77, 79, 158, 160, 414, 425, 468, 477, 478, 479, 480, 481, 482, 483, 563, 565, 567, 569, 580, 639, 640, 641, 642, 643, 644, 645, 646, 667, 668, 669, 712, 713, 720, 721, 722, 723, 724, 725, 726, 727]
# for i in range(0, len(new_a)-1):
#     # if sum_squared[i] > 0:
#     spectrum_example = i
#     plt.plot(wavelengths, grid[spectrum_example], label='Original Data', color='black')
#     plt.plot(wavelengths, fitted_grid[spectrum_example], label='Optimal Gaussian with Continuum', color='red')
#     plt.plot(wavelengths, fit_con[spectrum_example], label='Fitted Continuum', color='blue')
#     plt.scatter(wavelengths, fitted_grid[spectrum_example], color = 'red')
#     plt.plot(wavelengths, gaussian_with_continuum(wavelengths, *initials_all[spectrum_example]), label='Initial Gaussian with Continuum', color='grey', alpha=0.5)
#     plt.plot(wavelengths, sk_con_data[spectrum_example], label='Sklearn Fitted Continuum', color='green')
#     plt.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
#     plt.axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
#     plt.axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
#     plt.axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
#     plt.axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
#     plt.xlabel('Wavelength ($Å$)')
#     plt.ylabel('Flux')
#     #plt.xlim((H_alpha - blue_peak_mask[1]-20), (H_alpha + red_peak_mask[1]+20))
#     plt.xlim(6275, 6850)
#     #plt.xlim(wavelength_range[0], wavelength_range[1])
#     #plt.ylim(0,0.3e-12)
#     plt.title(f'Removing run {run_number[spectrum_example]}: Gaussian and Continuum')
#     plt.legend()
#     plt.show()
#     store_cuts.append(i)
# print(store_cuts)



# def equivalent_width_excess(wavelengths, data, gaussian_fit, continuum_fit, shift='blue', peak_mask=(0,1000)):
#     """ Returning the equivalent width excesses for each spectrum in the grid.
#     You can choose to return the blue or red wing excesses by setting the shift.
#     Formula: EW_excess = sum((data - gaussian_fit) / continuum_fit) * delta_wavelength
#     Parameters:
#         wavelengths (ndarray): wavelength array
#         data (ndarray): the original data
#         gaussian_fit (ndarray): the gaussian fit to the data
#         continuum_fit (ndarray): the continuum fit to the data
#         shift (str): 'blue' or 'red' for the blue or red wing excesses
#         peak_mask (tuple): tuple wavelengths to cut around the peak. Default is 0,1000 doesn't cut anything.
#             The value is the number of angstroms to cut around the peak, blue minus, red plus. First value is 
#             the value closer to H_alpha.
#     Returns:
#         ew (ndarray): equivalent width excesses for each spectrum in the grid
#     """
    
#     # H_alpha = 6562.819 #4861.333	#6562.819 # Å
    
#     # if shift == 'blue':
#     #     # interval boundary next to H_alpha
#     #     mask = (wavelengths < H_alpha - peak_mask[0]) # mask for wavelengths less than H_alpha
#     #     cut = len(mask[mask == True]) # number of wavelengths less than H_alpha
#     #     wavelengths = wavelengths[wavelengths < H_alpha - peak_mask[0]] # list of shorter wavelengths
#     #     wavelengths = np.append(wavelengths, (H_alpha - peak_mask[0])) # adding H_alpha to the list if included
#     #     wave_diff = np.diff(wavelengths) # delta wavelength between pixels
        
        
#     #     data_cut = [data[i][:cut] for i in range(len(data))] # cutting the data to the same length as the wavelengths
#     #     gaussian_fit_cut = [gaussian_fit[i][:cut] for i in range(len(data))]
#     #     continuum_fit_cut = [continuum_fit[i][:cut] for i in range(len(data))]
        
#     #     data = np.array(data_cut) # converting to numpy arrays
#     #     gaussian_fit = np.array(gaussian_fit_cut)
#     #     continuum_fit = np.array(continuum_fit_cut)
        
#     #     # Repeating for interval boundary far from H_alpha
#     #     mask = (wavelengths > H_alpha - peak_mask[1]) # mask for wavelengths less than H_alpha
#     #     cut = len(mask[mask == True]) # number of wavelengths less than H_alpha
#     #     original_min = min(wavelengths)
#     #     wavelengths = wavelengths[wavelengths > H_alpha - peak_mask[1]] # list of shorter wavelengths
#     #     if (H_alpha - peak_mask[1]) > original_min:
#     #         wavelengths = np.append((H_alpha - peak_mask[1]), wavelengths) # adding H_alpha to the list if included
#     #     wave_diff = np.diff(wavelengths) # delta wavelength between pixels
        
#     #     data_cut = [data[i][-cut:] for i in range(len(data))] # cutting the data to the same length as the wavelengths
#     #     gaussian_fit_cut = [gaussian_fit[i][-cut:] for i in range(len(data))]
#     #     continuum_fit_cut = [continuum_fit[i][-cut:] for i in range(len(data))]
        
#     #     data = np.array(data_cut) # converting to numpy arrays
#     #     gaussian_fit = np.array(gaussian_fit_cut)
#     #     continuum_fit = np.array(continuum_fit_cut)
#     #     #print('Blue Completed')
        
#     # if shift == 'red':
#     #     # interval boundary next to H_alpha
#     #     mask = (wavelengths > H_alpha + peak_mask[0]) # mask for wavelengths greater than H_alpha
#     #     cut = len(mask[mask == True]) # number of wavelengths greater than H_alpha
#     #     wavelengths = wavelengths[wavelengths > H_alpha + peak_mask[0]] # list of longer wavelengths
#     #     wavelengths = np.append((H_alpha + peak_mask[0]), wavelengths) # adding H_alpha to the list if included
#     #     wave_diff = np.diff(wavelengths) # delta wavelength between pixels

#     #     data_cut = [data[i][-cut:] for i in range(len(data))] # cutting the data to the same length as the wavelengths
#     #     gaussian_fit_cut = [gaussian_fit[i][-cut:] for i in range(len(data))]
#     #     continuum_fit_cut = [continuum_fit[i][-cut:] for i in range(len(data))]
        
#     #     data = np.array(data_cut) # converting to numpy arrays
#     #     gaussian_fit = np.array(gaussian_fit_cut)
#     #     continuum_fit = np.array(continuum_fit_cut)
        
#     #     # Repeating for interval boundary far from H_alpha
#     #     mask = (wavelengths < H_alpha + peak_mask[1]) # mask for wavelengths greater than H_alpha
#     #     cut = len(mask[mask == True]) # number of wavelengths greater than H_alpha
#     #     original_max = max(wavelengths)
#     #     wavelengths = wavelengths[wavelengths < H_alpha + peak_mask[1]] # list of longer wavelengths
#     #     if (H_alpha + peak_mask[1]) < original_max:
#     #         wavelengths = np.append(wavelengths, (H_alpha + peak_mask[1])) # adding H_alpha to the list if included
#     #     wave_diff = np.diff(wavelengths) # delta wavelength between pixels

        
#     #     data_cut = [data[i][:cut] for i in range(len(data))] # cutting the data to the same length as the wavelengths
#     #     gaussian_fit_cut = [gaussian_fit[i][:cut] for i in range(len(data))]
#     #     continuum_fit_cut = [continuum_fit[i][:cut] for i in range(len(data))]
        
#     #     data = np.array(data_cut) # converting to numpy arrays
#     #     gaussian_fit = np.array(gaussian_fit_cut)
#     #     continuum_fit = np.array(continuum_fit_cut)
#     #     #print('Red Completed')

#     # fraction = (data - gaussian_fit) / continuum_fit 
#     # ew = [np.sum(fraction[i] * wave_diff) for i in range(len(fraction))]
#     # return ew

# # %% KAN Symbolic regression
# ################################################################################
# print('TRENDS: SYMBOLIC REGRESSION VS SIMULATION PARAMETERS')
# ################################################################################
# import torch
# #from gplearn.genetic import SymbolicRegressor
# from kan import *

# log_X_train, log_X_test, log_Y_train, log_Y_test = train_test_split(log_X_all, log_Y_all, test_size=0.2, random_state=42)
# # convert data to import into another juypter notebok
# np.savetxt('log_X_train.csv', log_X_train, delimiter=',')
# np.savetxt('log_X_test.csv', log_X_test, delimiter=',')
# np.savetxt('log_Y_train.csv', log_Y_train, delimiter=',')
# np.savetxt('log_Y_test.csv', log_Y_test, delimiter=',')