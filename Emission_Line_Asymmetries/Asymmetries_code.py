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
################################################################################
################################################################################

# %%
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
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

plt.style.use('Solarize_Light2')
script_ran_once = False

# %%
################################################################################
print('STEP 2: LOADING DATA AND BUILDING GRID')
################################################################################

path_to_grid = '../Grids/Mixed_Ha_grid_spec_files/'
files = os.listdir(path_to_grid) # list of files in directory

# -------INPUTS-------- #
wavelength_range = (6435, 6685) # set desired wavelength range, start with the narrowest range
inclination_column = 10 # 10-14 = 20,45,60,72.5,85
grid_mixed = True # if grid is mixed, set to True to rerun the script at difference wavelength ranges
                  # you can change the wavelength range in the second input below.
# --------------------- #

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

for i in range(len(sorted_files)):
    flux = np.loadtxt(sorted_files[i], usecols=inclination_column, skiprows=81)[::-1] # flux values for each file
    grid[i] = flux[wave_mask] # adding fluxes into the grid
    wave = np.loadtxt(sorted_files[i], usecols=1, skiprows=81)[::-1] # wavelength values for each file
    wavelength_grid[i] = wave[wave_mask] # adding wavelengths into the grid

# %%
################################################################################
print('STEP 3: SKLEARN FITTING THE CONTINUUM ONLY, NOISE ERROR CALCULATED')
################################################################################


def continuum(wavelengths, spectrum):
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
    intervals = (6220, 6480, 6600, 6900) # wavelength intervals for continuum
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
for i in range(len(grid)): # Iterating through grid
    con, m, c, noise = continuum(wavelengths, grid[i])
    sk_con_data = np.append(sk_con_data, con)
    sk_slopes = np.append(sk_slopes, m)
    sk_intercepts = np.append(sk_intercepts, c)
    sk_noise_error = np.append(sk_noise_error, noise)
    
sk_con_data = sk_con_data.reshape(len(grid), len(wavelengths)) # append method back to grid shape

# %%
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

# %%
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
    fcont = m * (w - H_alpha) + b
    const = amp * (1.0 / (np.fabs(sigma)*(2*np.pi)**0.5))
    fline = const * np.exp(-0.5*((w - mu)/np.fabs(sigma))**2.0)
    if sigma < 0: # if trialling negative sigma, give it a super poor fit
        fline = -1000
    if amp < 0:
        fline = -1000
    
    return fcont + fline

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
    
    # fitting each spectrum in the grid
    for run, spectrum in enumerate(grid):
        # estimate for initial slope
        initial_m = (spectrum[-1]-spectrum[0]) / (wavelengths[-1]-wavelengths[0])
        
        # estimate for initial intercept
        initial_c = (spectrum[0]+spectrum[-1]) / 2
        
        # estimate for the initial Gaussian peak (mean) location
        gaussian_only = spectrum - (initial_m * (wavelengths - H_alpha) + initial_c) # removing continuum
        initial_mu = wavelengths[np.argmax(gaussian_only)]
        
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

# %% 
################################################################################
print('STEP 6: CALCULATING THE EQUIVALENT WIDTH EXCESSES')
################################################################################

def equivalent_width(wavelengths, data, gaussian_fit, continuum_fit, shift='blue', peak_mask=(0,1000)):
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
    fraction = (data - gaussian_fit) / continuum_fit
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

# -------INPUTS-------- #
# Be sensible with intervals or you'll break code. (near peak first, far peak second)
blue_peak_mask = (0,90) # number of angstroms to cut around the peak, blue minus.
red_peak_mask = (0,90) # number of angstroms to cut around the peak, red plus.
# --------------------- #

# Calculating the equivalent width excess data points for the PYTHON Grid
fitted_grid, fit_con, fit_parameters, initials_all, pcov_all, infodict_all = fit_procedure(wavelengths, grid) #TODO  this causes the covariance issue
blue_ew_excess = equivalent_width(wavelengths, grid, fitted_grid, fit_con, shift='blue', peak_mask=blue_peak_mask)
red_ew_excess = equivalent_width(wavelengths, grid, fitted_grid, fit_con, shift='red', peak_mask=red_peak_mask)

# Calculating the equivalent width excess errors from resampling grid
resampled_blue_ew_excess = np.array([])
resampled_red_ew_excess = np.array([])
samples = 150
for sample in tqdm(range(samples)):
    resampled_grid = resample_data(wavelengths, grid, sk_noise_error) # resampled grid
    fitted_grid, fit_con, fit_parameters, initials_all, _, _ = fit_procedure(wavelengths, resampled_grid)
    blue = equivalent_width(wavelengths, resampled_grid, fitted_grid, fit_con, shift='blue', peak_mask=blue_peak_mask)
    red = equivalent_width(wavelengths, resampled_grid, fitted_grid, fit_con, shift='red', peak_mask=red_peak_mask)
    resampled_blue_ew_excess = np.append(resampled_blue_ew_excess, blue)
    resampled_red_ew_excess = np.append(resampled_red_ew_excess, red)
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

# %%
################################################################################
print('TOOL: ANIMATED PLOTTING')
################################################################################

%matplotlib qt

# Plotting a grid of data as an animated plot. 
i_want_to_view_the_fits = False # right now, not later! You impatient mortal!

if i_want_to_view_the_fits:
    def slider_update(val):
        ax.clear()
        ax.plot(wavelengths, grid[int(val)], label='Original Data', color='black') # y=dictionary
        ax.plot(wavelengths, fitted_grid[int(val)], label='Optimal Gaussian with Continuum', color='red')
        ax.plot(wavelengths, fit_con[int(val)], label='Fitted Continuum', color='blue')
        ax.plot(wavelengths, 
                gaussian_with_continuum(wavelengths, *initials_all[int(val)]), 
                label='Initial Gaussian with Continuum', 
                color='grey'
                )
        ax.plot(wavelengths, sk_con_data[int(val)], label='Sklearn Fitted Continuum', color='green')
        sum_squared = sum((grid[int(val)]-fitted_grid[int(val)])**2)
        ax.scatter(wavelengths, fitted_grid[int(val)], color = 'red')
        ax.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
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

# %%
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
                        'pcov_all': pcov_all.tolist(), 
                        'infodict_all': infodict_all.tolist(), 
                        'blue_ew_excess': blue_ew_excess, 
                        'red_ew_excess': red_ew_excess, 
                        'blue_ew_excess_error': blue_ew_excess_error.tolist(), 
                        'red_ew_excess_error': red_ew_excess_error.tolist(),
                        'sk_con_data': sk_con_data.tolist(),
                         'sk_slopes': sk_slopes.tolist(),
                        'sk_intercepts': sk_intercepts.tolist(),
                        'sk_noise_error': sk_noise_error.tolist(),
                        'run_number': run_number.tolist()
                      }

    script_ran_once = True
    
if not grid_mixed:
    final_results = {'wavelength_grid': wavelength_grid.tolist(),
                        'grid': grid.tolist(),
                        'fitted_grid': fitted_grid.tolist(),
                        'fit_con': fit_con.tolist(),
                        'fit_parameters': fit_parameters.tolist(),
                        'initials_all': initials_all.tolist(),
                        'pcov_all': pcov_all.tolist(),
                        'infodict_all': infodict_all.tolist(),
                        'blue_ew_excess': blue_ew_excess,
                        'red_ew_excess': red_ew_excess,
                        'blue_ew_excess_error': blue_ew_excess_error.tolist(),
                        'red_ew_excess_error': red_ew_excess_error.tolist(),
                        'sk_con_data': sk_con_data.tolist(),
                        'sk_slopes': sk_slopes.tolist(),
                        'sk_intercepts': sk_intercepts.tolist(),
                        'sk_noise_error': sk_noise_error.tolist(),
                        'run_number': grid_length.tolist()
                        }
    print("You can skip Step 8 and move onto Step 9")
 
# %%
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
                        'pcov_all': pcov_all.tolist(),
                        'infodict_all': infodict_all.tolist(),
                        'blue_ew_excess': blue_ew_excess,
                        'red_ew_excess': red_ew_excess,
                        'blue_ew_excess_error': blue_ew_excess_error.tolist(),
                        'red_ew_excess_error': red_ew_excess_error.tolist(),
                        'sk_con_data': sk_con_data.tolist(),
                        'sk_slopes': sk_slopes.tolist(),
                        'sk_intercepts': sk_intercepts.tolist(),
                        'sk_noise_error': sk_noise_error.tolist(),
                        'run_number': mixed_runs
                        }
    
    # Combining any previous results together, use final_results dictionary
    # for code from here on out.
    final_results = {key : [] for key in mixed_results.keys()}                 

    for run in grid_length:
        if run in stored_results['run_number']:
            index = stored_results['run_number'].index(run)
            for key in final_results:
                final_results[key].append(stored_results[key][index])        
        elif run in mixed_runs:
            index = mixed_results['run_number'].index(run)
            for key in final_results:
                final_results[key].append(mixed_results[key][index])
            

# %%
################################################################################
print("STEP 9: PLOTTING THE EW EXCESSES FOR ALL RUNS AGAINST TEO'S DATA")
################################################################################

# EVERY RUN IS PLOTTED HERE FOR THE EQUIVALENT WIDTH EXCESSES, NO BAD FITS REMOVED
# YOU CAN POTENTIALLY SKIP THIS STEP IF YOU WANT TO REMOVE BAD FITS STRAIGHT AWAY.

%matplotlib qt

# loading Teo's data from csv files
bz_cam = np.loadtxt('Teo_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Teo_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Teo_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Teo_data/V751 Cyg.csv', delimiter=',')

# plot concentric circles of a given radius
def circle(radius):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    return (a, b)

ranges = slice(0,-1) # to select particular portions of the grid if desired
plt.errorbar(final_results['red_ew_excess'][ranges],
             final_results['blue_ew_excess'][ranges], 
             xerr=final_results['red_ew_excess_error'][ranges], 
             yerr=final_results['blue_ew_excess_error'][ranges], 
             fmt='none', 
             ecolor = 'grey', 
             alpha=0.5,
             zorder=2
             ) # error bars for scatterplot below
target = plt.scatter(final_results['red_ew_excess'][ranges],
                     final_results['blue_ew_excess'][ranges],
                     c=grid_length[ranges],
                     s=15,
                     label='Grid Data',
                     cmap='rainbow',
                     zorder=3
                     ) # scatter plot assigned as target for colour bar
plt.colorbar(target, label='Run Number') # colour bar

# Plotting Teo's data
plt.scatter(bz_cam[:,0], bz_cam[:,1], label='BZ Cam', color='black', s=10, marker='v')
plt.scatter(mv_lyr[:,0], mv_lyr[:,1], label='MV Lyr', color='black', s=10, marker='^')
plt.scatter(v425_cas[:,0], v425_cas[:,1], label='V425 Cas', color='black', s=10, marker='<')
plt.scatter(v751_cyg[:,0], v751_cyg[:,1], label='V751 Cyg', color='black', s=10, marker='>')

# vertical and horizontal lines at 0 i.e axes
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)

# plot formatting
plt.xlabel('Red Wing EW Excess ($Å$)')
plt.ylabel('Blue Wing EW Excess ($Å$)')
plt.title('Red vs Blue Wing Excess')
plt.xlim(min(final_results['red_ew_excess'])-2,max(final_results['red_ew_excess'])+2)
plt.ylim(min(final_results['blue_ew_excess'])-2,max(final_results['blue_ew_excess'])+2)
plt.legend()
plt.show()

# %%
################################################################################
print('STEP 10: CUTTING POOR DATA FITS LIKE BIMODAL EMISSION LINES OR NO VISIBLE EMISSION LINES')
################################################################################

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
    if np.max(cut_flux) < 2e-15: # flux limit to determine if flat
        cut_runs.append(run)

# finding the high error runs
# creating histrograms of the FRMS's and RMS's from the original data,gaussian fits and continuum data
fig, ax = plt.subplots(1,4, figsize=(15,5))

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

# working out error data for each spectra and fit
frms_data = frms(final_results['grid'], final_results['fitted_grid'], final_results['fit_con'])
rms_data = rms(final_results['grid'], final_results['fitted_grid'], final_results['fit_con'])
chi_2_data = chi_2(final_results['grid'], final_results['fitted_grid'], final_results['sk_noise_error'])
rss_data = rss(final_results['grid'], final_results['fitted_grid'])

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
plt.show()

# Plotting the error data as a function of run number (i.e which run is bad)
fig, ax = plt.subplots(1,4, figsize=(15,5))
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
        
cut_runs = np.append(cut_runs, cut_runs_2) # adding high error runs to flat runs
cut_runs = np.unique(cut_runs) # removing duplicates

# %%
################################################################################
print('STEP 11: REPLOTTING THE EW EXCESSES WITHOUT THE CUT RUNS')
################################################################################

%matplotlib qt

incs = [0 for i in range(10)] # to indent incs to the same column index as files
[incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

# removing bad and flat fits from the data for the final plot of the EW excesses
cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
cut_grid_length = np.delete(grid_length, cut_runs)

# Loading Teo's data from csv files
bz_cam = np.loadtxt('Teo_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Teo_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Teo_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Teo_data/V751 Cyg.csv', delimiter=',')

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
    # plotting the mask lines where the equalivalent width is calculated
    ax[1].axvline(x=H_alpha-blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha-blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha+red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
    ax[1].axvline(x=H_alpha+red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
    ax[1].set_xlabel('Wavelength ($Å$)') # plot formatting
    ax[1].set_ylabel('Flux')
    ax[1].legend()
    
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
                        c=cut_grid_length,
                        s=15,
                        label='Grid Data',
                        cmap='rainbow', 
                        zorder=3
                        ) # scatter plot assigned as target for colour bar
    plt.colorbar(target, label='Run Number', ax=ax[0], pad=0.1) # colour bar
    ax[0].scatter(bz_cam[:,0], bz_cam[:,1], label='BZ Cam', color='black', s=10, marker='v')
    ax[0].scatter(mv_lyr[:,0], mv_lyr[:,1], label='MV Lyr', color='black', s=10, marker='^')
    ax[0].scatter(v425_cas[:,0], v425_cas[:,1], label='V425 Cas', color='black', s=10, marker='<')
    ax[0].scatter(v751_cyg[:,0], v751_cyg[:,1], label='V751 Cyg', color='black', s=10, marker='>')
    # plot formatting
    ax[0].set_xlabel('Red Wing EW Excess ($Å$)')
    ax[0].set_ylabel('Blue Wing EW Excess ($Å$)')
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
    ax[0].text(xlim[0]-2.5, ylim[1]+0.8, 'Reverse\nP-Cygni?', fontsize=12, color='black', fontweight='bold')
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

# %%
################################################################################
print('STEP 12: FINDING TRENDS IN THE DATA')
################################################################################

#TODO work on finding trends in the data

# importing the run parameter combinations
# Loading run files parameter combinations from pretty table file
path_to_table = 'Grids/Wider_Ha_grid_spec_files/Grid_runs_full_table.txt'
ascii_table = np.genfromtxt(f'{path_to_table}',
                    delimiter='|',
                    skip_header=3,
                    skip_footer=1,
                    dtype=float
                    )

# removing nan column due to pretty table
ascii_table = np.delete(ascii_table, 0, 1) # array, index position, axis
parameter_table = np.delete(ascii_table, -1, 1)

sim_parameters = ['$\dot{M}_{disk}$',
        '$\dot{M}_{wind}$',
        '$KWD.d$',
        '$r_{exp}$',
        '$acc_{length}$',
        '$acc_{exp}$'
        ]

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
    fig.text(0.8, 0.70, 'The more faded the datapoint, \n the less confidence in the value', ha='center', fontsize=30)
    # plt.subplots_adjust(left=-0.3, wspace=0.05, hspace=0.05)
    # plt.tight_layout()
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
%matplotlib inline
blue_3d()
red_2d()

# %%

# create a even grid space between the interval of 1 and 5

for i in range(1,6):
    for j in range(1,6):
        plt.scatter(i, j, color='blue', s=50)
        
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Even Grid Space')
plt.show()


# %%
################################################################################
print('TOOL: Angstrom to km/s conversion and vice versa')
################################################################################
def angstrom_to_kms(wavelength):
    """Converts wavelength in angstroms from central h_alpha line to velocity in km/s.
    Args:
        wavelength (float): wavelength in angstroms"""
    kms = (wavelength - H_alpha) * 299792.458 / H_alpha
    return print(f'{wavelength}Å = {kms}km/s')
    
def kms_to_angstrom(velocity):
    """Converts velocity in km/s to wavelength in angstroms from central h_alpha line.
    Args:
        velocity (float): velocity in km/s"""  
    angstrom = H_alpha * (velocity / 299792.458) + H_alpha
    return print(f'{velocity}km/s = {angstrom}Å')

print(H_alpha)
kms_to_angstrom(1000) #22Å
kms_to_angstrom(4000) #54Å(2500) 88Å(4000)

################################################################################
print('END OF CODE')
################################################################################


































































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
# def equivalent_width(wavelengths, fluxes, continuum):
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
# equi_w = equivalent_width(wavelengths, grid[grid_index], con)
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



# def equivalent_width(wavelengths, data, gaussian_fit, continuum_fit, shift='blue', peak_mask=(0,1000)):
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