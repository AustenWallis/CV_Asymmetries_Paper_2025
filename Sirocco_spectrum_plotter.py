################################################################################
################################################################################
#  ____  _                           ____                  _             
# / ___|(_)_ __ ___   ___ ___ ___   / ___| _ __   ___  ___| |_ _ __ __ _ 
# \___ \| | '__/ _ \ / __/ __/ _ \  \___ \| '_ \ / _ \/ __| __| '__/ _` |
#  ___) | | | | (_) | (_| (_| (_) |  ___) | |_) |  __/ (__| |_| | | (_| |
# |____/|_|_|  \___/ \___\___\___/  |____/| .__/ \___|\___|\__|_|  \__,_|
#                                         |_|                            
################################################################################
################################################################################
# For plotting and inspecting the spectra of any sirocco output file
# Place your .spec files in the spectra folder and run the script
################################################################################
################################################################################

# %%
################################################################################
print('STEP 1: IMPORTING MODULES')
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm
import time
import matplotlib.widgets as widgets
from matplotlib.widgets import Button, Slider
import scienceplots
import pandas as pd
#import pysi

plt.style.use('science')

#plt.style.use('Solarize_Light2')
# %%
################################################################################
print('STEP 2: CHECKING THE FILES ARE PRESENT')
################################################################################

# --- USER INPUTS --- #
# Add the path from this sirocco file to your grid files
path_to_grids = "Release_Ha_grid_spec_files"

# Add the run numbers of the files you ran. This double checks all the spec
# files you expected are actually present. You likely have this list when you
# submit a job to a slurm computing cluster. 
run_number = np.arange(0,729)
#run_number = [77,79,158,160,240,241,320,322,414,425,468,477,478,479,480,481,482,483,563,565,567,569,580,639,640,641,642,643,644,645,646,667,668,669,712,713,720,721,722,723,724,725,726,727]
# 240,241,320,322
# Add the user chosen inclinations from your .spec files
inclinations = [20,45,60,72.5,85]

# ------------------- #

fluxes = {}
wavelengths = {}
print(f'You have {len(os.listdir(path_to_grids))} files in this directory')

# Checking if all the files exist
for run in run_number:
    file = f'{path_to_grids}/rerun{run}.spec' # Spec file name here
    if os.path.isfile(file):
        pass
    else:
        print(f'File {file} does not exist')
        continue
# %%
################################################################################
print('STEP 3: LOADING THE GRID')
################################################################################

# Loading run files data to variables
columns = np.arange(10, (10+len(inclinations))) # loadtxt column numbers
for run in tqdm(run_number):
    file = f'{path_to_grids}/rerun{run}.spec'
    wavelengths[run] = np.loadtxt(file, usecols=(1), skiprows=81)
    fluxes[run] = np.loadtxt(file, usecols=(columns), skiprows=81) # cols=incs

# csv file
parameter_table = np.genfromtxt(f'{path_to_grids}/Grid_runs_logfile.csv',
                    delimiter=',',
                    skip_header=1,
                    dtype=float
                    )
# Loading run files parameter combinations from pretty table file
# ascii_table = np.genfromtxt(f'{path_to_grids}/Grid_runs_logfile.txt',
#                     delimiter='|',
#                     skip_header=3,
#                     skip_footer=1,
#                     dtype=float
#                     )

# # removing nan column due to pretty table
# ascii_table = np.delete(ascii_table, 0, 1) # array, index position, axis
# parameter_table = np.delete(ascii_table, -1, 1)

#%%
# calculating the line luminosity of the H_alpha line from fluxes
wavelengths_increasing = [np.flip(wavelengths[run]) for run in run_number]
fluxes_increasing = [np.flip(fluxes[run], axis=0) for run in run_number]


# Converting fluxes to luminosities
distance_sq = (100 * 3.086e18)**2 # (100 parsecs in cm) ^2

# (erg/s/cm^2/Å --> ergs/s)
luminosity_spec = {run: fluxes_increasing[run][:,1] * wavelengths_increasing[run] * 4 * np.pi * distance_sq for run in run_number} 
total_luminosity = {run: np.trapz(luminosity_spec[run], wavelengths_increasing[run]) for run in run_number}
# convert to array
total_luminosity_array = np.array([total_luminosity[run] for run in run_number])
np.save('total_luminosity.npy', total_luminosity_array)

# # Create a function to compute the integration using bin centres
# def integrate_with_bin_centres(wavelengths, luminosity):
#     # Calculate bin centres as the average of adjacent wavelength points
#     bin_centres = 0.5 * (wavelengths[:-1] + wavelengths[1:])
#     # Similarly, compute the flux (luminosity per wavelength) at the bin centres
#     # Here we take the average of adjacent luminosity values.
#     luminosity_mid = 0.5 * (luminosity[:-1] + luminosity[1:])
#     # Integrate using the trapezoidal rule over the bin widths.
#     return np.sum(luminosity_mid * np.diff(wavelengths))

# # Now apply this to your dictionary for each run
# luminosity_spec = {
#     run: fluxes_increasing[run][:, 1] * wavelengths_increasing[run] * 4 * np.pi * distance_sq
#     for run in run_number
# }

# total_luminosity = {
#     run: integrate_with_bin_centres(wavelengths_increasing[run], luminosity_spec[run])
#     for run in run_number
# }

# total_luminosity_array = np.array([total_luminosity[run] for run in run_number])
# %%STEP 4: ANIMATED PLOT OF YOUR GRID
################################################################################
print('STEP 4: ANIMATED PLOT OF YOUR GRID')
################################################################################
%matplotlib qt

def slider_update(val):
    run = run_number[val]
    ax.clear()
    #ax.set_xlim(6425, 6700)
    y_flux_lim = 0
    for i in range(len(inclinations)):
        flux = fluxes[run][:, i]
        indexes = np.where((wavelengths[run] > 4000) & (wavelengths[run] < 7000))
        max_flux = np.max(flux[indexes[0][0]:indexes[0][-1]])
        if max_flux > y_flux_lim:
            y_flux_lim = max_flux
    #ax.set_ylim(0, y_flux_lim*1.3)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux ($erg/s/cm^2/Å$)')
    ax.set_title('H_α of CV for Run ' + str(run))
    ax.set_xlim(6450, 6650)
    
    #for i in range(len(inclinations)):
    for i in range(len(inclinations)):
        ax.plot(wavelengths[run], fluxes[run][:, i], label=f'{inclinations[i]}°')
        #ax.scatter(wavelengths[run], fluxes[run][:, i], s=5, color='black')
    # Add text box with parameter values
    # Construct each piece as a list
    vals = [
        rf'$\dot{{M}}_{{disk}}={parameter_table[run, 1]:.2e}$',
        rf'$\dot{{M}}_{{wind}}={parameter_table[run, 2]:.2e}$',
        rf'$d={parameter_table[run, 3]:.2f}$',
        rf'$r_{{exp}}={parameter_table[run, 4]:.2f}$',
        rf'$a_{{l}}={parameter_table[run, 5]:.2e}$',
        rf'$a_{{exp}}={parameter_table[run, 6]:.2f}$'
    ]

    # Join them with some spacing or separators
    textstr = '   '.join(vals)

    props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

    ax.text(
        0.02,                # x-position in axes fraction
        1.07,                # y-position in axes fraction (just above the top)
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='bottom',   # anchor so it stays above the axis
        bbox=props
    )
    # textstr = '\n'.join((
    #     r'$\dot{M}_{disk}=%.2e$' % (parameter_table[run, 1], ),
    #     r'$\dot{M}_{wind}=%.2e$' % (parameter_table[run, 2], ),
    #     r'$d=%.2f$' % (parameter_table[run, 3], ),
    #     r'$r_{exp}=%.2f$' % (parameter_table[run, 4], ),
    #     r'$a_{l}=%.2e$' % (parameter_table[run, 5], ),
    #     r'$a_{exp}=%.2f$' % (parameter_table[run, 6], )))
    # props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # ax.text(0.85, 1.15, textstr, transform=ax.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)
    ax.legend(bbox_to_anchor=(0.242, -0.08), loc='upper left', ncol=5)
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(frame):
    if anim.running:
        if grid_slider.val == len(run_number)-1:
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
grid_slider = Slider(ax_slider, 'Run', 0, len(run_number), valinit=0, valstep=1) 
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
                    frames=len(run_number),
                    interval=300
                    ) # setting up animation
anim.running = True # setting off animation

# %%plotting a single run 
# plotting a single run 
#%matplotlib inline

file3 = 'run233.spec'
run_num = 233
incs = [20,45,60,72.5,85]
wavelength3 = np.loadtxt(file3, usecols=(1), skiprows=81)
flux3 = np.loadtxt(file3, usecols=(10,11,12,13,14), skiprows=81)
fig, ax = plt.subplots(5, 1, figsize=(12, 25))
plt.tight_layout(pad=3.0)
for i in range(5):
    ax[i].plot(wavelength3, flux3[:, i])
    #ax[i].scatter(wavelength3, flux3[:, i], s=10)
    # ax[i].set_xlim(6350,6750)
    # ax[i].set_ylim(0, 5e-13)
    ax[i].set_xlabel('Wavelength (A)')
    ax[i].set_ylabel('Flux ($erg/s/cm^2/Å$)')
    ax[i].set_title('Spectrum of CV at ' + str(incs[i]) + ' degrees')

# ax[1].plot(wavelengths[run_num], fluxes[run_num][:, 3])
# ax[2].plot(wavelengths[run_num], fluxes[run_num][:, 6])
# ax[3].plot(wavelengths[run_num], fluxes[run_num][:, 8])
# ax[4].plot(wavelengths[run_num], fluxes[run_num][:, -1])
plt.show()

# %% FIGURE 4 CV PAPER 4 SIROCCO SUBPLOTS
################################################################################
print('FIGURE 4: CV PAPER 4 SIROCCO SUBPLOTS')
################################################################################
%matplotlib inline
# Multi plot several runs on the same plot
fig, ax = plt.subplots(2, 2, figsize=(20, 15))
#increase text size
plt.rcParams.update({'font.size': 20})

def to_latex_sci(value, precision=2):
    # e.g., "3.00e-09" -> "3.00\times 10^{-9}"
    s = f"{value:.{precision}e}"      # format as scientific e.g. "3.00e-09"
    mantissa, exponent = s.split('e') # split into "3.00" and "-09"
    exponent = exponent.replace('+', '')   # remove any "+"
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"

run_nums = [34, 251, 701, 652] # 34, 260(493), 503, 566
incs = [20,45,60,72.5,85]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(incs)))
for axis, run_num in enumerate(run_nums):
    for i in range(5):
        if axis == 0:
            ax[axis // 2, axis % 2].plot(wavelengths[run_num],
                                         fluxes[run_num][:, i],
                                         label=f'{incs[i]}°', 
                                         linewidth=3,
                                         color=colors[i])
        else:
            ax[axis // 2, axis % 2].plot(wavelengths[run_num],
                                         fluxes[run_num][:, i],
                                         linewidth=3,
                                         color = colors[i])
        # Add text box with parameter values
        textstr = '\n'.join((
            rf'$\dot{{M}}_{{disk}}:{to_latex_sci(parameter_table[run_num, 1])}$',
            rf'$\dot{{M}}_{{wind}}:{to_latex_sci(parameter_table[run_num, 2])}$',
            rf'$d:{parameter_table[run_num, 3]:.2f}$',
            rf'$r_{{exp}}:{parameter_table[run_num, 4]:.2f}$',
            rf'$a_{{l}}:{to_latex_sci(parameter_table[run_num, 5])}$',
            rf'$a_{{exp}}:{parameter_table[run_num, 6]:.2f}$'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey')
        if axis >= 2:
            ax[axis // 2, axis % 2].text(0.695, 
                                        0.97, 
                                        textstr, 
                                        transform=ax[axis // 2, axis % 2].transAxes,
                                        fontsize=18,
                                        verticalalignment='top',
                                        bbox=props
                                        )
        else:
            ax[axis // 2, axis % 2].text(0.685, 
                                    0.97, 
                                    textstr, 
                                    transform=ax[axis // 2, axis % 2].transAxes,
                                    fontsize=18,
                                    verticalalignment='top',
                                    bbox=props
                                    )
        ax[axis // 2, axis % 2].set_xlim(6400,6700)
        ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
        ax[axis // 2, axis % 2].set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathring{A}^{-1}$)')
        plt.rcParams.update({'font.size': 20})

        
# Add a text box with the word inclination to the top of the figure

fig.text(0.27, 0.9085, 'Inclinations:', ha='center', fontsize=22)
fig.text(0.14, 0.83, '(a)', fontsize=40)
fig.text(0.564, 0.83, '(b)', fontsize=40)
fig.text(0.14, 0.41, '(c)', fontsize=40)
fig.text(0.564, 0.41, '(d)', fontsize=40)

fig.legend(loc='upper center', bbox_to_anchor=(0.54, 0.94), ncol=5, fontsize=22)

plt.show()

# %%'STEP 5: SIROCCO LINE STATISTICS'
################################################################################
print('STEP 5: SIROCCO LINE STATISTICS')
################################################################################
# 45 degree inclination
# We are going to find the mean, median, mode, standard deviation, skewness and 
# kurtosis(with the mean/mode/median/stdev in velocity units).
# We are going to add the statistics to a table

from scipy.stats import skew, kurtosis, mode, describe

def angstrom_to_kms(wavelength):
    """Converts wavelength in angstroms from central h_alpha line to velocity in km/s.
    Args:
        wavelength (float): wavelength in angstroms"""
    kms = (wavelength - H_alpha) * 299792.458 / H_alpha
    return kms#, print(f'{wavelength}Å = {kms}km/s')
    
def kms_to_angstrom(velocity):
    """Converts velocity in km/s to wavelength in angstroms from central h_alpha line.
    Args:
        velocity (float): velocity in km/s"""  
    angstrom = H_alpha * (velocity / 299792.458) + H_alpha
    return angstrom, print(f'{velocity}km/s = {angstrom}Å')

table_df = pd.DataFrame(columns=['Run', 'Mean', 'Median', 'Mode', 'Stdev', 'Skewness', 'Kurtosis'])
H_alpha = 6562.819 # Å
velocities = {}

for run in run_number:
    indexes = np.where((wavelengths[run] > 6260) & (wavelengths[run] < 6860))
    flux = fluxes[run][indexes[0][0]:indexes[0][-1], 1]
    #converting wavelengths to velocities 
    velocities[run] = [angstrom_to_kms(w) for w in wavelengths[run]]
    mean = describe(flux).mean
    median = np.median(flux)
    mode_value = mode(flux).mode
    stdev = describe(flux).variance**0.5
    skewness_value = skew(flux)
    kurtosis_value = kurtosis(flux)

    table_df.loc[run] = {
        'Run': run,
        'Mean': mean,
        'Median': median,
        'Mode': mode_value,
        'Stdev': stdev,
        'Skewness': skewness_value,
        'Kurtosis': kurtosis_value
    }

#plot an individual run
run = 701
fig, ax = plt.subplots(figsize=(12, 8))
indexes = np.where((wavelengths[run] > 6260) & (wavelengths[run] < 6860))
plt.plot(velocities[run][indexes[0][0]:indexes[0][-1]], fluxes[run][indexes[0][0]:indexes[0][-1], 1])
#plot a virtual line at the central h_alpha line
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Radial Velocities ($m/s$)')
plt.ylabel('Flux ($erg/s/cm^2/Å$)')
plt.title('H alpha of CV for Run ' + str(run))
plt.show()

# #plot the cdf of the flux
# fig, ax = plt.subplots(figsize=(12, 8))
# plt.hist(flux, bins=1000, cumulative=True, histtype='step', density=True)
# plt.xlabel('Flux ($erg/s/cm^2/Å$)')
# plt.ylabel('Cumulative Probability')
# plt.title('CDF of Flux for Run ' + str(run))
# plt.show()

# # plot histogram of the flux
# fig, ax = plt.subplots(figsize=(12, 8))
# plt.hist(flux, bins=1000, histtype='step', density=True)
# plt.xlabel('Flux ($erg/s/cm^2/Å$)')
# plt.ylabel('Probability')
# plt.title('Histogram of Flux for Run ' + str(run))
# plt.show()



# %%# 'EMISSION MEASUREMENTS'
################################################################################
print('EMISSION MEASUREMENTS')
################################################################################
emission_measures = []
ne_sums = []
vol_sums = []
for run in range(0,729):
    if f'run{run}.master.txt' not in os.listdir('Sirocco_cv_grid_masters'):
        print(f'Run {run} does not exist')
        continue
    master_paths = f'Sirocco_cv_grid_masters/run{run}.master.txt'
    master_df = pd.read_csv(f'{master_paths}',  delim_whitespace=True)
    vol = master_df['vol'] # volume in cm^3
    vol_sum = vol.sum()
    vol_sums.append(vol_sum)

    ne = master_df['ne'] # electron density in cm^-3
    ne_sum = ne.sum()
    ne_sums.append(ne_sum)
    
    print(run, f'{vol_sum:.2e}', f'{ne_sum:.2e}')
    em = ne_sum**2 * vol_sum # EM = n_e^2 * V
    emission_measures.append(em)
# save
np.save('emission_measures.npy', emission_measures) # 618
# %%
emission_measures = []
ne_sums = []
vol_sums = []
for run in range(0,729):
    if f'run{run}.master.txt' not in os.listdir('Sirocco_cv_grid_masters'):
        print(f'Run {run} does not exist')
        continue
    master_paths = f'Sirocco_cv_grid_masters/run{run}.master.txt'
    master_df = pd.read_csv(f'{master_paths}',  delim_whitespace=True)
    vol = master_df['vol'] # volume in cm^3
    #vol_sum = vol.sum()
    #vol_sums.append(vol_sum)

    ne = master_df['ne'] # electron density in cm^-3
    #ne_sum = ne.sum()
    #ne_sums.append(ne_sum)
    
    #print(run, f'{vol:.2e}', f'{ne:.2e}')
    em = ne**2 * vol # EM = n_e^2 * V
    em_sum = em.sum()
    emission_measures.append(em_sum)
    #emission_measures.append(em)
# save
np.save('emission_measures.npy', emission_measures) # 618
# %%
run = 701
master_df = pd.read_csv(f'Sirocco_cv_grid_masters/run{run}.master.txt',  delim_whitespace=True)
description_dict = {
    "x": "left-hand lower cell corner x-coordinate, cm", 
    "z": "left-hand lower cell corner z-coordinate, cm", 
    "xcen": "cell centre x-coordinate, cm", 
    "zcen": "cell centre z-coordinate, cm", 
    "i": "cell index (column)", 
    "j": "cell index (row)", 
    "inwind": "is the cell in wind (0), partially in wind (1) or out of wind (<0)", 
    "converge": "how many convergence criteria is the cell failing?", 
    "v_x": "x-velocity, cm/s", 
    "v_y": "y-velocity, cm/s", 
    "v_z": "z-velocity, cm/s",  
    "vol": "volume in cm^3", 
    "rho": "density in g/cm^3", 
    "ne": "electron density in cm^-3", 
    "t_e": "electron temperature in K", 
    "t_r": "radiation temperature in K",  
    "h1": "H1 ion fraction", 
    "he2": "He2 ion fraction", 
    "c4": "C4 ion fraction",  
    "n5": "N5 ion fraction", 
    "o6": "O6 ion fraction", 
    "dmo_dt_x": "momentum rate, x-direction", 
    "dmo_dt_y": "momentum rate, y-direction", 
    "dmo_dt_z": "momentum rate, z-direction", 
    "ip": "U ionization parameter", 
    "xi": "xi ionization parameter", 
    "ntot": "total photons passing through cell", 
    "nrad": "total wind photons produced in cell", 
    "nioniz": "total ionizing photons passing through cell"
}

pivoted_ne = master_df.pivot(index='j', columns='i', values='ne')
pivoted_ne[pivoted_ne < 1e-10] = 0
pivoted_ne = pivoted_ne.replace(0, np.nan)
log_ne = np.log10(pivoted_ne)

# STEP 2: Convert to log scale if desired
# Avoid log(0) or negative issues by adding a small offset if needed
#log_ne = np.log10(pivoted_ne)

# STEP 3: Plot the 2D map
plt.figure(figsize=(8,6))
# Option A: imshow
plt.imshow(log_ne, 
           origin='lower',   # so that j=0 is at the bottom
           aspect='auto',    # or 'equal' depending on your preference
           cmap='viridis')

plt.colorbar(label='log ne')  # color scale legend
plt.xlabel('i')
plt.ylabel('j')
plt.title('Log Electron Density')
plt.show()

# %%
master_df['em'] = master_df['ne']**2 * master_df['vol']

# Pivot the data onto a 2D grid using cell indices 'i' and 'j'
pivoted_em = master_df.pivot(index='j', columns='i', values='em')

# Optional: Set very small values to 0 (e.g., below 1e-20) and then replace zeros with NaN
# Adjust the threshold as needed for your data
pivoted_em[pivoted_em < 1e2] = 0
pivoted_em = pivoted_em.replace(0, np.nan)

# Take the logarithm for a better visual dynamic range
log_em = np.log10(pivoted_em)

# Create the plot
plt.figure(figsize=(8,6))
plt.imshow(log_em, 
           origin='lower',   # so that j=0 is at the bottom
           aspect='auto', 
           cmap='plasma')    # using a different colormap for variety
plt.colorbar(label='log EM)')
plt.xlabel('i (cell index)')
plt.ylabel('j (cell index)')
plt.title(f'Log Emission Measure for Run {run}')
plt.show()
# %%
# Pivot the data using real cell center coordinates
pivoted_ne_real = master_df.pivot(index='zcen', columns='xcen', values='ne')

# Apply a threshold: values below 1e-10 become 0, then replace 0 with NaN
pivoted_ne_real[pivoted_ne_real < 1e-10] = 0
pivoted_ne_real = pivoted_ne_real.replace(0, np.nan)

# Take the log for better dynamic range
log_ne_real = np.log10(pivoted_ne_real)

# Determine the extent for imshow using the unique coordinate values
xcoords = np.sort(master_df['xcen'].unique())
zcoords = np.sort(master_df['zcen'].unique())
extent = [xcoords.min(), xcoords.max(), zcoords.min(), zcoords.max()]

# Plotting
plt.figure(figsize=(8,6))
plt.imshow(log_ne_real, 
           origin='lower', 
           aspect='auto', 
           cmap='viridis', 
           extent=extent)
plt.colorbar(label='log ne')
plt.xlabel('x (cm)')
plt.ylabel('z (cm)')
plt.title('Log Electron Density (Real Coordinates)')
plt.show()

pivoted_em_real = master_df.pivot(index='z', columns='x', values='em')

# Apply a threshold: values below 1e-10 become 0, then replace 0 with NaN
pivoted_em_real[pivoted_em_real < 1e2] = 0
pivoted_em_real = pivoted_em_real.replace(0, np.nan)

# Take the log for better dynamic range
log_em_real = np.log10(pivoted_em_real)

# Plotting
plt.figure(figsize=(8,6))
plt.imshow(log_em_real,
              origin='lower',
                aspect='auto',
                cmap='plasma',
                extent=extent)
plt.colorbar(label='log EM')
plt.xlabel('x (cm)')
plt.ylabel('z (cm)')
plt.title('Log Emission Measure (Real Coordinates)')
plt.show()


################################################################################
# END OF CODE
################################################################################
# %%



















































################################################################################
# OLD CODE I DON'T HAVE THE HEART TO DELETE INCASE I NEED IT LATER FOR SOMETHING
################################################################################

#file = 'run118_iridis_10m_photons_87b/run118_WMdot2p5e-8_d12_vinf2_time_test.spec' # 10m photons iridis
#file = '../large_optical_grid_tests_3/run154_low_large_optical_cv.spec'
# for run in run_number:
#     file = f'../optical_hypercube_spectra/run{run}.spec'
#     wavelength = np.loadtxt(file, usecols=(1), skiprows=81)
#     flux = np.loadtxt(file, usecols=(10,11,12,13,14,15,16,17,18,19,20,21), skiprows=81)
#inclinations = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

    # #file2 = 'broad_short_spec_cv_grid/run118_WMdot2p5e-08_d12_vinf2.spec' # 10m photons local
    # #file2 = '../large_optical_grid_tests_3/run155_mid_large_optical_cv.spec'
    # file2 = '../optical_hypercube_spectra/run1.spec'
    # wavelength2 = np.loadtxt(file2, usecols=(1), skiprows=81)
    # flux2 = np.loadtxt(file2, usecols=(10,11,12,13,14,15,16,17,18,19,20,21), skiprows=81)

    # #file3 = 'run118_100m_photons/run118_WMdot2p5e-8_d12_vinf2_time_test.spec' # 100m photons local
    # #file3 = '../large_optical_grid_tests_3/run156_high_large_optical_cv.spec'
    # wavelength3 = np.loadtxt(file2, usecols=(1), skiprows=81)
    # flux3 = np.loadtxt(file3, usecols=(10,11,12,13,14,15,16,17,18,19,20,21), skiprows=81)

    #Plotting a 11 flux plots for different inclinations
#     fig, ax = plt.subplots(6, 2, figsize=(25, 25))
#     fig.tight_layout(pad=3.0)
#     for i in range(12):

#         ax[i//2, i%2].loglog(wavelength, flux[:, i], label='low') # label='iridis run 10m photons'
#         #ax[i//2, i%2].loglog(wavelength2, flux2[:, i], label='mid') # label='local run 10m photons'
#         #ax[i//2, i%2].loglog(wavelength3, flux3[:, i]+1e-13, label='high') # label='local run 100m photons'
#         ax[i//2, i%2].set_xlim(4100,7900)
#         ax[i//2, i%2].set_ylim(0, 3e-12)
#         ax[i//2, i%2].set_xlabel('Wavelength (Angstroms)')
#         ax[i//2, i%2].set_ylabel('Flux (erg/s/cm^2/Angstrom)')
#         ax[i//2, i%2].set_title('Spectrum of CV at ' + str(inclinations[i]) + ' degrees')
#         ax[i//2, i%2].legend()

# plt.show()

# fig, ax = plt.subplots(12, 1, figsize=(10,45))
# fig.tight_layout(pad=3.0)
#     for i in range(12):
#     ax[i].plot(wavelength, flux[:, i], label='iridis run 10m photons')
#     ax[i].plot(wavelength2, flux2[:, i], label='local run 10m photons')
#     ax[i].plot(wavelength3, flux3[:, i], label='local run 100m photons')
#     ax[i].set_xlim(6000,7000)
#     ax[i].set_ylim(0, 7e-13)
#     ax[i].set_xlabel('Wavelength (Angstroms)')
#     ax[i].set_ylabel('Flux (erg/s/cm^2/Angstrom)')
#     ax[i].set_title(f'Spectrum of CV: inclination = {inclinations[i]} degrees')
#     ax[i].legend()

# Plot the data for a given wavelength  range
# plt.plot(wavelength, flux30, label='iridis run 10m photons')
# plt.plot(wavelength2, flux2, label='local run 10m photons')
# plt.plot(wavelength3, flux3, label='local run 100m photons')
# plt.xlim(6000,7000)
# plt.ylim(0, 3e-13)
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
# plt.title('Spectrum of CV')
# plt.legend()
# plt.show()

# %%
