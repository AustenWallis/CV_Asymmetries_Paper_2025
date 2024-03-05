################################################################################
################################################################################   
#  ____      _     _             _                ____          _         
# |  _ \ ___| |__ (_)_ __  _ __ (_)_ __   __ _   / ___|___   __| | ___   
# | |_) / _ | '_ \| | '_ \| '_ \| | '_ \ / _` | | |   / _ \ / _` |/ _ \  
# |  _ |  __| |_) | | | | | | | | | | | | (_| | | |__| (_) | (_| |  __/   
# |_| \_\___|_.__/|_|_| |_|_| |_|_|_| |_|\__, |  \____\___/ \__,_|\___|  
#                                        |___/                          
################################################################################
################################################################################
# Rebinning the Spec Tot (ionisation cycles) and Spec (Spectral cycles) spectra
# files to  check that there is minimal difference between the two. This is to 
# do with the TDE issue from Ed during Jan/Feb 2024 where discrepencies were   
# found between the two spectra files. This is to check that Austen's grids are
# largely unaffected by this issue.                                             
################################################################################
################################################################################

# %%
################################################################################
print('STEP 1: IMPORTING MODULES')
################################################################################

import numpy as np
from tqdm import tqdm
from pyinstrument import Profiler
import bisect
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

plt.style.use('Solarize_Light2')

# %%
################################################################################
print('STEP 2: LOADING THE DATA BETWEEN THE SPECIFIED WAVELENGTH RANGE')
################################################################################

# Importing run files data to variables
flux_spec = {}
flux_spec_tot = {}
run_number = np.arange(0, 729) # 729 grid runs in total

# Loading flux data for each spectral run
for run in tqdm(run_number):
    flux_spec[run] = np.loadtxt(
        f'../Grids/optical_grid_v87f_spec_files/run{run}.spec', # Units: flambda spectrum (erg/s/cm^2/A) at 100.0 parsecs
        usecols=(4), # emitted column
        skiprows=81  # skipping the header to data
        )
    flux_spec_tot[run] = np.loadtxt(
        f'../Grids/optical_grid_v87f_log_spec_tot_files/run{run}.log_spec_tot', # Units: L_nu spectrum (erg/s/Hz)
        usecols=(4), # emitted column
        skiprows=81  # skipping the header to data
        )

# Loading wavelength and frequency data for each spectral run
wavelength_spec = np.loadtxt(
    '../Grids/optical_grid_v87f_spec_files/run0.spec', # all runs use the same wavelengths and frequencies
    usecols=(1), # wavelength column
    skiprows=81  # skipping the header to data
    )
wavelength_spec_tot = np.loadtxt(
    '../Grids/optical_grid_v87f_log_spec_tot_files/run0.log_spec_tot',
    usecols=(1), # wavelength column
    skiprows=81  # skipping the header to data
    )
freq_spec = np.loadtxt(
    '../Grids/optical_grid_v87f_spec_files/run0.spec',
    usecols=(0), # frequency column
    skiprows=81  # skipping the header to data
    )
freq_spec_tot = np.loadtxt(
    '../Grids/optical_grid_v87f_log_spec_tot_files/run0.log_spec_tot',
    usecols=(0), # frequency column
    skiprows=81  # skipping the header to data
    )

wavelength_range = [900, 7850] # Taking fluxes only within the limits

# Placing a mask to truncate the wavelengths to the wavelength range for spec files
wave_spec_indexes = np.where((wavelength_spec > wavelength_range[0]) & (wavelength_spec < wavelength_range[1]))
wavelength_spec = wavelength_spec[wave_spec_indexes[0][0]:wave_spec_indexes[0][-1]]
freq_spec = freq_spec[wave_spec_indexes[0][0]:wave_spec_indexes[0][-1]] # truncating emitted frequencies

# Placing a mask to truncate the wavelengths to the wavelength range for spec tot files
wave_spec_tot_indexes = np.where((wavelength_spec_tot > wavelength_range[0]) & (wavelength_spec_tot < wavelength_range[1]))
wavelength_spec_tot = wavelength_spec_tot[wave_spec_tot_indexes[0][0]:wave_spec_tot_indexes[0][-1]]
freq_spec_tot = freq_spec_tot[wave_spec_tot_indexes[0][0]:wave_spec_tot_indexes[0][-1]] # truncating spec tot frequencies

# Truncating the fluxes to the same wavelength range for all spectral runs
flux_spec = {run: flux_spec[run][wave_spec_indexes[0][0]:wave_spec_indexes[0][-1]] for run in run_number}
flux_spec_tot = {run: flux_spec_tot[run][wave_spec_tot_indexes[0][0]:wave_spec_tot_indexes[0][-1]] for run in run_number}

# Converting fluxes to luminosities
distance_sq = (100 * 3.086e18)**2 # (100 parsecs in cm) ^2
# (erg/s/cm^2/Å --> ergs/s)
luminosity_spec = {run: flux_spec[run] * wavelength_spec * 4 * np.pi * distance_sq for run in run_number} 
# (erg/s/Hz --> ergs/s)
luminosity_spec_tot = {run: flux_spec_tot[run] * freq_spec_tot for run in run_number}

# %%
################################################################################
print("STEP 3: REBINNING THE SPEC SPECTRA TO SPEC TOT'S FREQUENCY BINS")
################################################################################

luminosity_spec_rebinned = {}
for run in tqdm(run_number):
    spec_lumin_rebinned = [[] for _ in range(len(freq_spec_tot))]
    #print(f'The length of the rebinning array is {len(spec_lumin_rebinned)}')

    for index, freq in enumerate(freq_spec): # for a frequency in the Spec Spectrum
        position = bisect.bisect_left(freq_spec_tot, freq)-1 # -1 Due to PYTHON left binning
        spec_lumin_rebinned[position].append(luminosity_spec[run][index]) # add luminosity to it's new bin
        
    for i, v in enumerate(spec_lumin_rebinned):
        if len(v) > 0:
            spec_lumin_rebinned[i] = sum(v) / len(v)
        else:
            spec_lumin_rebinned[i] = None

    luminosity_spec_rebinned[run] = spec_lumin_rebinned

# %%
################################################################################
print('STEP 4: PLOTTING AN INDIVIDUAL RUN OF BOTH ORIGINAL AND REBINED COMPARISONS')
################################################################################
%matplotlib inline
run = 450
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].loglog(freq_spec, luminosity_spec[run], label='Original Spec')
ax[0].loglog(freq_spec_tot, luminosity_spec_tot[run], label='Original Spec Tot', alpha=0.5)
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Luminosity (erg/s)')
ax[0].set_title(f'Run {run}: \n Original Spec vs Original Spec Tot Spectra')
ax[0].legend()
ax[1].loglog(freq_spec_tot, luminosity_spec_rebinned[run], label='Rebinned Spec')
ax[1].loglog(freq_spec_tot, luminosity_spec_tot[run], label='Original Spec Tot', alpha=0.5)
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Luminosity (erg/s)')
ax[1].set_title(f'Run {run}: \n Rebinned Spec vs Original Spec Tot Spectra')
ax[1].legend()
plt.tight_layout()
plt.show()


# %%
################################################################################
print('STEP 5: ANIMATION OF THE REBINNED SPEC SPECTRA VS SPEC TOT SPECTRA FOR EACH RUN')
################################################################################

# opens a pop-up window for the animation
%matplotlib qt 

def slider_update(val):
    """When the slider changes, this function updates the plot to the new value."""
    ax.clear()
    ax.loglog(freq_spec_tot, luminosity_spec_rebinned[int(val)], label='Rebinned Spec')
    ax.loglog(freq_spec_tot, luminosity_spec_tot[int(val)], label='Spec Tot', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Luminosity (erg/s)')
    ax.set_title(f'Run {int(val)}: Rebinned Emitted vs Spec Tot')
    ax.legend()
    fig.canvas.draw_idle()

def animation_setting_new_slider_value(_) -> None:
    """This function controls the animation. The change in slider value prompts
    the slider_update function to update the plot with the new value."""
    if anim.running:
        if grid_slider.val == 728:
            grid_slider.set_val(0)
        else:
            grid_slider.set_val(grid_slider.val + 1)
            
def play_pause(_) -> None:
    """Play and pause function for the animation on press of the button."""
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_) -> None:
    """Function to move the slider to the left by one unit."""
    anim.running = False
    grid_slider.set_val(grid_slider.val - 1)
    slider_update(grid_slider.val)

def right_button_func(_) -> None:
    """Function to move the slider to the right by one unit."""
    anim.running = False
    grid_slider.set_val(grid_slider.val + 1)
    slider_update(grid_slider.val)

# Initialising the figure for the animation    
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.03]) # Run Slider
grid_slider = Slider(ax_slider, 'Run', 0, 728, valinit=0, valstep=1)
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

# Running the animation
anim = FuncAnimation(fig, animation_setting_new_slider_value, frames=729, interval=400)
anim.running = True

# %%
################################################################################
print('STEP 6: WORKING OUT THE CHANGES IN FREQUENCY INTERVALS ACROSS THE SPECTRA')
################################################################################

# back to inline plots
%matplotlib inline

# Histogram for spec spectra
deltas = np.diff(freq_spec)
plt.hist(deltas, bins=20)
plt.xlabel('Change in frequency intervals')
plt.ylabel('Counts')
plt.title('Histogram of frequency deltas for Spec Spectra')
plt.show()

# Histogram for spec tot spectra
deltas = np.diff(freq_spec_tot)
plt.hist(deltas, bins=20)
plt.xlabel('Change in frequency intervals')
plt.ylabel('Counts')
plt.title('Histogram of frequency deltas for Spec Tot Spectra')
plt.show()

# Min and max frequency values for both spectra
print('Spec Spectra || Spec Tot Spectra')
print(f'{min(freq_spec):.2e} {min(freq_spec_tot):2e} min')
print(f'{max(freq_spec):.2e} {max(freq_spec_tot):.2e} max')

# Range of both spectra
range_emitted = max(freq_spec) - min(freq_spec)
range_spec_tot = max(freq_spec_tot) - min(freq_spec_tot)
print(f'{range_emitted:.2e} {range_spec_tot:.2e} range')

# Intervals of both spectra at a given number of bins
interval_emitted = range_emitted / 3000
interval_spec_tot = range_spec_tot / 3000
print(f'{interval_emitted:.2e} {interval_spec_tot:.2e} interval')

################################################################################
print('END OF CODE')
################################################################################

































































# %%

################################################################################
# OLD CODE I DON'T HAVE THE HEART TO DELETE INCASE I NEED IT LATER FOR SOMETHING
################################################################################

# # individual plot
# #%matplotlib inline
# from tqdm import tqdm
# run = 689
# rebin_frequencies = np.arange(min(freq_spec_tot), max(freq_spec)+1e13, 1e13)
# print(f'The length of the rebinning array is {len(rebin_frequencies)}')

# # rebinning the emitted data
# #rebinned_flux_emit = {}
# rebinned_lumin_emit = {} # dictionary of arrays initialised
# count = {}
# for run in tqdm(run_number): # for each spectral run 
#     rebinned_index = 0 # index to iterate the rebinning array
#     #temp_flux_emit = np.array([])
#     temp_lumin_emit = np.array([]) # temporary arrays to store the frequencies (flux/lumin values) smaller than the rebinning frequency
#     #new_flux_emit = np.array([])
#     new_lumin_emit = np.array([]) # new arrays to store the rebinned frequencies (flux/lumin values)
#     counter = 0
#     for index, value in enumerate(freq_spec): # for each frequency in the emitted data
#         if value <= rebin_frequencies[rebinned_index]: # if the frequency is smaller than the rebinning frequency
#             #temp_flux_emit = np.append(temp_flux_emit, flux_spec[run][index]) # append the flux value to the temporary array
#             temp_lumin_emit = np.append(temp_lumin_emit, luminosity_spec[run][index]) # append the luminosity value to the temporary array
#         else:
#             #new_flux_emit = np.append(new_flux_emit, np.mean(temp_flux_emit)) # append the mean of the temporary array to the new array
#             new_lumin_emit = np.append(new_lumin_emit, np.mean(temp_lumin_emit)) # append the mean of the temporary array to the new array
#             rebinned_index += 1 # increment the rebinning index
#             #temp_flux_emit = np.append([], flux_spec[run][index]) # reset the temporary array and append the current flux value
#             temp_lumin_emit = np.append([], luminosity_spec[run][index]) # reset the temporary array and append the current luminosity value
#             counter += 1
    
#     #rebinned_flux_emit[run] = new_flux_emit # add the new array to the spectral runs dictionary
#     rebinned_lumin_emit[run] = new_lumin_emit # add the new array to the spectral runs dictionary
#     count[run] = counter

# # rebinning the spec tot data, same as above
# #rebinned_flux_spec_tot = {}
# rebinned_lumin_spec_tot = {}
# for run in tqdm(run_number):
#     rebinned_index = 0
#     #temp_flux_spec_tot = np.array([])
#     temp_lumin_spec_tot = np.array([])
#     #new_flux_spec_tot = np.array([])
#     new_lumin_spec_tot = np.array([])
#     for index, value in enumerate(freq_spec_tot):
#         if value <= rebin_frequencies[rebinned_index]:
#             #temp_flux_spec_tot = np.append(temp_flux_spec_tot, flux_spec_tot[run][index])
#             temp_lumin_spec_tot = np.append(temp_lumin_spec_tot, luminosity_spec_tot[run][index])
#         else:
#             #new_flux_spec_tot = np.append(new_flux_spec_tot, np.mean(temp_flux_spec_tot))
#             new_lumin_spec_tot = np.append(new_lumin_spec_tot, np.mean(temp_lumin_spec_tot))
#             rebinned_index += 1
#             #temp_flux_spec_tot = np.append([], flux_spec_tot[run][index])
#             temp_lumin_spec_tot = np.append([], luminosity_spec_tot[run][index])
        
#     #rebinned_flux_spec_tot[run] = new_flux_spec_tot
#     rebinned_lumin_spec_tot[run] = new_lumin_spec_tot
    
#     #     temp_array = np.append(temp_array, np.mean(flux_spec[run][index:index+3]))
#     #     temp_array_2 = np.append(temp_array_2, freq_spec[index])
#     #     temp_array_3 = np.append(temp_array_3, np.mean(luminosity_spec[run][index:index+3]))
#     # rebinned_flux_emitted[run] = temp_array
#     # rebinned_freq_emitted[run] = temp_array_2
#     # rebinned_luminosity_emitted[run] = temp_array_3


# %%

# profiler.stop()
# profiler.print()
# # rebinning the emitted data
# rebinned_lumin_emit = {}        # dictionary of arrays initialised
# for run in tqdm(run_number):    # for each spectral run
#     rebinned_index = 0          # index to iterate the rebinning array
#     temp_lumin_emit = []        # temporary arrays to store the frequencies (flux/lumin values) smaller than the rebinning frequency
#     new_lumin_emit = []         # new arrays to store the rebinned frequencies (flux/lumin values)
#     for index, value in enumerate(freq_spec):                    # for each frequency in the emitted data
#         if value <= rebin_frequencies[rebinned_index]:              # if the frequency is smaller than the rebinning frequency
#             temp_lumin_emit.append(luminosity_spec[run][index])  # append the luminosity value to the temporary array
#         else:
#             #new_lumin_emit.append(np.mean(temp_lumin_emit))         # append the mean of the temporary array to the new array
#             new_lumin_emit.append(temp_lumin_emit)
#             rebinned_index += 1                                     # increment the rebinning index
#             temp_lumin_emit = [luminosity_spec[run][index]]      # reset the temporary array and append the current luminosity value
    
#     #change list matrix into a 2D numpy array
# #luminosity_spec_tot = {run: flux_spec_tot[run] * freq_spec_tot for run in run_number}

#     rebinned_lumin_emit[run] = np.mean(new_lumin_emit, axis=1)      # add the new array to the spectral runs dictionary



# # rebinning the spec tot data, same as above
# rebinned_lumin_spec_tot = {}
# for run in tqdm(run_number):
#     rebinned_index = 0
#     temp_lumin_spec_tot = []
#     new_lumin_spec_tot = []
#     for index, value in enumerate(freq_spec_tot):
#         if value <= rebin_frequencies[rebinned_index]:
#             temp_lumin_spec_tot.append(luminosity_spec_tot[run][index])
#         else:
#             new_lumin_spec_tot.append(np.mean(temp_lumin_spec_tot))
#             rebinned_index += 1
#             temp_lumin_spec_tot = [luminosity_spec_tot[run][index]]
        
#     rebinned_lumin_spec_tot[run] = new_lumin_spec_tot

    

# %%
# # individual plot reworking rebinning for speed

# #run = 689
# #rebin_frequencies = np.arange(min(freq_spec_tot), max(freq_spec)+1e14, 1e14)
# rebin_frequencies = freq_spec_tot
# print(f'The length of the rebinning array is {len(rebin_frequencies)}')

# # rebinning the emitted data
# rebinned_lumin_emit = {}        # dictionary of arrays initialised
# run_number = np.arange(0, 1)  # 729 grid runs in total
# for run in run_number:    # for each spectral run
#     rebinned_index = 0          # index to iterate the rebinning array
#     temp_lumin_emit = []        # temporary arrays to store the frequencies (flux/lumin values) smaller than the rebinning frequency
#     new_lumin_emit = []         # new arrays to store the rebinned frequencies (flux/lumin values)
#     for index, value in enumerate(freq_spec):                    # for each frequency in the emitted data
#         # print('-------------------')
#         # print(f'SPEC: index: {index} value: {value:2e}')
        
#         if value <= rebin_frequencies[rebinned_index]:              # if the frequency is smaller than the rebinning frequency
#             # print(f'SPEC TOT: index: {rebinned_index} frequency: {rebin_frequencies[rebinned_index]:2e}')
#             # print('if run')
#             # print(f'Temp list: {len(temp_lumin_emit)} --> {len(temp_lumin_emit)+1}')
            
#             temp_lumin_emit.append(luminosity_spec[run][index])  # append the luminosity value to the temporary array
            
            
#         elif len(temp_lumin_emit) != 0:                             # if the temporary array is not empty
#             # print(f'SPEC TOT: index: {rebinned_index} frequency: {rebin_frequencies[rebinned_index]:2e}')
#             # print('elif run')
#             # print(f'Temp list: {len(temp_lumin_emit)} --> 1')
            
#             new_lumin_emit.append(sum(temp_lumin_emit)/len(temp_lumin_emit))         # append the mean of the temporary array to the new array
#             rebinned_index += 1                                     # increment the rebinning index
#             temp_lumin_emit = [luminosity_spec[run][index]]    # reset the temporary array and append the current luminosity value
#             print('averaged and reset with new entry luminosity ')
#         # else:
#         #     new_lumin_emit.append(0)
    
#     rebinned_lumin_emit[run] = new_lumin_emit       # add the new array to the spectral runs dictionary

# # %%
# run = 0
# plt.plot(freq_spec_tot, luminosity_spec_rebinned[run], label='Rebinned Spec')
# plt.plot(freq_spec_tot, luminosity_spec_tot[run], label='Spec Tot', alpha = 0.5)
# plt.legend()
# plt.show()
# # %%
# %matplotlib qt
# plt.scatter(np.arange(len(rebin_frequencies)), rebin_frequencies, label='Spec Tot Frequencies', s=1)
# plt.scatter(np.arange(len(freq_spec)), freq_spec, label='Spec Frequencies', s=1)
# plt.title('Spec is rebinning into spec tot')
# plt.legend()
# plt.show()
# # %%
# plt.plot(wavelength_spec_tot, label='Spec Tot Wavelengths')
# plt.plot(wavelength_spec, label='Spec Wavelengths')
# plt.legend()
# plt.show()
# # %%
# run = 431
# plt.loglog(rebin_frequencies[1:-1], rebinned_lumin_spec_tot[run][1:], label='Spec Tot')
# plt.loglog(rebin_frequencies[1:-1], rebinned_lumin_emit[run][1:], label='Emitted')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Luminosity (erg/s)')
# plt.title(f'Run {run}: Rebinned Emitted vs Rebinned Spec Tot')
# plt.legend()
# plt.show()


# plt.loglog(wavelength_spec_tot, luminosity_spec_tot[run], label='Spec Tot')
# plt.loglog(wavelength_spec, luminosity_spec[run], label='Emitted')
# plt.xlabel('Wavelength ($\AA$)')
# plt.ylabel('Luminosity (erg/s)')
# plt.title(f'Run {run}: Emitted vs Spec Tot')
# plt.legend()
# plt.show()

# # %%
# #%matplotlib inline
# run = 630
# #plt.loglog(freq_spec_tot, luminosity_spec_tot[run], label='Spec Tot')
# #plt.loglog(freq_spec, luminosity_spec[run], label='Emitted')
# plt.loglog(rebin_frequencies[:-1], rebinned_lumin_spec_tot[run], label='Rebinned Spec Tot')
# plt.loglog(rebin_frequencies[1:-1], rebinned_lumin_emit[run][1:], label='Rebinned Emitted')

# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Luminosity (erg/s)')
# plt.title(f'Run {run}: Emitted vs Spec Tot')
# plt.legend()
# plt.show()
