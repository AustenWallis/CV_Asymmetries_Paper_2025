################################################################################
################################################################################
#  ____        _   _                   ____                  _             
# |  _ \ _   _| |_| |__   ___  _ __   / ___| _ __   ___  ___| |_ _ __ __ _ 
# | |_) | | | | __| '_ \ / _ \| '_ \  \___ \| '_ \ / _ \/ __| __| '__/ _` |
# |  __/| |_| | |_| | | | (_) | | | |  ___) | |_) |  __| (__| |_| | | (_| |
# |_|    \__, |\__|_| |_|\___/|_| |_| |____/| .__/ \___|\___|\__|_|  \__,_|
#        |___/                              |_|                            
################################################################################
################################################################################
# For plotting and inspecting the spectra of any python output file
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

plt.style.use('science')

#plt.style.use('Solarize_Light2')
# %%
################################################################################
print('STEP 2: CHECKING THE FILES ARE PRESENT')
################################################################################

# --- USER INPUTS --- #
# Add the path from this python file to your grid files
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

# %%
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
    #for i in range(len(inclinations)):
    for i in range(len(inclinations)):
        ax.plot(wavelengths[run], fluxes[run][:, i], label=f'{inclinations[i]}°')
        #ax.scatter(wavelengths[run], fluxes[run][:, i], s=5, color='black')
    # Add text box with parameter values
    textstr = '\n'.join((
        r'$\dot{M}_{disk}=%.2e$' % (parameter_table[run, 1], ),
        r'$\dot{M}_{wind}=%.2e$' % (parameter_table[run, 2], ),
        r'$d=%.2f$' % (parameter_table[run, 3], ),
        r'$r_{exp}=%.2f$' % (parameter_table[run, 4], ),
        r'$a_{l}=%.2e$' % (parameter_table[run, 5], ),
        r'$a_{exp}=%.2f$' % (parameter_table[run, 6], )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.85, 1.15, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.legend(bbox_to_anchor=(0.70, 1.17), loc='upper left')
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

# %%
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

# %%
# TODO Figure 8
%matplotlib inline
# Multi plot several runs on the same plot
fig, ax = plt.subplots(2, 2, figsize=(20, 15))
#increase text size
plt.rcParams.update({'font.size': 20})

run_nums = [34, 251, 701, 652] # 34, 260(493), 503, 566
incs = [20,45,60,72.5,85]
for axis, run_num in enumerate(run_nums):
    for i in range(5):
        if axis == 0:
            ax[axis // 2, axis % 2].plot(wavelengths[run_num],
                                         fluxes[run_num][:, i],
                                         label=f'{incs[i]}°', 
                                         linewidth=2)
        else:
            ax[axis // 2, axis % 2].plot(wavelengths[run_num],
                                         fluxes[run_num][:, i],
                                         linewidth=2)
        # Add text box with parameter values
        textstr = '\n'.join((
            r'$\dot{M}_{disk}:%.2e$' % (parameter_table[run_num, 1], ),
            r'$\dot{M}_{wind}:%.2e$' % (parameter_table[run_num, 2], ),
            r'$d:%.2f$' % (parameter_table[run_num, 3], ),
            r'$r_{exp}:%.2f$' % (parameter_table[run_num, 4], ),
            r'$a_{l}:%.2e$' % (parameter_table[run_num, 5], ),
            r'$a_{exp}:%.2f$' % (parameter_table[run_num, 6], )))
        props = dict(boxstyle='round', facecolor='palegoldenrod', alpha=0.5)
        ax[axis // 2, axis % 2].text(0.71, 
                                     0.97, 
                                     textstr, 
                                     transform=ax[axis // 2, axis % 2].transAxes,
                                     fontsize=18,
                                     verticalalignment='top',
                                     bbox=props
                                     )
        ax[axis // 2, axis % 2].set_xlim(6400,6700)
        ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
        ax[axis // 2, axis % 2].set_ylabel(r'Flux ($erg s^{-1} cm^{-2} \mathring{A}{-1}$)')
        plt.rcParams.update({'font.size': 20})

        
# Add a text box with the word inclination to the top of the figure
fig.text(0.275, 0.9125, 'Inclinations:', ha='center', fontsize=20)
fig.text(0.14, 0.83, '(A)', fontsize=30)
fig.text(0.564, 0.83, '(B)', fontsize=30)
fig.text(0.14, 0.41, '(C)', fontsize=30)
fig.text(0.564, 0.41, '(D)', fontsize=30)

fig.legend(loc='upper center', bbox_to_anchor=(0.525, 0.94), ncol=5)

plt.show()
# %%
################################################################################
# END OF CODE
################################################################################




















































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
