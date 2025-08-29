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
from matplotlib.ticker import FormatStrFormatter
#import pysi
dpi=300
plt.style.use('science')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe

#plt.style.use('Solarize_Light2')
# %%
################################################################################
print('STEP 2: CHECKING THE FILES ARE PRESENT')
################################################################################

# --- USER INPUTS --- #
# Add the path from this sirocco file to your grid files
path_to_grids = "H_alpha_models"

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

# %% Calculating total luminosity
########################################################################
print('CALCULATING TOTAL LUMINOSITY')
########################################################################
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
#np.save('total_luminosity.npy', total_luminosity_array)

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
# %%STEP 4: ANIMATED PLOT OF YOUR GRID (DO NOT USE IF PLT.STYLE IS SCIENCE, COMMENT OUT)
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

# %% Plotting a single run 
##############################################################################
print('Plotting a single sirocco run for all inclinations')
%matplotlib inline

run_num = 233
file = f'{path_to_grids}/rerun{run_num}.spec'
incs = [20,45,60,72.5,85]
wavelength3 = np.loadtxt(file, usecols=(1), skiprows=81)
flux3 = np.loadtxt(file, usecols=(10,11,12,13,14), skiprows=81)
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

# %% FIGURE 3 CV PAPER 4 CUENO SUBPLOTS Originial
################################################################################
print('FIGURE 3: CV PAPER 4 CUNEO SUBPLOTS')
################################################################################

index_limits = (0,62) # 0,17
grid_length = np.arange(index_limits[0],index_limits[1])
molly_data = np.load('Emission_Line_Asymmetries/molly_spectra.npy', allow_pickle=True)
run_numbers = molly_data.item().get('run_numbers')[index_limits[0]:index_limits[1]]
wavelength_grid = molly_data.item().get('wavelength_grid')[index_limits[0]:index_limits[1]]
grid = molly_data.item().get('flux_grid')[index_limits[0]:index_limits[1]]
times = molly_data.item().get('times')[index_limits[0]:index_limits[1]]
systems = molly_data.item().get('systems')[index_limits[0]:index_limits[1]]

# build a common wavelength grid for interpolation
# find the overlapping wavelength range across all observations
starts = [w[0] for w in wavelength_grid]
ends   = [w[-1] for w in wavelength_grid]
common_start = max(starts)
common_end   = min(ends)
# choose number of points equal to the minimum length among obs
min_len = min(len(w) for w in wavelength_grid)
common_wave = np.linspace(common_start, common_end, min_len)

# interpolate each spectrum onto the common grid
interp_fluxes = np.vstack([
    np.interp(common_wave, wl, spec)
    for wl, spec in zip(wavelength_grid, grid)
])

# compute min/max envelope
min_spectra = np.min(interp_fluxes, axis=0)
max_spectra = np.max(interp_fluxes, axis=0)
mean_spectra = np.mean(interp_fluxes, axis=0)
plus_one_standard_deviation = np.percentile(interp_fluxes, 84, axis=0)
minus_one_standard_deviation = np.percentile(interp_fluxes, 16, axis=0)
# The Molly data are already loaded just above:
#   molly_data, run_numbers, wavelength_grid, grid, times, systems
#   (see the code section titled “loading of the Molly data”)

# Identify the 4 unique systems present in this sample
unique_systems = np.unique(systems)

# fig, ax = plt.subplots(2, 2, figsize=(20, 15))
# plt.rcParams.update({'font.size': 20})

# for axis, sys_name in enumerate(unique_systems):
#     # Indices of observations for this system
#     sys_idx = np.where(systems == sys_name)[0]
#     n_obs   = len(sys_idx)

#     # Only plot every 2nd spectrum
#     sys_idx = sys_idx[::2]
#     n_obs   = len(sys_idx)
#     print(f'Plotting {n_obs} spectra for {sys_name}')

#     # Sort spectra by wavelength of peak flux (smallest first)
#     peak_wls = [wavelength_grid[i][np.argmax(grid[i])] for i in sys_idx]
#     sys_idx = [idx for idx, wl in sorted(zip(sys_idx, peak_wls), key=lambda x: x[1])]

#     # Colour ramp so each observation is distinguishable
#     colours = [plt.cm.viridis(np.linspace(0.0, 0.9, n_obs)), plt.cm.Oranges(np.linspace(0.5, 1.0, n_obs)), plt.cm.Greens(np.linspace(0.5, 1.0, n_obs)), plt.cm.Reds(np.linspace(0.5, 1.0, n_obs))]
#     colour = colours[axis]

#     # Draw lightest lines first, darkest last for front layering
#     for c, idx in sorted(enumerate(sys_idx), key=lambda x: x[0], reverse=True):
#         ax[axis // 2, axis % 2].plot(
#             wavelength_grid[idx] + (5*c),
#             grid[idx] + (0.16 * c),
#             lw=1.8,
#             color=colour[c]
#         )

#     ax[axis // 2, axis % 2].set_xlim(6450,6700)
#     ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
#     ax[axis // 2, axis % 2].set_ylabel(r'Normalised Flux')
#     #ax[axis // 2, axis % 2].set_ylim(bottom=0.7)
#     #ax[axis // 2, axis % 2].set_title(sys_name)

# # Sub‑panel labels to match the Sirocco figure style
# fig.text(0.14, 0.75, 'BZ Cam', fontsize=30)
# fig.text(0.564, 0.75, 'MV Lyr', fontsize=30)
# fig.text(0.14, 0.37, 'V425 Cas', fontsize=30)
# fig.text(0.564, 0.37, 'V751 Cyg', fontsize=30)

# import matplotlib as mpl
# sm = mpl.cm.ScalarMappable(cmap='Greys', norm=mpl.colors.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# cbar = fig.colorbar(
#     sm,
#     ax=ax.ravel().tolist(),
#     orientation='horizontal',
#     fraction=0.03,
#     pad=0.02,
#     location='top'
# )
# cbar.set_ticks([])  # no ticks for a vague effect
# cbar.ax.xaxis.set_ticks_position('top')   # move ticks to bottom
# cbar.ax.xaxis.set_label_position('top')   # move label to bottom
# cbar.set_label('Increasing Peak Redness →', labelpad=10, fontsize=30)

# #plt.savefig('plots/Figure_Cuneo_examples', dpi=dpi)
# plt.show()

# HEATMAP: Molly spectra spectrogram
H_alpha = 6562.819
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import TwoSlopeNorm
fig_hm, ax_hm = plt.subplots(2, 2, figsize=(20, 15))
plt.rcParams.update({'font.size': 20})

# manual observation times for each system (pulled from reference figures)
time_dict = {
    'BZ_Cam': [0.00, 0.04, 0.08, 0.12] 
              + [262.68, 262.74, 262.77, 262.79, 262.82] 
              + [308.62, 308.66, 308.70, 308.74, 309.51, 309.55, 309.59, 309.63],
    'MV_Lyr': [0.00, 0.02, 0.04, 0.06]
              + [23.73, 23.77, 23.81, 23.85]
              + [47.60, 47.65, 47.69, 47.73]
              + [1341.03, 1341.06, 1341.08, 1341.10],
    'V425_Cas': [0.00, 0.06, 0.12, 0.19]
                + [48.33, 48.39, 48.46, 48.52]
                + [95.99, 96.05, 96.12, 96.18],
    'V751_Cyg': [0.00, 0.05, 0.10, 0.14]
                + [23.92, 23.97, 24.02, 24.07]
                + [47.96, 48.01, 48.06, 48.11]
                + [71.85, 71.90, 71.94, 71.99]
}

# create a universal normalization across all systems
threshold = 1.0
# concatenate all spectra rows for every system to find global vmin/vmax
all_obs_flux = np.vstack([
    np.vstack([grid[i] for i in np.where(systems == sys_name)[0]])
    for sys_name in unique_systems
])
# global TwoSlopeNorm
norm = TwoSlopeNorm(vmin=all_obs_flux.min(),
                    vcenter=threshold,
                    vmax=all_obs_flux.max())

for axis, sys_name in enumerate(unique_systems):
    # all observation indices for this system
    sys_idx = np.where(systems == sys_name)[0]
    # build a 2D array: rows=epochs, cols=wavelength bins
    obs_flux = np.vstack([grid[i] for i in sys_idx])
    # define continuum threshold (normalized flux = 1.0)
    # threshold = 1.0
    # norm = TwoSlopeNorm(vmin=obs_flux.min(), vcenter=threshold, vmax=obs_flux.max())
    # assume all wavelength_grid rows have the same axis
    wave = wavelength_grid[sys_idx[0]]
    im = ax_hm[axis//2, axis%2].imshow(
        obs_flux,
        aspect='auto',
        origin='lower',
        extent=[wave.min(), wave.max(), 0, obs_flux.shape[0]],
        cmap='RdBu_r',
        norm=norm,
        interpolation='nearest',   # no smoothing
    )
    # mark the H_alpha rest wavelength
    ax = ax_hm[axis//2, axis%2]
    # thick black dashed line at true Hα wavelength
    ax.axvline(H_alpha, color='black', linestyle='--', linewidth=2)
    # enforce axis limits to the data bounds
    ax.set_xlim(6500, 6685)
    #ax_hm[axis//2, axis%2].set_title(sys_name)
    # Place system name outside the axes (left for left panels, right for right panels)
    if axis % 2 == 0:
        ax.text(0.03, 1.01, sys_name.replace('_', ' '),
                transform=ax.transAxes,
                ha='left', va='bottom',
                fontsize=25)
    else:
        ax.text(0.97, 1.01, sys_name.replace('_', ' '),
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=25)
    ax_hm[axis//2, axis%2].set_xlabel(r'Wavelength ($\mathring{A}$)')
    # add a true Hα tick alongside existing ticks
    ticks = list(ax.get_xticks())
    # include Hα position
    ticks.append(H_alpha)
    # sort the tick positions
    ticks = sorted(ticks)[1:-1]
    #remove the 6550 value
    ticks.remove(6550)
    ticks.remove(6575)
    # build labels: Hα for that position, numeric for others
    labels = [r'$\mathrm{H\,\alpha}$' if np.isclose(t, H_alpha) else f'{t:.0f}'for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', length=5, width=1)
    # label epochs centered in each heatmap row
    ax = ax_hm[axis//2, axis%2]
    ax.set_ylabel('Epoch')
    n_epochs = obs_flux.shape[0]
    # center ticks at row midpoints: 0.5, 1.5, ..., n_epochs-0.5
    yt = np.arange(n_epochs) + 0.5
    ax.set_yticks(yt)
    # label epochs from 1 to n_epochs
    ax.set_yticklabels([str(i) for i in range(1, n_epochs+1)], fontsize=20)
    # secondary y-axis for observation times (manual)
    times_sys = time_dict[sys_name]
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    yt = ax.get_yticks()
    # only keep ticks within available times
    yt_valid = yt[yt < len(times_sys)]
    ax2.set_yticks(yt_valid)
    # format times without trailing .0
    labels = []
    for y in yt_valid:
        t = times_sys[int(y)]
        lbl = f'{t:.2f}'.rstrip('0').rstrip('.')
        labels.append(lbl)
    ax2.set_yticklabels(labels, fontsize=14)
    ax2.set_ylabel('Time Since First Observation (hrs)', rotation=-90, labelpad=15, fontsize=14)
    # add a small inset plotting every spectrum
    locs = ['upper right', 'upper right', 'upper right', 'upper right']
    #if axis% 2 == 0:
    axins = inset_axes(
        ax,
        width="45%", height="45%",
        loc=locs[axis],
        bbox_to_anchor=(0.0,0,1,1),
        bbox_transform=ax.transAxes,
        borderpad=0.5
    )
    # fix inset y-axis to only ticks at 1 and 2
    if axis == 3:
        axins.set_yticks([1, 2])
        axins.tick_params(axis='y', which='major')
    for row in obs_flux:
        axins.plot(wave, row, lw=0.5, color='black', alpha=0.6)
    #axins.set_xticks([])
    #axins.set_yticks([1,2,3])
    axins.set_xlim(6515, 6590)
    axins.set_ylabel(r'$F/F_{\mathrm{cont}}$', fontsize=16)
    axins.set_xlabel(r'$\lambda$($\mathring{A}$)', fontsize=16)
    axins.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
    # thin frame for the inset
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)

# colorbar for all subplots
cbar_hm = fig_hm.colorbar(
    im,
    ax=ax_hm.ravel().tolist(),
    orientation='horizontal',
    fraction=0.03,
    pad=0.02,
    location='top'
)
cbar_hm.set_label(r'Continuum-Normalised Flux', labelpad=10)
# show numeric ticks at the continuum and ends
vmin, vcenter, vmax = im.norm.vmin, im.norm.vcenter, im.norm.vmax
cbar_hm.set_ticks([vmin, vcenter, vmax])
cbar_hm.set_ticklabels([f'{vmin:.2f}', f'{vcenter:.2f}', f'{vmax:.2f}'])
cbar_hm.ax.tick_params(axis='x', which='both', length=5, width=1)

#plt.tight_layout()
plt.savefig('plots/Figure_Cuneo_examples', dpi=dpi)
plt.show()

# %% FIGURE 4 CV PAPER 4 SIROCCO SUBPLOTS
################################################################################
print('FIGURE 4: CV PAPER 4 SIROCCO SUBPLOTS')
################################################################################
from sklearn.linear_model import LinearRegression
def continuum(wavelengths, spectrum, intervals):
    """Fit linear continuum on two wavelength windows."""
    low1, high1, low2, high2 = intervals
    mask1 = (wavelengths > low1) & (wavelengths < high1)
    mask2 = (wavelengths > low2) & (wavelengths < high2)
    con_mask = mask1 | mask2
    # fallback if too few points
    if np.sum(con_mask) < 2:
        peak = np.max(spectrum)
        return np.full_like(spectrum, peak), 0.0, peak, 0.0
    reg = LinearRegression()
    reg.fit(wavelengths[con_mask].reshape(-1,1), spectrum[con_mask])
    cont = reg.predict(wavelengths.reshape(-1,1))
    noise = np.std(spectrum - cont)
    return cont, reg.coef_[0], reg.intercept_, noise

%matplotlib inline
# Multi plot several runs on the same plot
# make room above panels for titles and horizontal info boxes
fig, ax = plt.subplots(2, 2, figsize=(20, 15))
fig.subplots_adjust(wspace=0.16, hspace=0.4)
#increase text size
plt.rcParams.update({'font.size': 20})
H_alpha = 6562.819

def to_latex_sci(value, precision=2):
    # e.g., "3.00e-09" -> "3.00\times 10^{-9}"
    s = f"{value:.{precision}e}"      # format as scientific e.g. "3.00e-09"
    mantissa, exponent = s.split('e') # split into "3.00" and "-09"
    exponent = exponent.replace('+', '')   # remove any "+"
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"

titles = ['(a)', '(b)', '(c)', '(d)']
run_nums = [34, 251, 431 , 652] # 34, 260(493), 503, 566 652
incs = [20,45,60,72.5,85]
#colors = plt.cm.gist_rainbow(np.linspace(0, 0.9, len(incs)))
#colors = ['tab:blue', 'tab:red','tab:orange', 'tab:green', 'tab:purple']
colors = plt.get_cmap('gist_ncar')([0.85,0.2,0.36,0.6,0.73])
for axis, run_num in enumerate(run_nums):
    # create inset for each panel
    cur_ax = ax[axis // 2, axis % 2]
    # Create inset at upper right, then lower it vertically without changing x-position
    axins = inset_axes(
        cur_ax,
        width="40%", height="40%",
        loc='upper right',
        bbox_to_anchor=(0.0, -0.05, 1, 1),
        bbox_transform=cur_ax.transAxes
    )
    # lower the inset without altering its x-position
    #pos = axins.get_position()
    #axins.set_position([pos.x0, pos.y0 - 1, pos.width, pos.height])
    wave = wavelengths[run_num]
    diff = H_alpha - common_wave[np.argmax(mean_spectra)]
    # envelope and mean in inset
    axins.fill_between(
        common_wave , min_spectra, max_spectra,
        lw=0.5, color='black', alpha=0.3, label = 'Min/Max'
    )
    axins.plot(
        common_wave, mean_spectra,
        lw=1.5, color='black', alpha=0.6, label = 'Mean'
    )
    axins.legend(loc ='upper left', fontsize=14, frameon=False)
    
    if axis == 2:
        # Build horizontal parameter string
        parts = [
            rf'$\dot{{M}}_{{disk}}:{to_latex_sci(parameter_table[run_num, 1])}\enspace\enspace$',
            rf'$\dot{{M}}_{{wind}}:{to_latex_sci(parameter_table[run_num, 2])}\enspace\enspace$',
            rf'$d:{parameter_table[run_num, 3]:.2f}\enspace\enspace$',
            rf'$\alpha:{parameter_table[run_num, 4]:.2f}\qquad\qquad\qquad\;$',
            rf'$R_s:{to_latex_sci(parameter_table[run_num, 5])}\qquad\,\,$',
            rf'$\beta:{parameter_table[run_num, 6]:.2f}\enspace\enspace$'
        ]

        first_line = '  '.join(parts[:3])  # first three parameters
        second_line = '  '.join(parts[3:])  # last three parameters
    else:
        # Build horizontal parameter string
        parts = [
            rf'$\dot{{M}}_{{disk}}:{to_latex_sci(parameter_table[run_num, 1])}\enspace\enspace$',
            rf'$\dot{{M}}_{{wind}}:{to_latex_sci(parameter_table[run_num, 2])}\enspace\enspace$',
            rf'$d:{parameter_table[run_num, 3]:.2f}\enspace\enspace$',
            rf'$\alpha:{parameter_table[run_num, 4]:.2f}\qquad\qquad\qquad\;$',
            rf'$R_s:{to_latex_sci(parameter_table[run_num, 5])}\qquad\;\,\,$',
            rf'$\beta:{parameter_table[run_num, 6]:.2f}\enspace\enspace$'
        ]

        first_line = '  '.join(parts[:3])
        second_line = '  '.join(parts[3:])  # last three parameters
    # Combine into a single string with two lines
    textstr = f'{first_line}\n{second_line}'
    for i in range(5):
        if axis == 0:
            line, = ax[axis // 2, axis % 2].plot(
                wavelengths[run_num],
                fluxes[run_num][:, i],
                label=f'{incs[i]}°',
                linewidth=2,
                color=colors[i],
                zorder=0
            )
            line.set_path_effects([
                pe.Stroke(linewidth=line.get_linewidth()+1, foreground='black'),
                pe.Normal()
            ])
        else:
            line, = ax[axis // 2, axis % 2].plot(
                wavelengths[run_num],
                fluxes[run_num][:, i],
                linewidth=2,
                color=colors[i],
                zorder=0
            )
            line.set_path_effects([
                pe.Stroke(linewidth=line.get_linewidth()+1, foreground='black'),
                pe.Normal()
            ])
        ax[axis // 2, axis % 2].set_xlim(6500, 6685)#6450,6700
        ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
        ax[axis // 2, axis % 2].set_ylabel(r'Flux (erg\,s$^{-1}\,$cm$^{-2}\,$$\mathring{A}^{-1}$)')
        ax[axis // 2, axis % 2].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)

        #ax[axis // 2, axis % 2].yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        plt.rcParams.update({'font.size': 20})
        # Set title and info box only once per panel (not per inclination)
        if i == 0:
            cur_ax = ax[axis // 2, axis % 2]
            #cur_ax.set_title(f'Panel {titles[axis]}', pad=40)
            # place horizontal info box below title
            if axis // 2 == 0:
                # top row
                cur_ax.text(
                    0.245, 1.03,
                    textstr,
                    transform=cur_ax.transAxes,
                    ha='left', va='bottom',
                    fontsize=18,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
                    clip_on=False)
            else:
                cur_ax.text(
                    0.255, 1.03,
                    textstr,
                    transform=cur_ax.transAxes,
                    ha='left', va='bottom',
                    fontsize=18,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
                    clip_on=False)
        # add normalized peak profile to inset for middle inclination
        if i == 1: # if 45 degrees
            cont, _, _, _ = continuum(
                wave, fluxes[run_num][:, i],
                intervals=(6450, 6475, 6625, 6650)
            )
            norm_spec = fluxes[run_num][:, i] / cont
            line, = axins.plot(wave, norm_spec, color=colors[i], lw=2)
            line.set_path_effects([
                pe.Stroke(linewidth=line.get_linewidth()+1, foreground='black'),
                pe.Normal()
            ])
    # finalize inset styling
    axins.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
    axins.set_ylabel(r'$F/F_{\mathrm{cont}}$', fontsize=16)
    axins.set_xlabel(r'$\lambda$($\mathring{A}$)', fontsize=16)
    #axins.set_xticks([])
    #axins.set_yticks()
    axins.set_xlim(6515, 6590)
    axins.set_title('Observations vs Model', fontsize=16, pad=3)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)

    ticks = list(ax[axis // 2, axis % 2].get_xticks())
    # include Hα position
    ticks.append(H_alpha)
    # sort the tick positions
    ticks = sorted(ticks)[1:-1]
    #remove the 6550 value
    ticks.remove(6550)
    ticks.remove(6575)
    # build labels: Hα for that position, numeric for others
    labels = [r'$\mathrm{H\,\alpha}$' if np.isclose(t, H_alpha) else f'{t:.0f}'for t in ticks]
    ax[axis // 2, axis % 2].set_xticks(ticks)
    ax[axis // 2, axis % 2].set_xticklabels(labels)
    ax[axis // 2, axis % 2].tick_params(axis='x', which='major', length=5, width=1)
# Panel labels and top text
fig.text(0.170, 0.9, '(a)', fontsize=40)
fig.text(0.585, 0.9, '(b)', fontsize=40)
fig.text(0.175, 0.45, '(c)', fontsize=40)
fig.text(0.593, 0.45, '(d)', fontsize=40)
# Add a text box with the word inclination to the top of the figure
fig.text(0.27, 0.95, 'Inclinations:', ha='center', fontsize=22)

# gather handles & labels from main panels only
main_handles, main_labels = [], []
for main_ax in ax.flatten():
    h, l = main_ax.get_legend_handles_labels()
    main_handles.extend(h)
    main_labels.extend(l)

# create and assign the legend
leg = fig.legend(
    main_handles,
    main_labels,
    loc='upper center',
    bbox_to_anchor=(0.54, 0.981),
    ncol=5,
    fontsize=22
)
# thicken legend lines without altering the plotted lines
for legline in leg.get_lines():
    legline.set_linewidth(4.0)  # set desired legend line width
plt.savefig('plots/Figure_sirocco_examples', dpi=dpi)
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

# plot an individual run
# run = 701
# fig, ax = plt.subplots(figsize=(12, 8))
# indexes = np.where((wavelengths[run] > 6260) & (wavelengths[run] < 6860))
# plt.plot(velocities[run][indexes[0][0]:indexes[0][-1]], fluxes[run][indexes[0][0]:indexes[0][-1], 1])
# #plot a virtual line at the central h_alpha line
# plt.axvline(x=0, color='red', linestyle='--')
# plt.xlabel('Radial Velocities ($m/s$)')
# plt.ylabel('Flux ($erg/s/cm^2/Å$)')
# plt.title('H alpha of CV for Run ' + str(run))
# plt.show()

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
# Calculating the emission measures of each Sirocco run
emission_measures = []
ne_sums = []
vol_sums = []
for run in tqdm(range(0,729)):
    if f'run{run}.master.txt' not in os.listdir('Sirocco_cv_grid_masters'):
        print(f'Run {run} does not exist')
        continue
    master_paths = f'Sirocco_cv_grid_masters/run{run}.master.txt'
    master_df = pd.read_csv(f'{master_paths}', sep=r'\s+')

    vol = master_df['vol'] # volume in cm^3
    ne = master_df['ne'] # electron density in cm^-3
    
    #print(run, f'{vol:.2e}', f'{ne:.2e}')
    em = ne**2 * vol # EM = n_e^2 * V
    em_sum = em.sum()
    emission_measures.append(em_sum)

#np.save('emission_measures.npy', emission_measures) # 618
# %% Plotting a Selected Parameter
###########################################################################
print("Plotting a Selected Parameter in Index Coordinates")
###########################################################################
run = 562
master_df = pd.read_csv(f'Sirocco_cv_grid_masters/run{run}.master.txt', sep=r'\s+')
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

pivoted_ne = master_df.pivot(index='j', columns='i', values='t_r')
pivoted_ne[pivoted_ne < 1e-10] = 0
pivoted_ne = pivoted_ne.replace(0, np.nan)
log_ne = np.log(pivoted_ne) #np.log10(

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

# %% Plotting Emission Measure
###########################################################################
print("Plotting Emission Measure")
###########################################################################
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
plt.colorbar(label='log (EM)')
plt.xlabel('i (cell index)')
plt.ylabel('j (cell index)')
plt.title(f'Log Emission Measure for Run {run}')
plt.show()

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

## FIGURE 4 CV PAPER 4 SIROCCO SUBPLOTS
# ################################################################################
# print('FIGURE 4: CV PAPER 4 SIROCCO SUBPLOTS')
# ################################################################################
# from sklearn.linear_model import LinearRegression
# def continuum(wavelengths, spectrum, intervals):
#     """Fit linear continuum on two wavelength windows."""
#     low1, high1, low2, high2 = intervals
#     mask1 = (wavelengths > low1) & (wavelengths < high1)
#     mask2 = (wavelengths > low2) & (wavelengths < high2)
#     con_mask = mask1 | mask2
#     # fallback if too few points
#     if np.sum(con_mask) < 2:
#         peak = np.max(spectrum)
#         return np.full_like(spectrum, peak), 0.0, peak, 0.0
#     reg = LinearRegression()
#     reg.fit(wavelengths[con_mask].reshape(-1,1), spectrum[con_mask])
#     cont = reg.predict(wavelengths.reshape(-1,1))
#     noise = np.std(spectrum - cont)
#     return cont, reg.coef_[0], reg.intercept_, noise

# %matplotlib inline
# # Multi plot several runs on the same plot
# fig, ax = plt.subplots(2, 2, figsize=(20, 15))
# #increase text size
# plt.rcParams.update({'font.size': 20})
# H_alpha = 6562.819

# def to_latex_sci(value, precision=2):
#     # e.g., "3.00e-09" -> "3.00\times 10^{-9}"
#     s = f"{value:.{precision}e}"      # format as scientific e.g. "3.00e-09"
#     mantissa, exponent = s.split('e') # split into "3.00" and "-09"
#     exponent = exponent.replace('+', '')   # remove any "+"
#     return rf"{mantissa}\times 10^{{{int(exponent)}}}"

# titles = ['(a)', '(b)', '(c)', '(d)']
# run_nums = [34, 251, 431 , 652] # 34, 260(493), 503, 566 652
# incs = [20,45,60,72.5,85]
# colors = plt.cm.viridis(np.linspace(0, 0.9, len(incs)))
# for axis, run_num in enumerate(run_nums):
#     if axis == 2:
#         axins_2 = inset_axes(
#             ax[1,0],
#             width="40%", height="40%",
#             loc='upper left',
#             #bbox_to_anchor=(0.0,0,1,1),
#             #bbox_transform=ax.transAxes,
#             #borderpad=0.5
#         )
#         # wavelength array for this run
#         wave = wavelengths[run_num]
#         index = np.argmax(mean_spectra)
#         diff = H_alpha - common_wave[index] # As they fit Gaussian mean and we fix
#         axins_2.fill_between(common_wave+diff, min_spectra, max_spectra, lw=0.5, color='black', alpha=0.3)
#         axins_2.plot(wavelength_grid[i]+diff, mean_spectra, lw=1.5, color='black', alpha=0.6)
#         #axins.plot(wavelength_grid[i], plus_one_standard_deviation, lw=1.5, color='black', alpha=0.6, linestyle='--')
#         #axins.plot(wavelength_grid[i], minus_one_standard_deviation, lw=1.5, color='black', alpha=0.6, linestyle='--')
#     if axis == 1:
#         axins_1 = inset_axes(
#             ax[0,1],
#             width="40%", height="40%",
#             loc='upper left',
#             #bbox_to_anchor=(0.0,0,1,1),
#             #bbox_transform=ax.transAxes,
#             #borderpad=0.5
#         )
#         # wavelength array for this run
#         wave = wavelengths[run_num]
#         index = np.argmax(mean_spectra)
#         diff = H_alpha - common_wave[index] # As they fit Gaussian mean and we fix
#         axins_1.fill_between(common_wave+diff, min_spectra, max_spectra, lw=0.5, color='black', alpha=0.3)
#         axins_1.plot(wavelength_grid[i]+diff, mean_spectra, lw=1.5, color='black', alpha=0.6)
#         #axins.plot(wavelength_grid[i], plus_one_standard_deviation, lw=1.5, color='black', alpha=0.6, linestyle='--')
#         #axins.plot(wavelength_grid[i], minus_one_standard_deviation, lw=1.5, color='black', alpha=0.6, linestyle='--')

#     for i in range(5):
#         if axis == 0:
#             ax[axis // 2, axis % 2].plot(wavelengths[run_num],
#                                          fluxes[run_num][:, i],
#                                          label=f'{incs[i]}°', 
#                                          linewidth=3,
#                                          color=colors[i],
#                                          zorder=0)
#         else:
#             ax[axis // 2, axis % 2].plot(wavelengths[run_num],
#                                          fluxes[run_num][:, i],
#                                          linewidth=3,
#                                          color = colors[i],
#                                          zorder=0)
#         # Add text box with parameter values
#         textstr = '\n'.join((
#             rf'$\dot{{M}}_{{disk}}:{to_latex_sci(parameter_table[run_num, 1])}$',
#             rf'$\dot{{M}}_{{wind}}:{to_latex_sci(parameter_table[run_num, 2])}$',
#             rf'$d:{parameter_table[run_num, 3]:.2f}$',
#             rf'$\alpha:{parameter_table[run_num, 4]:.2f}$',
#             rf'$l:{to_latex_sci(parameter_table[run_num, 5])}$',
#             rf'$\beta:{parameter_table[run_num, 6]:.2f}$'
#         ))
#         props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey')
#         if axis >= 2:
#             ax[axis // 2, axis % 2].text(0.695, 
#                                         0.97, 
#                                         textstr, 
#                                         transform=ax[axis // 2, axis % 2].transAxes,
#                                         fontsize=18,
#                                         verticalalignment='top',
#                                         bbox=props
#                                         )
#         else:
#             ax[axis // 2, axis % 2].text(0.685, 
#                                     0.97, 
#                                     textstr, 
#                                     transform=ax[axis // 2, axis % 2].transAxes,
#                                     fontsize=18,
#                                     verticalalignment='top',
#                                     bbox=props
#                                     )
#         ax[axis // 2, axis % 2].set_xlim(6480, 6630)#6450,6700
#         ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
#         ax[axis // 2, axis % 2].set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathring{A}^{-1}$)')
#         ax[axis // 2, axis % 2].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5)
#         plt.rcParams.update({'font.size': 20})
#         ax[axis // 2, axis % 2].set_title(f'Panel {titles[axis]}')
#         if axis == 2:
#             if i == 1:
#                 # fit and divide by linear continuum in two windows
#                 cont, _, _, _ = continuum(wave, fluxes[run_num][:, i],
#                                         intervals=(6450, 6475, 6625, 6650))
#                 norm_spec = fluxes[run_num][:, i] / cont
                
#                 axins_2.plot(wave, norm_spec, color=colors[i], lw=3)
#                 #axins.plot(wavelength_grid[0], min_spectra, lw=0.5, color='black', alpha=0.6)

#                 axins_2.set_xticks([])
#                 axins_2.set_yticks([])
#                 axins_2.set_xlim(6480, 6630)
#                 # thin frame for the inset
#                 for spine in axins_2.spines.values():
#                     spine.set_linewidth(0.5)
#         if axis == 1:
#             if i == 1:
#                 # fit and divide by linear continuum in two windows
#                 cont, _, _, _ = continuum(wave, fluxes[run_num][:, i],
#                                         intervals=(6450, 6475, 6625, 6650))
#                 norm_spec = fluxes[run_num][:, i] / cont
                
#                 axins_1.plot(wave, norm_spec, color=colors[i], lw=3)
#                 #axins.plot(wavelength_grid[0], min_spectra, lw=0.5, color='black', alpha=0.6)

#                 axins_1.set_xticks([])
#                 axins_1.set_yticks([])
#                 axins_1.set_xlim(6480, 6630)
#                 # thin frame for the inset
#                 for spine in axins_1.spines.values():
#                     spine.set_linewidth(0.5)
        
# # Add a text box with the word inclination to the top of the figure

# fig.text(0.27, 0.92, 'Inclinations:', ha='center', fontsize=22)
# # fig.text(0.14, 0.83, '(a)', fontsize=40)
# # fig.text(0.564, 0.83, '(b)', fontsize=40)
# # fig.text(0.14, 0.41, '(c)', fontsize=40)
# # fig.text(0.564, 0.41, '(d)', fontsize=40)

# fig.legend(loc='upper center', bbox_to_anchor=(0.54, 0.951), ncol=5, fontsize=22)
# plt.savefig('plots/Figure_sirocco_examples', dpi=dpi)
# plt.show()

# # %%
# run = 627

# master_df = pd.read_csv(f'../../../Sirocco_test_runs/cv_asym_weird_profiles_check_2/run{run}.master.txt', sep=r'\s+')
# # Pivot the data using real cell center coordinates
# parameter = 't_e'
# pivoted_ne_real = master_df.pivot(index='z', columns='x', values=parameter)

# # Apply a threshold: values below 1e-10 become 0, then replace 0 with NaN
# #pivoted_ne_real[pivoted_ne_real < 1e-10] = 0
# #pivoted_ne_real = pivoted_ne_real.replace(0, np.nan)

# # Take the log for better dynamic range
# log_ne_real = np.log10(pivoted_ne_real)

# # Determine the extent for imshow using the unique coordinate values
# xcoords = np.sort(master_df['x'].unique())
# zcoords = np.sort(master_df['z'].unique())
# extent = [xcoords.min(), xcoords.max(), zcoords.min(), zcoords.max()]

# # Plotting
# plt.figure(figsize=(8,6))
# plt.imshow(log_ne_real, 
#            origin='lower', 
#            aspect='auto', 
#            cmap='viridis', 
#            extent=extent)
# plt.axvline(x=2.17555e10, color='red', linestyle='--', alpha=0.5, label='Disc_RadMax')
# plt.axvline(x=1.00e+12, color='black', linestyle='--', alpha=0.5, label='Wind_RadMax')
# plt.colorbar(label=f'log10({parameter})')
# plt.xlabel('x (cm)')
# plt.ylabel('z (cm)')
# plt.xscale('log')
# plt.yscale('log')

# plt.title(f'Log10({parameter}) (Real Coordinates)')
# plt.legend()
# plt.show()


# # %%
# run = 627
# master_df = pd.read_csv(
#     f"../../../Sirocco_test_runs/cv_asym_weird_profiles_check/run{run}.master.txt",
#     sep=r'\s+'
# )

# # 1) Pivot on the true cell-corner coordinates using x and z
# parameter = "ne"
# pivoted = master_df.pivot(index="z", columns="x", values=parameter)

# # 2) Build edges directly from x and z corners
# x_edges = np.array(sorted(pivoted.columns))
# z_edges = np.array(sorted(pivoted.index))

# # Also capture the full coordinate range for axis limits
# xcoords_full = np.sort(master_df['x'].unique())
# zcoords_full = np.sort(master_df['z'].unique())

# # Compute cell widths (use the last delta if spacing varies)
# dx = np.diff(x_edges)
# dx_last = dx[-1] if not np.allclose(dx, dx[0]) else dx[0]
# x_edges = np.append(x_edges, x_edges[-1] + dx_last)

# dz = np.diff(z_edges)
# dz_last = dz[-1] if not np.allclose(dz, dz[0]) else dz[0]
# z_edges = np.append(z_edges, z_edges[-1] + dz_last)

# # 3) Convert to log scale, masking zeros
# pivoted[pivoted < 1e-10] = 0  # Set values below threshold to 0
# pivoted = pivoted.replace(0, np.nan)
# log_data = np.log10(pivoted.values)

# # # Apply a threshold: values below 1e-10 become 0, then replace 0 with NaN
# # log_data[log_data < 1e0] = 0
# # log_data = log_data.replace(0, np.nan)

# # 4) Plot with pcolormesh on log-log axes
# plt.figure(figsize=(8,6))
# pcm = plt.pcolormesh(
#     x_edges,
#     z_edges,
#     log_data,
#     cmap="turbo",
#     shading="auto"
# )
# cbar = plt.colorbar(pcm, label=f"log10({parameter})")
# plt.axvline(x=2.17555e10, color="red", linestyle="--", alpha=0.5, label="Disc_RadMax")
# #plt.axvline(x=1.00e12, color="black", linestyle="--", alpha=0.5, label="Wind_RadMax")

# plt.xscale("log")
# plt.yscale("log")
# # plt.xlim(0, 6e10)
# # plt.ylim(0, 6e10)
# # Set axis limits to the smallest strictly positive coordinate and max
# x_min_plot = xcoords_full[xcoords_full > 0].min()
# z_min_plot = zcoords_full[zcoords_full > 0].min()
# plt.xlim(x_min_plot, xcoords_full.max())
# plt.ylim(z_min_plot, zcoords_full.max())

# plt.xlabel("x (cm)")
# plt.ylabel("z (cm)")
# plt.title(f"Log10({parameter}) (Real Coordinates using corners)")
# plt.legend()
# plt.tight_layout()
# plt.show()
# # %%
# pivoted_em_real = master_df.pivot(index='z', columns='x', values='em')

# # Apply a threshold: values below 1e-10 become 0, then replace 0 with NaN
# pivoted_em_real[pivoted_em_real < 1e2] = 0
# pivoted_em_real = pivoted_em_real.replace(0, np.nan)

# # Take the log for better dynamic range
# log_em_real = np.log10(pivoted_em_real)

# # Plotting
# plt.figure(figsize=(8,6))
# plt.imshow(log_em_real,
#               origin='lower',
#                 aspect='auto',
#                 cmap='plasma',
#                 extent=extent)
# plt.colorbar(label='log EM')
# plt.xlabel('x (cm)')
# plt.ylabel('z (cm)')

# plt.title('Log Emission Measure (Real Coordinates)')
# plt.show()








# %%
# # %% FIGURE 4+ CV PAPER 4 CUENO SUBPLOTS
# ################################################################################
# print('FIGURE 4: CV PAPER 4 CUNEO SUBPLOTS')
# ################################################################################

# index_limits = (0,62) # 0,17
# grid_length = np.arange(index_limits[0],index_limits[1])
# molly_data = np.load('Emission_Line_Asymmetries/molly_spectra.npy', allow_pickle=True)
# run_numbers = molly_data.item().get('run_numbers')[index_limits[0]:index_limits[1]]
# wavelength_grid = molly_data.item().get('wavelength_grid')[index_limits[0]:index_limits[1]]
# grid = molly_data.item().get('flux_grid')[index_limits[0]:index_limits[1]]
# times = molly_data.item().get('times')[index_limits[0]:index_limits[1]]
# systems = molly_data.item().get('systems')[index_limits[0]:index_limits[1]]

# grid_length   = np.arange(*index_limits)

# # The Molly data are already loaded just above:
# #   molly_data, run_numbers, wavelength_grid, grid, times, systems
# #   (see the code section titled “loading of the Molly data”)

# # Identify the 4 unique systems present in this sample
# unique_systems = np.unique(systems)

# fig, ax = plt.subplots(2, 2, figsize=(20, 15))
# plt.rcParams.update({'font.size': 20})

# for axis, sys_name in enumerate(unique_systems):
#     # Indices of observations for this system
#     sys_idx = np.where(systems == sys_name)[0]
#     n_obs   = len(sys_idx)

#     # Only plot every 2nd spectrum
#     sys_idx = sys_idx[::1]
#     n_obs   = len(sys_idx)
#     print(f'Plotting {n_obs} spectra for {sys_name}')

#     # Colour ramp so each observation is distinguishable
#     colours = [plt.cm.Oranges_r(np.linspace(0.0, 0.6, n_obs)), plt.cm.Blues_r(np.linspace(0.0, 0.6, n_obs)), plt.cm.Greens_r(np.linspace(0.0, 0.6, n_obs)), plt.cm.Reds_r(np.linspace(0.0, 0.6, n_obs))]
#     colour = colours[axis]

#     # Draw lightest lines first, darkest last for front layering
#     for c, idx in sorted(enumerate(sys_idx), key=lambda x: x[0], reverse=True):
#         ax[axis // 2, axis % 2].plot(
#             wavelength_grid[idx],
#             grid[idx] + 0.13 * c,
#             lw=1.8,
#             color=colour[c]
#         )

#     ax[axis // 2, axis % 2].set_xlim(6450,6700)
#     ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
#     ax[axis // 2, axis % 2].set_ylabel(r'Normalised Flux')
#     #ax[axis // 2, axis % 2].set_ylim(bottom=0.7)
#     #ax[axis // 2, axis % 2].set_title(sys_name)

# # Sub‑panel labels to match the Sirocco figure style
# fig.text(0.14, 0.75, 'BZ Cam', fontsize=30)
# fig.text(0.564, 0.75, 'MV Lyr', fontsize=30)
# fig.text(0.14, 0.37, 'V425 Cas', fontsize=30)
# fig.text(0.564, 0.37, 'V751 Cyg', fontsize=30)

# import matplotlib as mpl
# sm = mpl.cm.ScalarMappable(cmap='Greys_r', norm=mpl.colors.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# cbar = fig.colorbar(
#     sm,
#     ax=ax.ravel().tolist(),
#     orientation='horizontal',
#     fraction=0.03,
#     pad=0.02,
#     location='top'
# )
# cbar.set_ticks([])  # no ticks for a vague effect
# cbar.ax.xaxis.set_ticks_position('top')   # move ticks to bottom
# cbar.ax.xaxis.set_label_position('top')   # move label to bottom
# cbar.set_label('Increasing Epoch →', labelpad=10, fontsize=30)

# plt.savefig('plots/Figure_4_Molly', dpi=600)
# plt.show()


# # %% FIGURE 4+ CV PAPER 4 CUENO SUBPLOTS Original
# ################################################################################
# print('FIGURE 4: CV PAPER 4 CUNEO SUBPLOTS')
# ################################################################################

# index_limits = (0,62) # 0,17
# grid_length = np.arange(index_limits[0],index_limits[1])
# molly_data = np.load('Emission_Line_Asymmetries/molly_spectra.npy', allow_pickle=True)
# run_numbers = molly_data.item().get('run_numbers')[index_limits[0]:index_limits[1]]
# wavelength_grid = molly_data.item().get('wavelength_grid')[index_limits[0]:index_limits[1]]
# grid = molly_data.item().get('flux_grid')[index_limits[0]:index_limits[1]]
# times = molly_data.item().get('times')[index_limits[0]:index_limits[1]]
# systems = molly_data.item().get('systems')[index_limits[0]:index_limits[1]]

# grid_length   = np.arange(*index_limits)

# # The Molly data are already loaded just above:
# #   molly_data, run_numbers, wavelength_grid, grid, times, systems
# #   (see the code section titled “loading of the Molly data”)

# # Identify the 4 unique systems present in this sample
# unique_systems = np.unique(systems)

# fig, ax = plt.subplots(2, 2, figsize=(20, 15))
# plt.rcParams.update({'font.size': 20})

# for axis, sys_name in enumerate(unique_systems):
#     # Indices of observations for this system
#     sys_idx = np.where(systems == sys_name)[0]
#     n_obs   = len(sys_idx)

#     # Only plot every 2nd spectrum
#     sys_idx = sys_idx[::2]
#     n_obs   = len(sys_idx)
#     print(f'Plotting {n_obs} spectra for {sys_name}')

#     # Sort spectra by wavelength of peak flux (smallest first)
#     peak_wls = [wavelength_grid[i][np.argmax(grid[i])] for i in sys_idx]
#     sys_idx = [idx for idx, wl in sorted(zip(sys_idx, peak_wls), key=lambda x: x[1])]

#     # Colour ramp so each observation is distinguishable
#     colours = [plt.cm.viridis(np.linspace(0.0, 0.9, n_obs)), plt.cm.Oranges(np.linspace(0.5, 1.0, n_obs)), plt.cm.Greens(np.linspace(0.5, 1.0, n_obs)), plt.cm.Reds(np.linspace(0.5, 1.0, n_obs))]
#     colour = colours[axis]

#     # Draw lightest lines first, darkest last for front layering
#     for c, idx in sorted(enumerate(sys_idx), key=lambda x: x[0], reverse=True):
#         ax[axis // 2, axis % 2].plot(
#             wavelength_grid[idx],
#             grid[idx],
#             lw=1.8,
#             color=colour[c]
#         )

#     ax[axis // 2, axis % 2].set_xlim(6450,6700)
#     ax[axis // 2, axis % 2].set_xlabel(r'Wavelength ($\mathring{A}$)')
#     ax[axis // 2, axis % 2].set_ylabel(r'Normalised Flux')
#     #ax[axis // 2, axis % 2].set_ylim(bottom=0.7)
#     #ax[axis // 2, axis % 2].set_title(sys_name)

# # Sub‑panel labels to match the Sirocco figure style
# fig.text(0.14, 0.75, 'BZ Cam', fontsize=30)
# fig.text(0.564, 0.75, 'MV Lyr', fontsize=30)
# fig.text(0.14, 0.37, 'V425 Cas', fontsize=30)
# fig.text(0.564, 0.37, 'V751 Cyg', fontsize=30)

# import matplotlib as mpl
# sm = mpl.cm.ScalarMappable(cmap='Greys', norm=mpl.colors.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# cbar = fig.colorbar(
#     sm,
#     ax=ax.ravel().tolist(),
#     orientation='horizontal',
#     fraction=0.03,
#     pad=0.02,
#     location='top'
# )
# cbar.set_ticks([])  # no ticks for a vague effect
# cbar.ax.xaxis.set_ticks_position('top')   # move ticks to bottom
# cbar.ax.xaxis.set_label_position('top')   # move label to bottom
# cbar.set_label('Increasing Peak Redness →', labelpad=10, fontsize=30)

# plt.savefig('plots/Figure_4_Molly', dpi=600)
# plt.show()

















