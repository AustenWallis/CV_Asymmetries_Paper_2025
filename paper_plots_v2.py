# %% IMPORTING MODULES AND DATA
################################################################################
print('IMPORTING MODULES AND DATA')
################################################################################

import pyinstrument
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.lines import Line2D
import scienceplots
from sklearn.neighbors import KernelDensity
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from IPython.display import display, Math, Latex, Markdown
import statsmodels.api as sm
import scipy.optimize as opt
# import gaussian smoothing 1d
from scipy.integrate import quad
import itertools

plt.style.use('science')

all_results = {}
inclination_columns = [10,11,12,13,14]
#inclination_columns = [11]  # 45° inclination
mask = '22_55_mask' # 11-88 = 500-4000, 22-88 = 1000-4000, 22-55 = 1000-2500, 22-90 = 1000-4100
for inclination_column in tqdm(inclination_columns):
    if os.path.exists(f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy'):
        all_results[inclination_column] = np.load(f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()

# %% APPENDIX 1/2 LARGE- LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS
################################################################################
print('APPENDIX 1/2: LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS')
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.scale import SymmetricalLogLocator

# Define inclination columns and incs
inclination_columns = [10, 11, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
incs = [0 for _ in range(10)]  # to align indices
incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# Create the figure and a 2×3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=True)
axs = axs.flatten()

# Remove the unused 6th subplot
fig.delaxes(axs[5])
plt.rcParams.update({'font.size': 15})

# Load Teo's data once
bz_cam   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv',   delimiter=',')
mv_lyr   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv',   delimiter=',')
v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# Prepare legend handles
handles = []
labels = []

for idx, inclination_column in enumerate(inclination_columns):
    final_results = all_results[inclination_column]
    cut_runs = final_results['cut_runs']
    peak_colour_map = final_results['peak_colour_map']
    grid_length = np.arange(0, 729)

    cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
    cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
    cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
    cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
    cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
    # Create masks for single-peaked and double-peaked spectra
    cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
    # Boolean masks for single- vs double-peaked
    single_mask = (cut_peak_colour_map == 'black')
    double_mask = (cut_peak_colour_map == 'red')

    ax = axs[idx]
    #ax.set_axisbelow(True)
    ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

    # Plot error bars (all points together for simplicity)
    ax.errorbar(
        cut_red_ew_excess,
        cut_blue_ew_excess,
        xerr=cut_red_ew_excess_error,
        yerr=cut_blue_ew_excess_error,
        fmt='none',
        ecolor='black',
        alpha=0.7,
        zorder=1
    )

    # Plot single-peaked (black) vs double-peaked (red)
    # Add legend entries only in the first subplot
    if idx == 0:
        single_scatter = ax.scatter(
            cut_red_ew_excess[single_mask],
            cut_blue_ew_excess[single_mask],
            c='red',
            s=10,
            label='Single-peaked Sirocco Spectra',
            zorder=3,
            alpha=0.7
        )
        double_scatter = ax.scatter(
            cut_red_ew_excess[double_mask],
            cut_blue_ew_excess[double_mask],
            c='black',
            s=10,
            label='Double-peaked Sirocco Spectra',
            zorder=2,
            alpha=0.7
        )
        handles.append(single_scatter)
        labels.append('Single-peaked Sirocco Spectra')
        handles.append(double_scatter)
        labels.append('Double-peaked Sirocco Spectra')
    else:
        ax.scatter(
            cut_red_ew_excess[single_mask],
            cut_blue_ew_excess[single_mask],
            c='red',
            s=10,
            zorder=3,
            alpha=0.7
        )
        ax.scatter(
            cut_red_ew_excess[double_mask],
            cut_blue_ew_excess[double_mask],
            c='black',
            s=10,
            zorder=2,
            alpha=0.7
        )

    # Only plot Teo’s data on the 45° inclination
    # if inclination_column == 11:
    #     teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1], color='red', s=10, marker='o', label='Cúneo et al. (2023)')
    #     ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
    #     ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
    #     ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
    #     handles.append(teo_scatter)
    #     labels.append('Cúneo et al. (2023)')

    # Reference lines
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

    # Dashed box for linear/log threshold
    linear_thrs = 0.1
    ax.plot(
        [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
        [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
        color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0,
        label='Linear/Logarithmic Threshold'
    )
    #linear threshold lines
    ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    
    # ax.axvline(linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
    # ax.axhline(-linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
    # ax.axhline(linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
    

    # Axes formatting
    ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
    # Put a y-label on the left column of subplots (idx=0 for top-left, idx=3 for bottom-left)
    if idx in [0, 3]:
        ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')

    ax.set_title(f'{incs[inclination_column]}° inclination')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xscale('symlog', linthresh=linear_thrs)
    ax.set_yscale('symlog', linthresh=linear_thrs)

    # Minor tick locators for symlog
    ax.xaxis.set_minor_locator(
        SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
    )

# Add final threshold handle/label
labels.append('Linear/Logarithmic Threshold')
handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

fig.text(0.895, 0.32, 
        '$\pm\,1000-2500\,km\,s^{-1}$ Masking Window',  #change for mask
        #transform=ax2.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
#labels.append('$1000-2500 kms^{-1}$ Mask Window')
#handles.append(Line2D([0], [0], color='white', linestyle='-', linewidth=2.0))

# Adjust subplot spacing (both horizontally and vertically)
fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)

# Position the legend closer to the top
fig.legend(
    handles, labels,
    loc='lower right', ncol=1,
    bbox_to_anchor=(0.9, 0.19)
)

# Manually reposition the bottom row subplots so they are centered
pos3 = axs[3].get_position()
pos4 = axs[4].get_position()
axs[3].set_position([0.125, pos3.y0-0.04, 0.26, pos3.height])  # shift bottom-left a bit to the right
axs[4].set_position([0.385, pos4.y0-0.04, 0.255, pos4.height])  # shift bottom-right a bit left

plt.show()


# %% APPENDIX 3 - FWHM DIAGNOSTIC PLOTS
################################################################################
print('APPENDIX 3: FWHM DIAGNOSTIC PLOTS')
################################################################################

# Define the 5 inclination columns (these indices correspond to the model inclinations)
inclination_columns = [10, 11, 12, 13, 14]  # e.g. 20°, 45°, 60°, 72.5°, 85°
# Build the "incs" list so that incs[inclination_column] gives the physical value.
incs = [0 for _ in range(10)]
incs.extend([20, 45, 60, 72.5, 85])  # so, incs[10]=20, incs[11]=45, etc.

# Create the figure and a 2×3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=True)
axs = axs.flatten()
fig.delaxes(axs[5])  # remove the unused subplot
plt.rcParams.update({'font.size': 15})

# Prepare lists for legend handles and labels
handles = []
labels = []

for idx, inclination_column in enumerate(inclination_columns):
    # Load the FWHM diagnostic results for this inclination from the FWHM folder
    filepath = f'Emission_Line_Asymmetries/FWHM_1p0_5_mask_data/final_results_inc_col_{inclination_column}.npy'
    if not os.path.exists(filepath):
        continue
    final_results = np.load(filepath, allow_pickle=True).item()
    
    cut_runs = final_results['cut_runs']
    peak_colour_map = final_results['peak_colour_map']
    grid_length = np.arange(0, 729)
    
    cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
    cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
    cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
    cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
    cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
    
    # Create masks for single-peaked and double-peaked spectra
    cut_peak_colour_map = np.array(cut_peak_colour_map)
    single_mask = (cut_peak_colour_map == 'black')
    double_mask = (cut_peak_colour_map == 'red')
    
    ax = axs[idx]
    ax.set_axisbelow(True)
    ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
    # Plot error bars for all data points
    ax.errorbar(cut_red_ew_excess,
                cut_blue_ew_excess,
                xerr=cut_red_ew_excess_error,
                yerr=cut_blue_ew_excess_error,
                fmt='none',
                ecolor='black',
                alpha=0.7,
                zorder=2)
    
    # Plot single-peaked and double-peaked Sirocco points. Add legend entries only once.
    if idx == 0:
        single_scatter = ax.scatter(cut_red_ew_excess[single_mask],
                                    cut_blue_ew_excess[single_mask],
                                    c='red',
                                    s=10,
                                    label='Single-peaked Sirocco Spectra',
                                    zorder=3,
                                    alpha=0.7)
        double_scatter = ax.scatter(cut_red_ew_excess[double_mask],
                                    cut_blue_ew_excess[double_mask],
                                    c='black',
                                    s=10,
                                    label='Double-peaked Sirocco Spectra',
                                    zorder=2,
                                    alpha=0.7)
        handles.extend([single_scatter, double_scatter])
        labels.extend(['Single-peaked Sirocco Spectra', 'Double-peaked Sirocco Spectra'])
    else:
        ax.scatter(cut_red_ew_excess[single_mask],
                   cut_blue_ew_excess[single_mask],
                   c='red',
                   s=10,
                   zorder=3,
                   alpha=0.7)
        ax.scatter(cut_red_ew_excess[double_mask],
                   cut_blue_ew_excess[double_mask],
                   c='black',
                   s=10,
                   zorder=2,
                   alpha=0.7)

    # Draw reference axes (vertical/horizontal lines)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    
    # Draw the dashed box and threshold lines
    linear_thrs = 0.1
    ax.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
            [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
            color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0,
            label='Linear/Logarithmic Threshold')
    ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    
    # Axes formatting
    ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
    if idx in [0, 3]:
        ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')
    ax.set_title(f'{incs[inclination_column]}° inclination')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xscale('symlog', linthresh=linear_thrs)
    ax.set_yscale('symlog', linthresh=linear_thrs)
    ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
    ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))

fig.text(0.875, 0.32, 
        r'$1-5\times$ FWHM Masking Window', 
        #transform=ax2.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')

# Add final threshold legend entry
labels.append('Linear/Logarithmic Threshold')
handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)
fig.legend(
    handles, labels,
    loc='lower right', ncol=1,
    bbox_to_anchor=(0.9, 0.19)
)

# Manually reposition the bottom row subplots (if needed)
pos3 = axs[3].get_position()
pos4 = axs[4].get_position()
axs[3].set_position([0.125, pos3.y0-0.04, 0.26, pos3.height])  # shift bottom-left a bit to the right
axs[4].set_position([0.385, pos4.y0-0.04, 0.255, pos4.height])  # shift bottom-right a bit left
plt.show()

# %% FIGURE 1 - THEORETICAL DIAGNOSTIC PLOTS
################################################################################
print('FIGURE 1: THEORETICAL DIAGNOSTIC PLOTS')
################################################################################
%matplotlib inline
fig, axs = plt.subplots(figsize=(7, 7))
plt.rcParams.update({'font.size': 17})
axs.scatter(0,0, color='black', marker='+')
x = [-30, -30, 30, 30, -30, 0, 0, 30]
y = [-30, 30, 30, -30, 0, -30, 30, 0]
axs.scatter(x, y, color='black', marker='o')

axs.set_xlabel('Red Wing EW Excess ($\mathring{A}$)')
axs.set_ylabel('Blue Wing EW Excess ($\mathring{A}$)')

axs.axvline(x=0, color='black', linestyle='--', alpha=0.5)
axs.axhline(y=0, color='black', linestyle='--', alpha=0.5)
axs.plot(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100), color='black', linestyle='--', alpha=1.0, linewidth=2.0)
# add symmetric line profile text at a 45 degree rotation
axs.text(-17.0, -13.5, 'Symmetric', fontsize=17, ha='center', va='center', rotation=45)
axs.text(13.5, 17, 'Line Profiles', fontsize=17, ha='center', va='center', rotation=45)

# axs.plot(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100), color='black', linestyle='--', alpha=0.5)

axs.plot(np.linspace(-50, 50, 100), -np.linspace(-50, 50, 100), color='black', linestyle='--', alpha=0.5)

axs.set_xlim(-40, 40)
axs.set_ylim(-40, 40)

# title labels
labels = ['Enhanced Blue/\nSuppressed Red Wings', 'Suppressed Red Wing', 'Suppressed Wings', 'Enhanced Wings', 'Enhanced Red Wing', 'Suppressed Blue/\nEnhanced Red Wings', 'Enhanced Blue Wing', 'Suppressed Blue Wing' ]

x_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
theory_datasets = [
    [0, 0, 0, 0.1, 0.5, 1.15, 0.4, -0.15, 0, 0, 0], # Inverse P-Cygni
    [0, 0, 0, 0, 0.25, 1.2, 0, 0, 0, 0, 0], # Increased Red Absorption
    [0, 0, 0, 0, 0, 1.25, 0, 0, 0, 0, 0], # Broad Absorption Wings
    [0, 0, 0, 0.1, 0.4, 1.15, 0.4, 0.1, 0, 0, 0], # Broad Emission Wings
    [0, 0, 0, 0, 0.25, 1.15, 0.5, 0.1, 0, 0, 0], # Increased Red Emission
    [0, 0, 0, -0.15, 0.44, 1.15, 0.5, 0.1, 0, 0, 0], # P-Cygni
    [0, 0, 0, 0.1, 0.5, 1.15, 0.25, 0, 0, 0, 0], # Increased Blue Emission
    [0, 0, 0, 0, 0, 1.2, 0.25, 0, 0, 0, 0], # Increased Blue Absorption
]

x_values_detailed = np.linspace(-5, 5, 100)
theory_datasets_detailed = []
for i in range(8):
    array = np.interp(x_values_detailed, x_values, theory_datasets[i])
    theory_datasets_detailed.append(gaussian_filter1d(array, 2.5))

plt.xticks(visible=False)
plt.yticks(visible=False)

# Define positions for the mini graphs
positions = [
    (-85, 22),
    (-85, -18.5),
    (-85, -60),
    (47, 22),
    (47, -18.5),
    (47, -60),
    (-17.5, 43),
    (-17.5, -85)
]

# Plot temporary Gaussian in mini graphs
i = 0

for x_pos, y_pos in positions:
    # Create an inset axis at each position
    ax_inset = axs.inset_axes([x_pos, y_pos, 35, 35],
                              transform=axs.transData)
    # Generate data for Gaussian plot
    x = np.linspace(-5, 5, 100)
    y = np.exp(-x**2)
    # Plot Gaussian in inset axes
    ax_inset.plot(x, y, color='red', linestyle='--')
    ax_inset.plot(x_values_detailed, theory_datasets_detailed[i], color='black')
    ax_inset.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    # Keep axes on but remove the numbers surrounding the plot
    ax_inset.tick_params(
        axis='both',          # Apply changes to both axes
        which='both',         # Apply to both major and minor ticks
        labelbottom=False,    # Hide x-axis tick labels
        labelleft=False       # Hide y-axis tick labels
    )
    ax_inset.set_title(labels[i], fontdict={'fontsize': 14})
    ax_inset.set_ylim(-0.3, 1.2)
    i += 1

# Create custom legend
custom_lines = [Line2D([0], [0], color='red', linestyle='--'),
                Line2D([0], [0], color='black'),
                Line2D([0], [0], color='black', linestyle='--', alpha=0.5)]

fig.legend(custom_lines, ['Fixed Gaussian', 'Hypothetical Line Profile', 'Rest Wavelength'], loc='upper center', ncol=1, bbox_to_anchor=(-0.1, -0.13))
plt.show()

# %% Figure 2 - MASK FITTING METHODOLOGY
################################################################################
print('FIGURE 2: MASK FITTING METHODOLOGY')
################################################################################

H_alpha = 6562.819
blue_peak_mask = (22, 55)  # number of angstroms to cut around the peak, blue minus.
red_peak_mask = (22, 55)  # number of angstroms to cut around the peak, red plus.

final_results = all_results[11]  # 45° inclination
run =701

fig, ax = plt.subplots(1, 1, figsize=(7, 7), sharey=True)
ax.plot(final_results['wavelength_grid'][run],
            final_results['grid'][run],
            label='Original Data',
            color='black'
            )
ax.plot(final_results['wavelength_grid'][run],
            final_results['fitted_grid'][run],
            label='Optimal Gaussian',
            color='red'
            )
ax.plot(final_results['wavelength_grid'][run],
            final_results['fit_con'][run],
            label='Fitted Continuum',
            color='blue'
            )
ax.axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5, label=r'$H\alpha$')
ax.set_xlabel('Wavelength ($\mathring{A}$)')
ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathring{A}^{-1}$)')

ax.axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
ax.axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
ax.axvspan(H_alpha - blue_peak_mask[1], H_alpha - blue_peak_mask[0], color='blue', alpha=0.1, label='Blue Window')
ax.axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
ax.axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
ax.axvspan(H_alpha + red_peak_mask[0], H_alpha + red_peak_mask[1], color='red', alpha=0.1, label='Red Window')

# Get the y-position for the annotations
y_min, y_max = ax.get_ylim()
y_pos = y_min + 0.9 * (y_max - y_min)  # Adjust the 0.1 to move the arrow up or down

# Add double-sided arrow for blue mask
ax.annotate(
    '',
    xy=(H_alpha - blue_peak_mask[1], y_pos),
    xytext=(H_alpha - blue_peak_mask[0], y_pos),
    arrowprops=dict(arrowstyle='<->', color='blue', linewidth=2)
)

# Add double-sided arrow for red mask
ax.annotate(
    '',
    xy=(H_alpha + red_peak_mask[0], y_pos),
    xytext=(H_alpha + red_peak_mask[1], y_pos),
    arrowprops=dict(arrowstyle='<->', color='red', linewidth=2)
)

# Get existing handles and labels
handles, labels = ax.get_legend_handles_labels()

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0, top=0.85)

# Add global legend
fig.legend(handles, labels, loc='upper center', ncol=len(labels)/3, bbox_to_anchor=(0.48, 1.03))
# add white space on the right side of the plot
plt.subplots_adjust(right=0.83)
plt.show()

# %% FIGURE 4 - CV PAPER 4 SIROCCO SUBPLOTS
################################################################################
print('FIGURE 3 - CV PAPER 4 SIROCCO SUBPLOTS')
################################################################################
print('This plot is in the Sirocco_spectrum_plotter.py script')

# %% FWHM histograms calculations
################################################################################
print('FWHM histograms calculations')
################################################################################

# Define inclination columns and incs
inclination_columns = [10, 11, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
incs = [0 for _ in range(10)]  # to align indices
incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models
mask = 'FWHM_Mask'
ew_results = {}
fwhm_results = {}
plotting = False

numbers = [10, 11, 12, 13, 14]
for number in tqdm(numbers):
    inclination_column = number  # 45° inclination
    if os.path.exists(f'Emission_Line_Asymmetries/FWHM_0p7_3_mask_data/final_results_inc_col_{inclination_column}.npy'):
        final_results = np.load(f'Emission_Line_Asymmetries/FWHM_0p7_3_mask_data/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()
        
    fwhm_bounds_array = np.array(final_results['fwhm_bounds']) # anstrom bounds
    cut_runs = final_results['cut_runs']
    blue_fwhm_bounds = fwhm_bounds_array[:, 0]
    red_fwhm_bounds = fwhm_bounds_array[:, 1]
    fwhm = red_fwhm_bounds - blue_fwhm_bounds
    #fwhm = np.delete(fwhm, cut_runs)
    
    #FWHM errors 
    fwhm_error = np.array(final_results['fwhm_error'])

    cuneo_data = np.load('Emission_Line_Asymmetries/Cuneo_FWHM/final_results_replicating_Cuneo_0p7_3FWHM.npy', allow_pickle=True).item()
    cueno_fwhm_bounds_array = np.array(cuneo_data['fwhm_bounds']) # anstrom bounds
    cuneo_blue_fwhm_bounds = cueno_fwhm_bounds_array[:, 0]
    cuneo_red_fwhm_bounds = cueno_fwhm_bounds_array[:, 1]
    cuneo_fwhm = cuneo_red_fwhm_bounds - cuneo_blue_fwhm_bounds

    # Ensure that the FWHM values are positive for logarithmic binning
    sirocco_positive = fwhm[fwhm > 0]
    cuneo_positive = cuneo_fwhm[cuneo_fwhm > 0]

    # Define logarithmic bins with 0.1 dex spacing for Sirocco data
    min_fwhm = np.min(sirocco_positive)
    max_fwhm = np.max(sirocco_positive)
    bins = 10 ** np.arange(np.log10(min_fwhm), np.log10(max_fwhm) + 0.1, 0.1)

    # Define logarithmic bins for Cúneo data
    min_c_fwhm = np.min(cuneo_positive)
    max_c_fwhm = np.max(cuneo_positive)
    bins_c = 10 ** np.arange(np.log10(min_c_fwhm), np.log10(max_c_fwhm) + 0.1, 0.1)

    if plotting:
        plt.figure(figsize=(10, 6))
        plt.hist(fwhm, bins=bins, alpha=0.5, label='Sirocco', color='blue')
        plt.hist(cuneo_fwhm, bins=bins_c, alpha=0.5, label='Cúneo et al. (2023)', color='red')
        plt.axvline(x=44, color='black', linestyle='--', alpha=0.5, zorder=1, label='2x1000km/s')
        plt.xlabel('Å')
        plt.ylabel('Frequency')
        plt.title(f'FWHM Histograms {incs[inclination_column]}° inclination')
        plt.xscale('log')
        plt.legend()
        plt.show()
        
    # print('Figure X - EW histograms')
    #inclination_column = 14  # 45° inclination
    if os.path.exists(f'Emission_Line_Asymmetries/new_data/ew_data/ew_data_all_inc{inclination_column}.npy'):
        ew_dict = np.load(f'Emission_Line_Asymmetries/new_data/ew_data/ew_data_all_inc{inclination_column}.npy', allow_pickle=True).item()

    ew = np.array(ew_dict['ew_data_all'])
    ew_error = ew_dict['ew_data_all_error']
    cut_runs = final_results['cut_runs']

    #ew = np.delete(ew, cut_runs)
    #ew_error = np.delete(ew_error, cut_runs)

    if os.path.exists(f'Emission_Line_Asymmetries/new_data/cuneo_ew_data/cueno_ew_data_all.npy'):
        cuneo_ew_dict = np.load(f'Emission_Line_Asymmetries/new_data/cuneo_ew_data/cueno_ew_data_all.npy', allow_pickle=True).item()

    cuneo_ew = np.array(cuneo_ew_dict['ew_data_all'])
    cuneo_ew_error = cuneo_ew_dict['ew_data_all_error']


    # Ensure that the EW values are positive for logarithmic binning
    sirocco_ew_positive = ew[ew > 0]
    cuneo_ew_positive = cuneo_ew[cuneo_ew > 0]

    # Define logarithmic bins with 0.1 dex spacing for Sirocco EW data
    min_ew = np.min(sirocco_ew_positive)
    max_ew = np.max(sirocco_ew_positive)
    bins_ew = 10 ** np.arange(np.log10(min_ew), np.log10(max_ew) + 0.1, 0.1)

    # Define logarithmic bins for Cúneo EW data
    min_c_ew = np.min(cuneo_ew_positive)
    max_c_ew = np.max(cuneo_ew_positive)
    bins_c_ew = 10 ** np.arange(np.log10(min_c_ew), np.log10(max_c_ew) + 0.1, 0.1)
    if plotting:
        plt.figure(figsize=(10, 6))
        plt.hist(ew, bins=bins_ew, alpha=0.5, label='Sirocco', color='blue')
        plt.hist(cuneo_ew, bins=bins_c_ew, alpha=0.5, label='Cúneo et al. (2023)', color='red')
        plt.xlabel('Å')
        plt.ylabel('Frequency')
        plt.title(f'EW Histograms {incs[inclination_column]}° inclination')
        plt.xscale('log')
        plt.legend()
        plt.show()

    #print('Figure X - EW vs fwhm scatter plot')
    # Example: Assuming fwhm and ew are NumPy arrays for the Sirocco data,
    # and cuneo_fwhm and cuneo_ew are the corresponding Cúneo et al. (2023) data.
    if plotting: 
        plt.figure(figsize=(10, 6))

        # Plot Sirocco data
        plt.scatter(fwhm, ew, color='blue', alpha=0.7, label='Sirocco')

        # Plot Cúneo data
        plt.scatter(cuneo_fwhm, cuneo_ew, color='red', alpha=0.7, label='Cúneo et al. (2023)')

        plt.xlabel('FWHM Å')
        plt.ylabel('Equivalent Width (Å)')
        plt.title(f'EW vs FWHM Scatter Plot{incs[inclination_column]}° inclination')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(0.01, 300)
        plt.xlim(1, 500)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.show()
    
    ew_results[inclination_column] = {'ew': ew, 'ew_error': ew_error}
    fwhm_results[inclination_column] = {'fwhm': fwhm, 'fwhm_error': fwhm_error}

# %% Figure 5 - EW vs FWHM All Incs Scatter plot
################################################################################
print('Figure 5 - EW vs FWHM All Incs Scatter plot')
################################################################################
inclination_columns = [10, 11, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
incs = [0 for _ in range(10)]  # to align indices
incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

fwhm_mrs, ew_mrs = np.loadtxt('Emission_Line_Asymmetries/Zhao_data/MRS.csv', delimiter=',', unpack=True) # Zhao et al. (2025)
fwhm_lrs, ew_lrs = np.loadtxt('Emission_Line_Asymmetries/Zhao_data/LRS.csv', delimiter=',', unpack=True)
#convert fwhm to angstroms
def kms_to_angstrom(velocity):
    """Converts velocity in km/s to wavelength in angstroms from central h_alpha line.
    Args:
        velocity (float): velocity in km/s"""  
    H_alpha = 6562.819
    angstrom = H_alpha * (velocity / 299792.458)
    return angstrom
fwhm_mrs = kms_to_angstrom(fwhm_mrs)
fwhm_lrs = kms_to_angstrom(fwhm_lrs)

# Create the figure and a 2×3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=True)
axs = axs.flatten()

# Remove the unused 6th subplot
fig.delaxes(axs[5])
plt.rcParams.update({'font.size': 15})

relevent_runs = {}
for idx, inclination_column in enumerate(inclination_columns):
    final_results = all_results[inclination_column]
    cut_runs = final_results['cut_runs']
    cut_runs_mask = np.ones(729, dtype=bool)
    anti_cut_runs_mask = np.zeros(729, dtype=bool)
    anti_cut_runs_mask[cut_runs] = True
    cut_runs_mask[cut_runs] = False
    
    peak_colour_map = final_results['peak_colour_map']
    grid_length = np.arange(0, 729)
    ew = np.array(ew_results[inclination_column]['ew'])
    fwhm = np.array(fwhm_results[inclination_column]['fwhm'])
    fwhm_threshold = (3,30)
    ew_threshold = (3,70)
    
    relevent_runs[inclination_column] = (ew >= ew_threshold[0]) & (ew <= ew_threshold[1]) & (fwhm <= fwhm_threshold[1]) & (fwhm >= fwhm_threshold[0])
    # merge relevent_runs mask and cut_runs_mask
    relevent_runs[inclination_column] = relevent_runs[inclination_column] & cut_runs_mask
    print(f'Number of relevent runs: {len(ew[relevent_runs[inclination_column]])}')
    
    # exclude the relevent runs from cut_runs_mask
    cut_runs_plotting_mask = cut_runs_mask & ~relevent_runs[inclination_column]
    ax = axs[idx]
    
    ax.errorbar(
        ew[cut_runs_mask], fwhm[cut_runs_mask],
        xerr=ew_results[inclination_column]['ew_error'][cut_runs_mask],
        yerr=fwhm_results[inclination_column]['fwhm_error'][cut_runs_mask],
        fmt='none',
        ecolor='grey',
        alpha=0.5, 
        zorder=0
    )
    ax.errorbar(
        ew[anti_cut_runs_mask], fwhm[anti_cut_runs_mask],
        xerr=ew_results[inclination_column]['ew_error'][anti_cut_runs_mask],
        yerr=fwhm_results[inclination_column]['fwhm_error'][anti_cut_runs_mask],
        fmt='none',
        ecolor='dimgrey',
        alpha=0.1, 
        zorder=0
    )
    #print(len(ew), len(fwhm), f'inclination{incs[inclination_column]}')
    med_res_colour = '#FF9200'#FF7400'
    low_res_colour = '#FF0000'#FFB800'
    sirocco_colour = '#0c01c7'#0c01c7''#223CFF'
    in_seln_box_colour = '#00AE15'
    if idx == 0:
        ax.scatter(ew[relevent_runs[inclination_column]], fwhm[relevent_runs[inclination_column]], s=10, alpha=0.7, c=in_seln_box_colour, label='Selected Sirocco Spectra')
        ax.scatter(ew[cut_runs_plotting_mask], fwhm[cut_runs_plotting_mask], c=sirocco_colour, s=10, alpha=0.7, label='Retained Sirocco Spectra')
        ax.scatter(ew[anti_cut_runs_mask], fwhm[anti_cut_runs_mask], c='dimgrey', s=10, alpha=0.2, label='Excluded Sirocco Spectra')
        ax.scatter(cuneo_ew, cuneo_fwhm, color='cyan', s=45, marker='o', edgecolor='navy', alpha=0.5, label='Cúneo et al. (2023)')
        ax.scatter(ew_lrs, fwhm_lrs, color=low_res_colour, s=45, marker='o', edgecolor='black', alpha=0.2, label='Zhao et al. (2025) Low-Res Spectra')
        ax.scatter(ew_mrs, fwhm_mrs, color=med_res_colour, s=45, marker='o', edgecolor='black', alpha=0.2, label='Zhao et al. (2025) Med-Res Spectra')
        ax.scatter(ew[cut_runs_plotting_mask], fwhm[cut_runs_plotting_mask], c=sirocco_colour, s=10, alpha=0.7)
    else:
        #ax.scatter(ew[cut_runs_mask], fwhm[cut_runs_mask], c=sirocco_colour, s=10, alpha=0.7)
        ax.scatter(ew[anti_cut_runs_mask], fwhm[anti_cut_runs_mask], c='dimgrey', s=10, alpha=0.2)
        ax.scatter(cuneo_ew, cuneo_fwhm, color='cyan', s=45, marker='o', edgecolor='navy', alpha=0.5)
        ax.scatter(ew_lrs, fwhm_lrs, color=low_res_colour, s=45, marker='o', edgecolor='black', alpha=0.2)
        ax.scatter(ew_mrs, fwhm_mrs, color=med_res_colour, s=45, marker='o', edgecolor='black', alpha=0.2)
        ax.scatter(ew[cut_runs_plotting_mask], fwhm[cut_runs_plotting_mask], c=sirocco_colour, s=10, alpha=0.7)
    
    ax.set_xlabel('Equivalent Width (Å)')
    if idx in [0, 3]:
        ax.set_ylabel('Full-Width Half-Maximum (Å)')
    ax.set_title(f'{incs[inclination_column]}° inclination')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 400)
    ax.set_ylim(2, 500)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    if idx == 0:
        ax.plot(
            [ew_threshold[0], ew_threshold[1], ew_threshold[1], ew_threshold[0], ew_threshold[0]], [fwhm_threshold[0], fwhm_threshold[0], fwhm_threshold[1], fwhm_threshold[1], fwhm_threshold[0]],
            color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0,
            label=rf'\space${ew_threshold[0]}\mathring{{A}} \leq EW \leq {ew_threshold[1]}\mathring{{A}}$,\\ ${fwhm_threshold[0]}\mathring{{A}} \leq FWHM \leq {fwhm_threshold[1]}\mathring{{A}}$'
        )
    else: 
        ax.plot(
            [ew_threshold[0], ew_threshold[1], ew_threshold[1], ew_threshold[0], ew_threshold[0]], [fwhm_threshold[0],fwhm_threshold[0], fwhm_threshold[1], fwhm_threshold[1], fwhm_threshold[0]],
            color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0,
        )
    

    ax.scatter(ew[relevent_runs[inclination_column]], fwhm[relevent_runs[inclination_column]], s=10, alpha=0.7, c=in_seln_box_colour)
    # add text to top left of plot 
    if len(ew[relevent_runs[inclination_column]])<100:
        ax.text(0.54, 0.82, f'Retained: {len(ew[cut_runs_mask])}/729\n'+'In Sel$^{\mathrm{n}}$ Box:\enspace\enspace'+f'{len(ew[relevent_runs[inclination_column]])}/{len(ew[cut_runs_mask])}',
            transform=ax.transAxes,
            fontsize=15, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
            ha='right', 
            va='bottom')
    else: 
        ax.text(0.54, 0.82, f'Retained: {len(ew[cut_runs_mask])}/729\n'+'In Sel$^{\mathrm{n}}$ Box:\enspace'+f'{len(ew[relevent_runs[inclination_column]])}/{len(ew[cut_runs_mask])}',
            transform=ax.transAxes,
            fontsize=15, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
            ha='right', 
            va='bottom')

    ax.set_axisbelow(True) 
# Adjust subplot spacing (both horizontally and vertically)
fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)

# Position the legend closer to the top
fig.legend(
    loc='lower right', ncol=1,
    bbox_to_anchor=(0.92, 0.1075),
)
# Manually reposition the bottom row subplots so they are centered
pos3 = axs[3].get_position()
pos4 = axs[4].get_position()
axs[3].set_position([0.125, pos3.y0-0.04, 0.26, pos3.height])  # shift bottom-left a bit to the right
axs[4].set_position([0.385, pos4.y0-0.04, 0.255, pos4.height])  # shift bottom-right a bit left

plt.show()

# %% FIGURE 6 - ALL INCS, REALISTIC RUNS, CUENO DATA
################################################################################
print('FIGURE 6 - ALL INCS, REALISTIC RUNS')
################################################################################

# Define inclination columns and incs
inclination_columns = [10, 11, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
incs = [0 for _ in range(10)]  # to align indices
incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# Create the figure and a 2×3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=True)
axs = axs.flatten()

# Remove the unused 6th subplot
fig.delaxes(axs[5])
plt.rcParams.update({'font.size': 15})

# Load Cuneo's data once
cueno_refitted = np.load(f'Emission_Line_Asymmetries/Cuneo_2023_data/Cuneo_refitted_final_results.npy', allow_pickle=True).item()
# bz_cam   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv',   delimiter=',')
# mv_lyr   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv',   delimiter=',')
# v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# Prepare legend handles
handles = []
labels = []

for idx, inclination_column in enumerate(inclination_columns):
    final_results = all_results[inclination_column]
    cut_runs = final_results['cut_runs']
    peak_colour_map = final_results['peak_colour_map']
    grid_length = np.arange(0, 729)

    cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
    cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
    cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
    cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
    cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
    # Create masks for single-peaked and double-peaked spectra
    cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
    # Boolean masks for single- vs double-peaked
    single_mask = (cut_peak_colour_map == 'black')
    double_mask = (cut_peak_colour_map == 'red')

    ax = axs[idx]
    #ax.set_axisbelow(True)
    ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
    cut_relevent_runs = np.delete(relevent_runs[inclination_column], cut_runs)
    s_mask = cut_relevent_runs & single_mask
    d_mask = cut_relevent_runs & double_mask
    both_mask = cut_relevent_runs & (single_mask | double_mask)

    # Plot error bars (all points together for simplicity)
    ax.errorbar(
        cut_red_ew_excess[both_mask],
        cut_blue_ew_excess[both_mask],
        xerr=cut_red_ew_excess_error[both_mask],
        yerr=cut_blue_ew_excess_error[both_mask],
        fmt='none',
        ecolor='black',
        alpha=0.5,
        zorder=1
    )

    # Plot single-peaked (black) vs double-peaked (red)
    # Add legend entries only in the first subplot
    if idx == 0:
        single_scatter = ax.scatter(
            cut_red_ew_excess[s_mask],
            cut_blue_ew_excess[s_mask],
            c='red',
            s=10,
            label='Single-peaked Sirocco Spectra',
            zorder=3,
            alpha=0.7
        )
        double_scatter = ax.scatter(
            cut_red_ew_excess[d_mask],
            cut_blue_ew_excess[d_mask],
            c='black',
            s=10,
            label='Double-peaked Sirocco Spectra',
            zorder=2,
            alpha=0.7
        )
        handles.append(single_scatter)
        labels.append('Single-peaked Sirocco Spectra')
        handles.append(double_scatter)
        labels.append('Double-peaked Sirocco Spectra')
    else:
        ax.scatter(
            cut_red_ew_excess[s_mask],
            cut_blue_ew_excess[s_mask],
            c='red',
            s=10,
            zorder=3,
            alpha=0.7
        )
        ax.scatter(
            cut_red_ew_excess[d_mask],
            cut_blue_ew_excess[d_mask],
            c='black',
            s=10,
            zorder=2,
            alpha=0.7
        )
    base_colour='cyan'# Define a mapping from system name to colour
    mapping = {
        'MV_Lyr': 'navy',
        'V425_Cas': 'navy',
        'V751_Cyg': 'navy',
        'BZ_Cam': 'navy'
    }

    if idx == 0:
        teo_scatter = ax.scatter(cueno_refitted['red_ew_excess'], 
                                 cueno_refitted['blue_ew_excess'],
                                 color='cyan', s=45, marker='o',edgecolor='navy',
                                 alpha=0.7,
                                 label='Cúneo et al. (2023)', 
                                 zorder=4)
        ax.errorbar(
            cueno_refitted['red_ew_excess'],
            cueno_refitted['blue_ew_excess'],
            xerr=cueno_refitted['red_ew_excess_error'],
            yerr=cueno_refitted['blue_ew_excess_error'],
            fmt='none',
            ecolor='black',
            alpha=0.5,
            zorder=1
        )
        # ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
        #                             color=mapping['BZ_Cam'], s=45, marker='o',edgecolor='navy',
        #                             alpha=0.7,
        #                             label='Cúneo et al. (2023)', 
        #                             zorder=4)
        # # Plot additional datasets without separate legend labels
        # ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=mapping['MV_Lyr'], s=45, marker='o',edgecolor='navy', alpha=0.7, zorder=4)
        # ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=mapping['V425_Cas'], s=45, marker='o',edgecolor='navy', alpha=0.7, zorder=4)
        # ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=mapping['V751_Cyg'], s=45, marker='o',edgecolor='navy', alpha=0.7, zorder=4)
        handles.append(teo_scatter)
        labels.append('Cúneo et al. (2023)')
    elif idx == 1: 
        teo_scatter = ax.scatter(cueno_refitted['red_ew_excess'],
                                cueno_refitted['blue_ew_excess'],
                                color='cyan', s=45, marker='o',edgecolor='navy',
                                alpha=0.7, zorder=4)
        ax.errorbar(
            cueno_refitted['red_ew_excess'],
            cueno_refitted['blue_ew_excess'],
            xerr=cueno_refitted['red_ew_excess_error'],
            yerr=cueno_refitted['blue_ew_excess_error'],
            fmt='none',
            ecolor='black',
            alpha=0.5,
            zorder=1
        )
        # ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
        #                             color=mapping['BZ_Cam'], s=45, marker='o',
        #                             alpha=0.7, edgecolor='navy', zorder=4)                         
        # # Plot additional datasets without separate legend labels
        # ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=mapping['MV_Lyr'], s=45, marker='o', alpha=0.7, edgecolor='navy', zorder=4)
        # ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=mapping['V425_Cas'], s=45, marker='o', alpha=0.7, edgecolor='navy', zorder=4)
        # ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=mapping['V751_Cyg'], s=45, marker='o', alpha=0.7, edgecolor='navy', zorder=4)

    # Reference lines
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

    # Dashed box for linear/log threshold
    linear_thrs = 0.1
    ax.plot(
        [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
        [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
        color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0,
        label='Linear/Logarithmic Threshold'
    )
    #linear threshold lines
    ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)

    # Axes formatting
    ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
    # Put a y-label on the left column of subplots (idx=0 for top-left, idx=3 for bottom-left)
    if idx in [0, 3]:
        ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')

    ax.set_title(f'{incs[inclination_column]}° inclination')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xscale('symlog', linthresh=linear_thrs)
    ax.set_yscale('symlog', linthresh=linear_thrs)

    # Minor tick locators for symlog
    ax.xaxis.set_minor_locator(
        SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
    )
    if len(cut_red_ew_excess[s_mask]) < 100 and idx in [3,4]:
        ax.text(0.39, 0.82, f'Single:\space\space{len(cut_red_ew_excess[s_mask])}/{len(cut_red_ew_excess[both_mask])}\nDouble: {len(cut_red_ew_excess[d_mask])}/{len(cut_red_ew_excess[both_mask])}',
        transform=ax.transAxes,
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
    elif len(cut_red_ew_excess[s_mask]) < 100: 
        ax.text(0.42, 0.82, f'Single:\space\space{len(cut_red_ew_excess[s_mask])}/{len(cut_red_ew_excess[both_mask])}\nDouble: {len(cut_red_ew_excess[d_mask])}/{len(cut_red_ew_excess[both_mask])}',
        transform=ax.transAxes,
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
    else:
        ax.text(0.42, 0.82, f'Single: {len(cut_red_ew_excess[s_mask])}/{len(cut_red_ew_excess[both_mask])}\nDouble: {len(cut_red_ew_excess[d_mask])}/{len(cut_red_ew_excess[both_mask])}',
        transform=ax.transAxes,
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
# Add final threshold handle/label
labels.append('Linear/Logarithmic Threshold')
handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

fig.text(0.895, 0.32, 
        '$\pm\,1000-2500\,km\,s^{-1}$ Masking Window', 
        #transform=ax2.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
#labels.append('$1000-2500 kms^{-1}$ Mask Window')
#handles.append(Line2D([0], [0], color='white', linestyle='-', linewidth=2.0))

# Adjust subplot spacing (both horizontally and vertically)
fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)

# Position the legend closer to the top
fig.legend(
    handles, labels,
    loc='lower right', ncol=1,
    bbox_to_anchor=(0.9, 0.17)
)

# Manually reposition the bottom row subplots so they are centered
pos3 = axs[3].get_position()
pos4 = axs[4].get_position()
axs[3].set_position([0.125, pos3.y0-0.04, 0.26, pos3.height])  # shift bottom-left a bit to the right
axs[4].set_position([0.385, pos4.y0-0.04, 0.255, pos4.height])  # shift bottom-right a bit left
plt.show()


# %% FIGURE 7 - QUADRANTED MASKING ARROW PLOT
################################################################################
print('FIGURE 7 - QUADRANTED MASKING ARROW PLOT')
################################################################################
mask_results = {}
inclination_column = 11  # 45° inclination
#masks = ['11_88_mask', '22_88_mask', '22_55_mask']  # ...
masks = ['18_55_mask', '22_55_mask', '26_55_mask']  # 20-55, 22-55, 24-55

# Load results for each mask
for mask in tqdm(masks):
    filepath = f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy'
    if os.path.exists(filepath):
        all_results[mask] = np.load(filepath, allow_pickle=True).item()

plt.figure(figsize=(7, 7))
plt.rcParams.update({'font.size': 15})

# Get the list of runs to exclude (bad runs) across all masks
to_axe_runs = []
for mask in masks:
    final_results = all_results[mask]
    to_axe_runs.append(final_results['cut_runs'])
to_axe_runs = np.unique(np.concatenate(to_axe_runs))

# Use the reference mask (choose '22_55_mask') to assign quadrants
ref_results = all_results[masks[1]]
red_vals = ref_results['red_ew_excess']
blue_vals = ref_results['blue_ew_excess']

# All valid runs are those not in "to_axe_runs"
all_runs = [i for i in range(729) if i not in to_axe_runs]
print('possible runs:', len(all_runs))
real_runs = np.delete(relevent_runs[inclination_column], to_axe_runs)
print('real runs:', len(real_runs))
#ONly include all_runs that return true with real_runs
selected_runs = np.array([val for i, val in enumerate(all_runs) if real_runs[i]])
print('selected runs:', len(selected_runs))


# Split runs into quadrants:
# Quadrant 1: positive x, positive y
q1 = [i for i in selected_runs if red_vals[i] > 0 and blue_vals[i] > 0]
print('# in Upper right quadrant:', len(q1))
# Quadrant 2: negative x, positive y
q2 = [i for i in selected_runs if red_vals[i] < 0 and blue_vals[i] > 0]
print('# in Upper left quadrant:', len(q2))
# Quadrant 3: negative x, negative y
q3 = [i for i in selected_runs if red_vals[i] < 0 and blue_vals[i] < 0]
print('# in Lower left quadrant:', len(q3))
# Quadrant 4: positive x, negative y
q4 = [i for i in selected_runs if red_vals[i] > 0 and blue_vals[i] < 0]
print('# in Lower right quadrant:', len(q4))

# Set a seed for reproducibility if desired
#seed = np.random.choice(np.arange(1, 1000, 1))
seed = 103 # for paper
#seed = 0
print('seed:', seed)
np.random.seed(seed)


# Select 10 runs from each quadrant (if possible; if not, select all)
def sample_runs(runs, n=10):
    if len(runs) >= n:
        return list(np.random.choice(runs, n, replace=False))
    else:
        return runs

keep_samples = sample_runs(q1, 20) + sample_runs(q2, 10) + sample_runs(q3, 10) + sample_runs(q4, 10)
keep_samples = np.array(keep_samples)

# Define cut_runs as those runs NOT chosen
cut_runs = np.array([i for i in range(729) if i not in keep_samples])

print("Selected runs:", keep_samples)

colours = ['red', 'black', 'blue']
mask_labels = ['$\pm\,800-2500\,km\,s^{-1}$', '$\pm\,1000-2500\,km\,s^{-1}$', '$\pm\,1200-2500\,km\,s^{-1}$']

# For one of the masks (here the second mask) we plot error bars for context
for i, mask in enumerate(masks):
    final_results = all_results[mask]
    cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
    cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
    cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
    cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)

    if i == 1:
        plt.errorbar(
            cut_red_ew_excess,
            cut_blue_ew_excess,
            xerr=cut_red_ew_excess_error,
            yerr=cut_blue_ew_excess_error,
            fmt='none',
            ecolor='grey',
            alpha=0.5,
            zorder=-1
        )
        scatter_plot = plt.scatter(
            cut_red_ew_excess,
            cut_blue_ew_excess,
            c=colours[i],
            s=10,
            label=mask_labels[i]
        )
    # else:
    #     plt.scatter(
    #         np.delete(final_results['red_ew_excess'], cut_runs),
    #         np.delete(final_results['blue_ew_excess'], cut_runs),
    #         c=colours[i],
    #         s=10,
    #         #label=mask_labels[i]
    #     )

# Build a dictionary to store data points for each selected run from each mask
data_points = {}
for run_index in keep_samples:
    data_points[run_index] = []
    for mask in masks:
        final_results = all_results[mask]
        red_val = final_results['red_ew_excess'][run_index]
        blue_val = final_results['blue_ew_excess'][run_index]
        data_points[run_index].append((red_val, blue_val))

# Define arrow colors
arrow_colors = ['red', 'blue']

# Get existing handles and labels for the legend
handles, labels = plt.gca().get_legend_handles_labels()

# Custom legend handler for arrows
def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrowPatch(
        (0, 4.5),
        (30, 4.5),
        mutation_scale=fontsize * 0.75,
        arrowstyle='<|-',
        color=orig_handle.get_edgecolor(),
        linewidth=orig_handle.get_linewidth()
    )


# Create custom arrow handles
arrow_black_line = FancyArrowPatch((0, 0.5), (1, 0.5), mutation_scale=15,
                                   arrowstyle='-|>', color='red', linewidth=1)
arrow_red_line = FancyArrowPatch((0, 0.5), (1, 0.5), mutation_scale=15,
                                 arrowstyle='-|>', color='blue', linewidth=1)
linlogthereshold = Line2D([0], [0], color='black', linestyle='--', linewidth=2.0)

handles.extend([linlogthereshold, arrow_black_line])
labels.extend([
    'Linear/Logarithmic Threshold',
    f'{mask_labels[0]}'
])

# Draw arrows between data points across masks
for idx, run_index in enumerate(keep_samples):
    points = data_points[run_index]
    # Arrow from mask '22_55_mask' (index 1) to '20_55_mask' (index 0)
    start_point = points[1]
    end_point = points[0]
    plt.annotate(
        '',
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(arrowstyle='-|>', color=arrow_colors[0], lw=1),
        annotation_clip=False,
        zorder=0
    )
    # Arrow from mask '22_55_mask' (index 1) to '24_55_mask' (index 2)
    start_point = points[1]
    end_point = points[2]
    plt.annotate(
        '',
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(arrowstyle='-|>', color=arrow_colors[1], lw=1),
        annotation_clip=False,
        zorder=0
    )

# Plot vertical and horizontal zero lines
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

# Format plot axes and labels
incs = [0] * 10
incs.extend([20, 45, 60, 72.5, 85])
plt.xlabel('Red Wing EW Excess ($Å$)')
plt.ylabel('Blue Wing EW Excess ($Å$)')
plt.title(f'{incs[inclination_column]}° inclination')

linear_thrs = 0.1
plt.plot(
    [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
    [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
    color='black',
    linestyle='--',
    alpha=1.0,
    zorder=1,
    linewidth=2.0
)
plt.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
plt.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
plt.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
plt.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)

plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.xscale('symlog', linthresh=linear_thrs)
plt.yscale('symlog', linthresh=linear_thrs)

handles.extend([arrow_red_line])
labels.extend([f'{mask_labels[2]}'])

# Rearranging handles/labels if desired (example: moving first handle to index 2)
handles.insert(2, handles.pop(0))
labels.insert(2, labels.pop(0))

ax = plt.gca()
ax.set_axisbelow(True)
ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

legend = plt.legend(
    handles,
    labels,
    handler_map={
        arrow_black_line: HandlerPatch(patch_func=make_legend_arrow),
        arrow_red_line: HandlerPatch(patch_func=make_legend_arrow)
    },
    loc='lower left',
    bbox_to_anchor=(-0.01, 0.74),
    frameon=True
)
legend.get_frame().set_facecolor('white')
plt.show()

# %% FIGURE 8 - FWHM WINDOW Comparison with Cúneo Data for 20° and 45°
################################################################################
print('FIGURE 8 - FWHM WINDOW Comparison with Cúneo Data for 20° and 45°')
################################################################################

# Example: define your inclination columns and incs
inclination_columns = [10, 11]#, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
incs = [0 for _ in range(10)]  # to align indices
incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# Create the figure and a 2×3 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True)
axs = axs.flatten()

# Remove the unused 6th subplot
#fig.delaxes(axs[5])
plt.rcParams.update({'font.size': 15})
cuneo_data = np.load('Emission_Line_Asymmetries/Cuneo_FWHM/final_results_replicating_Cuneo_1p0_5FWHM.npy', allow_pickle=True).item()
cuneo_red_ew_excess = cuneo_data['red_ew_excess']
cuneo_blue_ew_excess = cuneo_data['blue_ew_excess']
cuneo_red_ew_excess_error = cuneo_data['red_ew_excess_error']
cuneo_blue_ew_excess_error = cuneo_data['blue_ew_excess_error']
mask = 'FWHM_Mask'
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
    'V425_Cas': 'yellow',
    'V751_Cyg': 'green',
    'BZ_Cam': 'cyan'
}
# Create a new array of colours matching the system_labels
system_colours_extended = [mapping[label] for label in system_labels]

# Prepare legend handles
handles = []
labels = []

for idx, inclination_column in enumerate(inclination_columns):
    if os.path.exists(f'Emission_Line_Asymmetries/FWHM_1p0_5_mask_data/final_results_inc_col_{inclination_column}.npy'):
        final_results = np.load(f'Emission_Line_Asymmetries/FWHM_1p0_5_mask_data/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()
    
    cut_runs = final_results['cut_runs']
    peak_colour_map = final_results['peak_colour_map']
    grid_length = np.arange(0,729)

    cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
    cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
    cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
    cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
    cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
    cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
    cut_grid_length = np.delete(grid_length, cut_runs)

    # A mask that hides all values with EW excess of 0
    for i in range(len(cut_red_ew_excess)):
        if cut_red_ew_excess[i] == 0 or cut_blue_ew_excess[i] == 0:
            cut_red_ew_excess[i] = np.nan
            cut_blue_ew_excess[i] = np.nan
 
    # create a mask for data with EW between 1 and 100 and FWHM < 30
    mask_cut = np.delete(relevent_runs[inclination_column], cut_runs)
    
    # Create masks for single-peaked and double-peaked spectra
    cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
    # Boolean masks for single- vs double-peaked
    single_mask = (cut_peak_colour_map == 'black')
    double_mask = (cut_peak_colour_map == 'red')
    mask_cut_plus_single = mask_cut & single_mask
    mask_cut_plus_double = mask_cut & double_mask
    ax = axs[idx]

    # Plot error bars (all points together for simplicity)
    ax.errorbar(
        cut_red_ew_excess[mask_cut],
        cut_blue_ew_excess[mask_cut],
        xerr=cut_red_ew_excess_error[mask_cut],
        yerr=cut_blue_ew_excess_error[mask_cut],
        fmt='none',
        ecolor='black',
        alpha=0.5,
        zorder=-1
    )
    ax.errorbar(
        cuneo_red_ew_excess,
        cuneo_blue_ew_excess,
        xerr=cuneo_red_ew_excess_error,
        yerr=cuneo_blue_ew_excess_error,
        fmt='none',
        ecolor='black',
        alpha=0.5,
        zorder=-1
    )
    # Plot single-peaked (black) vs double-peaked (red)
    # Add legend entries only in the first subplot
    if idx == 0:
        double_near_scatter = ax.scatter(
            cut_red_ew_excess[mask_cut_plus_double],
            cut_blue_ew_excess[mask_cut_plus_double],
            c='black',
            s=10,
            label='Potential Sirocco Spectra',
            zorder=1,
            alpha=0.7
        )
        single_near_scatter = ax.scatter(
            cut_red_ew_excess[mask_cut_plus_single],
            cut_blue_ew_excess[mask_cut_plus_single],
            c='red',
            s=10,
            label='Potential Sirocco Spectra',
            zorder=1,
            alpha=1
        )
        cuneo_scatter = ax.scatter(
            cuneo_red_ew_excess,
            cuneo_blue_ew_excess,
            c='cyan',
            s=45,
            marker='o',
            edgecolor='navy',
            alpha=0.7,
            label='Cúneo et al. (2023)'
        )

        handles.append(single_near_scatter)
        labels.append('Single-Peaked Sirocco Spectra')
        handles.append(double_near_scatter)
        labels.append('Double-Peaked Sirocco Spectra')
        handles.append(cuneo_scatter)
        labels.append('Cúneo et al. (2023)')
        
    else:
        ax.scatter(
            cut_red_ew_excess[mask_cut_plus_double],
            cut_blue_ew_excess[mask_cut_plus_double],
            c='black',
            s=10,
            zorder=1,
            alpha=0.7
        )
        ax.scatter(
            cut_red_ew_excess[mask_cut_plus_single],
            cut_blue_ew_excess[mask_cut_plus_single],
            c='red',
            s=10,
            zorder=1,
            alpha=1
        )
        ax.scatter(
            cuneo_red_ew_excess,
            cuneo_blue_ew_excess,
            c='cyan',
            s=45,
            marker='o',
            edgecolor='navy',
            alpha=0.7
        )
        
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

    # Dashed box for linear/log threshold
    linear_thrs = 0.1
    ax.plot(
        [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
        [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
        color='black', linestyle='--', alpha=1.0, zorder=0, linewidth=2.0,
        label='Linear/Logarithmic Threshold'
    )
    #linear threshold lines
    ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
    ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
    ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
    ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)

    # Axes formatting
    ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
    # Put a y-label on the left column of subplots (idx=0 for top-left, idx=3 for bottom-left)
    if idx in [0, 3]:
        ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')

    ax.set_title(f'{incs[inclination_column]}° inclination')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xscale('symlog', linthresh=linear_thrs)
    ax.set_yscale('symlog', linthresh=linear_thrs)
    ax.set_axisbelow(True)
    ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=-1)

    # Minor tick locators for symlog
    ax.xaxis.set_minor_locator(
        SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
    )
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
    )

# Add final threshold handle/label
labels.append('Linear/Logarithmic Threshold')
handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

# Adjust subplot spacing (both horizontally and vertically)
fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)

# Position the legend closer to the top
fig.legend(
    handles, labels,
    loc='upper center', ncol=2,
    bbox_to_anchor=(0.4, 1.05)
)
fig.text(0.82, 1.013, 
"1 – 5× FWHM\nMasking Window",
        #transform=ax2.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='center', 
        va='top',
        multialignment='center')
plt.show()


# %% EMISSION MEASUREMENTS, CURVE OF GROWTH FIT AND LINEAR REGRESSION CALCULATION
################################################################################
print('EMISSION MEASUREMENTS, CURVE OF GROWTH FIT AND LINEAR REGRESSION CALCULATION')
################################################################################
plt.rcParams.update({'font.size': 15})
inclination = 11
# load data
emission_measures = np.load('Emission_Line_Asymmetries/Sirocco_based_data/emission_measures.npy')
remove_negative_ews = []
for i, val in enumerate(ew_results[inclination]['ew']):
    if val <= 0: 
        remove_negative_ews.append(i)
emission_measures = np.delete(emission_measures, remove_negative_ews)
equivalent_widths = np.delete(ew_results[inclination]['ew'], remove_negative_ews)
equivalent_widths_error = np.delete(ew_results[inclination]['ew_error'], remove_negative_ews)
log_ew = np.log10(equivalent_widths)
log_em = np.log10(emission_measures)
log_ew_error = (1/np.log(10))*(equivalent_widths_error/equivalent_widths)

path_to_grids = "Release_Ha_grid_spec_files"
parameter_table = pd.read_csv(f'{path_to_grids}/Grid_runs_logfile.csv')

# remove run number columns 
X = parameter_table.iloc[:, 1:].values
# only log10 the 0th, 1st, 2nd, and 4th columns. keep the rest the same

X_copy = X.copy() # for use later in script
X = np.delete(X, remove_negative_ews, axis=0)
log_X_all = np.column_stack((np.log10(X[:,0]),
                             np.log10(X[:,1]),
                             np.log10(X[:,2]),
                             X[:,3],
                             np.log10(X[:,4]), 
                             X[:,5]
                             ))

sim_parameters = [r'$\dot{M}_{disk}$',
        r'$\dot{M}_{wind}$',
        r'$KWD.d$',
        r'$r_{exp}$',
        r'$acc_{length}$',
        r'$acc_{exp}$'
        ] # Sirocco Parameters


# Ordinary Least Squares (OLS) model
ols_X = sm.add_constant(log_X_all) # add an intercept value to the X parameters
ols = sm.WLS(log_em, ols_X) # statsmodels OLS model
ols_result = ols.fit() # fitting the model
print(ols_result.summary()) # summary of the model (errors, coefficients, etc)


predicted_log_Y_all_ols = ols_result.predict(ols_X)

eqn_all = f'log(EM) = {ols_result.params[0]:.3f} + {ols_result.params[1]:.3f}log({sim_parameters[0]}) + {ols_result.params[2]:.3f}log({sim_parameters[1]}) + {ols_result.params[3]:.3f}log({sim_parameters[2]}) + {ols_result.params[4]:.3f}{sim_parameters[3]} + {ols_result.params[5]:.3f}log({sim_parameters[4]}) + {ols_result.params[6]:.3f}{sim_parameters[5]}'

print('Linear Regression model for the entire EW line') # in LaTeX style
display(Math(f'{eqn_all}')) # Latex formatted string to LaTeX output

# # OLS error calculation using standard error of the coefficients
# predicted_log_Y_all_ols_err = []
# for i, _ in enumerate(ols_X):
#     x_err_plus_c = [ols_result.bse[j]*abs(ols_X[i][j]) for j in range(len(ols_X[i]))]
#     y_err = np.sqrt(np.sum(np.array(x_err_plus_c)**2))
#     predicted_log_Y_all_ols_err.append(y_err)
    
plt.figure(figsize=(8, 8))
plt.scatter(predicted_log_Y_all_ols, log_em, alpha=0.5)
plt.xlabel('True $\mathrm{Log_{10}}$(Emission Measure)')
plt.ylabel('Predicted $\mathrm{Log_{10}}$(Emission Measure)')
plt.plot([min(log_em), max(log_em)],
         [min(log_em), max(log_em)],
         color='red',
         linestyle='--',
         label='$EM_{pred} = EM$'
         )
plt.title(f'Inclination {incs[inclination]}°')
plt.legend()
plt.show()


# CURVE OF GROWTH
# over plotting the Curve of growth in astronomy 
# https://en.wikipedia.org/wiki/Curve_of_growth_(astronomy)
# Eq11.4.4 https://www.astro.uvic.ca/~tatum/stellatm/atm11.pdf

def curve_of_growth(all_logEM, K1, K2): # K2 inputted as log to help with fitting
    def integrand(delta, tau_0):
        f = 1 - np.exp(-tau_0 * np.exp(-delta*delta))
        return f

    #all_logEM = np.arange(52,60,0.1) # logEM values found
    all_EM = 10.0**all_logEM # EM = 10^logEM
    all_tau_0 = 10**K2 * all_EM # 𝜏_0 = K2 * EM

    W = []
    for tau_0 in all_tau_0:
        W.append(K1 * quad(integrand, 0, np.inf, args=(tau_0))[0])
    return np.log10(W) # to reduce dynamic range of W

    
# Fit the curve of growth to the data with an initial guess for the parameters
popt, pcov, _, _, _= opt.curve_fit(curve_of_growth, log_em, log_ew, p0=[7.0, -60],maxfev=100000, method='lm', full_output=True)
print(f'Curve of growth fit parameters true: {popt}')

# Generate the curve of growth values using vectorized computation 
N_values = 10**np.linspace(52, 61, 500)

W_values = curve_of_growth(np.log10(N_values), *popt)
plt.figure(figsize=(7, 7))
plt.plot(np.log10(N_values), W_values, color='red', linestyle='--', label='Curve of Growth')
#plt.plot(np.log10(N_values), W_values, color='blue', linestyle='--', label='Initial Guess')
plt.scatter(log_em, log_ew, alpha=0.5)
plt.xlabel('True $\mathrm{Log_{10}}$(Emission Measure)')
plt.ylabel('$\mathrm{Log_{10}}$(Equivalent Width)')
plt.title(f'Inclination: {incs[inclination]}°')
#add the growth parameters to legend
ax = plt.gca()
ax.text(0.95, 0.05, 
        f'CoG Coefficients:\n$K_1$: {popt[0]:.2f}\n$K_2$: {popt[1]:.2f}', 
        transform=ax.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
plt.xlim(52.8, 60.2)
plt.legend()
plt.show()

# kernel density estimate of log_em values 
kde = KernelDensity(kernel='gaussian', bandwidth=1.8).fit(log_ew.reshape(-1, 1))
density = np.exp(kde.score_samples(log_ew.reshape(-1, 1)))
plt.plot(log_ew, density, 'o')
plt.show()

# Fit the curve of growth to the data with an initial guess for the parameters
# Generate the curve of growth values using vectorized computation
popt_pred, pcov_pred, infodict_pred, _, _= opt.curve_fit(curve_of_growth, predicted_log_Y_all_ols, log_ew, sigma=density, absolute_sigma=True, p0=[7.0, -60], maxfev=100000, method='lm', full_output=True)
print(f'Curve of growth fit parameters predicted: {popt_pred}')

W_values_pred = curve_of_growth(np.log10(N_values), *popt_pred)
plt.figure(figsize=(7, 7))
plt.plot(np.log10(N_values), W_values_pred, color='red', linestyle='--', label='Curve of Growth')
plt.errorbar(predicted_log_Y_all_ols, log_ew, yerr=log_ew_error, alpha=0.2,fmt='none', zorder=0)
#plt.plot(np.log10(N_values), curve_of_growth(np.log10(N_values), 4.7,-55.28), color='blue', linestyle='--', label='Initial Guess')
plt.scatter(predicted_log_Y_all_ols, log_ew, alpha=0.5)
plt.xlabel('Predicted $\mathrm{Log_{10}}$(Emission Measure)')
plt.ylabel('$\mathrm{Log_{10}}$(Equivalent Width)')
plt.title(f'Inclination: {incs[inclination]}°')
#add the growth parameters to legend
ax = plt.gca()
ax.text(0.95, 0.05, 
        f'CoG Coefficients:\n$K_1$: {popt_pred[0]:.2f}\n$K_2$: {popt_pred[1]:.2f}', 
        transform=ax.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
plt.ylim(min(log_ew)-0.2, max(log_ew)+0.2)
plt.legend()
plt.show()

# %% FIGURE 9 - LINEAR REGRESSORS OF EM AND EW
################################################################################
print('FIGURE 9 - LINEAR REGRESSORS OF EM AND EW')
################################################################################

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

not_relevant_runs_linear = np.delete(~relevent_runs[inclination], remove_negative_ews)
relevent_runs_linear = np.delete(relevent_runs[inclination], remove_negative_ews)

a=0.03
b=1.91
c=0.02
d=0.75
e=1.14
f=0.61
g=59.56
test_EM = a*log_X_all[:,0] + b*log_X_all[:,1] + c*log_X_all[:,2] + d*log_X_all[:,3] + e*log_X_all[:,4] + f*log_X_all[:,5] + g

h1 = ax1.scatter(log_em[not_relevant_runs_linear], predicted_log_Y_all_ols[not_relevant_runs_linear], alpha=0.5, c='navy')
h2 = ax1.scatter(log_em[relevent_runs_linear], predicted_log_Y_all_ols[relevent_runs_linear], alpha=0.5, c='#00AE15')
ax1.set_xlabel('True $\mathrm{Log_{10}}$(Emission Measure)')
ax1.set_ylabel('Predicted$\mathrm{Log_{10}}$(Emission Measure)')
ax1.plot(
    [min(log_em), max(log_em)],
    [min(log_em), max(log_em)],
     linestyle='--', linewidth=3, label='$EM = EM_{pred}$',
     c='red'
)
#ax1.set_title(f'Inclination {incs[inclination]}°')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.text(-0.255, 0.05, 
        f'$R^2$ Score: {ols_result.rsquared:.2f}', 
        transform=ax2.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
ax1.scatter(log_em, test_EM, alpha=0.5, c='orange', label='Test EM')
ax1.legend()
# Right subplot: Curve-of-Growth fit on linear data using a log-log plot

plt.figure(figsize=(7, 7))
ax2.plot(np.log10(N_values), W_values,  linestyle='--', linewidth=3, label='Curve of Growth',c='red')
ax2.scatter(log_em[not_relevant_runs_linear], log_ew[not_relevant_runs_linear], alpha=0.5, c='navy')
ax2.scatter(log_em[relevent_runs_linear], log_ew[relevent_runs_linear], alpha=0.5, c='#00AE15')
ax2.errorbar(log_em, log_ew, yerr=log_ew_error, alpha=0.3,fmt='none', zorder=0)
ax2.set_xlabel('True $\mathrm{Log_{10}}$(Emission Measure)')
ax2.set_ylabel('$\mathrm{Log_{10}}$(Equivalent Width)')
#add the growth parameters to legend
ax2.text(0.95, 0.05, 
        f'CoG Coefficients:\n$K_1$: {popt[0]:.1f}\n$K_2$: {popt[1]:.1f}', 
        transform=ax2.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
ax2.set_xlim(min(predicted_log_Y_all_ols)-0.2, max(predicted_log_Y_all_ols)+0.2)
ax2.set_ylim(min(log_ew)-0.2, max(log_ew)+0.2)
ax2.legend()
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
#ax2.set_title(f'Inclination: {incs[inclination]}°')
#fig.suptitle(f'Inclination: {incs[inclination]}°')

ax3.plot(np.log10(N_values), W_values_pred,  linestyle='--', linewidth=3, label='Curve of Growth',c='red')
ax3.scatter(predicted_log_Y_all_ols[not_relevant_runs_linear], log_ew[not_relevant_runs_linear], alpha=0.5, c='navy')
ax3.scatter(predicted_log_Y_all_ols[relevent_runs_linear], log_ew[relevent_runs_linear], alpha=0.5, c='#00AE15')
ax3.errorbar(predicted_log_Y_all_ols, log_ew, yerr=log_ew_error, alpha=0.3,fmt='none', zorder=0)
ax3.set_xlabel('Predicted $\mathrm{Log_{10}}$(Emission Measure)')
ax3.set_ylabel('$\mathrm{Log_{10}}$(Equivalent Width)')
#add the growth parameters to legend
ax3.text(0.95, 0.05, 
        f'CoG Coefficients:\n$K_1$: {popt_pred[0]:.1f}\n$K_2$: {popt_pred[1]:.1f}', 
        transform=ax3.transAxes, 
        fontsize=15, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        ha='right', 
        va='bottom')
ax3.set_xlim(min(predicted_log_Y_all_ols)-0.2, max(predicted_log_Y_all_ols)+0.2)
ax3.set_ylim(min(log_ew)-0.2, max(log_ew)+0.2)
ax3.legend()
ax3.grid(True, which='both', linestyle='--', alpha=0.5)
#ax3.set_title(f'Inclination: {incs[inclination]}°')
# Create a single, figure-level legend for the common labels.
# fig.legend([h2, h1],
#            ['In Selection Box Sirocco Spectra', 'Remaining Positive EW Sirocco Spectra'],
#            loc='upper center', ncol=2, fontsize=15, bbox_to_anchor=(0.43, 0.97))
# fig.suptitle(f'Inclination: {incs[inclination]}°', x=0.8, y=0.94, fontsize=15)
fig.legend([h2, h1],
           ['In Selection Box Sirocco Spectra', 'Remaining Positive EW Sirocco Spectra'],
           loc='upper center', ncol=2, fontsize=15, bbox_to_anchor=(0.375, 0.97))
fig.suptitle(f'Inclination: {incs[inclination]}°', x=0.787, y=0.938, fontsize=15)

plt.show()

# %% FIGURE 3 Histogram of removed spectra
################################################################################
print('FIGURE 3 Histogram of removed spectra')
################################################################################
# Load grid parameter table (not used for plotting but available if needed)
path_to_grids = "Release_Ha_grid_spec_files"
parameter_table = pd.read_csv(f'{path_to_grids}/Grid_runs_logfile.csv')

# Define categorical grid values for each parameter (6 parameters)
param1_name = ['Low', 'Medium', 'High']  # Disk.mdot
param2_name = ['Low', 'Medium', 'High']  # wind.mdot
param3_name = ['Low', 'Medium', 'High']  # KWD.d
param4_name = ['Low', 'Medium', 'High']  # KWD.mdot_r_exponent
param5_name = ['Low', 'Medium', 'High']  # KWD.acceleration_length
param6_name = ['Low', 'Medium', 'High']  # KWD.acceleration_exponent

# Build all unique grid combinations (each combination is a list of 6 categories)
temp_grid = np.array([param1_name, param2_name, param3_name, param4_name, param5_name, param6_name])
unique_combinations = []
for comb in itertools.product(*temp_grid):
    unique_combinations.append(list(comb))
all_combos = np.array(unique_combinations)  # shape (num_combos, 6)

# Define the inclinations.
# The file names are given for inclination columns (e.g., 10, 11, 12, 13, 14),
# and we map these to physical inclinations:
inclination_columns = [10, 11, 12, 13, 14]
inclination_map = {10: 20, 11: 45, 12: 60, 13: 72.5, 14: 85}

# For each inclination, load the "cut_runs" information from the final_results file.
# Then, for the grid the "Retained" (included) combinations are all_combos except those in cut_runs.
# We will record counts per grid parameter (6 parameters) for each category for each inclination.
num_params = 6
categories = ['Low', 'Medium', 'High']

# Initialize a dictionary: for each parameter (index 0..5), we will store data per inclination.
# counts[i][incl] = {'incl': [count_low, count_medium, count_high],
#                    'rem':  [count_low, count_medium, count_high],
#                    'rel':  [count_low, count_medium, count_high]}   <-- additional counts for qualifying (relevant) runs
counts = {i: {} for i in range(num_params)}

for incl in inclination_columns:
    # It is assumed that relevent_runs is defined globally as a boolean array of length 729 for each inclination.
    # Get the global relevant runs mask for this inclination.
    rel_mask = relevent_runs[incl]
    
    filepath = f'Emission_Line_Asymmetries/FWHM_1p0_5_mask_data/final_results_inc_col_{incl}.npy'
    if not os.path.exists(filepath):
        print(f"File not found for inclination {incl}. Skipping.")
        continue
    final_results = np.load(filepath, allow_pickle=True).item()
    cut_runs = final_results['cut_runs']
    
    # The removed combinations are indexed by cut_runs
    unique_removed = all_combos[cut_runs]
    # Included (retained) runs are all runs that are not in cut_runs.
    included_mask = np.ones(len(all_combos), dtype=bool)
    included_mask[cut_runs] = False
    unique_included = all_combos[included_mask]
    
    # Compute the subset among the retained runs that also qualify as 'relevant'
    unique_relevant = all_combos[included_mask & rel_mask]
    
    # For each parameter (index from 0 to 5), compute counts per category for included, removed, and relevant sets.
    for i in range(num_params):
        incl_counts = [np.sum(unique_included[:, i] == cat) for cat in categories]
        rem_counts  = [np.sum(unique_removed[:, i]  == cat) for cat in categories]
        rel_counts  = [np.sum(unique_relevant[:, i]  == cat) for cat in categories]
        counts[i][incl] = {'incl': incl_counts, 'rem': rem_counts, 'rel': rel_counts}

# Now create one figure with 6 subplots (one per parameter).
# In each subplot, we plot grouped bars (groups by category) showing the counts (stacked: "Retained" and "Removed")
# for each of the 5 inclinations and then overlay an additional set of bars (hatched) for the subset that qualifies as relevant.
num_incl = len(inclination_columns)
bar_width = 0.8 / num_incl

plt.rcParams.update({'font.size': 15})
fig, axes = plt.subplots(1, num_params, figsize=(3*num_params, 5), sharey=True)
param_names = ['Disk Mass Accretion Rate', 
               'Wind Mass-Loss Rate', 
               'Degree of Collimation ($d$)', 
               'Wind Mass-Loss Rate per\nUnit Area Exponent ($\\alpha$)', 
               'Acceleration Length', 
               'Acceleration Exponent']

# Pick colors for the inclinations – here we use a viridis colormap.
colors = plt.get_cmap('Blues')(np.linspace(1.0, 0.3, num_incl))
colors2 = plt.get_cmap('Greens')(np.linspace(1.0, 0.3, num_incl))
x = np.arange(len(categories))  # x positions for each category group

# For legend purposes, collect one handle per inclination (only once)
leg_handles = []
leg_labels = []

for param_idx in range(num_params):
    ax = axes[param_idx]
    # For each inclination, get counts for parameter param_idx.
    for j, incl in enumerate(inclination_columns):
        if incl not in counts[param_idx]:
            continue
        # Get the counts for the current inclination
        incl_counts = np.array(counts[param_idx][incl]['incl'])
        rem_counts  = np.array(counts[param_idx][incl]['rem'])
        total_counts = incl_counts + rem_counts
        # Calculate position offset so that bars for different inclinations are side by side.
        offset = (j - num_incl/2) * bar_width + bar_width/2
        # Plot the "retained" portion first.
        bars = ax.bar(x + offset, incl_counts, width=bar_width, color=colors[j], edgecolor='black',
                      label=f'{inclination_map[incl]}°' if param_idx == 0 else None)
        # Then plot the "removed" portion stacked on top (with light gray transparent color).
        ax.bar(x + offset, rem_counts, width=bar_width, bottom=incl_counts, color='dimgray', alpha=0.4)
        # Overlay an additional set of bars for runs that are relevant (qualify),
        # plotted with a hatched pattern and no fill.
        rel_counts = np.array(counts[param_idx][incl]['rel'])
        rel_bars = ax.bar(x + offset, rel_counts, width=bar_width, color=colors2[j],
               linewidth=1.5, edgecolor='black')
        if param_idx == 0:
            leg_handles.append(bars)
            leg_labels.append('')
            leg_handles.append(rel_bars)
            leg_labels.append('')
            
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_title(param_names[param_idx], fontsize=15)
    if param_idx == 0:
        ax.set_ylabel('Number of Spectra', fontsize=15)
    # Set y-axis limit based on the maximum total counts among all inclinations for this parameter.
    max_total = 0
    for incl in inclination_columns:
        if incl in counts[param_idx]:
            cat_total = np.array(counts[param_idx][incl]['incl']) + np.array(counts[param_idx][incl]['rem'])
            max_total = max(max_total, cat_total.max())
    ax.set_ylim(0, 250)

# Add a single legend for the inclinations
fig.legend(leg_handles, leg_labels, fontsize=16, loc='upper right', ncol=5, bbox_to_anchor=(0.48, 0.95))
fig.suptitle('Light Grey = Removed Spectra', fontsize=16, x=0.775, y=0.88, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'))
fig.text(0.226, 0.84, 
        '20° \qquad\quad 45° \qquad\quad  60° \qquad\quad 72.5° \qquad\ 85°', 
        #transform=ax3.transAxes, 
        fontsize=18, 
        ha='left', 
        va='bottom')
fig.text(0.18, 0.82, 
        'Blues = Retained Spectra \nGreens = Selected Spectra', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
        fontsize=16, 
        ha='right', 
        va='bottom')
plt.tight_layout(rect=[0, 0, 0.9, 0.93])
plt.show()
# # correct unique_combinations built-in code rounding issue
# for i, _ in enumerate(unique_combinations):
#     unique_combinations[i][0] = float(f'{unique_combinations[i][0]:.3g}')
# #generating low, medium, high lists corresponding to the combinations values
# lmh_array = np.array()
# for i, val in enumerate(unique_combinations):
#     lmh_list = []
#     for j, item in enumerate(val):
#         # index position in params_list
        
    


#for inclination_column in inclination_columns:
    

# %%

# popt, pcov, infodict, errmsg, ier= opt.curve_fit(curve_of_growth, predicted_log_Y_all_ols, log_ew, p0=[7.0, -55.5], maxfev=1000000, method='lm', full_output=True)
# print(f'Curve of growth fit parameters: {popt}')
# # Generate the curve of growth values using vectorized computation

# N_values = 10**np.linspace(52, 61, 500)
# W_values = curve_of_growth(np.log10(N_values), *popt)
# #W_values = curve_of_growth(N_values, 9e+55, 9e+57, 1.5e-55, 1,9e-29) #N, b1, b2, k1, k2, k3

# plt.figure(figsize=(7, 7))
# plt.plot(np.log10(N_values), W_values, color='red', linestyle='--', label='Curve of Growth')

# W_values = curve_of_growth(np.log10(N_values), 7.0, -55.5)
# plt.plot(np.log10(N_values), W_values, color='blue', linestyle='--', label='Curve of Growth HandPicked')
# plt.scatter(predicted_log_Y_all_ols, log_ew, alpha=0.5)
# plt.xlabel('Predicted $\mathrm{Log_{10}}$(Emission Measure)')
# plt.ylabel('Log10(Equivalent Width)')
# plt.title(f'Inclination: {incs[inclination]}°')
# #add the growth parameters to legend
# ax = plt.gca()
# ax.text(0.95, 0.05, 
#         f'CoG Coefficients:\n$K_1$: {popt[0]:.1f}\n$K_2$: {popt[1]:.1f}', 
#         transform=ax.transAxes, 
#         fontsize=15, 
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey'),
#         ha='right', 
#         va='bottom')
# plt.xlim(52.8, 60.2)
# plt.legend()
# plt.show()

# # residual plot
# residuals = log_ew - curve_of_growth(predicted_log_Y_all_ols, *popt)
# residuals_picked = log_ew - curve_of_growth(predicted_log_Y_all_ols, 7.0, -55.5)
# mse = np.mean(residuals**2)
# mse_picked = np.mean(residuals_picked**2)
# plt.figure(figsize=(7, 7))
# plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
# plt.scatter(predicted_log_Y_all_ols, residuals, alpha=0.5, label='curve_fit')
# plt.scatter(predicted_log_Y_all_ols, residuals_picked, alpha=0.5, label='hand-picked')
# plt.xlabel('Predicted $\mathrm{Log_{10}}$(Emission Measure)')
# plt.ylabel('Residuals')
# plt.title(f'MSE: = {mse}, Picked_MSE = {mse_picked}')
# plt.legend()
# plt.show()
# # residual plot
# residuals = log_ew - curve_of_growth(predicted_log_Y_all_ols, *popt)
# residuals_picked = log_ew - curve_of_growth(predicted_log_Y_all_ols, 7.0, -55.5)
# mse = np.mean(residuals**2)
# mse_picked = np.mean(residuals_picked**2)
# plt.figure(figsize=(20, 7))
# plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
# plt.scatter(np.arange(len(residuals)), residuals, alpha=0.5, label='curve_fit')
# plt.scatter(np.arange(len(residuals_picked)), residuals_picked, alpha=0.5, label='hand-picked')
# plt.xlabel('in order ')
# plt.ylabel('Residuals')
# plt.title(f'MSE: = {mse}, Picked_MSE = {mse_picked}')
# plt.legend()
# plt.show()
# %% LINE MOMENTS
################################################################################
print('TABLE: LINE MOMENTS')
################################################################################

%matplotlib inline
import time
import pandas as pd
from scipy.stats import skew, kurtosis, mode, describe, moment
import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import numpy as np
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

plt.style.use('science')

# Loading Sirocco Profiles
all_results = {}
inclination_columns = [11]  # 45° inclination
mask = '22_55_mask' # 11-88 = 500-4000, 22-88 = 1000-4000, 22-55 = 1000-2500, 22-90 = 1000-4100
for inclination_column in inclination_columns:
    if os.path.exists(f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy'):
        all_results[inclination_column] = np.load(f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()

#load existing grid parameters to add statistics.
path_to_grids = "Release_Ha_grid_spec_files"
parameter_table = pd.read_csv(f'{path_to_grids}/Grid_runs_logfile.csv')
table_df = pd.DataFrame(columns=['Intergated Flux', 'Mean', 'Median', 'Mode', 'Stdev', 'Skewness', 'Kurtosis', 'Flat Spectrum (1=True, 0=False)'])

H_alpha = 6562.819 # Å

def angstrom_to_kms(wavelength):
    """Converts wavelength in angstroms from central h_alpha line to velocity in km/s.
    Args:
        wavelength (float): wavelength in angstroms"""
    kms = (wavelength - H_alpha) * 299792.458 / H_alpha
    return kms

def zero_moment(flux, velocities):
    """Calculating the zeroth moment of the line.
    Args:
        flux (array): continuum-nomalised flux values, ie f_line/f_continuum
        velocity (array): Radial velocities from -4000 to 4000 km/s around Halpha"""
        
    zero_moment = np.trapz(flux-1, velocities)
    return zero_moment

def mean_raw_moment(flux, velocities):
    """Calculating the first raw moment of the emission line profile.

    Args:
        flux (array): continuum-nomalised flux values, ie f_line/f_continuum
        velocities (array): Radial velocities from -4000 to 4000 km/s around Halpha
    """
    #mu_1_sum = np.sum(velocities * (flux - 1)) / np.sum(flux - 1)
    if np.sum(flux - 1.0 < 0) > 50 and printing: # if more than 50 negative values
        print('Warning: This is going to break bad')
    mu_1 = np.trapz(velocities * (flux - 1.0), velocities) / zero_moment(flux, velocities)
    return mu_1

def variance_central_moment(flux, velocities):
    """Calculating the second central moment of the emission line profile.

    Args:
        flux (array): continuum-nomalised flux values, ie f_line/f_continuum
        velocities (array): Radial velocities from -4000 to 4000 km/s around Halpha
    """
    mu_1 = mean_raw_moment(flux, velocities)
    variance = np.trapz((velocities - mu_1)**2 * (flux - 1), velocities) / zero_moment(flux, velocities)
    return variance

def skewness_standardised_moment(flux, velocities):
    """Calculating the third standardised moment of the emission line profile.

    Args:
        flux (array): continuum-nomalised flux values, ie f_line/f_continuum
        velocities (array): Radial velocities from -4000 to 4000 km/s around Halpha
    """
    mu_1 = mean_raw_moment(flux, velocities)
    variance = variance_central_moment(flux, velocities)
    skewness = np.trapz((velocities - mu_1)**3 * (flux - 1), velocities) / (zero_moment(flux, velocities) * np.sqrt(variance)**3)
    return skewness

def kurtosis_standardised_moment(flux, velocities):
    """Calculating the fourth standardised moment of the emission line profile.

    Args:
        flux (array): continuum-nomalised flux values, ie f_line/f_continuum
        velocities (array): Radial velocities from -4000 to 4000 km/s around Halpha
    """
    mu_1 = mean_raw_moment(flux, velocities)
    variance = variance_central_moment(flux, velocities)
    kurtosis = np.trapz((velocities - mu_1)**4 * (flux - 1), velocities) / (zero_moment(flux, velocities) * variance**2)
    return kurtosis

save_sirocco_data = True
printing = False
sirocco_dictionary = {}

for run_num in tqdm(range(0, 729)): #605
    
    # Load velocity data
    wavelength_data = np.array(all_results[11]['wavelength_grid'][run_num])
    velocity_data = angstrom_to_kms(wavelength_data)
    
    #Load flux data
    flux_data = np.array(all_results[11]['grid'][run_num])
    continuum_data = np.array(all_results[11]['sk_con_data'][run_num])
    
    # Convert data to radial velocities and continuum normalised flux
    flux_norm = flux_data / continuum_data
    
    indexes = np.where((velocity_data > -4000) & (velocity_data < 4000))
    flux_norm = flux_norm[indexes]
    velocity_data = velocity_data[indexes]
    
    if save_sirocco_data:
        sirocco_dictionary[run_num] = {'flux': np.array(all_results[11]['grid'][run_num]), 'wavelengths': np.array(all_results[11]['wavelength_grid'][run_num]), 'continuum': np.array(all_results[11]['sk_con_data'][run_num])}
    
    m_0 = zero_moment(flux_norm, velocity_data)
    if printing: print(f'm_0 = {zero_moment(flux_norm, velocity_data)}')

    m_1 = mean_raw_moment(flux_norm, velocity_data)
    if printing: print(f'm_1 (mean) = {mean_raw_moment(flux_norm, velocity_data)} km/s')

    var = variance_central_moment(flux_norm, velocity_data)
    if printing: print(f'var = {var} (km/s)^2')
    
    std = np.sqrt(var)
    if printing: print(f'std = {std} km/s')

    skew = skewness_standardised_moment(flux_norm, velocity_data)
    if printing: print(f'skew = {skew}')

    kurt = kurtosis_standardised_moment(flux_norm, velocity_data)
    if printing: print(f'kurt = {kurt}')
    
    mode = velocity_data[np.argmax(flux_norm)]
    if printing: print(f'mode = {mode} km/s')
    
    #Median
    total_flux = np.trapz(flux_norm, velocity_data) # flux_norm is flux/continuum
    norm_spectrum = flux_norm / total_flux

    # Compute the CDF and ensure it ends at 1
    cdf = cumulative_trapezoid(norm_spectrum, velocity_data, initial=0)
    cdf = cdf / cdf[-1]  # Normalize CDF to end at 1, this is a check to ensure the CDF ends at 1

    # Generate random numbers from the uniform distribution [0, 1]
    ndata = 5_000_000
    uniform = np.random.uniform(0.0, 1.0, ndata)

    # Use inverse transform sampling to get random numbers from the PDF
    hist_data = np.interp(uniform, cdf, velocity_data)
    median = np.median(hist_data)

    if printing: print(f'median = {median} km/s')


    # finding the flat spectra with no emission lines
    flat = 0
    gradient = all_results[11]['sk_slopes'][run_num] # continuum gradient
    intercept = all_results[11]['sk_intercepts'][run_num] # continuum intercept
    # converting continuum function to flux values at the right wavelengths
    rebased_con_fluxes = gradient*wavelength_data + intercept
    flux_without_continuum = flux_data - rebased_con_fluxes # flux without continuum
    cut_flux = flux_without_continuum[100:-100] # cutting the edges of the flux as unphysical
    cut_wavelength = wavelength_data[100:-100] # cutting the edges of the wavelength as unphysical
    if np.max(cut_flux) < 4e-15: # flux limit to determine if flat
        flat = 1
    
    table_df.loc[run_num] = {
        'Intergated Flux': m_0,
        'Mean': m_1,
        'Median': median,
        'Mode': mode,
        'Stdev': std,
        'Skewness': skew,
        'Kurtosis_excess': kurt,
        'Flat Spectrum (1=True, 0=False)': flat
    }
    
parameter_table = parameter_table.merge(table_df, left_index=True, right_index=True)

#save the table
parameter_table.to_csv(f'Line_moments.csv', index=False)

#save the sirocco data
if save_sirocco_data:
    np.save('Emission_Line_Asymmetries/Sirocco_based_data/sirocco_data.npy', sirocco_dictionary)
#%% Cuneo spectra processing to friendly format

import trm.molly as molly

# list all file names wihin a directory
paths = os.listdir('Emission_Line_Asymmetries/AWDs_spec')

wavelength_range = (6435, 6685)

run_numbers = []
wavelength_grid = []
flux_grid = []
times = []
systems = []
stored_run_number = 0
total_count = 0

# plotting molly spectra
for path in paths:
    # Load the molly file
    if path.endswith('.mol'):
        molly_file_path = f'Emission_Line_Asymmetries/AWDs_spec/{path}'
    else:
        continue
    print(molly_file_path)
    molly_data = molly.rmolly(molly_file_path)

    # Plot the molly data
    plt.figure(figsize=(10, 6))
    for spectrum in molly_data:
        plt.plot(spectrum.wave, spectrum.f, label=f'{spectrum.head["UTC"], spectrum.head["Day"]}')
        total_count += 1
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'{spectrum.head["Object"]}')
    plt.legend()
    plt.show()
# %%
# storing data to numpy friendly format
w_min, w_max = wavelength_range
for path in paths:
    if not path.endswith('.mol'):
        continue
    molly_file_path = f'Emission_Line_Asymmetries/AWDs_spec/{path}'
    print(molly_file_path)
    molly_data = molly.rmolly(molly_file_path)
    
    for spectrum in molly_data:
        mask = (spectrum.wave >= w_min) & (spectrum.wave <= w_max)
        if path == 'V751_Cyg_Halfa_clip8000kms_norm.mol':
            mask = (spectrum.wave >= 6435) & (spectrum.wave <= 6684.3) # to avoid rugged arrays. 
        filtered_wavelength = spectrum.wave[mask]
        filtered_flux = spectrum.f[mask]
        
        run_numbers.append(stored_run_number + spectrum.head['Run'])
        wavelength_grid.append(filtered_wavelength)
        flux_grid.append(filtered_flux)
        times.append((spectrum.head['Day'], spectrum.head['UTC']))
        systems.append(spectrum.head['Object'])
        print(spectrum.head['Day'], spectrum.head['Month'], spectrum.head['Year'], spectrum.head['UTC'], spectrum.head['Run'])
    stored_run_number += spectrum.head['Run']

# convert to ndarrays
run_numbers = np.array(run_numbers)
wavelength_grid = np.array(wavelength_grid)
flux_grid = np.array(flux_grid)
times = np.array(times)
systems = np.array(systems)

# store in a npy file
#np.save('molly_spectra.npy', {'run_numbers': run_numbers, 'wavelength_grid': wavelength_grid, 'flux_grid': flux_grid, 'times': times, 'systems': systems})

# %% LINE LUMINOSITY CALCULATIONS
################################################################################
print('LINE LUMINOSITY CALCULATIONS')
################################################################################
inclination = 12
wave_grid = all_results[inclination]['wavelength_grid']
flux_grid = all_results[inclination]['grid']
continuum_grid = all_results[inclination]['sk_con_data']
run_number = np.arange(0, 729)


# Converting fluxes to luminosities
distance_sq = (100 * 3.086e18)**2 # (100 parsecs in cm) ^2

store_total_luminosity = []
# (erg/s/cm^2/Å --> ergs/s)
for run in run_number:
    line_flux = np.array(flux_grid[run])-np.array(continuum_grid[run])
    luminosity_spec = line_flux * np.array(wave_grid[run]) * 4 * np.pi * distance_sq
    total_luminosity = np.trapz(luminosity_spec, wave_grid[run])
    #print(f'Total luminosity for run {run}: {total_luminosity:.2e} erg/s')
    store_total_luminosity.append(total_luminosity)
    
# luminosity_spec = {run: fluxes_increasing[run][:,1] * wavelengths_increasing[run] * 4 * np.pi * distance_sq for run in run_number} 
# total_luminosity = {run: np.trapz(luminosity_spec[run], wavelengths_increasing[run]) for run in run_number}
# # convert to array
# total_luminosity_array = np.array([total_luminosity[run] for run in run_number])
# np.save('total_luminosity.npy', total_luminosity_array)





















# %matplotlib inline
# import numpy as np
# import time
# from scipy import stats
# from matplotlib import pyplot as plt
# import matplotlib
# import scipy.stats
# from scipy.integrate import cumulative_trapezoid
# import pandas as pd
# from scipy.stats import skew, kurtosis, mode, describe, moment

# # drawing random numbers from arbitrary pdfs just by using the cdf
# path_to_grids = "Release_Ha_grid_spec_files"
# parameter_table = pd.read_csv(f'{path_to_grids}/Grid_runs_logfile.csv')
# table_df = pd.DataFrame(columns=['Mean', 'Median', 'Mode', 'Stdev', 'Skewness', 'Kurtosis_excess', 'Flat Spectrum (1=True, 0=False)'])

# def angstrom_to_kms(wavelength):
#     """Converts wavelength in angstroms from central h_alpha line to velocity in km/s.
#     Args:
#         wavelength (float): wavelength in angstroms"""
#     kms = (wavelength - H_alpha) * 299792.458 / H_alpha
#     return kms

# # def moment_0(flux_con_norm, velocity):
# #     """Calculate the zeroth moment
# #     m_0 = ∫ (1-F(v)) dv"""
# #     m_0 = np.trapz(flux_con_norm - 1, velocity)
# #     return m_0

# # def moment_i(flux_con_norm, velocity, i):
# #     """Calculate the ith moment
# #     m_i = (∫ v^i(1-F(v)) dv)/m_0"""
# #     m_i = np.trapz(velocity**i * (flux_con_norm - 1), velocity) / moment_0(flux_con_norm, velocity)
# #     return m_i

# def raw_moment_1(flux_con_norm, velocity):
#     """Calculate the first raw moment (mean velocity)
#     m_1 = μ = ∫ v(F(v)-1) dv = E[V]"""
#     m_1 = np.trapz(velocity * (flux_con_norm - 1), velocity)
#     return m_1

# def raw_moment_2(flux_con_norm, velocity):
#     """Calculate the second raw moment (variance)
#     m_2 = σ^2 = ∫ v^2(F(v)-1) dv = E[V^2]"""
#     m_2 = np.trapz(velocity**2 * (flux_con_norm - 1), velocity)
#     return m_2

# def raw_moment_3(flux_con_norm, velocity):
#     """Calculate the third raw moment (skewness)
#     m_3 = ∫ v^3(F(v)-1) dv"""
#     m_3 = np.trapz(velocity**3 * (flux_con_norm - 1), velocity)
#     return m_3

# def raw_moment_4(flux_con_norm, velocity):
#     """Calculate the fourth raw moment (kurtosis)
#     m_4 = ∫ v^4(F(v)-1) dv"""
#     m_4 = np.trapz(velocity**4 * (flux_con_norm - 1), velocity)
#     return m_4

# def std(flux_con_norm, velocity):
#     """Calculate the standard deviation
#     σ = sqrt(∫ (v)^2(F(v)-1) dv)"""
#     #mean = raw_moment_1(flux_con_norm, velocity)
#     var = np.trapz((velocity)**2 * (flux_con_norm - 1), velocity)
#     sigma = np.sqrt(var)
#     return sigma


# H_alpha = 6562.819 # Å
# plotting = True
# printing = False

# for run_num in tqdm(range(605, 606)):
#     wavelength_data = np.array(all_results[11]['wavelength_grid'][run_num])
#     flux_data = np.array(all_results[11]['grid'][run_num])
#     continuum_data = np.array(all_results[11]['sk_con_data'][run_num])
    
#     # Convert data to radial velocities and continuum normalised flux
#     flux_norm = flux_data / continuum_data
#     velocity_data = angstrom_to_kms(wavelength_data)
    
#     #bound arrays to only be -4000 to 4000km/s
#     indexes = np.where((velocity_data > -4000) & (velocity_data < 4000))
#     flux_norm = flux_norm[indexes]
#     velocity_data = velocity_data[indexes]
    
#     print(f'raw_moment_1: {raw_moment_1(flux_norm, velocity_data)/std(flux_norm, velocity_data)}')
#     print(f'raw_moment_2: {raw_moment_2(flux_norm, velocity_data)}')
#     print(f'raw_moment_3: {raw_moment_3(flux_norm, velocity_data)}')
#     print(f'raw_moment_4: {raw_moment_4(flux_norm, velocity_data)}')
    
#     if plotting:
#         plt.figure(figsize=(7,7))
#         plt.subplot(3,1,1)
#         plt.plot(velocity_data, flux_norm, label='Original Data')
#         plt.vlines(0, 0, np.max(flux_norm), color='black', linestyle='--', alpha=0.5)
#         plt.xlabel('Radial Velocities ($km/s$)')
#         plt.ylabel('Flux/Continuum')
#         plt.ylim(1,np.max(flux_norm))
    
#     # Raw moments
#     # moment_0_value = moment_0(flux_norm, velocity_data)
#     # moment_1_value = moment_i(flux_norm, velocity_data, 1)
#     # moment_2_value = moment_i(flux_norm, velocity_data, 2)
#     # moment_3_value = moment_i(flux_norm, velocity_data, 3)
#     # moment_4_value = moment_i(flux_norm, velocity_data, 4)
#     mode = velocity_data[np.argmax(flux_norm)]
    
    
    
#     # Normalize the PDF so it integrates to 1
#     norm = np.trapz(flux_norm, velocity_data)
#     y = flux_norm / norm

#     # Compute the CDF and ensure it ends at 1
#     cdf = cumulative_trapezoid(y, velocity_data, initial=0)
#     cdf = cdf / cdf[-1]  # Normalize CDF to end at 1

#     # Generate random numbers from the uniform distribution [0, 1]
#     ndata = 10_000_000
#     uniform = np.random.uniform(0.0, 1.0, ndata)

#     # Use inverse transform sampling to get random numbers from the PDF
#     hist_data = np.interp(uniform, cdf, velocity_data)
    
#     if plotting:
#         plt.subplot(3,1,2)
#         plt.hist(hist_data, bins=200, color="blue", density=True, alpha=0.6, label='Sampled Data')
#         plt.plot(velocity_data, y, label='Original Data')
#         plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
#         plt.xlabel('Radial Velocities ($km/s$)')
#         plt.ylabel('Frequency')

#         plt.subplot(3,1,3)
#         plt.plot(velocity_data, cdf, label='CDF')
#         plt.xlabel('Radial Velocities ($km/s$)')
#         plt.ylabel('Cumulative Probability')
#         plt.tight_layout()
    
#     #mean_flux_dist= np.mean(hist_data)
#     median = np.median(hist_data)
    
#     if printing:
#         print('Run:', run_num)
#         #print(f'Mean: {mean_flux_dist}')
#         print(f'Median: {median}')
#         print(f'Mode: {mode}')
#         # print(f'Moment 0: {moment_0_value}')
#         # print(f'Moment 1: {moment_1_value}')
#         # print(f'Moment 2: {moment_2_value}')
#         # print(f'Moment 3: {moment_3_value}')
#         # print(f'Moment 4: {moment_4_value}')
        
    
#     # Standardised moments (equations)
#     integrand = flux_norm - 1.0
#     norm_factor = np.trapz(integrand, velocity_data)
#     p = integrand / norm_factor
    
#     mean_continuum_diff = np.trapz(velocity_data * p, velocity_data)
#     var = np.trapz((velocity_data - mean_continuum_diff)**2 * p, velocity_data)
#     sigma = np.sqrt(var)
    
#     m3 = np.trapz((velocity_data )**3 * p, velocity_data)
#     m4 = np.trapz((velocity_data )**4 * p, velocity_data)
#     skewness = m3 / (sigma**3)
#     kurtosis_excess = (m4 / (sigma**4))
    
#     if printing:
#         print(f'Mean: {mean_continuum_diff}')
#         print(f'Variance: {var}')
#         print(f'skewness: {skewness}')
#         print(f'kurtosis: {kurtosis_excess}')
    
#     if plotting:
#         plt.subplot(3,1,1)
#         plt.axvline(x=mean_continuum_diff, color='red', linestyle='--', alpha=0.5)
#         plt.axvline(x=mode, color='green', linestyle='--', alpha=0.5)
#         plt.axvline(x=median, color='blue', linestyle='--', alpha=0.5)
#         plt.title(f'Run {run_num}')
#         plt.legend()
#         plt.show()
        
#     # finding the flat spectra with no emission lines
#     flat = 0
#     flux = all_results[11]['grid'][run_num]
#     wavelength = np.array(all_results[11]['wavelength_grid'][run_num])
#     gradient = all_results[11]['sk_slopes'][run_num] # continuum gradient
#     intercept = all_results[11]['sk_intercepts'][run_num] # continuum intercept
#     # converting continuum function to flux values at the right wavelengths
#     rebased_con_fluxes = gradient*wavelength + intercept
#     flux_without_continuum = flux - rebased_con_fluxes # flux without continuum
#     cut_flux = flux_without_continuum[100:-100] # cutting the edges of the flux as unphysical
#     cut_wavelength = wavelength[100:-100] # cutting the edges of the wavelength as unphysical
#     if np.max(cut_flux) < 4e-15: # flux limit to determine if flat
#         flat = 1
    
#     table_df.loc[run_num] = {
#         'Mean': mean_continuum_diff,
#         'Median': median,
#         'Mode': mode,
#         'Stdev': sigma,
#         'Skewness': skewness,
#         'Kurtosis_excess': kurtosis_excess, 
#         'Flat Spectrum (1=True, 0=False)': flat
#     }
    
# parameter_table = parameter_table.merge(table_df, left_index=True, right_index=True)

# #save the table
# parameter_table.to_csv(f'Line_moments.csv', index=False)

    










# %% Figure 1 - LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS
# ################################################################################
# print('FIGURE 1: LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS')
# ################################################################################
# # Define inclinations for low, medium, and high
# inclination_columns = [11, 12, 13]  # 45°, 60°, 72.5°
# incs = [0 for _ in range(10)]  # to align indices
# incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# # Set up subplots
# fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
# plt.rcParams.update({'font.size': 15})

# # Load Teo's data (only need to load once)
# bz_cam = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
# mv_lyr = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv', delimiter=',')
# v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# # Initialize lists to collect handles and labels for legend
# handles = []
# labels = []

# for idx, inclination_column in enumerate(inclination_columns):
#     final_results = all_results[inclination_column]
#     cut_runs = final_results['cut_runs']
#     peak_colour_map = final_results['peak_colour_map']
#     grid_length = np.arange(0, 729)

#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
#     cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#     cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
#     cut_grid_length = np.delete(grid_length, cut_runs)

#     # Create masks for single-peaked and double-peaked spectra
#     cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array

#     ax = axs[idx]
#     ranges = slice(0, -1)  # to select particular portions of the grid if desired

#     # Plot error bars (no label)
#     ax.errorbar(
#         cut_red_ew_excess,
#         cut_blue_ew_excess, 
#         xerr=cut_red_ew_excess_error, 
#         yerr=cut_blue_ew_excess_error, 
#         fmt='none', 
#         ecolor='black', 
#         alpha=0.5,
#         zorder=-1
#     )

#     # Plot grid data, get handle (only need to get handle once)
#     if idx == 0:
#         grid_scatter = ax.scatter(
#             cut_red_ew_excess,
#             cut_blue_ew_excess,
#             c='black',
#             s=10,
#             label='Grid Data',
#             zorder=1
#         )
#         handles.append(grid_scatter)
#         labels.append('Grid Data')
#     else:
#         ax.scatter(
#             cut_red_ew_excess,
#             cut_blue_ew_excess,
#             c='black',
#             s=10,
#             zorder=1
#         )

#     kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
#     kde.fit(np.vstack([cut_red_ew_excess, cut_blue_ew_excess]).T)

#     xcenters = np.linspace(min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2, 500)
#     ycenters = np.linspace(min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2, 500)
#     X, Y = np.meshgrid(xcenters, ycenters)
#     xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
#     Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

#     contour_list = np.exp(np.arange(-3.0, 2.5, 0.5)) # circular shapes
#     #contour_list = np.exp(np.arange(-4.0, 3.5, 0.5)) # square shapes
#     #ax.contourf(X, Y, Z, levels=contour_list, cmap='Greys', alpha=0.6, zorder=0)
#     #ax.contour(X, Y, Z, levels=contour_list, colors='black', zorder=0)

#     # Only plot Teo's data on the medium inclination (45 degrees, inclination_column==11)
#     if inclination_column == 11:
#         # Plot Teo's data and collect handle for legend
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1], color='red', s=10, marker='o', label='Cúneo et al. (2023)')
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
#         handles.append(teo_scatter)
#         labels.append('Cúneo et al. (2023)')

#     # Vertical and horizontal lines at 0 (axes)
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     #plot a square dashed box around the linear threshold boundary
#     linear_thrs = 0.1
#     ax.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)
    
#     # Plot formatting
#     ax.set_xlabel('Red Wing EW Excess ($\mathring{A}$)')
#     if idx == 0:
#         ax.set_ylabel('Blue Wing EW Excess ($\mathring{A}$)')
#     ax.set_title(f'{incs[inclination_column]}° inclination')

#     ax.set_xlim(-30, 30)
#     ax.set_ylim(-30, 30)
#     ax.set_xscale('symlog', linthresh=linear_thrs) #change this for circular square shapes
#     ax.set_yscale('symlog', linthresh=linear_thrs)

# # Get the current axis
#     ax = plt.gca()
#     # ax.set_axisbelow(True)
#     # ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
#     # Set minor tick locators for symlog scale
#     ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
#     ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
    
# for ax in axs:
#     #Add grid lines for both major and minor ticks
#     ax.set_axisbelow(True)
#     ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=-1)

# labels.append('Linear/Logarithmic Threshold')
# handles.append(Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0))
    
# # Adjust spacing between subplots to remove the gap
# fig.subplots_adjust(wspace=0)

# # Add global legend
# fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.525, 1.05))

# # Adjust layout to make room for the legend
# #plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()





# # %% Figure 7 LARGE- FWHM DIAGNOSTIC PLOTS
# ################################################################################
# print('FIGURE 7: LARGE- FWHM DIAGNOSTIC PLOTS')
# ################################################################################

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.scale import SymmetricalLogLocator

# # Define the 5 inclination columns (these indices correspond to the model inclinations)
# inclination_columns = [10, 11, 12, 13, 14]  # e.g. 20°, 45°, 60°, 72.5°, 85°
# # Build the "incs" list so that incs[inclination_column] gives the physical value.
# incs = [0 for _ in range(10)]
# incs.extend([20, 45, 60, 72.5, 85])  # so, incs[10]=20, incs[11]=45, etc.

# # Create the figure and a 2×3 grid of subplots
# fig, axs = plt.subplots(2, 3, figsize=(16, 11), sharex=False, sharey=True)
# axs = axs.flatten()
# fig.delaxes(axs[5])  # remove the unused subplot
# plt.rcParams.update({'font.size': 15})

# # (Optional) Load Teo's data if needed for diagnostics (here not plotted):
# cuneo_data = np.load('Emission_Line_Asymmetries/Cuneo_FWHM/final_results_replicating_Cuneo_2.npy', allow_pickle=True).item()
# cuneo_red_ew_excess = cuneo_data['red_ew_excess']
# cuneo_blue_ew_excess = cuneo_data['blue_ew_excess']
# cuneo_red_ew_excess_error = cuneo_data['red_ew_excess_error']
# cuneo_blue_ew_excess_error = cuneo_data['blue_ew_excess_error']

# # Prepare lists for legend handles and labels
# handles = []
# labels = []

# for idx, inclination_column in enumerate(inclination_columns):
#     # Load the FWHM diagnostic results for this inclination from the FWHM folder
#     filepath = f'Emission_Line_Asymmetries/new_FWHM_data/final_results_inc_col_{inclination_column}.npy'
#     if not os.path.exists(filepath):
#         continue
#     final_results = np.load(filepath, allow_pickle=True).item()
    
#     cut_runs = final_results['cut_runs']
#     peak_colour_map = final_results['peak_colour_map']
#     grid_length = np.arange(0, 729)
    
#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
#     cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
    
#     # Create masks for single-peaked and double-peaked spectra
#     cut_peak_colour_map = np.array(cut_peak_colour_map)
#     single_mask = (cut_peak_colour_map == 'black')
#     double_mask = (cut_peak_colour_map == 'red')
    
#     ax = axs[idx]
    
#     # Plot error bars for all data points
#     ax.errorbar(cut_red_ew_excess,
#                 cut_blue_ew_excess,
#                 xerr=cut_red_ew_excess_error,
#                 yerr=cut_blue_ew_excess_error,
#                 fmt='none',
#                 ecolor='black',
#                 alpha=0.5,
#                 zorder=-1)
    
#     # Plot single-peaked and double-peaked Sirocco points. Add legend entries only once.
#     if idx == 0:
#         single_scatter = ax.scatter(cut_red_ew_excess[single_mask],
#                                     cut_blue_ew_excess[single_mask],
#                                     c='red',
#                                     s=10,
#                                     label='Single-peaked Sirocco Spectra',
#                                     zorder=1,
#                                     alpha=1)
#         double_scatter = ax.scatter(cut_red_ew_excess[double_mask],
#                                     cut_blue_ew_excess[double_mask],
#                                     c='black',
#                                     s=10,
#                                     label='Double-peaked Sirocco Spectra',
#                                     zorder=0,
#                                     alpha=0.5)
#         handles.extend([single_scatter, double_scatter])
#         labels.extend(['Single-peaked Sirocco Spectra', 'Double-peaked Sirocco Spectra'])
#     else:
#         ax.scatter(cut_red_ew_excess[single_mask],
#                    cut_blue_ew_excess[single_mask],
#                    c='red',
#                    s=10,
#                    zorder=1,
#                    alpha=1)
#         ax.scatter(cut_red_ew_excess[double_mask],
#                    cut_blue_ew_excess[double_mask],
#                    c='black',
#                    s=10,
#                    zorder=0,
#                    alpha=0.5)
    
#     base_colour = 'cyan'
#     if inclination_column in [10,11]:
#         if idx == 0:
#             cuneo_scatter = ax.scatter(cuneo_red_ew_excess,
#                                        cuneo_blue_ew_excess,
#                                        color=base_colour,
#                                        s=45,
#                                        marker='o',
#                                        edgecolor='navy',
#                                        alpha=0.7,
#                                        label='Cúneo et al. (2023)')
#             handles.append(cuneo_scatter)
#             labels.append('Cúneo et al. (2023)')
#         else:
#             ax.scatter(cuneo_red_ew_excess,
#                     cuneo_blue_ew_excess,
#                     color=base_colour, 
#                     s=45, 
#                     marker='o',
#                     edgecolor='navy', 
#                     alpha=0.7)
#             ax.errorbar(cuneo_red_ew_excess,
#                         cuneo_blue_ew_excess,
#                         xerr=cuneo_red_ew_excess_error,
#                         yerr=cuneo_blue_ew_excess_error,
#                         fmt='none',
#                         ecolor='navy',
#                         alpha=0.5,
#                         zorder=-1)
#     # Draw reference axes (vertical/horizontal lines)
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    
#     # Draw the dashed box and threshold lines
#     linear_thrs = 0.1
#     ax.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#             [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#             color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0,
#             label='Linear/Logarithmic Threshold')
#     ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
#     ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
#     ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
#     ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
    
#     # Axes formatting
#     ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
#     if idx in [0, 3]:
#         ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')
#     ax.set_title(f'{incs[inclination_column]}° inclination')
#     ax.set_xlim(-30, 30)
#     ax.set_ylim(-30, 30)
#     ax.set_xscale('symlog', linthresh=linear_thrs)
#     ax.set_yscale('symlog', linthresh=linear_thrs)
#     ax.set_axisbelow(True)
#     ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=-1)
#     ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
#     ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))

# # Add final threshold legend entry
# labels.append('Linear/Logarithmic Threshold')
# handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

# fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)
# fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.96))

# # Manually reposition the bottom row subplots (if needed)
# pos3 = axs[3].get_position()
# pos4 = axs[4].get_position()
# axs[3].set_position([0.252, pos3.y0 - 0.04, 0.26, pos3.height])
# axs[4].set_position([0.513, pos4.y0 - 0.04, 0.26, pos4.height])
# plt.show()








































#%%
# # %% LINE MOMENTS
# ################################################################################
# print('TABLE: LINE MOMENTS')
# ################################################################################

# %matplotlib inline
# import numpy as np
# import time
# from scipy import stats
# from matplotlib import pyplot as plt
# import matplotlib
# import scipy.stats
# from scipy.integrate import cumulative_trapezoid
# import pandas as pd
# from scipy.stats import skew, kurtosis, mode, describe, moment

# # drawing random numbers from arbitrary pdfs just by using the cdf
# path_to_grids = "Release_Ha_grid_spec_files"
# parameter_table = pd.read_csv(f'{path_to_grids}/Grid_runs_logfile.csv')
# table_df = pd.DataFrame(columns=['Mean', 'Median', 'Mode', 'Stdev', 'Skewness', 'Kurtosis', 'Flat Spectrum (1=True, 0=False)'])

# def angstrom_to_kms(wavelength):
#     """Converts wavelength in angstroms from central h_alpha line to velocity in km/s.
#     Args:
#         wavelength (float): wavelength in angstroms"""
#     kms = (wavelength - H_alpha) * 299792.458 / H_alpha
#     return kms

# H_alpha = 6562.819 # Å

# for run_num in range(701,702):
#     x = all_results[11]['wavelength_grid'][run_num]
#     y = all_results[11]['grid'][run_num]
#     continuum = all_results[11]['sk_con_data'][run_num]
#     #y = np.array(all_results[11]['grid'][run_num]) / np.array(all_results[11]['sk_con_data'][run_num]) # subtracting the continuum

#     # Convert x and y to NumPy arrays
#     x = np.array([angstrom_to_kms(val) for val in x])
#     y = np.array(y)
#     continuum = np.array(continuum)
    
#     #slice arrays to only be -4000 to 4000km/s
#     indexes = np.where((x > -4000) & (x < 4000))
#     x = x[indexes]
#     y = y[indexes]

#     # Ensure x is in increasing order
#     if np.any(np.diff(x) < 0):
#         x = x[::-1]
#         y = y[::-1]
#         print('Reversed arrays')

#     # Ensure y (PDF) is non-negative
#     #y[y < 0] = 0
#     fluxes = y.copy()
#     # Normalize the PDF so it integrates to 1
#     norm = np.trapz(y, x)
#     y = y / norm

#     # Compute the CDF and ensure it ends at 1
#     cdf = cumulative_trapezoid(y, x, initial=0)
#     cdf = cdf / cdf[-1]  # Normalize CDF to end at 1

#     # Generate random numbers from the uniform distribution [0, 1]
#     ndata = 1_000_000
#     uniform = np.random.uniform(0.0, 1.0, ndata)

#     # Use inverse transform sampling to get random numbers from the PDF
#     wave_data = np.interp(uniform, cdf, x)

#     if True: #and run_num%70 ==0: # to see a plot for diagnostic purposes if required
#         # Plot the PDF and the histogram of the sampled data
#         plt.figure(figsize=(8, 8))
#         plt.subplot(3, 1, 1)
#         plt.plot(x, y, linestyle='-', color='red', label='PDF')
#         plt.vlines(0, 0, np.max(y), color='black', linestyle='--', alpha=0.5)
#         plt.hist(wave_data, bins=100, color="blue", density=True, alpha=0.6, label='Sampled Data')
#         plt.ylabel('Density')
#         plt.legend()

#         plt.subplot(3, 1, 2)
#         plt.plot(x, cdf, linestyle='-', color='red', label='CDF')
#         plt.vlines(0, 0, 1, color='black', linestyle='--', alpha=0.5)
#         plt.xlabel('x')
#         plt.ylabel('Cumulative Probability')
#         plt.legend()

#         plt.subplot(3, 1, 3)
#         plt.plot(x, fluxes/np.array(all_results[11]['sk_con_data'][run_num])[indexes], label='Original Data')
#         plt.vlines(0, 0, np.max(y), color='black', linestyle='--', alpha=0.5)
#         plt.xlabel('Radial Velocities ($km/s$)')
#         plt.ylabel('Flux/Continuum')
#         plt.tight_layout()
#         plt.show()
        
#     # 45 degree inclination
#     # We are going to find the mean, median, mode, standard deviation, skewness and 
#     # kurtosis(with the mean/mode/median/stdev in velocity units).
#     # We are going to add the statistics to a table
#     # we are using flux normialised by the continuum so equations slightly different
    
    
    
#     # mean = np.mean(wave_data)
#     # #moment_1 = moment(wave_data, moment=1)
#     # median = np.median(wave_data)
#     # #mode_value = mode(wave_data).mode
#     # # the wavelength index which correcspoinds to the peak of the pdf
#     # mode_peak = x[np.argmax(y)]
#     # stdev = np.std(wave_data)
#     # moment_2 = moment(wave_data, moment=2)
#     # skewness_value = skew(wave_data)
#     # moment_3 = moment(wave_data, moment=3)
#     # kurtosis_value = kurtosis(wave_data)
#     # moment_4 = moment(wave_data, moment=4)
#     # print(mean, 'mean')
#     # print(median, 'median')
#     # print(mode_peak, 'mode ')
#     # print(stdev, moment_2**0.5, 'stdev')
#     # print(skewness_value, moment_3/(moment_2**1.5), 'skewness')
#     # print(kurtosis_value, moment_4/(moment_2**2)-3, 'kurtosis')
    
#     # finding the flat spectra with no emission lines
#     flat = 0
#     flux = all_results[11]['grid'][run_num]
#     wavelength = np.array(all_results[11]['wavelength_grid'][run_num])
#     gradient = all_results[11]['sk_slopes'][run_num] # continuum gradient
#     intercept = all_results[11]['sk_intercepts'][run_num] # continuum intercept
#     # converting continuum function to flux values at the right wavelengths
#     rebased_con_fluxes = gradient*wavelength + intercept
#     flux_without_continuum = flux - rebased_con_fluxes # flux without continuum
#     cut_flux = flux_without_continuum[100:-100] # cutting the edges of the flux as unphysical
#     cut_wavelength = wavelength[100:-100] # cutting the edges of the wavelength as unphysical
#     if np.max(cut_flux) < 4e-15: # flux limit to determine if flat
#         flat = 1
    
#     table_df.loc[run_num] = {
#         'Mean': mean,
#         'Median': median,
#         'Mode': mode_peak,
#         'Stdev': stdev,
#         'Skewness': skewness_value,
#         'Kurtosis': kurtosis_value, 
#         'Flat Spectrum (1=True, 0=False)': flat
#     }
    
# parameter_table = parameter_table.merge(table_df, left_index=True, right_index=True)

# #save the table
# parameter_table.to_csv(f'Line_moments.csv', index=False)
#     #print(f'Run {run_num} - Mean: {mean}, Median: {median}, Mode: {mode_value}, Stdev: {stdev}, Skewness: {skewness_value}, Kurtosis: {kurtosis_value}')
    
    
    

#     # for run in run_number:
#     #     indexes = np.where((wavelengths[run] > 6260) & (wavelengths[run] < 6860))
#     #     flux = fluxes[run][indexes[0][0]:indexes[0][-1], 1]
#     #     #converting wavelengths to velocities 
#     #     velocities[run] = [angstrom_to_kms(w) for w in wavelengths[run]]
#     #     mean = describe(flux).mean
#     #     median = np.median(flux)
#     #     mode_value = mode(flux).mode
#     #     stdev = describe(flux).variance**0.5
#     #     skewness_value = skew(flux)
#     #     kurtosis_value = kurtosis(flux)

#     #     table_df.loc[run] = {
#     #         'Run': run,
#     #         'Mean': mean,
#     #         'Median': median,
#     #         'Mode': mode_value,
#     #         'Stdev': stdev,
#     #         'Skewness': skewness_value,
#     #         'Kurtosis': kurtosis_value
#     #     }

#     # #plot an individual run
#     # run = 701
#     # fig, ax = plt.subplots(figsize=(12, 8))
#     # indexes = np.where((wavelengths[run] > 6260) & (wavelengths[run] < 6860))
#     # plt.plot(velocities[run][indexes[0][0]:indexes[0][-1]], fluxes[run][indexes[0][0]:indexes[0][-1], 1])
#     # #plot a virtual line at the central h_alpha line
#     # plt.axvline(x=0, color='red', linestyle='--')
#     # plt.xlabel('Radial Velocities ($m/s$)')
#     # plt.ylabel('Flux ($erg/s/cm^2/Å$)')
#     # plt.title('H alpha of CV for Run ' + str(run))
#     # plt.show()
    
    

# def curve_of_growth(N, b1, b2, k1, k2, k3):
#     """
    
#       1) W ~ k1 * N,                for N < b1
#       2) W ~ k2 * sqrt( ln(N) ),    for b1 <= N < b2
#       3) W ~ k3 * sqrt(N),          for N >= b2

#     Parameters
#     ----------
#     N   : array-like
#         1D array of emission measures (N).
#     b1  : float
#         Boundary between the low-N and middle-N regime.
#     b2  : float
#         Boundary between the middle-N and high-N regime.
#     k1, k2, k3 : float
#         Proportionality constants for each segment.

#     Returns
#     -------
#     W : ndarray
#         1D array of the same shape as N, containing the curve-of-growth values W.
#     """
#     # Convert input to numpy array
#     Ns = N
#     Ws = []

#     for n in Ns:
#         if n < b1:
#             W = k1 * n
#             Ws.append(W)
#         elif b1 <= n < b2:
#             W = k2 * np.sqrt(np.log(n))
#             Ws.append(W)
#         else:
#             W = k3 * np.sqrt(n)
#             Ws.append(W)
#     #smoothen
#     Ws = gaussian_filter1d(Ws, sigma=15)
#     return np.array(Ws)



# # %% Figure 10 HIGHLIGHTING EWvsfwhm- LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS
# ################################################################################
# print('FIGURE 10 HIGHLIGHTING EWvsfwhm: LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS')
# ################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.scale import SymmetricalLogLocator

# # Example: define your inclination columns and incs
# inclination_columns = [10, 11]#, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
# incs = [0 for _ in range(10)]  # to align indices
# incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# # Create the figure and a 2×3 grid of subplots
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True)
# axs = axs.flatten()

# # Remove the unused 6th subplot
# #fig.delaxes(axs[5])
# plt.rcParams.update({'font.size': 15})

# # Load Teo's data once
# bz_cam   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv',   delimiter=',')
# mv_lyr   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv',   delimiter=',')
# v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# # Prepare legend handles
# handles = []
# labels = []

# for idx, inclination_column in enumerate(inclination_columns):
#     final_results = all_results[inclination_column]
#     cut_runs = final_results['cut_runs']
#     peak_colour_map = final_results['peak_colour_map']
#     grid_length = np.arange(0, 729)
#     ew = np.array(ew_results[inclination_column]['ew'])
#     fwhm = np.array(fwhm_results[inclination_column]['fwhm'])
#     # create a mask for data with EW between 1 and 100 and FWHM < 30
#     mask = (ew > 1) & (ew < 100) & (fwhm < 30)
    

#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
#     cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#     mask_cut = np.delete(mask, cut_runs)
#     # Create masks for single-peaked and double-peaked spectra
#     cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
#     # Boolean masks for single- vs double-peaked
#     single_mask = (cut_peak_colour_map == 'black')
#     double_mask = (cut_peak_colour_map == 'red')
#     mask_cut_plus_single = mask_cut & single_mask
#     mask_cut_plus_double = mask_cut & double_mask
#     ax = axs[idx]

#     # Plot error bars (all points together for simplicity)
#     ax.errorbar(
#         cut_red_ew_excess[mask_cut],
#         cut_blue_ew_excess[mask_cut],
#         xerr=cut_red_ew_excess_error[mask_cut],
#         yerr=cut_blue_ew_excess_error[mask_cut],
#         fmt='none',
#         ecolor='black',
#         alpha=0.5,
#         zorder=-1
#     )

#     # Plot single-peaked (black) vs double-peaked (red)
#     # Add legend entries only in the first subplot
#     if idx == 0:
#         # single_scatter = ax.scatter(
#         #     cut_red_ew_excess[single_mask],
#         #     cut_blue_ew_excess[single_mask],
#         #     c='red',
#         #     s=10,
#         #     label='Single-peaked Sirocco Spectra',
#         #     zorder=1,
#         #     alpha=1
#         # )
#         # double_scatter = ax.scatter(
#         #     cut_red_ew_excess[double_mask],
#         #     cut_blue_ew_excess[double_mask],
#         #     c='black',
#         #     s=10,
#         #     label='Double-peaked Sirocco Spectra',
#         #     zorder=0,
#         #     alpha=0.5
#         # )
#         double_near_scatter = ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_double],
#             cut_blue_ew_excess[mask_cut_plus_double],
#             c='dimgray',
#             s=10,
#             label='Potential Sirocco Spectra',
#             zorder=1,
#             alpha=0.4
#         )
#         single_near_scatter = ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_single],
#             cut_blue_ew_excess[mask_cut_plus_single],
#             c='green',
#             s=10,
#             label='Potential Sirocco Spectra',
#             zorder=1,
#             alpha=1
#         )
#         # handles.append(single_scatter)
#         # labels.append('Single-peaked Sirocco Spectra')
#         # handles.append(double_scatter)
#         # labels.append('Double-peaked Sirocco Spectra')
#         handles.append(single_near_scatter)
#         labels.append('Similar EW-FWHM Single-Peaked Sirocco Spectra')
#         handles.append(double_near_scatter)
#         labels.append('Similar EW-FWHM Double-Peaked Sirocco Spectra')
#     else:
#         # ax.scatter(
#         #     cut_red_ew_excess[single_mask],
#         #     cut_blue_ew_excess[single_mask],
#         #     c='red',
#         #     s=10,
#         #     zorder=1,
#         #     alpha=1
#         # )
#         # ax.scatter(
#         #     cut_red_ew_excess[double_mask],
#         #     cut_blue_ew_excess[double_mask],
#         #     c='black',
#         #     s=10,
#         #     zorder=0,
#         #     alpha=0.5
#         # )
#         ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_double],
#             cut_blue_ew_excess[mask_cut_plus_double],
#             c='dimgray',
#             s=10,
#             zorder=1,
#             alpha=0.4
#         )
#         ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_single],
#             cut_blue_ew_excess[mask_cut_plus_single],
#             c='green',
#             s=10,
#             zorder=1,
#             alpha=1
#         )
#     base_colour = 'cyan' 
#     if idx == 0:
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color=base_colour, s=45, marker='o',edgecolor='navy',
#                                     alpha=0.7,
#                                     label='Cúneo et al. (2023)')
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         handles.append(teo_scatter)
#         labels.append('Cúneo et al. (2023)')
#     elif idx == 1: 
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color=base_colour, s=45, marker='o',
#                                     alpha=0.7, edgecolor='navy')                         
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy', )
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy')
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy')
#     # Only plot Teo’s data on the 45° inclination
#     # if inclination_column == 11:
#     #     teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1], color='red', s=10, marker='o', label='Cúneo et al. (2023)')
#     #     ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
#     #     ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
#     #     ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
#     #     handles.append(teo_scatter)
#     #     labels.append('Cúneo et al. (2023)')

#     # Reference lines
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

#     # Dashed box for linear/log threshold
#     linear_thrs = 0.1
#     ax.plot(
#         [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#         [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#         color='black', linestyle='--', alpha=1.0, zorder=0, linewidth=2.0,
#         label='Linear/Logarithmic Threshold'
#     )
#     #linear threshold lines
#     ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
#     ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
#     ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
#     ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
    
#     # ax.axvline(linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     # ax.axhline(-linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     # ax.axhline(linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
    

#     # Axes formatting
#     ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
#     # Put a y-label on the left column of subplots (idx=0 for top-left, idx=3 for bottom-left)
#     if idx in [0, 3]:
#         ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')

#     ax.set_title(f'{incs[inclination_column]}° inclination')
#     ax.set_xlim(-30, 30)
#     ax.set_ylim(-30, 30)
#     ax.set_xscale('symlog', linthresh=linear_thrs)
#     ax.set_yscale('symlog', linthresh=linear_thrs)
#     ax.set_axisbelow(True)
#     ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=-1)

#     # Minor tick locators for symlog
#     ax.xaxis.set_minor_locator(
#         SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
#     )
#     ax.yaxis.set_minor_locator(
#         SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
#     )

# # Add final threshold handle/label
# labels.append('Linear/Logarithmic Threshold')
# handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

# # Adjust subplot spacing (both horizontally and vertically)
# fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)

# # Position the legend closer to the top
# fig.legend(
#     handles, labels,
#     loc='upper center', ncol=2,
#     bbox_to_anchor=(0.5, 1.05)
# )

# # Manually reposition the bottom row subplots so they are centered
# #pos3 = axs[3].get_position()
# #pos4 = axs[4].get_position()
# #axs[3].set_position([0.252, pos3.y0-0.04, 0.26, pos3.height])  # shift bottom-left a bit to the right
# #axs[4].set_position([0.513, pos4.y0-0.04, 0.26, pos4.height])  # shift bottom-right a bit left

# plt.show()

# # %% Figure 5 -DOUBLE MASK FITTING METHODOLOGY
# ################################################################################
# print('FIGURE 5: DOUBLE MASK FITTING METHODOLOGY')
# ################################################################################

# H_alpha = 6562.819
# blue_peak_mask = (22, 88)  # number of angstroms to cut around the peak, blue minus.
# red_peak_mask = (22, 88)  # number of angstroms to cut around the peak, red plus.

# blue_peak_mask_2 = (11, 55)
# red_peak_mask_2 = (11, 55) # 5 45

# final_results = all_results[11]  # 45° inclination
# run = 710

# fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
# for i in [0, 1]:
#     ax[i].plot(final_results['wavelength_grid'][run],
#                final_results['grid'][run],
#                label='Original Data',
#                color='black'
#                )
#     ax[i].plot(final_results['wavelength_grid'][run],
#                final_results['fitted_grid'][run],
#                label='Optimal Gaussian',
#                color='red'
#                )
#     ax[i].plot(final_results['wavelength_grid'][run],
#                final_results['fit_con'][run],
#                label='Fitted Continuum',
#                color='blue'
#                )
#     ax[i].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5, label=r'$H\alpha$')
#     ax[i].set_xlabel('Wavelength ($\mathring{A}$)')
# ax[0].set_ylabel('Flux')

# ax[0].axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
# ax[0].axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
# ax[0].axvspan(H_alpha - blue_peak_mask[1], H_alpha - blue_peak_mask[0], color='blue', alpha=0.1, label='Blue Window')
# ax[0].axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
# ax[0].axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
# ax[0].axvspan(H_alpha + red_peak_mask[0], H_alpha + red_peak_mask[1], color='red', alpha=0.1, label='Red Window')

# ax[1].axvline(x=H_alpha - blue_peak_mask_2[0], color='blue', linestyle='--', alpha=0.5)
# ax[1].axvline(x=H_alpha - blue_peak_mask_2[1], color='blue', linestyle='--', alpha=0.5)
# ax[1].axvspan(H_alpha - blue_peak_mask_2[1], H_alpha - blue_peak_mask_2[0], color='blue', alpha=0.1)
# ax[1].axvline(x=H_alpha + red_peak_mask_2[0], color='red', linestyle='--', alpha=0.5)
# ax[1].axvline(x=H_alpha + red_peak_mask_2[1], color='red', linestyle='--', alpha=0.5)
# ax[1].axvspan(H_alpha + red_peak_mask_2[0], H_alpha + red_peak_mask_2[1], color='red', alpha=0.1)

# # Get the y-position for the annotations
# y_min, y_max = ax[0].get_ylim()
# y_pos = y_min + 0.9 * (y_max - y_min)  # Adjust the 0.1 to move the arrow up or down

# # Add double-sided arrow for blue mask
# ax[0].annotate(
#     '',
#     xy=(H_alpha - blue_peak_mask[1], y_pos),
#     xytext=(H_alpha - blue_peak_mask[0], y_pos),
#     arrowprops=dict(arrowstyle='<->', color='blue', linewidth=2)
# )

# # Add double-sided arrow for red mask
# ax[0].annotate(
#     '',
#     xy=(H_alpha + red_peak_mask[0], y_pos),
#     xytext=(H_alpha + red_peak_mask[1], y_pos),
#     arrowprops=dict(arrowstyle='<->', color='red', linewidth=2)
# )

# # Get the y-position for the annotations
# y_min, y_max = ax[1].get_ylim()
# y_pos = y_min + 0.9 * (y_max - y_min)  # Adjust the 0.1 to move the arrow up or down

# # Add double-sided arrow for blue mask
# ax[1].annotate(
#     '',
#     xy=(H_alpha - blue_peak_mask_2[1], y_pos),
#     xytext=(H_alpha - blue_peak_mask_2[0], y_pos),
#     arrowprops=dict(arrowstyle='<->', color='blue', linewidth=2)
# )

# # Add double-sided arrow for red mask
# ax[1].annotate(
#     '',
#     xy=(H_alpha + red_peak_mask_2[0], y_pos),
#     xytext=(H_alpha + red_peak_mask_2[1], y_pos),
#     arrowprops=dict(arrowstyle='<->', color='red', linewidth=2)
# )

# # Get existing handles and labels
# handles, labels = ax[0].get_legend_handles_labels()

# # Adjust spacing between subplots
# fig.subplots_adjust(wspace=0, top=0.85)

# # Add global legend
# fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 0.98))

# plt.show()





# # %% FIGURE 6 - 20°, 45° SIMILAR LINE CUENO COMPARISON
# ################################################################################
# print('FIGURE 6 - 20°, 45° SIMILAR LINE CUENO COMPARISON')
# ################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.scale import SymmetricalLogLocator

# # Example: define your inclination columns and incs
# inclination_columns = [10, 11]#, 12, 13, 14]  # 20°, 45°, 60°, 72.5°, 85°
# incs = [0 for _ in range(10)]  # to align indices
# incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# # Create the figure and a 2×3 grid of subplots
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True)
# axs = axs.flatten()

# # Remove the unused 6th subplot
# #fig.delaxes(axs[5])
# plt.rcParams.update({'font.size': 15})

# # Load Teo's data once
# bz_cam   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv',   delimiter=',')
# mv_lyr   = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv',   delimiter=',')
# v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# # Prepare legend handles
# handles = []
# labels = []

# for idx, inclination_column in enumerate(inclination_columns):
#     final_results = all_results[inclination_column]
#     cut_runs = final_results['cut_runs']
#     peak_colour_map = final_results['peak_colour_map']
#     grid_length = np.arange(0, 729)
#     ew = np.array(ew_results[inclination_column]['ew'])
#     fwhm = np.array(fwhm_results[inclination_column]['fwhm'])
#     # create a mask for data with EW between 1 and 100 and FWHM < 30
#     mask_cut = np.delete(relevent_runs[inclination_column], cut_runs)
#     #mask = (ew > 1) & (ew < 100) & (fwhm < 30)
    

#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
#     cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#     #mask_cut = np.delete(mask, cut_runs)
#     # Create masks for single-peaked and double-peaked spectra
#     cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
#     # Boolean masks for single- vs double-peaked
#     single_mask = (cut_peak_colour_map == 'black')
#     double_mask = (cut_peak_colour_map == 'red')
#     mask_cut_plus_single = mask_cut & single_mask
#     mask_cut_plus_double = mask_cut & double_mask
#     ax = axs[idx]

#     # Plot error bars (all points together for simplicity)
#     ax.errorbar(
#         cut_red_ew_excess[mask_cut],
#         cut_blue_ew_excess[mask_cut],
#         xerr=cut_red_ew_excess_error[mask_cut],
#         yerr=cut_blue_ew_excess_error[mask_cut],
#         fmt='none',
#         ecolor='black',
#         alpha=0.5,
#         zorder=-1
#     )

#     # Plot single-peaked (black) vs double-peaked (red)
#     # Add legend entries only in the first subplot
#     if idx == 0:
#         # single_scatter = ax.scatter(
#         #     cut_red_ew_excess[single_mask],
#         #     cut_blue_ew_excess[single_mask],
#         #     c='red',
#         #     s=10,
#         #     label='Single-peaked Sirocco Spectra',
#         #     zorder=1,
#         #     alpha=1
#         # )
#         # double_scatter = ax.scatter(
#         #     cut_red_ew_excess[double_mask],
#         #     cut_blue_ew_excess[double_mask],
#         #     c='black',
#         #     s=10,
#         #     label='Double-peaked Sirocco Spectra',
#         #     zorder=0,
#         #     alpha=0.5
#         # )
#         double_near_scatter = ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_double],
#             cut_blue_ew_excess[mask_cut_plus_double],
#             c='dimgray',
#             s=10,
#             label='Potential Sirocco Spectra',
#             zorder=1,
#             alpha=0.4
#         )
#         single_near_scatter = ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_single],
#             cut_blue_ew_excess[mask_cut_plus_single],
#             c='red',
#             s=10,
#             label='Potential Sirocco Spectra',
#             zorder=1,
#             alpha=1
#         )
#         # handles.append(single_scatter)
#         # labels.append('Single-peaked Sirocco Spectra')
#         # handles.append(double_scatter)
#         # labels.append('Double-peaked Sirocco Spectra')
#         handles.append(single_near_scatter)
#         labels.append('Single-Peaked Sirocco Spectra')
#         handles.append(double_near_scatter)
#         labels.append('Double-Peaked Sirocco Spectra')
#     else:
#         # ax.scatter(
#         #     cut_red_ew_excess[single_mask],
#         #     cut_blue_ew_excess[single_mask],
#         #     c='red',
#         #     s=10,
#         #     zorder=1,
#         #     alpha=1
#         # )
#         # ax.scatter(
#         #     cut_red_ew_excess[double_mask],
#         #     cut_blue_ew_excess[double_mask],
#         #     c='black',
#         #     s=10,
#         #     zorder=0,
#         #     alpha=0.5
#         # )
#         ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_double],
#             cut_blue_ew_excess[mask_cut_plus_double],
#             c='dimgray',
#             s=10,
#             zorder=1,
#             alpha=0.4
#         )
#         ax.scatter(
#             cut_red_ew_excess[mask_cut_plus_single],
#             cut_blue_ew_excess[mask_cut_plus_single],
#             c='red',
#             s=10,
#             zorder=1,
#             alpha=1
#         )
#     base_colour = 'cyan' 
#     if idx == 0:
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color=base_colour, s=45, marker='o',edgecolor='navy',
#                                     alpha=0.7,
#                                     label='Cúneo et al. (2023)')
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         handles.append(teo_scatter)
#         labels.append('Cúneo et al. (2023)')
#     elif idx == 1: 
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color=base_colour, s=45, marker='o',
#                                     alpha=0.7, edgecolor='navy')                         
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy', )
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy')
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy')
#     # Only plot Teo’s data on the 45° inclination
#     # if inclination_column == 11:
#     #     teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1], color='red', s=10, marker='o', label='Cúneo et al. (2023)')
#     #     ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
#     #     ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
#     #     ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
#     #     handles.append(teo_scatter)
#     #     labels.append('Cúneo et al. (2023)')

#     # Reference lines
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

#     # Dashed box for linear/log threshold
#     linear_thrs = 0.1
#     ax.plot(
#         [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#         [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#         color='black', linestyle='--', alpha=1.0, zorder=0, linewidth=2.0,
#         label='Linear/Logarithmic Threshold'
#     )
#     #linear threshold lines
#     ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
#     ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
#     ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
#     ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=0)
    
#     # ax.axvline(linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     # ax.axhline(-linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     # ax.axhline(linear_thrs, -linear_thrs, linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
    

#     # Axes formatting
#     ax.set_xlabel('Red Wing EW Excess ($\\mathring{A}$)')
#     # Put a y-label on the left column of subplots (idx=0 for top-left, idx=3 for bottom-left)
#     if idx in [0, 3]:
#         ax.set_ylabel('Blue Wing EW Excess ($\\mathring{A}$)')

#     ax.set_title(f'{incs[inclination_column]}° inclination')
#     ax.set_xlim(-30, 30)
#     ax.set_ylim(-30, 30)
#     ax.set_xscale('symlog', linthresh=linear_thrs)
#     ax.set_yscale('symlog', linthresh=linear_thrs)
#     ax.set_axisbelow(True)
#     ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=-1)

#     # Minor tick locators for symlog
#     ax.xaxis.set_minor_locator(
#         SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
#     )
#     ax.yaxis.set_minor_locator(
#         SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10))
#     )

# # Add final threshold handle/label
# labels.append('Linear/Logarithmic Threshold')
# handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2.0))

# # Adjust subplot spacing (both horizontally and vertically)
# fig.subplots_adjust(wspace=0, hspace=0.15, top=0.88)

# # Position the legend closer to the top
# fig.legend(
#     handles, labels,
#     loc='upper center', ncol=2,
#     bbox_to_anchor=(0.5, 1.05)
# )

# # Manually reposition the bottom row subplots so they are centered
# #pos3 = axs[3].get_position()
# #pos4 = axs[4].get_position()
# #axs[3].set_position([0.252, pos3.y0-0.04, 0.26, pos3.height])  # shift bottom-left a bit to the right
# #axs[4].set_position([0.513, pos4.y0-0.04, 0.26, pos4.height])  # shift bottom-right a bit left

# plt.show()

# # %% Figure 4 - SINGLE VS DOUBLE PEAK DISTRIBUTIONS
# ################################################################################
# print('FIGURE 4: SINGLE VS DOUBLE PEAK DISTRIBUTIONS')
# ################################################################################

# inclination_column = 11  # 45° inclination
# final_results = all_results[inclination_column]
# # EVERY RUN IS PLOTTED HERE FOR THE EQUIVALENT WIDTH EXCESSES, NO BAD FITS REMOVED
# # YOU CAN POTENTIALLY SKIP THIS STEP IF YOU WANT TO REMOVE BAD FITS STRAIGHT AWAY.
# cut_runs = final_results['cut_runs']
# peak_colour_map = final_results['peak_colour_map']
# grid_length = np.arange(0,729)


# cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
# cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
# cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
# cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
# cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
# #cut_sk_con_data = np.delete(final_results['sk_con_data'], cut_runs)
# cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
# cut_grid_length = np.delete(grid_length, cut_runs)


# # Create masks for single-peaked and double-peaked spectra
# cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
# single_peak_mask = cut_peak_colour_map == 'black'
# double_peak_mask = cut_peak_colour_map == 'red'

# plt.figure(figsize=(7,7))
# plt.rcParams.update({'font.size': 15})
# ranges = slice(0,-1) # to select particular portions of the grid if desired
# plt.errorbar(cut_red_ew_excess,
#              cut_blue_ew_excess, 
#              xerr=cut_red_ew_excess_error, 
#              yerr=cut_blue_ew_excess_error, 
#              fmt='none', 
#              ecolor = 'grey', 
#              alpha=0.5,
#              zorder=-1
#              ) # error bars for scatterplot below

# # Plot double-peaked spectra
# plt.scatter(cut_red_ew_excess[double_peak_mask],
#             cut_blue_ew_excess[double_peak_mask],
#             c='red',
#             s=10,
#             label='Double-peaked Spectra')

# # Plot single-peaked spectra
# plt.scatter(cut_red_ew_excess[single_peak_mask],
#             cut_blue_ew_excess[single_peak_mask],
#             c='black',
#             s=10,
#             label='Single-peaked Spectra')

# kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
# kde.fit(np.vstack([cut_red_ew_excess[double_peak_mask], cut_blue_ew_excess[double_peak_mask]]).T)

# xcenters = np.linspace(min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2, 500)
# ycenters = np.linspace(min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2, 500)
# X, Y = np.meshgrid(xcenters, ycenters)
# xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
# Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

# contour_list = np.exp(np.arange(-3.0, 2.5, 0.5))
# #plt.contourf(X, Y, Z, levels=contour_list, cmap='Reds', alpha=0.6, zorder=0)
# #plt.contour(X, Y, Z, levels=contour_list, colors='red', zorder=0)

# kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
# kde.fit(np.vstack([cut_red_ew_excess[single_peak_mask], cut_blue_ew_excess[single_peak_mask]]).T)

# xcenters = np.linspace(min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2, 500)
# ycenters = np.linspace(min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2, 500)
# X, Y = np.meshgrid(xcenters, ycenters)
# xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
# Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

# contour_list = np.exp(np.arange(-3.0, 2.5, 0.5))
# #plt.contourf(X, Y, Z, levels=contour_list, cmap='Grays', alpha=0.6, zorder=0)
# #plt.contour(X, Y, Z, levels=contour_list, colors='black', zorder=0)

# # vertical and horizontal lines at 0 i.e axes
# plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)

# linear_thrs = 0.1
# plt.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)

# # plot formatting
# incs = [0 for i in range(10)] # to indent incs to the same column index as files
# [incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

# plt.xlabel('Red Wing EW Excess ($\mathring{A}$)')
# plt.ylabel('Blue Wing EW Excess ($\mathring{A}$)')
# plt.title(f'Red vs Blue Wing Excess at {incs[inclination_column]}° inclination')
# # sigma clip the data to remove outliers
# max_red = np.mean(cut_red_ew_excess) + 3*np.std(cut_red_ew_excess)
# min_red = np.mean(cut_red_ew_excess) - 3*np.std(cut_red_ew_excess)
# max_blue = np.mean(cut_blue_ew_excess) + 3*np.std(cut_blue_ew_excess)
# min_blue = np.mean(cut_blue_ew_excess) - 3*np.std(cut_blue_ew_excess)

# plt.xlim(-50,50)
# plt.ylim(-50,50)
# plt.xscale('symlog', linthresh=linear_thrs)
# plt.yscale('symlog', linthresh=linear_thrs)

# # Get the current axis
# ax = plt.gca()
# ax.set_axisbelow(True)

# # Set minor tick locators for symlog scale
# ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
# ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))

# # Add grid lines for both major and minor ticks
# plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

# plt.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.02))
# #plt.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.02))
# plt.show()

# # %% Checking Zhao et al 2025, data 
# import numpy as np
# import matplotlib.pyplot as plt

# fwhm_mrs, ew_mrs = np.loadtxt('MRS.csv', delimiter=',', unpack=True)
# fwhm_lrs, ew_lrs = np.loadtxt('LRS.csv', delimiter=',', unpack=True)
# plt.figure(figsize=(8, 8))
# plt.scatter(fwhm_mrs, ew_mrs, marker = '^', color = 'red', s=65)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(20, 1100)
# plt.ylim(0.1, 50)
# plt.xlabel('FWHM ($km/s$)')
# plt.ylabel('EW ($\mathring{A}$)')
# plt.title('Zhao et al. 2025 - MRS')
# plt.show()

# fwhm_lrs, ew_lrs = np.loadtxt('LRS.csv', delimiter=',', unpack=True)
# plt.figure(figsize=(8, 8))
# plt.scatter(fwhm_lrs, ew_lrs, marker = '^', color = 'red', s=65)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(80, 2100)
# plt.ylim(0.4, 110)
# plt.xlabel('FWHM ($km/s$)')
# plt.ylabel('EW ($\mathring{A}$)')
# plt.title('Zhao et al. 2025 - LRS')
# plt.show()



# # %% Figure 9 - 45 degree comparison to Cuneo data. 
# ################################################################################
# print('FIGURE 9: 20/45 degree comparison to Cuneo data')
# ################################################################################
# %matplotlib inline
# import matplotlib.pyplot as plt
# # handles = []
# # labels = []
# # Load Teo's data (only need to load once)
# bz_cam = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
# mv_lyr = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv', delimiter=',')
# v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# # Create a figure with two adjacent subplots
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True)
# plt.rcParams.update({'font.size': 15})

# # Define the two inclination columns:
# # For our incs list, index 10 corresponds to 20° and index 11 to 45°
# inclination_list = [10, 11]
# incs = [0] * 10
# incs.extend([20, 45, 60, 72.5, 85])  # incs[10]=20, incs[11]=45, etc.

# # Prepare lists for legend handles and labels
# legend_handles = []
# legend_labels = []

# # Loop over the two desired inclinations and capture handles only once.
# for ax, incl in zip(axs, inclination_list):
#     final_results = all_results[incl]
#     cut_runs = final_results['cut_runs']
#     peak_colour_map = final_results['peak_colour_map']
#     grid_length = np.arange(0, 729)
    
#     # Remove bad runs
#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
#     cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#     cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
#     cut_grid_length = np.delete(grid_length, cut_runs)
    
#     # Ensure peak colour array is a NumPy array and create mask for single-peaked spectra
#     cut_peak_colour_map = np.array(cut_peak_colour_map)
#     single_peak_mask = cut_peak_colour_map == 'black'
#     double_peak_mask = cut_peak_colour_map == 'red'
    
#     # Plot the linear/log threshold box and lines
#     linear_thrs = 0.1
#     ax.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#             [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#             color='black', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0)
#     ax.axvline(x=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
#     ax.axvline(x=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
#     ax.axhline(y=-linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)
#     ax.axhline(y=linear_thrs, color='black', linestyle='--', alpha=1.0, zorder=1)

#     # Plot error bars and single-peaked Sirocco data.
#     # For the first (20°) subplot we capture the handle.
#     sc = ax.errorbar(cut_red_ew_excess[single_peak_mask],
#                      cut_blue_ew_excess[single_peak_mask],
#                      xerr=cut_red_ew_excess_error[single_peak_mask],
#                      yerr=cut_blue_ew_excess_error[single_peak_mask],
#                      fmt='none', ecolor='grey', alpha=0.5, zorder=-1)
#     sp_scatter = ax.scatter(cut_red_ew_excess[single_peak_mask],
#                             cut_blue_ew_excess[single_peak_mask],
#                             c='red', s=10,
#                             label='Single-Peaked Sirocco Spectra')
#     #Add double peaked
#     dp_scatter = ax.scatter(cut_red_ew_excess[double_peak_mask],
#                cut_blue_ew_excess[double_peak_mask],
#                label='Double-Peaked Sirocco Spectra',
#                c='dimgray', s=10, alpha=0.4, zorder=0)
    
               
#     if incl == 10:
#         legend_handles.append(sp_scatter)
#         legend_labels.append('Single-Peaked Sirocco Spectra')
#         legend_handles.append(dp_scatter)
#         legend_labels.append('Double-Peaked Sirocco Spectra')
        

#     # Only display Teo's data on the 45° (incl==11) plot. Capture the handle.
#     base_colour = 'cyan' 
#     if incl == 11:
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color=base_colour, s=45, marker='o',edgecolor='navy',
#                                     alpha=0.7,
#                                     label='Cúneo et al. (2023)')
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=base_colour, s=45, marker='o',edgecolor='navy', alpha=0.7)
#         legend_handles.append(teo_scatter)
#         legend_labels.append('Cúneo et al. (2023)')
#     else: 
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color=base_colour, s=45, marker='o',
#                                     alpha=0.7, edgecolor='navy')
                                    
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy', )
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy')
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color=base_colour, s=45, marker='o', alpha=0.7, edgecolor='navy')

        
#     # Plot vertical and horizontal axes
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    
#     # Formatting the axes
#     ax.set_xlabel('Red Wing EW Excess ($\mathring{A}$)')
#     if incl != 11:
#         ax.set_ylabel('Blue Wing EW Excess ($\mathring{A}$)')
#     ax.set_title(f'{incs[incl]}° inclination')
#     ax.set_xlim(-30, 30)
#     ax.set_ylim(-30, 30)
#     ax.set_xscale('symlog', linthresh=linear_thrs)
#     ax.set_yscale('symlog', linthresh=linear_thrs)
#     ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
#     ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
#     ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
#     ax.set_axisbelow(True)

# # Adjust space between the subplots
# fig.subplots_adjust(wspace=0)

# # Add the linear/log threshold handle (only one needed)
# threshold_handle = Line2D([0], [0], color='black', linestyle='--', linewidth=2.0)
# legend_handles.append(threshold_handle)
# legend_labels.append('Linear/Logarithmic Threshold')

# # Add global legend
# fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_labels)/2, bbox_to_anchor=(0.5, 1.05))
# plt.show()
# # %% FIGURE 9: 20/45 degree comparison to Cuneo data
# ################################################################################
# print('FIGURE 9: 20/45 degree comparison to Cuneo data')
# ################################################################################
# %matplotlib inline
# import matplotlib.pyplot as plt
# # handles = []
# # labels = []
# # Load Teo's data (only need to load once)
# bz_cam = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
# mv_lyr = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv', delimiter=',')
# v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
# v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# # Create a figure with two adjacent subplots
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True)
# plt.rcParams.update({'font.size': 15})

# # Define the two inclination columns:
# # For our incs list, index 10 corresponds to 20° and index 11 to 45°
# inclination_list = [10, 11]
# incs = [0] * 10
# incs.extend([20, 45, 60, 72.5, 85])  # incs[10]=20, incs[11]=45, etc.

# # Prepare lists for legend handles and labels
# legend_handles = []
# legend_labels = []

# # Loop over the two desired inclinations and capture handles only once.
# for ax, incl in zip(axs, inclination_list):
#     final_results = all_results[incl]
#     cut_runs = final_results['cut_runs']
#     peak_colour_map = final_results['peak_colour_map']
#     grid_length = np.arange(0, 729)
    
#     # Remove bad runs
#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
#     cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#     cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
#     cut_grid_length = np.delete(grid_length, cut_runs)
    
#     # Ensure peak colour array is a NumPy array and create mask for single-peaked spectra
#     cut_peak_colour_map = np.array(cut_peak_colour_map)
#     single_peak_mask = cut_peak_colour_map == 'black'
    
#     # Plot the linear/log threshold box and lines
#     linear_thrs = 0.1
#     ax.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#             [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#             color='blue', linestyle='--', alpha=1.0, zorder=1, linewidth=2.0)
#     ax.axvline(x=linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     ax.axvline(x=-linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     ax.axhline(y=-linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
#     ax.axhline(y=linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)

#     # Plot error bars and single-peaked Sirocco data.
#     # For the first (20°) subplot we capture the handle.
#     sc = ax.errorbar(cut_red_ew_excess[single_peak_mask],
#                      cut_blue_ew_excess[single_peak_mask],
#                      xerr=cut_red_ew_excess_error[single_peak_mask],
#                      yerr=cut_blue_ew_excess_error[single_peak_mask],
#                      fmt='none', ecolor='grey', alpha=0.5, zorder=-1)
#     sp_scatter = ax.scatter(cut_red_ew_excess[single_peak_mask],
#                             cut_blue_ew_excess[single_peak_mask],
#                             c='black', s=10,
#                             label='Single-Peaked Sirocco Spectra')
#     if incl == 10:
#         legend_handles.append(sp_scatter)
#         legend_labels.append('Single-Peaked Sirocco Spectra')

#     # Only display Teo's data on the 45° (incl==11) plot. Capture the handle.
#     if incl == 11:
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color='red', s=10, marker='o',
#                                     label='Cúneo et al. (2023)')
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
#         legend_handles.append(teo_scatter)
#         legend_labels.append('Cúneo et al. (2023)')
#     else: 
#         teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1],
#                                     color='red', s=10, marker='o'
#                                     )
#         # Plot additional datasets without separate legend labels
#         ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
#         ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
#         ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
        
#     # Plot vertical and horizontal axes
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    
#     # Formatting the axes
#     ax.set_xlabel('Red Wing EW Excess ($\mathring{A}$)')
#     if incl != 11:
#         ax.set_ylabel('Blue Wing EW Excess ($\mathring{A}$)')
#     ax.set_title(f'{incs[incl]}° inclination')
#     ax.set_xlim(-0.1, 1)
#     ax.set_ylim(-5, 5)
#     #ax.set_xscale('symlog', linthresh=linear_thrs)
#     #ax.set_yscale('symlog', linthresh=linear_thrs)
#     #ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
#     #ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
#     ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
#     ax.set_axisbelow(True)

# # Adjust space between the subplots
# fig.subplots_adjust(wspace=0)

# # Add the linear/log threshold handle (only one needed)
# threshold_handle = Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0)
# legend_handles.append(threshold_handle)
# legend_labels.append('Linear/Logarithmic Threshold')

# # Add global legend
# fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_labels), bbox_to_anchor=(0.5, 1.02))
# plt.show()


# # %% Figure 6 - MASKING ARROW PLOT
# ################################################################################
# print('FIGURE 6 - MASKING ARROW PLOT')
# ################################################################################
# mask_results = {}
# inclination_column = 11  # 45° inclination
# #masks = ['11_88_mask', '22_88_mask', '22_55_mask'] # 11-88 = 500-4000, 22-88 = 1000-4000, 22-55 = 1000-2500, 22-90 = 1000-4100
# masks = ['20_55_mask', '22_55_mask', '24_55_mask']  # 20-55 = 900-2500, 22-55 = 1000-2500, 24-55 = 1100-2500

# for mask in masks:
#     if os.path.exists(f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy'):
#         all_results[mask] = np.load(
#             f'Emission_Line_Asymmetries/new_data/{mask}/final_results_inc_col_{inclination_column}.npy', allow_pickle=True
#         ).item()

# plt.figure(figsize=(7, 7))
# plt.rcParams.update({'font.size': 15})

# # Add a run number only if the run is present in all masks
# to_axe_runs = []
# for mask in masks:
#     final_results = all_results[mask]
#     to_axe_runs.append(final_results['cut_runs'])
# to_axe_runs = np.unique(np.concatenate(to_axe_runs))

# # Randomly select 30 data points to plot
# seed = np.random.choice(np.arange(1, 1000, 1))
# print(seed)
# np.random.seed(880)  # 927
# to_keep_runs = [i for i in range(729) if i not in to_axe_runs]
# keep_samples = np.random.choice(to_keep_runs, 155, replace=False)
# cut_runs = np.array([i for i in range(729) if i not in keep_samples])

# print(keep_samples)

# colours = ['red', 'black', 'blue']
# #mask_labels = ['$500-4000ms^{-1}$', '$1000-4000ms^{-1}$', '$1000-2500ms^{-1}$']
# mask_labels = ['$900-2500kms^{-1}$', '$1000-2500kms^{-1}$', '$1100-2500kms^{-1}$']

# for i, mask in enumerate(masks):
#     final_results = all_results[mask]
#     cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
#     cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
#     cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
#     cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)

#     ranges = slice(0, -1)
#     if i == 1:
#         plt.errorbar(
#             cut_red_ew_excess,
#             cut_blue_ew_excess,
#             xerr=cut_red_ew_excess_error,
#             yerr=cut_blue_ew_excess_error,
#             fmt='none',
#             ecolor='grey',
#             alpha=0.5,
#             zorder=-1
#         )

#         scatter_plot = plt.scatter(
#             cut_red_ew_excess,
#             cut_blue_ew_excess,
#             c=colours[i],
#             s=10,
#             label=mask_labels[i]
#         )

# # Build a dictionary to store data points for each run
# data_points = {}

# for run_index in keep_samples:
#     data_points[run_index] = []
#     for mask in masks:
#         final_results = all_results[mask]
#         red_ew_excess = final_results['red_ew_excess'][run_index]
#         blue_ew_excess = final_results['blue_ew_excess'][run_index]
#         data_points[run_index].append((red_ew_excess, blue_ew_excess))

# # Define arrow colors
# arrow_colors = ['red', 'blue']
# # Get existing handles and labels
# handles, labels = plt.gca().get_legend_handles_labels()

# # Create custom legend entries

# # Custom handler for arrows
# def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
#     return FancyArrowPatch(
#         (0, 4.5),
#         (30, 4.5),
#         mutation_scale=fontsize * 0.75,
#         arrowstyle='<|-',
#         color=orig_handle.get_edgecolor(),
#         linewidth=orig_handle.get_linewidth()
#     )

# # Custom handles
# arrow_black_line = FancyArrowPatch((0, 0.5), (1, 0.5), mutation_scale=15, arrowstyle='-|>', color='red', linewidth=1)
# arrow_red_line = FancyArrowPatch((0, 0.5), (1, 0.5), mutation_scale=15, arrowstyle='-|>', color='blue', linewidth=1)
# linlogthereshold = Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0)
# # Append custom entries to the handles and labels
# handles.extend([linlogthereshold, arrow_black_line])
# labels.extend([
#     'Linear/Logarithmic Threshold',
#     f'{mask_labels[0]}',#900-2500
# ])

# # Draw arrows between the data points across masks
# for idx, run_index in enumerate(keep_samples):
#     points = data_points[run_index]
#     # Arrow from mask '22_55_mask' to '20_55_mask'
#     start_point = points[1]
#     end_point = points[0]
#     plt.annotate(
#         '',
#         xy=end_point,
#         xytext=start_point,
#         arrowprops=dict(arrowstyle='-|>', color=arrow_colors[0], lw=1),
#         annotation_clip=False,
#         zorder=0
#     )
#     # Arrow from mask '22_55_mask' to '24_55_mask'
#     start_point = points[1]
#     end_point = points[2]
#     plt.annotate(
#         '',
#         xy=end_point,
#         xytext=start_point,
#         arrowprops=dict(arrowstyle='-|>', color=arrow_colors[1], lw=1),
#         annotation_clip=False,
#         zorder=0
#     )

# # Vertical and horizontal lines at 0 (axes)
# plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

# # Plot formatting
# incs = [0] * 10  # To indent incs to the same column index as files
# incs.extend([20, 45, 60, 72.5, 85])  # Inclinations from models

# plt.xlabel('Red Wing EW Excess ($Å$)')
# plt.ylabel('Blue Wing EW Excess ($Å$)')
# plt.title(f'{incs[inclination_column]}° inclination')

# linear_thrs = 0.1
# plt.plot(
#     [-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#     [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#     color='blue',
#     linestyle='--',
#     alpha=1.0,
#     zorder=1,
#     #label='Linear/Logarithmic Threshold',
#     linewidth=2.0
# )
# #linear threshold lines
# plt.axvline(x=linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
# plt.axvline(x=-linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
# plt.axhline(y=-linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
# plt.axhline(y=linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)

# plt.xlim(-30, 30)
# plt.ylim(-30, 30)
# plt.xscale('symlog', linthresh=linear_thrs)
# plt.yscale('symlog', linthresh=linear_thrs)


# # Append custom entries to the handles and labels
# handles.extend([arrow_red_line])
# labels.extend([
#     f'{mask_labels[2]}',#1100-2500
# ])

# # move handle and label index 0 to index 2 position
# handles.insert(2, handles.pop(0))
# labels.insert(2, labels.pop(0))

# # Get the current axis
# ax = plt.gca()
# ax.set_axisbelow(True)

# # Set minor tick locators for symlog scale
# ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
# ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))

# # Add grid lines for both major and minor ticks
# plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

# # Set the legend with custom handlers inside a white background box
# legend = plt.legend(
#     handles,
#     labels,
#     handler_map={
#         arrow_black_line: HandlerPatch(patch_func=make_legend_arrow),
#         arrow_red_line: HandlerPatch(patch_func=make_legend_arrow)
#     },
#     loc='lower left',
#     bbox_to_anchor=(-0.01, 0.74),
#     frameon=True
# )
# legend.get_frame().set_facecolor('white')

# plt.show()

# # %% Figure 7 - FWHM MASKING PROFILES
# ################################################################################
# print('FIGURE 7: FWHM MASKING PROFILES')
# ################################################################################
# mask = 'FWHM_Mask'
# inclination_column = 11  # 45° inclination
# if os.path.exists(f'Emission_Line_Asymmetries/new_FWHM_data/final_results_inc_col_{inclination_column}.npy'):
#     final_results = np.load(f'Emission_Line_Asymmetries/new_FWHM_data/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()
    
# cut_runs = final_results['cut_runs']
# peak_colour_map = final_results['peak_colour_map']
# grid_length = np.arange(0,729)

# cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
# cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
# cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
# cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
# cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
# cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
# cut_grid_length = np.delete(grid_length, cut_runs)

# # A mask that hides all values with EW excess of 0
# for i in range(len(cut_red_ew_excess)):
#     if cut_red_ew_excess[i] == 0 or cut_blue_ew_excess[i] == 0:
#         cut_red_ew_excess[i] = np.nan
#         cut_blue_ew_excess[i] = np.nan

# # Create masks for single-peaked and double-peaked spectra
# cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array
# single_peak_mask = cut_peak_colour_map == 'black'
# double_peak_mask = cut_peak_colour_map == 'red'

# plt.figure(figsize=(7,7))
# plt.rcParams.update({'font.size': 15})
# ranges = slice(0,-1)  # to select particular portions of the grid if desired

# plt.errorbar(cut_red_ew_excess,
#              cut_blue_ew_excess, 
#              xerr=cut_red_ew_excess_error, 
#              yerr=cut_blue_ew_excess_error, 
#              fmt='none', 
#              ecolor='grey', 
#              alpha=0.5,
#              zorder=-1)

# # Plot single-peaked spectra (black) and double-peaked spectra (red)
# plt.scatter(cut_red_ew_excess[single_peak_mask],
#             cut_blue_ew_excess[single_peak_mask],
#             c='black',
#             s=10,
#             label='Single-peaked')
# plt.scatter(cut_red_ew_excess[double_peak_mask],
#             cut_blue_ew_excess[double_peak_mask],
#             c='red',
#             s=10,
#             label='Double-peaked')

# # Vertical and horizontal lines at 0 (axes)
# plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)

# linear_thrs = 0.1
# plt.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs],
#          [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs],
#          color='blue', linestyle='--', alpha=1.0, zorder=1,
#          label='Linear/Logarithmic Threshold', linewidth=2.0)

# # Linear threshold lines
# plt.axvline(x=linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
# plt.axvline(x=-linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
# plt.axhline(y=-linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)
# plt.axhline(y=linear_thrs, color='blue', linestyle='--', alpha=1.0, zorder=1)

# incs = [0 for i in range(10)]
# incs.extend([20,45,60,72.5,85])

# plt.xlabel('Red Wing EW Excess ($\\mathring{A}$)')
# plt.ylabel('Blue Wing EW Excess ($\\mathring{A}$)')
# plt.title(f'0.7x and 3.0x FWHM Window at {incs[inclination_column]}° inclination')
# plt.xlim(-50, 50)
# plt.ylim(-50, 50)
# plt.xscale('symlog', linthresh=linear_thrs)
# plt.yscale('symlog', linthresh=linear_thrs)
# plt.minorticks_on()
# legend = plt.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.02), frameon=True)
# legend.get_frame().set_facecolor('white')

# ax = plt.gca()
# ax.set_axisbelow(True)
# ax.xaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
# ax.yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=linear_thrs, base=10, subs=np.arange(1, 10)))
# plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
# plt.show()


# ################################################################################
# print('EMISSION MEASUREMENTS, CURVE OF GROWTH FIT AND LINEAR REGRESSION CALCULATION')
# ################################################################################
# # Weighted Least Squares (WLS) model
# wls = sm.WLS(log_em, ols_X, weights=1/log_ew_error) # including y errors effect
# wls_result = wls.fit() # fitting the model
# print(wls_result.summary()) # summary of the model (errors, coefficients, etc)


# predicted_log_Y_all_wls = wls_result.predict(ols_X)

# # WLS error calculation using standard error of the coefficients
# predicted_log_Y_all_wls_err = []
# for i, _ in enumerate(ols_X):
#     x_err_plus_c = [wls_result.bse[j]*abs(ols_X[i][j]) for j in range(len(ols_X[i]))]
#     y_err = np.sqrt(np.sum(np.array(x_err_plus_c)**2))
#     predicted_log_Y_all_wls_err.append(y_err)



# store_total_luminosity = np.delete(store_total_luminosity, remove_negative_ews)

# for_christian = {
#     'ew': equivalent_widths,
#     'em': emission_measures,
#     'predicted_em': np.power(10, predicted_log_Y_all),
#     'L_line': store_total_luminosity
# }
# np.save('for_christian.npy', for_christian)


# data = np.load('for_christian.npy', allow_pickle=True).item()
# df = pd.DataFrame(data)
# df.to_csv('for_christian.csv', index=False)

# plt.figure(figsize=(7, 7))
# plt.scatter(log_em, log_ew, alpha=0.5)
# plt.xlabel('Log10(Emission Measure)')
# plt.ylabel('Log10(Equivalent Width)')
# plt.title(f'Inclination: {incs[inclination]}°')
# plt.show()

# log_X_all = np.delete(log_X_all, nan_values_all, axis=0)

#log_ew_over_em = log_ew/log_em

# model_all = LinearRegression()
# model_all.fit(log_X_all, log_em)#, 1/log_Y_all_error) # 1/unc for weights from errors

# print(f'R^2 Score: {model_all.score(log_X_all, log_em)}')#, 1/log_Y_all_error)}')
# print('Model fitting equation:')
# display(Math(r'log(EM) = alog(p1) + blog(p2) + clog(p3) + d * p4 + elog(p5) + f * p6 + g'))
# eqn_all = f'log(EM) = {model_all.coef_[0]:.3f}log({sim_parameters[0]}) + {model_all.coef_[1]:.3f}log({sim_parameters[1]}) + {model_all.coef_[2]:.3f}log({sim_parameters[2]}) + {model_all.coef_[3]:.3f}{sim_parameters[3]} + {model_all.coef_[4]:.3f}log({sim_parameters[4]}) + {model_all.coef_[5]:.3f}{sim_parameters[5]} + {model_all.intercept_:.3f}$'
# display(Math(f'{eqn_all}')) # Latex formatted string to LaTeX output

# predicted_log_Y_all = model_all.predict(log_X_all)


# # print the intercept and coefficient values ols
# print(f'Intercept: {ols_result.params[0]}')
# print(f'Coefficients: {ols_result.params[1:]}')
# print(f'std_err: {ols_result.bse}')