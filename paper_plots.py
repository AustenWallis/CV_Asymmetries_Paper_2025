# %%
################################################################################
print('STEP 1: IMPORTING MODULES AND DATA')
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
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import scipy.signal as signal

plt.style.use('science')

all_results = {}
inclination_columns = [10,11,12,13,14]
mask = '22_55_mask' # 11-88 = 500-4000, 22-88 = 1000-4000, 22-55 = 1000-2500, 22-90 = 1000-4100
for inclination_column in inclination_columns:
    if os.path.exists(f'Emission_Line_Asymmetries/{mask}/final_results_inc_col_{inclination_column}.npy'):
        all_results[inclination_column] = np.load(f'Emission_Line_Asymmetries/{mask}/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()


# %% Figure 1
################################################################################
print('FIGURE 1: LOW, MEDIUM, HIGH INCLINATION DIAGNOSTIC PLOTS')
################################################################################
# Define inclinations for low, medium, and high
inclination_columns = [11, 12, 13]  # 45°, 60°, 72.5°
incs = [0 for _ in range(10)]  # to align indices
incs.extend([20, 45, 60, 72.5, 85])  # inclinations from models

# Set up subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
plt.rcParams.update({'font.size': 15})

# Load Teo's data (only need to load once)
bz_cam = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/BZ Cam.csv', delimiter=',') 
mv_lyr = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/MV Lyr.csv', delimiter=',')
v425_cas = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V425 Cas.csv', delimiter=',')
v751_cyg = np.loadtxt('Emission_Line_Asymmetries/Cuneo_2023_data/V751 Cyg.csv', delimiter=',')

# Initialize lists to collect handles and labels for legend
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
    cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
    cut_grid_length = np.delete(grid_length, cut_runs)

    # Create masks for single-peaked and double-peaked spectra
    cut_peak_colour_map = np.array(cut_peak_colour_map)  # ensure it's an array

    ax = axs[idx]
    ranges = slice(0, -1)  # to select particular portions of the grid if desired

    # Plot error bars (no label)
    ax.errorbar(
        cut_red_ew_excess,
        cut_blue_ew_excess, 
        xerr=cut_red_ew_excess_error, 
        yerr=cut_blue_ew_excess_error, 
        fmt='none', 
        ecolor='grey', 
        alpha=0.5,
        zorder=-1
    )

    # Plot grid data, get handle (only need to get handle once)
    if idx == 0:
        grid_scatter = ax.scatter(
            cut_red_ew_excess,
            cut_blue_ew_excess,
            c='black',
            s=10,
            label='Grid Data'
        )
        handles.append(grid_scatter)
        labels.append('Grid Data')
    else:
        ax.scatter(
            cut_red_ew_excess,
            cut_blue_ew_excess,
            c='black',
            s=10,
        )

    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    kde.fit(np.vstack([cut_red_ew_excess, cut_blue_ew_excess]).T)

    xcenters = np.linspace(min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2, 500)
    ycenters = np.linspace(min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2, 500)
    X, Y = np.meshgrid(xcenters, ycenters)
    xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

    contour_list = np.exp(np.arange(-3.0, 2.5, 0.5)) # circular shapes
    #contour_list = np.exp(np.arange(-4.0, 3.5, 0.5)) # square shapes
    #ax.contourf(X, Y, Z, levels=contour_list, cmap='Greys', alpha=0.6, zorder=0)
    #ax.contour(X, Y, Z, levels=contour_list, colors='black', zorder=0)

    # Only plot Teo's data on the medium inclination (45 degrees, inclination_column==11)
    if inclination_column == 11:
        # Plot Teo's data and collect handle for legend
        teo_scatter = ax.scatter(bz_cam[:, 0], bz_cam[:, 1], color='red', s=10, marker='o', label='Cúneo et al. (2023)')
        ax.scatter(mv_lyr[:, 0], mv_lyr[:, 1], color='red', s=10, marker='o')
        ax.scatter(v425_cas[:, 0], v425_cas[:, 1], color='red', s=10, marker='o')
        ax.scatter(v751_cyg[:, 0], v751_cyg[:, 1], color='red', s=10, marker='o')
        handles.append(teo_scatter)
        labels.append('Cúneo et al. (2023)')

    # Vertical and horizontal lines at 0 (axes)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=1)
    #plot a square dashed box around the linear threshold boundary
    linear_thrs = 0.1
    ax.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)
    
    # Plot formatting
    ax.set_xlabel('Red Wing EW Excess ($\mathring{A}$)')
    if idx == 0:
        ax.set_ylabel('Blue Wing EW Excess ($\mathring{A}$)')
    ax.set_title(f'{incs[inclination_column]}° inclination')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xscale('symlog', linthresh=linear_thrs) #change this for circular square shapes
    ax.set_yscale('symlog', linthresh=linear_thrs)

labels.append('Linear/Logarithmic Threshold')
handles.append(Line2D([0], [0], color='blue', linestyle='--', linewidth=2.0))
    
# Adjust spacing between subplots to remove the gap
fig.subplots_adjust(wspace=0)

# Add global legend
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.525, 1.05))

# Adjust layout to make room for the legend
#plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %% Figure 4
################################################################################
print('FIGURE 4: SINGLE VS DOUBLE PEAK DISTRIBUTIONS')
################################################################################

inclination_column = 11  # 45° inclination
final_results = all_results[inclination_column]
# EVERY RUN IS PLOTTED HERE FOR THE EQUIVALENT WIDTH EXCESSES, NO BAD FITS REMOVED
# YOU CAN POTENTIALLY SKIP THIS STEP IF YOU WANT TO REMOVE BAD FITS STRAIGHT AWAY.
cut_runs = final_results['cut_runs']
peak_colour_map = final_results['peak_colour_map']
grid_length = np.arange(0,729)


cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
#cut_sk_con_data = np.delete(final_results['sk_con_data'], cut_runs)
cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
cut_grid_length = np.delete(grid_length, cut_runs)


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
#plt.contourf(X, Y, Z, levels=contour_list, cmap='Reds', alpha=0.6, zorder=0)
#plt.contour(X, Y, Z, levels=contour_list, colors='red', zorder=0)

kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
kde.fit(np.vstack([cut_red_ew_excess[single_peak_mask], cut_blue_ew_excess[single_peak_mask]]).T)

xcenters = np.linspace(min(cut_red_ew_excess)-2, max(cut_red_ew_excess)+2, 500)
ycenters = np.linspace(min(cut_blue_ew_excess)-2, max(cut_blue_ew_excess)+2, 500)
X, Y = np.meshgrid(xcenters, ycenters)
xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
Z = np.exp(kde.score_samples(xy_sample)).reshape(X.shape)

contour_list = np.exp(np.arange(-3.0, 2.5, 0.5))
#plt.contourf(X, Y, Z, levels=contour_list, cmap='Grays', alpha=0.6, zorder=0)
#plt.contour(X, Y, Z, levels=contour_list, colors='black', zorder=0)

# vertical and horizontal lines at 0 i.e axes
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)

linear_thrs = 0.1
plt.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)

# plot formatting
incs = [0 for i in range(10)] # to indent incs to the same column index as files
[incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

plt.xlabel('Red Wing EW Excess ($\mathring{A}$)')
plt.ylabel('Blue Wing EW Excess ($\mathring{A}$)')
plt.title(f'Red vs Blue Wing Excess at {incs[inclination_column]}° inclination')
# sigma clip the data to remove outliers
max_red = np.mean(cut_red_ew_excess) + 3*np.std(cut_red_ew_excess)
min_red = np.mean(cut_red_ew_excess) - 3*np.std(cut_red_ew_excess)
max_blue = np.mean(cut_blue_ew_excess) + 3*np.std(cut_blue_ew_excess)
min_blue = np.mean(cut_blue_ew_excess) - 3*np.std(cut_blue_ew_excess)

plt.xlim(-50,50)
plt.ylim(-50,50)
plt.xscale('symlog', linthresh=linear_thrs)
plt.yscale('symlog', linthresh=linear_thrs)

plt.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.02))
#plt.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.02))
plt.show()

# %% Figure 5
################################################################################
print('FIGURE 5: MASK FITTING METHODOLOGY')
################################################################################

H_alpha = 6562.819
blue_peak_mask = (22, 88)  # number of angstroms to cut around the peak, blue minus.
red_peak_mask = (22, 88)  # number of angstroms to cut around the peak, red plus.

blue_peak_mask_2 = (11, 55)
red_peak_mask_2 = (11, 55) # 5 45

final_results = all_results[11]  # 45° inclination
run = 701

fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for i in [0, 1]:
    ax[i].plot(final_results['wavelength_grid'][run],
               final_results['grid'][run],
               label='Original Data',
               color='black'
               )
    ax[i].plot(final_results['wavelength_grid'][run],
               final_results['fitted_grid'][run],
               label='Optimal Gaussian',
               color='red'
               )
    ax[i].plot(final_results['wavelength_grid'][run],
               final_results['fit_con'][run],
               label='Fitted Continuum',
               color='blue'
               )
    ax[i].axvline(x=H_alpha, color='black', linestyle='--', alpha=0.5, label=r'$H\alpha$')
    ax[i].set_xlabel('Wavelength ($\mathring{A}$)')
ax[0].set_ylabel('Flux')

ax[0].axvline(x=H_alpha - blue_peak_mask[0], color='blue', linestyle='--', alpha=0.5)
ax[0].axvline(x=H_alpha - blue_peak_mask[1], color='blue', linestyle='--', alpha=0.5)
ax[0].axvspan(H_alpha - blue_peak_mask[1], H_alpha - blue_peak_mask[0], color='blue', alpha=0.1, label='Blue Mask')
ax[0].axvline(x=H_alpha + red_peak_mask[0], color='red', linestyle='--', alpha=0.5)
ax[0].axvline(x=H_alpha + red_peak_mask[1], color='red', linestyle='--', alpha=0.5)
ax[0].axvspan(H_alpha + red_peak_mask[0], H_alpha + red_peak_mask[1], color='red', alpha=0.1, label='Red Mask')

ax[1].axvline(x=H_alpha - blue_peak_mask_2[0], color='blue', linestyle='--', alpha=0.5)
ax[1].axvline(x=H_alpha - blue_peak_mask_2[1], color='blue', linestyle='--', alpha=0.5)
ax[1].axvspan(H_alpha - blue_peak_mask_2[1], H_alpha - blue_peak_mask_2[0], color='blue', alpha=0.1)
ax[1].axvline(x=H_alpha + red_peak_mask_2[0], color='red', linestyle='--', alpha=0.5)
ax[1].axvline(x=H_alpha + red_peak_mask_2[1], color='red', linestyle='--', alpha=0.5)
ax[1].axvspan(H_alpha + red_peak_mask_2[0], H_alpha + red_peak_mask_2[1], color='red', alpha=0.1)

# Get the y-position for the annotations
y_min, y_max = ax[0].get_ylim()
y_pos = y_min + 0.9 * (y_max - y_min)  # Adjust the 0.1 to move the arrow up or down

# Add double-sided arrow for blue mask
ax[0].annotate(
    '',
    xy=(H_alpha - blue_peak_mask[1], y_pos),
    xytext=(H_alpha - blue_peak_mask[0], y_pos),
    arrowprops=dict(arrowstyle='<->', color='blue', linewidth=2)
)

# Add double-sided arrow for red mask
ax[0].annotate(
    '',
    xy=(H_alpha + red_peak_mask[0], y_pos),
    xytext=(H_alpha + red_peak_mask[1], y_pos),
    arrowprops=dict(arrowstyle='<->', color='red', linewidth=2)
)

# Get the y-position for the annotations
y_min, y_max = ax[1].get_ylim()
y_pos = y_min + 0.9 * (y_max - y_min)  # Adjust the 0.1 to move the arrow up or down

# Add double-sided arrow for blue mask
ax[1].annotate(
    '',
    xy=(H_alpha - blue_peak_mask_2[1], y_pos),
    xytext=(H_alpha - blue_peak_mask_2[0], y_pos),
    arrowprops=dict(arrowstyle='<->', color='blue', linewidth=2)
)

# Add double-sided arrow for red mask
ax[1].annotate(
    '',
    xy=(H_alpha + red_peak_mask_2[0], y_pos),
    xytext=(H_alpha + red_peak_mask_2[1], y_pos),
    arrowprops=dict(arrowstyle='<->', color='red', linewidth=2)
)

# Get existing handles and labels
handles, labels = ax[0].get_legend_handles_labels()

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0, top=0.85)

# Add global legend
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 0.98))

plt.show()
# %%
################################################################################
print('FIGURE 6: THEORETICAL DIAGNOSTIC PLOTS')
################################################################################
fig, axs = plt.subplots(figsize=(7, 7))
plt.rcParams.update({'font.size': 15})
axs.scatter(0,0, color='black', marker='+')
x = [-30, -30, 30, 30, -30, 0, 0, 30]
y = [-30, 30, 30, -30, 0, -30, 30, 0]
axs.scatter(x, y, color='black', marker='o')

axs.set_xlabel('Red Wing EW Excess ($\mathring{A}$)')
axs.set_ylabel('Blue Wing EW Excess ($\mathring{A}$)')

axs.axvline(x=0, color='black', linestyle='--', alpha=0.5)
axs.axhline(y=0, color='black', linestyle='--', alpha=0.5)

axs.plot(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100), color='black', linestyle='--', alpha=0.5)

axs.plot(np.linspace(-50, 50, 100), -np.linspace(-50, 50, 100), color='black', linestyle='--', alpha=0.5)

axs.set_xlim(-40, 40)
axs.set_ylim(-40, 40)

# # Add text to plot
# axs.text(-78, 45, 'Inverse P-Cygni', fontsize=15, color='black')
# axs.text(49, 45, 'Broad Emission Wings', fontsize=15, color='black')
# axs.text(-84, -50, 'Broad Absorption Wings', fontsize=15, color='black')
# axs.text(59, -50, 'P-Cygni', fontsize=15, color='black')
# axs.text(-17, -50, 'Increased Blue Absorption', fontsize=15, color='black')
# axs.text(-17, 70, 'Increased Blue Emission', fontsize=15, color='black')
# axs.text(48, 0, 'Increased Red Emission', fontsize=15, color='black')
# axs.text(-85, 0, 'Increased Red Absorption', fontsize=15, color='black')

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
   # (0, -25)

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
    # Keep axes on but remove the numbers surrounding the plot
    ax_inset.tick_params(
        axis='both',          # Apply changes to both axes
        which='both',         # Apply to both major and minor ticks
        labelbottom=False,    # Hide x-axis tick labels
        labelleft=False       # Hide y-axis tick labels
    )
    ax_inset.set_title(labels[i], fontdict={'fontsize': 12})
    ax_inset.set_ylim(-0.3, 1.2)

    i += 1

# Create custom legend
custom_lines = [Line2D([0], [0], color='red', linestyle='--'),
                Line2D([0], [0], color='black')]

fig.legend(custom_lines, ['Fixed Gaussian', 'Hypothetical Line Profile'], loc='upper center', ncol=1, bbox_to_anchor=(-0.1, -0.13))

plt.show()
# %%
################################################################################
print('FIGURE 3: DIFFERENT MASKING PROFILES')
################################################################################

%matplotlib inline

mask_results = {}
inclination_column = 11  # 45° inclination
#masks = ['11_88_mask', '22_88_mask', '22_55_mask'] # 11-88 = 500-4000, 22-88 = 1000-4000, 22-55 = 1000-2500, 22-90 = 1000-4100
masks = ['20_55_mask', '22_55_mask', '24_55_mask'] # 20-55 = 900-2500, 22-55 = 1000-2500, 24-55 = 1100-2500

for mask in masks:
    if os.path.exists(f'Emission_Line_Asymmetries/{mask}/final_results_inc_col_{inclination_column}.npy'):
        all_results[mask] = np.load(f'Emission_Line_Asymmetries/{mask}/final_results_inc_col_{inclination_column}.npy', allow_pickle=True).item()
   

plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 15})

# add a run number only if the run is present in all masks
to_axe_runs = []
for mask in masks:
    final_results = all_results[mask]
    to_axe_runs.append(final_results['cut_runs'])
to_axe_runs = np.unique(np.concatenate(to_axe_runs))

# randomly select 30 data points to plot
seed = np.random.choice(np.arange(1,1000,1))
print(seed)
np.random.seed(927) # 927
to_keep_runs = [i for i in range(729) if i not in to_axe_runs]
keep_samples = np.random.choice(to_keep_runs, 30, replace=False)
cut_runs = np.array([i for i in range(729) if i not in keep_samples])
        
print(keep_samples)

colours = ['red', 'black', 'blue']
#mask_labels = ['$500-4000ms^{-1}$', '$1000-4000ms^{-1}$', '$1000-2500ms^{-1}$']
mask_labels = ['$900-2500ms^{-1}$', '$1000-2500ms^{-1}$', '$1100-2500ms^{-1}$']

for i, mask in enumerate(masks):
    final_results = all_results[mask]
    cut_red_ew_excess = np.delete(final_results['red_ew_excess'], cut_runs)
    cut_blue_ew_excess = np.delete(final_results['blue_ew_excess'], cut_runs)
    cut_red_ew_excess_error = np.delete(final_results['red_ew_excess_error'], cut_runs)
    cut_blue_ew_excess_error = np.delete(final_results['blue_ew_excess_error'], cut_runs)
    #cut_peak_colour_map = np.delete(peak_colour_map, cut_runs)
    #cut_grid = [i for j, i in enumerate(final_results['grid']) if j not in cut_runs]
    #cut_grid_length = np.delete(grid_length, cut_runs)

    ranges = slice(0, -1)
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

        plt.scatter(
            cut_red_ew_excess,
            cut_blue_ew_excess,
            c=colours[i],
            s=10,
            label=mask_labels[i]
        )

# Build a dictionary to store data points for each run
data_points = {}

for run_index in keep_samples:
    data_points[run_index] = []
    for mask in masks:
        final_results = all_results[mask]
        red_ew_excess = final_results['red_ew_excess'][run_index]
        blue_ew_excess = final_results['blue_ew_excess'][run_index]
        data_points[run_index].append((red_ew_excess, blue_ew_excess))

# Use a selected list of colours or a qualitative colormap

# Choose a qualitative colormap and generate colors
#cmap = cm.get_cmap('tab10', len(keep_samples))
#arrow_colors = [cmap(i) for i in range(len(keep_samples))]
arrow_colors = ['black', 'red', 'blue']

# Draw arrows between the data points across masks
for idx, run_index in enumerate(keep_samples):
    points = data_points[run_index]
    #color = arrow_colors[idx]
    # for i in range(len(points) - 1):
    #     start_point = points[i]
    #     end_point = points[i + 1]
    #     color = arrow_colors[i]
    #     plt.annotate(
    #         '',
    #         xy=end_point,
    #         xytext=start_point,
    #         arrowprops=dict(arrowstyle='->', color=color, lw=1),
    #         annotation_clip=False
    #     )
    start_point = points[1]
    end_point = points[0]
    color = arrow_colors[0]
    plt.annotate(
        '',
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(arrowstyle='-|>', color=color, lw=1),
        annotation_clip=False
    )
    start_point = points[1]
    end_point = points[2]
    color = arrow_colors[1]
    plt.annotate(
        '',
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(arrowstyle='-|>', color=color, lw=1),
        annotation_clip=False
    )
    # start_point = points[2]
    # end_point = points[0]
    # color = arrow_colors[2]
    # plt.annotate(
    #     '',
    #     xy=end_point,
    #     xytext=start_point,
    #     arrowprops=dict(arrowstyle='->', color=color, lw=1),
    #     annotation_clip=False
    # )
    # Add a colorbar to represent the run indices
#sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
#sm.set_array([])
#cbar = plt.colorbar(sm, ax=plt.gca())

# vertical and horizontal lines at 0 i.e axes
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder = 1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder = 1)

# plot formatting
incs = [0 for i in range(10)] # to indent incs to the same column index as files
[incs.append(i) for i in [20,45,60,72.5,85]] # inclinations from PYTHON models

plt.xlabel('Red Wing EW Excess ($Å$)')
plt.ylabel('Blue Wing EW Excess ($Å$)')
plt.title(f'Red vs Blue Wing Excess at {incs[inclination_column]}° inclination')

linear_thrs = 0.1
plt.plot([-linear_thrs, linear_thrs, linear_thrs, -linear_thrs, -linear_thrs], [-linear_thrs, -linear_thrs, linear_thrs, linear_thrs, -linear_thrs], color='blue', linestyle='--', alpha=1.0, zorder=1, label='Linear/Logarithmic Threshold', linewidth=2.0)


plt.xlim(-10,10)
plt.ylim(-10,10)
plt.xscale('symlog', linthresh=linear_thrs)
plt.yscale('symlog', linthresh=linear_thrs)

plt.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.02))
plt.show()

# %%
