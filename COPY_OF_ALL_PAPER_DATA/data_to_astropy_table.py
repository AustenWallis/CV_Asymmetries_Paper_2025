#%%
import numpy as np
from astropy.table import Table
import os

path_to_fluxes = '../Online_PyScript/cv_spectra_data.npz'
if not os.path.exists(path_to_fluxes):
    raise FileNotFoundError(f"File not found: {path_to_fluxes}")
# Load the data from the npz file
data = np.load(path_to_fluxes)
# Extract the Sirocco arrays from the loaded data
wavelengths = np.array(data['wavelengths'])
fluxes = data['fluxes']
normalised_fluxes = data['normalized_fluxes']
run_numbers = data['run_number']
inclinations = data['inclinations']
parameter_table = data['parameter_table']
sim_parameters = ['Mdot_disk',
                  'Mdot_wind',
                  'KWD_d',
                  'r_exp(alpha)',
                  'acc_length(l)',
                  'acc_exp(beta)']

#Emission measures
path_to_emission_measures = '../Emission_Line_Asymmetries/Sirocco_based_data/emission_measures.npy'
if not os.path.exists(path_to_emission_measures):
    raise FileNotFoundError(f"File not found: {path_to_emission_measures}")
# Load the emission measures
emission_measures = np.array(np.load(path_to_emission_measures))

# Equivalent widths
path_to_equivalent_widths = '../Emission_Line_Asymmetries/Sirocco_based_data/ew_results.npy'
if not os.path.exists(path_to_equivalent_widths):
    raise FileNotFoundError(f"File not found: {path_to_equivalent_widths}")
# Load the equivalent widths
ew_results = np.load(path_to_equivalent_widths, allow_pickle=True).item()

# FWHMs
path_to_fwhms = '../Emission_Line_Asymmetries/Sirocco_based_data/fwhm_results.npy'
if not os.path.exists(path_to_fwhms):
    raise FileNotFoundError(f"File not found: {path_to_fwhms}")
# Load the FWHMs
fwhm_results = np.load(path_to_fwhms, allow_pickle=True).item()

# Sample Labels 
path_to_labels = '../Online_PyScript/gold_silver_bronze_labels.npz'
if not os.path.exists(path_to_labels):
    raise FileNotFoundError(f"File not found: {path_to_labels}")
# Load the labels
labels_data = np.load(path_to_labels, allow_pickle=True)
labels_data = labels_data['labels'].item()

#load line luminoisities 
line_luminosities = {}
keys = [10,11,12,13,14]
for i, val in enumerate(['20', '45', '60', '72p5', '85']):
    path_to_line_luminosities = f'../Line_Luminosities/for_christian_{val}.csv'
    if not os.path.exists(path_to_line_luminosities):
        raise FileNotFoundError(f"File not found: {path_to_line_luminosities}")
    # Load the line luminosities for each inclination
    line_lum = np.loadtxt(path_to_line_luminosities, delimiter=',', skiprows=1, usecols=(2))
    line_luminosities[keys[i]] = line_lum


#Load excess ews
excess_ews_22_55 = {}
excess_ews_11_88 = {}
excess_ews_fwhm_1_5 = {}
keys = [10,11,12,13,14]
for i, val in enumerate(keys):
    path_to_excess_ews = f'../Emission_Line_Asymmetries/final_data/22_55_mask/final_results_inc_col_{val}.npy'
    if not os.path.exists(path_to_excess_ews):
        raise FileNotFoundError(f"File not found: {path_to_excess_ews}")
    # Load the excess equivalent widths for each inclination
    results = np.load(path_to_excess_ews, allow_pickle=True).item()
    excess_ews_22_55[val] = {'blue':results['blue_ew_excess'], 'red':results['red_ew_excess']}
for i, val in enumerate(keys):
    path_to_excess_ews = f'../Emission_Line_Asymmetries/final_data/11_88_mask/final_results_inc_col_{val}.npy'
    if not os.path.exists(path_to_excess_ews):
        raise FileNotFoundError(f"File not found: {path_to_excess_ews}")
    # Load the excess equivalent widths for each inclination
    results = np.load(path_to_excess_ews, allow_pickle=True).item()
    excess_ews_11_88[val] = {'blue':results['blue_ew_excess'], 'red':results['red_ew_excess']}
for i, val in enumerate(keys):
    path_to_excess_ews = f'../Emission_Line_Asymmetries/final_data/FWHM_1p0_5_mask_data/final_results_inc_col_{val}.npy'
    if not os.path.exists(path_to_excess_ews):
        raise FileNotFoundError(f"File not found: {path_to_excess_ews}")
    # Load the excess equivalent widths for each inclination
    results = np.load(path_to_excess_ews, allow_pickle=True).item()
    excess_ews_fwhm_1_5[val] = {'blue':results['blue_ew_excess'], 'red':results['red_ew_excess']}



# Create an Astropy Table
table = Table()
table['run_number'] = run_numbers
for i, param in enumerate(sim_parameters):
    table[param] = parameter_table[:, i+1]  # +1 to skip the first column which is run_number
table['emission_measure'] = emission_measures
# ensure emission_measure displays with five decimal places
table['emission_measure'].format = '%.5e'
table['wavelength'] = wavelengths
for i, inc in enumerate(inclinations):
    table[f'flux_inc_{inc}'] = fluxes[:, :, i]
for i, inc in enumerate(inclinations):
    table[f'norm_flux_inc_{inc}'] = normalised_fluxes[:, :, i]
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'ew_inc_{inc}'] = ew_results[keys[i]]['ew']
    table[f'ew_inc_{inc}'].format = '%.5e'  # ensure ew displays with five decimal places
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'fwhm_inc_{inc}'] = fwhm_results[keys[i]]['fwhm']
    table[f'fwhm_inc_{inc}'].format = '%.5e'  # ensure fwhm displays with five decimal places
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'Sample_Label_inc_{inc}'] = labels_data[keys[i]]
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'line_luminosity_inc_{inc}'] = line_luminosities[keys[i]]
    table[f'line_luminosity_inc_{inc}'].format = '%.5e'  # ensure line luminosities display with five decimal places
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'excess_ew_blue_22_55_inc_{inc}'] = excess_ews_22_55[keys[i]]['blue']
    table[f'excess_ew_blue_22_55_inc_{inc}'].format = '%.5e'  # ensure excess ew displays with five decimal places
    table[f'excess_ew_red_22_55_inc_{inc}'] = excess_ews_22_55[keys[i]]['red']
    table[f'excess_ew_red_22_55_inc_{inc}'].format = '%.5e'  # ensure excess ew displays with five decimal places
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'excess_ew_blue_11_88_inc_{inc}'] = excess_ews_11_88[keys[i]]['blue']
    table[f'excess_ew_blue_11_88_inc_{inc}'].format = '%.5e'  # ensure excess ew displays with five decimal places
    table[f'excess_ew_red_11_88_inc_{inc}'] = excess_ews_11_88[keys[i]]['red']
    table[f'excess_ew_red_11_88_inc_{inc}'].format = '%.5e'  # ensure excess ew displays with five decimal places
for i, inc in enumerate(inclinations):
    keys = [10,11,12,13,14]
    table[f'excess_fwhm_blue_inc_{inc}'] = excess_ews_fwhm_1_5[keys[i]]['blue']
    table[f'excess_fwhm_blue_inc_{inc}'].format = '%.5e'  # ensure excess fwhm displays with five decimal places
    table[f'excess_fwhm_red_inc_{inc}'] = excess_ews_fwhm_1_5[keys[i]]['red']
    table[f'excess_fwhm_red_inc_{inc}'].format = '%.5e'  # ensure excess fwhm displays with five decimal places

# Save the table to a file in current location
output_file = 'ALL_PAPER_DATA.fits'
table.write(output_file, format='fits', overwrite=True)
# %%
