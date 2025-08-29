# CV Asymmetries Paper 2025
[![DOI](https://zenodo.org/badge/767447082.svg)](https://doi.org/10.5281/zenodo.15257396)

PYSCRIPT WEB-BASED INTERACTIVE TOOL TO VIEW SPECTRA: https://austenwallis.pyscriptapps.com/h-alpha-grid-inspector/latest/

### Sirocco Model Access:
- 729 spectra files can be found under the `H_alpha_models` folder. Sirocco (formally Python) v87f was used. Bear in mind that if generating your own data, Sirocco uses Monte Carlo processes; therefore, your results may vary slightly. A Python script (dynamic_grid.py) is available in this repository to generate the .pf files required to run matching Sirocco models. The template pf files used to generate the models is under 'Template_pf_files'.

### All Plotting Data:
- This can be found in `COPY_OF_ALL_PAPER_DATA` folder. Contains information such as sirocco model parameters, EWs, FWHMs Exceeses, wavelength and flux (normalised and unnormalised) data, line luminosities, sample labels, 

This is an open research store for my CV asymmetries paper 2025. 
The Emission_Line_Asymmetries.py script is not designed to be run straight as a Python file. I operate the script like a Jupyter notebook by using VSCode, IPyKernel and magic commands (# %%). This is why the Python script is sectioned. The user should run the script/notebook in a similar format. This code isn't intended for use or production as a tool to run the analysis, only for paper reproducibility.

### Abstract
Blueshifted absorption is the classic spectroscopic signature of an accretion disc wind in X-ray binaries and cataclysmic variables (CVs). However, outflows can also create pure emission lines, especially at optical wavelengths. Therefore, developing other outflow diagnostics for these types of lines is worthwhile. With this in mind, we construct a systematic grid of 3645 synthetic wind-formed $\mathrm{H\alpha}$ line profiles for CVs with the radiative transfer code `sirocco`. Our grid yields a variety of line shapes: symmetric, asymmetric, single‐ to quadruple‐peaked, and even P-Cygni profiles. About 20 per cent of these lines -- our 'Gold' sample -- have strengths and widths consistent with observations. We use this grid to test a recently proposed method for identifying wind-formed emission lines based on deviations in the wing profile shape: the 'excess equivalent width diagnostic diagram'. We find that our Gold sample can preferentially populate the suggested 'wind regions' of this diagram. However, the method is highly sensitive to the adopted definition of the line profile 'wing'. Hence, we propose a refined definition based on the full-width at half maximum to improve the interpretability of the diagnostic diagram. Furthermore, we define an approximate scaling relation for the strengths of wind-formed CV emission lines in terms of the outflow parameters. This relation provides a fast way to assess whether -- and what kind of -- outflow can produce an observed emission line. All our wind-based models are open-source and we provide an easy-to-use web-based tool to browse our full set of $\mathrm{H\alpha}$ spectral profiles.

### The repository currently includes:
- Asymmetries_code_normalised.py/unnormalised.py
  - These are the scripts that generated the excess EW data, normalised is for spectra whos continuum flux is 1. 
  - The script is designed to be run like a notebook with magic commands # %%. Hence, the step naming. The process happens twice due to several wider bound spectra and ragged arrays.
  - Data source is required from authors. Too large for a Github repo. 
  
- Paper_plots_v2.py
  - A script solely for generating the paper figures. All labels as such. 

- Rebinning_Spec_Tot_vs_Spec_Spectra:
    - A script to rebin the ionisation cycles spectra vs the spectral cycles spectra for PYTHON. This was due to a discrepancy
        identified by Ed on ~Jan 24. Shown to have no (or negligible) effects on the grids I am using.
- Figure_4_Panel_D_Emissivities:
  	- A script to find the emissivities, velocitis, and other model parameters such as electron density etc. Helpful in diagnosing more interesting spectra. 

### Cueno Data Access: 
Although we do have this data, this should be requested from the original authors. 
