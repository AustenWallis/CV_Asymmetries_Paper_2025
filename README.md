# CV_Asymmetries_Paper_2025

This is a open research store for my CV asymmetries paper 2025 {link to be added}. 
The Emission_Line_Asymmetries.py script is not designed to be run straight as a python file. I operate the script like a juypter notebook by using vscode, Ipykernal and magic commands (# %%). This is why the python script is sectioned. The user should run the script-notebook in a similar format. This code isn't indented for production, only for paper reproducibility.  

ChatGPT summary of the paper below (o1 model):
🌌 Main Objective:

To evaluate how well asymmetric line profile diagnostic plots detect outflows and identify line characteristics in cataclysmic variable (CV) systems using 729 wind-only simulated spectra from the Sirocco radiative transfer code.

⸻

🔑 Key Findings:
	•	P-Cygni profiles are classic outflow indicators, but not all outflows exhibit them.
	•	Asymmetric diagnostic plots, which assess deviations from Gaussian profiles in spectral lines (mainly Hα), have limited reliability in isolation.
	•	Masking window choice (velocity range around the line core) significantly affects results; a fixed window causes sensitivity and inconsistencies.
	•	A more robust method is to scale the masking region with the line’s full-width half maximum (FWHM), improving the diagnostic’s consistency.
	•	Inclination angle plays a big role — higher inclinations lead to double-peaked profiles and larger equivalent width excesses.
	•	Developed a linear regressor to predict emission measure and equivalent width based on physical parameters of the system — useful for observers.

⸻

📉 Implications:

The study shows that outflow detection via line asymmetries needs to be used alongside other diagnostics, especially due to the high sensitivity to user input (e.g., masking range). Simulations like those from Sirocco are critical for interpreting observational data more robustly.

⸻

📝 Longer Description

This paper investigates how asymmetric emission-line profiles can serve as indicators of outflows in cataclysmic variables (CVs), where a white dwarf accretes material from a disk. The authors note that while classic P-Cygni signatures are strong evidence for winds, many CVs produce subtler features that are harder to classify. By focusing on line asymmetries—especially in Hα—they aim to refine existing diagnostic plots and reduce dependence on user-chosen velocity masking windows that can skew results.

To do this, they generate 729 purely wind-based spectra using Sirocco, a Monte Carlo radiative transfer code configured for CV outflows. Physical parameters such as disk mass accretion rate, wind geometry, and acceleration laws are varied systematically to explore how different setups shape emission lines. The team then applies “asymmetric diagnostic plots,” adapting the masking window to the full-width half maximum of each line so that outflow signatures, rather than arbitrary velocity bounds, drive the measurement of asymmetries.

Their findings show that wind-driven emission alone can populate nearly any part of the diagnostic plot, confirming that no single quadrant automatically implies a strong outflow. However, the revised, adaptive approach makes asymmetry signals clearer, particularly for lower-inclination systems with single-peaked lines. The authors conclude that comparing observed spectra to these advanced simulations—and employing new tools like linear regressors to link physical parameters and emission strengths—offers the most reliable way to decode how disk winds shape CV emission lines.

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

### Sirocco Model Access:
- You can request grids from the Authors or generate your own with Sirocco. Sirocco (formally Python) v87f was used. Bare in mind if generating your own data, Sirocco uses Monte Carlo processes, therefore your results may vary verys slightly. A python script (dynamic_grid.py) is available in this repository to generate the .pf files required to run matching sirocco models.

### Cueno Data Access: 
Although we do have this data, this should be requested from the original authors. 
