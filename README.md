# CV_Asymmetries_Paper_2024

This is a open research store for my CV asymmetries paper 2024 {link to be added}. 
The Emission_Line_Asymmetries.py script is not designed to be run straight as a python file. I operate the script like a juypter notebook by using vscode, Ipykernal and magic commands (# %%). This is why the python script is sectioned. The user should run the script-notebook in a similar format. This code isn't indented for production, only for paper reproducibility.  

ChatGPT summary of the paper below:

TODO Add when completed. 


### The repository currently includes:
- Emission Line Asymmetries:
    - Seeing excess equivalent widths either side of emission lines to prove or disprove asymmetries from wind signatures.
  
- Rebinning_Spec_Tot_vs_Spec_Spectra:
    - A script to rebin the ionisation cycles spectra vs the spectral cycles spectra for PYTHON. This was due to a discrepancy
        identified by Ed on ~Jan 24. Shown to have no (or negligible) effects on the grids I am using.

### Sirocco Model Access:
- The H_alpha models are hosted under the Release_Ha_grid_spec_files directory. 
- If you are internal Southampton, click the link where you can download more grids that the script has been tested with when building up to the release grid. Diverts to a OneDrive Repo. Place the 'Grids' directory within the 'Emission_Line_Asymmetries' directory. Due to Soton policy. This link is only shareable with Soton staff. Request for a different link if external. 
[Grids](https://sotonac-my.sharepoint.com/:f:/g/personal/agww1g17_soton_ac_uk/EkjOSNgu3WJOvSFfVXUsyyoBMVbc6yMg08ke7vCYuVzZWQ?e=i5Vt1m)
- You can also request grids from the Authors or generate your own with Sirocco. Sirocco (formally Python) v87f was used. Bare in mind if generating your own data, Sirocco uses Monte Carlo processes, therefore your results may vary verys slightly. A python script (dynamic_grid.py) is available in this repository to generate the .pf files required to run matching sirocco models.
