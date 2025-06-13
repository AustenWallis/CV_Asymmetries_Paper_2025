# %%
# spec_to_array.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
# runs requiring alternate continuum intervals
mixed_runs = [77,79,158,160,240,241,320,322,414,425,468,477,478,
              479,480,481,482,483,563,565,567,569,580,639,640,
              641,642,643,644,645,646,667,668,669,712,713,
              720,721,722,723,724,725,726,727]

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

# ------------------- USER CONFIGURATION ------------------- #
path_to_grids = "../H_alpha_models"
data_file = "cv_spectra_data.npz"
run_number = np.arange(0, 729, dtype=np.int32)
inclinations = np.array([20, 45, 60, 72.5, 85], dtype=np.float32)
# ----------------------------------------------------------- #

if not os.path.isfile(data_file):
    print("Building compressed .npz archive from scratch...")
    # Load all spectra into lists
    norms = []
    wls = []
    fxs = []
    for run in tqdm(run_number, desc="Loading spectra"):
        spec_file = os.path.join(path_to_grids, f"rerun{run}.spec")
        wl = np.loadtxt(spec_file, usecols=(1,), skiprows=81).astype(np.float32)
        fx = np.loadtxt(
            spec_file,
            usecols=np.arange(10, 10 + len(inclinations)),
            skiprows=81
        ).astype(np.float32)
        # compute continuum-normalized flux for this run on full spectrum
        norm_full = np.zeros_like(fx)
        intervals = (6300,6325,6775,6800) if run in mixed_runs else (6450,6475,6625,6650)
        for k in range(fx.shape[1]):
            spec_full = fx[:, k]
            cont, _, _, _ = continuum(wl, spec_full, intervals)
            norm_full[:, k] = spec_full / cont

            # use a half-open interval to get exactly 400 points
        mask = (wl >= 6450.2) & (wl < 6650)
        wls.append(wl[mask])
        fxs.append(fx[mask])
        norms.append(norm_full[mask])
    # Stack into arrays: shape (n_runs, n_wl, n_incs)
    wavelengths_arr = np.stack(wls, axis=0)
    fluxes_arr = np.stack(fxs, axis=0)
    norm_fluxes_arr = np.stack(norms, axis=0)
    # Load parameter table
    param_table = np.genfromtxt(
        os.path.join(path_to_grids, "Grid_runs_logfile.csv"),
        delimiter=',', skip_header=1
    ).astype(np.float32)
    # Save compressed .npz
    np.savez_compressed(
        data_file,
        wavelengths=wavelengths_arr,
        fluxes=fluxes_arr,
        normalized_fluxes=norm_fluxes_arr,
        parameter_table=param_table,
        run_number=run_number,
        inclinations=inclinations
    )
    print(f"Data archive saved to {data_file}")

# Your code here

data = np.load(data_file)
# For .npz, arrays are stored directly
wavelengths_all = data['wavelengths']  # shape (n_runs, n_wl)
fluxes_all = data['fluxes']           # shape (n_runs, n_wl, n_incs)
parameter_table = data['parameter_table']
run_number = data['run_number']
inclinations = data['inclinations']
print(f"Loaded data from {data_file}")

# %%
