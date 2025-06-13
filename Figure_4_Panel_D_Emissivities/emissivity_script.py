# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from astropy.table import Table

plot_real_coords = True  # Set to False to plot in grid coordinates

# %%
############################################################################
print("Plotting Macro-Atom Emissivity")
############################################################################
%matplotlib inline
run = 'run652' #652 #685
path = f'matom_linelum_{run}/linelums_{run}_lev2.txt'
# reading in the file with pandas
master_data = pd.read_csv(f"{run}.master.txt",delim_whitespace=True)
data_emiss = pd.read_csv(path, delim_whitespace=True, skiprows=4, header=0)

emiss_data = data_emiss[['LowerLev001']].copy()
emiss_data['x'] = master_data['x']
emiss_data['z'] = master_data['z']
emiss_data['i'] = master_data['i']
emiss_data['j'] = master_data['j']

# set any values below a threshold to 1e8
emiss_data['x'] = emiss_data['x'].replace(0, 1e8)  # Avoid log(0)
emiss_data['z'] = emiss_data['z'].replace(0, 1e8)  # Avoid log(0)
#x and z are values for the cell centres, i and j are the grid indices
if plot_real_coords:
    emiss_matrix = emiss_data.pivot_table(index='z', columns='x', values='LowerLev001')
    x_centers = np.array(sorted(emiss_matrix.columns))
    z_centers = np.array(sorted(emiss_matrix.index))

    # Mask for values below a threshold
    emiss_matrix[emiss_matrix < 1e24] = 0  # Set values below threshold to 0
    emiss_matrix = emiss_matrix.replace(0, np.nan)  # Replace 0 with NaN for log scaling

    # Plot using pcolormesh on real coordinates
    plt.figure(figsize=(8, 6))

    pcm = plt.pcolormesh(
        x_centers,
        z_centers,
        emiss_matrix.values,
        norm=LogNorm(vmin=1e24),  # Set a minimum value for log scaling
        cmap="turbo",
        shading="auto"
    )
    cbar = plt.colorbar(pcm, label='Total Emissivity (erg/s)')
    plt.axvline(x=7.25182e8, color='blue', linestyle='--', alpha=0.5, label='Disc_RadMin')
    plt.axvline(x=2.17555e10, color='red', linestyle='--', alpha=0.5, label='Disc_RadMax')
    plt.axvline(x=1.00e+12, color='black', linestyle='--', alpha=0.5, label='Wind_RadMax')
    plt.xlabel('x (cm)')
    plt.ylabel('z (cm)')
    plt.title('Total Macro-Atom Emissivity per Cell (Real Coordinates)')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(x_centers.min(), x_centers.max())
    plt.ylim(z_centers.min(), z_centers.max())
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()


emiss_matrix2 = data_emiss.pivot_table(index='j', columns='i', values='LowerLev001')
# Mask for values below a threshold
emiss_matrix2[emiss_matrix2 < 1e24] = 0  # Set values below threshold to 0
emiss_matrix2 = emiss_matrix2.replace(0, np.nan)  # Replace 0 with NaN for log scaling

plt.figure(figsize=(8, 6))
plt.imshow(
    emiss_matrix2.values,
    origin='lower',
    aspect='equal', 
    cmap='turbo',
    norm=LogNorm(vmin=1e24),
    extent=(
        emiss_matrix2.columns.min(),  # j_min
        emiss_matrix2.columns.max(),  # j_max
        emiss_matrix2.index.min(),    # i_min
        emiss_matrix2.index.max()     # i_max
    )
)
#plt.axvline(x=2.17555e10, color="red", linestyle="--", alpha=0.5, label="Disc_RadMax")
plt.colorbar(label='Total Emissivity (erg/s)')
plt.ylabel('j (grid index)')
plt.xlabel('i (grid index)')
plt.title('Total Macro-Atom Emissivity per Cell')
plt.grid(False)
plt.tight_layout()
plt.show()

#%%
##################################################################################
print("Plotting Projected Rotational Velocity")
##################################################################################
run = 'run652' #652 #685
path = f'matom_linelum_{run}/linelums_{run}_lev2.txt'

ha_df = pd.read_csv(path, delim_whitespace=True, skiprows=4, header=0)
master_data = pd.read_csv(f"{run}.master.txt",delim_whitespace=True)
#Merge velocity data with the line‐luminosity data on (i, j)
vel_data = master_data[['i', 'j', 'v_x', 'v_y']].copy()

#Join linelums data (which has i, j) with vel_data to get per‐cell velocities
merged = pd.merge(ha_df,vel_data,on=['i', 'j'],how='left')

#Treat v_y as the azimuthal (rotational) speed in a 2D (x,z) run
merged['v_phi'] = merged['v_y']

#Project with inclination i_inc
i_inc = 20
#Convert projected velocity from cm/s to km/s by dividing by 1e5
merged['v_proj'] = (merged['v_phi']/ 1e5 * np.sin(np.radians(i_inc))) 

# Pivot the projected velocity onto (x, z) for plotting
vel_matrix = merged.pivot(index='z', columns='x', values='v_proj')

# reading in the file with pandas
ha_df = pd.read_csv(path, delim_whitespace=True, skiprows=4, header=0)

# 1) Pivot on the true cell-corner coordinates using x and z
parameter = "LowerLev001"
pivoted = ha_df.pivot(index="j", columns="i", values=parameter)

# 2) Build edges directly from x and z corners
x_edges = np.array(sorted(pivoted.columns))
z_edges = np.array(sorted(pivoted.index))

# Also capture the full coordinate range for axis limits
xcoords_full = np.sort(ha_df['i'].unique())
zcoords_full = np.sort(ha_df['j'].unique())

# Compute cell widths (use the last delta if spacing varies)
dx = np.diff(x_edges)
dx_last = dx[-1] if not np.allclose(dx, dx[0]) else dx[0]
x_edges = np.append(x_edges, x_edges[-1] + dx_last)

dz = np.diff(z_edges)
dz_last = dz[-1] if not np.allclose(dz, dz[0]) else dz[0]
z_edges = np.append(z_edges, z_edges[-1] + dz_last)

# Pivot the projected velocity onto (i, j) for plotting
velocity_matrix_cells = merged.pivot(index='j', columns='i', values='v_proj')
i_centers = np.array(sorted(velocity_matrix_cells.columns))
j_centers = np.array(sorted(velocity_matrix_cells.index))

# --- build a boolean mask of wind cells from the emissivity map ---
# True where the emissivity matrix is *not* NaN, False elsewhere.
wind_mask = ~np.isnan(emiss_matrix2.values)

# 3/4) Plot projected rotational velocity (v_proj) with pcolormesh on log-log axes
plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(
    x_edges,
    z_edges,
    vel_matrix.values,
    cmap="gist_ncar",
    shading="auto"
)
cbar = plt.colorbar(pcm, label="Projected v_phi (km/s)")
#plt.axvline(x=2.17555e10, color="red", linestyle="--", alpha=0.5, label="Disc_RadMax")
# --- add a white outline showing the wind footprint ---
# The contour is drawn at 0.5 on the boolean mask (True==1, False==0).
plt.contour(
    i_centers,
    j_centers,
    wind_mask.astype(int),
    levels=[0.5],
    colors='white',
    linewidths=1.2
)
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("i")
plt.ylabel("j")
plt.title("Projected Rotational Velocity (v_proj)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
##################################################################################
print("Plotting Projected Streamline Velocity")
##################################################################################
%matplotlib qt
run = 'run652' #652 #685
path = f'matom_linelum_{run}/linelums_{run}_lev2.txt'

ha_df = pd.read_csv(path, delim_whitespace=True, skiprows=4, header=0)
master_data = pd.read_csv(f"{run}.master.txt",delim_whitespace=True)

# calculating projected velocities
inclination = 20  # inclination angle in degrees
inc_radians = np.radians(inclination)
velocity_cols = master_data[['v_x', 'v_y', 'v_z']].copy()
for key in velocity_cols.columns:
    # Convert velocities from cm/s to km/s
    velocity_cols[key] = velocity_cols[key] / 1e5  # Convert cm/s to km/s

# Add i, j coordinates to velocity data
velocity_cols['i'] = master_data['i']
velocity_cols['j'] = master_data['j']
# Add x, z coordinates to velocity data
velocity_cols['x'] = master_data['x']
velocity_cols['z'] = master_data['z']
# all y coordinates are 0 in a 2D run
velocity_cols['y'] = np.zeros(len(velocity_cols), dtype=float)


# Compute cylindrical radius in x–z plane
r = np.sqrt(velocity_cols['x']**2 + velocity_cols['z']**2)

# 1) Radial outflow component and its LOS projection
v_r = (velocity_cols['v_x'] * velocity_cols['x'] + velocity_cols['v_z'] * velocity_cols['z']) / r
velocity_cols['v_radial_proj'] = v_r * (
    (velocity_cols['x'] / r) * np.sin(inc_radians)
    + (velocity_cols['z'] / r) * np.cos(inc_radians)
)

# 2) Azimuthal (rotational) component about the y-axis and its LOS projection
v_phi = (-velocity_cols['z'] * velocity_cols['v_x'] + velocity_cols['x'] * velocity_cols['v_z']) / r
velocity_cols['v_rot_proj'] = v_phi * (velocity_cols['x'] / r) * np.cos(inc_radians)

# 3) Select which projected velocity to plot:
#    use ['v_radial_proj'] for radial, or ['v_rot_proj'] for rotation
velocity_cols['v_proj'] = velocity_cols['v_radial_proj']


# Set any values below a threshold to 1e8
velocity_cols['x'] = velocity_cols['x'].replace(0, 1e8)  # Avoid log(0)
velocity_cols['z'] = velocity_cols['z'].replace(0, 1e8)  # Avoid log(0)


if plot_real_coords:
    velocity_matrix_real = velocity_cols.pivot(index='z', columns='x', values='v_proj')
    x_centers = np.array(sorted(velocity_matrix_real.columns))
    z_centers = np.array(sorted(velocity_matrix_real.index))

    plt.figure(figsize=(8, 6))
    pcm = plt.pcolormesh(
        x_centers,
        z_centers,
        velocity_matrix_real.values,
        cmap="gist_ncar",
        shading="auto"
        )
    
    cbar = plt.colorbar(pcm, label='Projected radial velocity (km/s)')
    plt.axvline(x=7.25182e8, color='blue', linestyle='--', alpha=0.5, label='Disc_RadMin')
    plt.axvline(x=2.17555e10, color='red', linestyle='--', alpha=0.5, label='Disc_RadMax')
    plt.axvline(x=1.00e+12, color='black', linestyle='--', alpha=0.5, label='Wind_RadMax')
    plt.xlabel('x (cm)')
    plt.ylabel('z (cm)')
    plt.title('Projected Radial Velocity (v_proj) (Real Coordinates)')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(x_centers[0], x_centers[-1])
    plt.ylim(z_centers[0], z_centers[-1])
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()

# Pivot the projected velocity onto (i, j) for plotting
velocity_matrix_cells = velocity_cols.pivot(index='j', columns='i', values='v_proj')
i_centers = np.array(sorted(velocity_matrix_cells.columns))
j_centers = np.array(sorted(velocity_matrix_cells.index))

# --- build a boolean mask of wind cells from the emissivity map ---
# True where the emissivity matrix is *not* NaN, False elsewhere.
wind_mask = ~np.isnan(emiss_matrix2.values)

plt.figure(figsize=(8, 6))
imshow = plt.imshow(velocity_matrix_cells.values,
              origin='lower',
                aspect='equal',
                cmap='gist_ncar',
                extent=(
                    i_centers.min(),  # j_min
                    i_centers.max(),  # j_max
                    j_centers.min(),  # i_min
                    j_centers.max()   # i_max
                )
)
plt.colorbar(imshow, label='Projected radial velocity (km/s)')
# --- add a white outline showing the wind footprint ---
# The contour is drawn at 0.5 on the boolean mask (True==1, False==0).
plt.contour(
    i_centers,
    j_centers,
    wind_mask.astype(int),
    levels=[0.5],
    colors='white',
    linewidths=1.2
)

plt.xlabel('i (grid index)')
plt.ylabel('j (grid index)')
plt.title(f'Projected Radial Velocity (v_proj) per Cell\nwith Wind Outline - {inclination}°')
plt.grid(False)
plt.tight_layout()
plt.show()


# %%
###########################################################################
print("Plotting a Selected Parameter in Real Coordinates")
###########################################################################
"""Here you can change index and column strings to z/j and x/i
to plot any parameter in real coordinates for index coordinates.
If in real, you can uncomment the lines for plotting
"""

run = 'run652' #652 #685
master_df = pd.read_csv(
    f"{run}.master.txt",
    delim_whitespace=True
)
# 1) Pivot on the true cell-corner coordinates using x and z
parameter = "ne"
pivoted = master_df.pivot(index="j", columns="i", values=parameter)

# 2) Build edges directly from x and z corners
x_edges = np.array(sorted(pivoted.columns))
z_edges = np.array(sorted(pivoted.index))

# Also capture the full coordinate range for axis limits
xcoords_full = np.sort(master_df['x'].unique())
zcoords_full = np.sort(master_df['z'].unique())

# Compute cell widths (use the last delta if spacing varies)
dx = np.diff(x_edges)
dx_last = dx[-1] if not np.allclose(dx, dx[0]) else dx[0]
x_edges = np.append(x_edges, x_edges[-1] + dx_last)

dz = np.diff(z_edges)
dz_last = dz[-1] if not np.allclose(dz, dz[0]) else dz[0]
z_edges = np.append(z_edges, z_edges[-1] + dz_last)

# 3) Convert to log scale, masking zeros
pivoted[pivoted < 1e-10] = 0  # Set values below threshold to 0
pivoted = pivoted.replace(0, np.nan)
log_data = np.log10(pivoted.values)

# # Apply a threshold: values below 1e-10 become 0, then replace 0 with NaN
# log_data[log_data < 1e0] = 0
# log_data = log_data.replace(0, np.nan)

# 4) Plot with pcolormesh on log-log axes
plt.figure(figsize=(8,6))
pcm = plt.pcolormesh(
    x_edges,
    z_edges,
    log_data,
    cmap="turbo",
    shading="auto"
)
cbar = plt.colorbar(pcm, label=f"log10({parameter})")
# plt.axvline(x=7.25182e8, color='blue', linestyle='--', alpha=0.5, label='Disc_RadMin')
# plt.axvline(x=2.17555e10, color='red', linestyle='--', alpha=0.5, label='Disc_RadMax')
# plt.axvline(x=1.00e+12, color='black', linestyle='--', alpha=0.5, label='Wind_RadMax')

# plt.xscale("log")
# plt.yscale("log")
# plt.xlim(0, 6e10)
# plt.ylim(0, 6e10)
# Set axis limits to the smallest strictly positive coordinate and max
x_min_plot = xcoords_full[xcoords_full > 0].min()
z_min_plot = zcoords_full[zcoords_full > 0].min()
# plt.xlim(x_min_plot, xcoords_full.max())
# plt.ylim(z_min_plot, zcoords_full.max())

plt.xlabel("x (cm)")
plt.ylabel("z (cm)")
plt.title(f"Log10({parameter}) (Real Coordinates using corners)")
plt.tight_layout()
plt.show()


#%%
###########################################################################
print("Converting Sirocco Master Text to FITS")
###########################################################################
# read the master txt
diag = pd.read_csv(f"{run}.master.txt", delim_whitespace=True)

# convert to an Astropy Table and write out a FITS file
tbl = Table.from_pandas(diag)
tbl.write(f"{run}.master.fits", format="fits", overwrite=True)















































# %%
######################################################################
# OLD CODE - FOR REFERENCE
######################################################################

# # using the txt file
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# # reading in the file with pandas
# file = 'run652.txt'
# data = pd.read_csv(file, delim_whitespace=True, header=2)
# # remove row 0 
# data = data.drop(index=0)
# # only include rows where variable is matom_emiss
# data = data[data['Variable'] == 'matom_emiss']

# # Immediately after loading:
# for col in data.columns:
#     if col.startswith("MacLev"):
#         data[col] = data[col].astype(float)
# # Ensure i and j are numeric for correct sorting
# data['i'] = data['i'].astype(int)
# data['j'] = data['j'].astype(int)

# # Create a list of all MacLev columns
# maclev_cols = [c for c in data.columns if c.startswith("MacLev")]

# # Sum across those columns to form a single "TotalEmiss" column
# data['TotalEmiss'] = data[maclev_cols].sum(axis=1)
# # 4) Pivot onto (i,j)
# emiss_matrix = data.pivot_table(index='j', columns='i', values='TotalEmiss', fill_value=0)
# emiss_matrix = emiss_matrix.sort_index(axis=0).sort_index(axis=1)

# # 5) Plot
# plt.figure(figsize=(10, 8))
# plt.imshow(
#     emiss_matrix.values,
#     origin='lower',
#     aspect='equal',
#     cmap='turbo',
#     #norm=LogNorm(vmin=1e16, vmax=1e30),
#     extent=(
#         emiss_matrix.columns.min(),  # j_min
#         emiss_matrix.columns.max(),  # j_max
#         emiss_matrix.index.min(),    # i_min
#         emiss_matrix.index.max()     # i_max
#     )
# )
# plt.colorbar(label='Total Emissivity (erg/s)')
# plt.ylabel('j (grid index)')
# plt.xlabel('i (grid index)')
# plt.title('Total Macro-Atom Emissivity per Cell')
# plt.grid(False)
# plt.tight_layout()
# plt.show()




# #Merge velocity data with the line‐luminosity data on (i, j)
# vel_data = master_data[['i', 'j', 'v_x', 'v_y']].copy()

# #Join linelums data (which has i, j) with vel_data to get per‐cell velocities
# merged = pd.merge(ha_df,vel_data,on=['i', 'j'],how='left')

# #Treat v_y as the azimuthal (rotational) speed in a 2D (x,z) run
# merged['v_phi'] = merged['v_y']

# #Project with inclination i_inc
# i_inc = 90
# #Convert projected velocity from cm/s to km/s by dividing by 1e5
# merged['v_proj'] = (merged['v_phi']/ 1e5 * np.sin(np.radians(i_inc))) 

# # Pivot the projected velocity onto (x, z) for plotting
# vel_matrix = merged.pivot(index='z', columns='x', values='v_proj')

# # reading in the file with pandas
# ha_df = pd.read_csv(path, delim_whitespace=True, skiprows=4, header=0)

# # 1) Pivot on the true cell-corner coordinates using x and z
# parameter = "LowerLev001"
# pivoted = ha_df.pivot(index="j", columns="i", values=parameter)

# # 2) Build edges directly from x and z corners
# x_edges = np.array(sorted(pivoted.columns))
# z_edges = np.array(sorted(pivoted.index))

# # Also capture the full coordinate range for axis limits
# xcoords_full = np.sort(ha_df['i'].unique())
# zcoords_full = np.sort(ha_df['j'].unique())

# # Compute cell widths (use the last delta if spacing varies)
# dx = np.diff(x_edges)
# dx_last = dx[-1] if not np.allclose(dx, dx[0]) else dx[0]
# x_edges = np.append(x_edges, x_edges[-1] + dx_last)

# dz = np.diff(z_edges)
# dz_last = dz[-1] if not np.allclose(dz, dz[0]) else dz[0]
# z_edges = np.append(z_edges, z_edges[-1] + dz_last)

# # 3/4) Plot projected rotational velocity (v_proj) with pcolormesh on log-log axes
# plt.figure(figsize=(8, 6))
# pcm = plt.pcolormesh(
#     x_edges,
#     z_edges,
#     vel_matrix.values,
#     cmap="gist_ncar",
#     shading="auto"
# )
# cbar = plt.colorbar(pcm, label="Projected v_phi (km/s)")
# #plt.axvline(x=2.17555e10, color="red", linestyle="--", alpha=0.5, label="Disc_RadMax")

# # plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel("i")
# plt.ylabel("j")
# plt.title("Projected Rotational Velocity (v_proj)")
# plt.legend()
# plt.tight_layout()
# plt.show()




# #############################################################################
# print("Plotting Projected Keplerian Velocity")
# #############################################################################
# TESTING PURPOSES ONLY - DOESN"T CONSERVE ALL STREAMLINES BUT THROUGH Z AXIS
# run = 'run652'
# path = f'matom_linelum_{run}/linelums_{run}_lev2.txt'

# ha_df = pd.read_csv(path, delim_whitespace=True, skiprows=4, header=0)
# vel_data_pure = ha_df[['i','j','x','z']].copy()

# # --- override with pure Keplerian rotation for testing ---
# G_cgs = 6.674e-8             # gravitational constant in cgs
# M_wd = 0.8 * 1.989e33        # WD mass = 0.8 M_sun in grams
# r_min = 7.25e8
# # in 2D, the cylindrical radius is r (cm)
# vel_data_pure['r'] = vel_data_pure['x'].abs().replace(0, 1e-20)

# # set v_phi = sqrt(G M / r)
# vel_data_pure['v_phi'] = np.sqrt(G_cgs * M_wd / vel_data_pure['r'])

# i_inc = 20.0
# vel_data_pure['v_proj'] = (vel_data_pure['v_phi'] * np.sin(np.radians(i_inc))) / 1e5

# vel_matrix = vel_data_pure.pivot(index='j', columns='i', values='v_proj')

# plt.figure(figsize=(8, 6))
# plt.imshow(
#     vel_matrix.values,
#     origin='lower',
#     aspect='equal',
#     cmap='gist_ncar'
# )
# plt.colorbar(label='Projected Rotational Velocity (km/s)')
# plt.xlabel('i (grid index)')
# plt.ylabel('j (grid index)')
# plt.title('Projected Keplerian v_proj Map')
# plt.tight_layout()
# plt.show()



# ###########################################################################
# print("Plotting Projected Keplerian Velocity in (r_phys, j) Space # DONT TRUST YET")
# ###########################################################################

# run = 'run685'
# path = f'matom_linelum_{run}/linelums_{run}_lev2.txt'

# ha_df = pd.read_csv(path, delim_whitespace=True, skiprows=4)
# diag = pd.read_csv(f"{run}.master.txt", delim_whitespace=True)

# # Rename the diag radial coordinate so it doesn’t collide with ha_df’s 'x'
# diag = diag.rename(columns={'x': 'r_phys'})

# # Merge Hα data with diag to get the true cylindrical radius 'r_phys'
# ha_vel = ha_df.merge(
#     diag[['i','j','r_phys']],
#     on=['i','j'],
#     how='left'
# )

# # --- override with pure Keplerian rotation for testing ---
# G_cgs = 6.674e-8             # gravitational constant in cgs
# M_wd = 0.8 * 1.989e33        # WD mass = 0.8 M_sun in grams
# r_min = 7.25e8
# # in 2D, the cylindrical radius is r_phys (cm)
# ha_vel['r_phys'] = ha_vel['r_phys'].abs().replace(0, 1e-20)
# # set v_phi = sqrt(G M / r)
# ha_vel['v_phi'] = np.sqrt(G_cgs * M_wd / ha_vel['r_phys'])

# i_inc = 90.0
# ha_vel['v_proj'] = (ha_vel['v_phi'] * np.sin(np.radians(i_inc))) / 1e5
# ha_vel = ha_vel[ ha_vel['r_phys'] > r_min ]
# # Pivot by j vs. physical radius r_phys
# vel_matrix = ha_vel.pivot_table(
#     index='j',
#     columns='r_phys',
#     values='v_proj',
#     fill_value=0
# ).sort_index(axis=0).sort_index(axis=1)

# # Build radial edges from sorted unique r_phys values for plotting
# r_vals = np.array(vel_matrix.columns)
# dr = np.diff(r_vals)
# dr_last = dr[-1] if not np.allclose(dr, dr[0]) else dr[0]
# r_edges = np.append(r_vals, r_vals[-1] + dr_last)

# # Compute z_edges from j index spacing (unchanged)
# j_vals = np.array(vel_matrix.index)
# dj = np.diff(j_vals)
# dj_last = dj[-1] if not np.allclose(dj, dj[0]) else dj[0]
# z_edges = np.append(j_vals, j_vals[-1] + dj_last)

# # Plot projected velocity in (r_phys, j) space
# plt.figure(figsize=(8, 6))
# pcm = plt.pcolormesh(
#     r_edges,
#     z_edges,
#     vel_matrix.values,
#     cmap='gist_ncar',
#     shading='auto'
# )
# plt.xscale('log')
# plt.yscale('linear')
# plt.colorbar(pcm, label='Projected Rotational Velocity (km/s)')
# plt.xlabel('r (cm)')
# plt.ylabel('j (grid index)')
# plt.title('Projected Keplerian v_proj vs. Physical Radius')
# plt.tight_layout()
# plt.show()