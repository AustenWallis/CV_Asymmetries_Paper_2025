# %%
%matplotlib qt
# ------ importing all modules ------ 
# ------ importing all modules ------ 
import numpy as np
import matplotlib.pyplot as plt
# Add path effects import
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 50  # allow up to 50 MB embedded animations
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe
# from pyscript import current_target, document, display
#from matplotlib_pyodide.browser_backend import TimerWasm
import bisect

print('Imported Modules')

# ------ Loading Required Data --------
data_file = "cv_spectra_data.npz"
data = np.load(data_file)
# For .npz, arrays are stored directly
wavelengths_all = data['wavelengths']  # shape (n_runs, n_wl)
fluxes_all = data['fluxes']           # shape (n_runs, n_wl, n_incs)
parameter_table = data['parameter_table'][:,1:7]
run_number = data['run_number']
inclinations = data['inclinations']
original_inclinations = list(inclinations)
original_parameter_table = parameter_table.copy()
grid_slider_list = list(run_number)
print(f"Loaded data from {data_file}")
label_data = np.load('gold_silver_bronze_labels.npz', allow_pickle=True)
run_list_labels = label_data['labels'].item()  # Use .item() to get the dictionary
# replace 10,11,12,13,14 dictionary keys with 20.0,45.0,60.0,72.5,85.0
run_list_labels = {
    20.0: run_list_labels[10],
    45.0: run_list_labels[11],
    60.0: run_list_labels[12],
    72.5: run_list_labels[13],
    85.0: run_list_labels[14]
}
# load precomputed continuum-normalized fluxes
dimensionless_grid_fluxes = data['normalized_fluxes']
print(f"Loaded labels from gold_silver_bronze_labels.npz")


# # ------ Preparing Animations --------
# class Timer(TimerWasm):
#     def __init__(self, interval=None):
#         self._timer = None
#         super().__init__(interval=interval)

y_max = max(np.max(fluxes_all[i]) for i in range(len(run_number))) * 1.3
# fontsize
plt.rcParams.update({'font.size': 12, 'legend.fontsize': 10})
# generate a colormap for each inclination
colors = plt.get_cmap('gist_ncar')([0.85,0.2,0.36,0.6,0.73])
colours = {inc: c for inc, c in zip(inclinations, colors)}
y_max = max(np.max(fluxes_all[i]) for i in range(len(run_number))) * 1.3
continuum_normalised = False # If True, normalise the continuum flux to 1.0
# store permanently-added spectra as tuples: (wavelengths, flux, label, color)
add_grid_plots = []
# counter to assign a new style on each Add
add_plot_iter = 0

linestyles   = ['--', '-.', ':', (0,(3,1,1,1)), (0,(5,2))]

# current sample filter: 'full', 'bronze', 'silver', or 'gold'
sample_selected = 'full'
# handle for "no spectra" message
no_spectra_text = None

def rebuild_active_runs():
    """Rebuild grid_slider_list from frozen params and sample/inclination filters."""
    global grid_slider_list
    # 1) Parameter-based filter
    param_runs = set(run_number)
    for lbl, is_frozen in frozen.items():
        if not is_frozen:
            continue
        idx_p = int(lbl[-1]) - 1
        v     = freeze_sliders[lbl].val
        if idx_p == 1:
            rel = original_parameter_table[:,1] / original_parameter_table[:,0]
            param_runs &= {i for i in param_runs if np.isclose(rel[i], v)}
        else:
            param_runs &= {
                i for i in param_runs
                if original_parameter_table[i, idx_p] == v
            }
    # 2) Sample/inclination filter
    if sample_selected == 'full':
        sample_runs = set(run_number)
    else:
        if len(inclinations) == 1:
            inc = inclinations[0]
            sample_runs = {
                i for i in run_number
                if run_list_labels[inc][i] == sample_selected
            }
        else:
            sample_runs = {
                i for i in run_number
                if any(run_list_labels[inc][i] == sample_selected
                       for inc in inclinations)
            }
    # 3) Intersect and update slider list
    new_runs = sorted(param_runs & sample_runs)
    curr     = grid_slider.val
    grid_slider_list[:] = new_runs
    grid_slider.valstep = grid_slider_list
    if grid_slider_list:
        grid_slider.set_val(curr if curr in grid_slider_list else new_runs[0])


# Local interactive version with slider and buttons
fig, ax = plt.subplots(figsize=(22, 10), dpi=80)
plt.subplots_adjust(left=0.26,bottom=0.2, right=0.75)
# Add permanent rest wavelength line for H-alpha
ax.axvline(6562.8, color='lightgrey', linestyle='--', linewidth=1)
ax.set_xlim(6500, 6625)
ax.set_ylim(0, y_max)
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Flux ($erg/s/cm^2/Å$)')
ax.set_title('H_α CV Spectra Animation')

# Initial artists
lines = []
for i, inc in enumerate(inclinations):
    line, = ax.plot(
        wavelengths_all[0],
        fluxes_all[0][:, i],
        label=f'{inc}°',
        color=colours[inc],
        lw=2
    )
    # add black outline behind the colored line
    line.set_path_effects([
        pe.Stroke(linewidth=line.get_linewidth()+1, foreground='black'),
        pe.Normal()
    ])
    lines.append(line)

# Static legend
ax.legend(loc='upper right')

# Text box background props
props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

# Initial text
params0 = parameter_table[0]
text = ax.text(
    0.02, 1.05,
    '   '.join([
        rf'$\dot{{M}}_{{disk}}={params0[0]:.2e}$',
        rf'$\dot{{M}}_{{wind}}={params0[1]:.2e}$',
        rf'$d={params0[2]:.2f}$',
        rf'$r_{{exp}}={params0[3]:.2f}$',
        rf'$a_{{l}}={params0[4]:.2e}$',
        rf'$a_{{exp}}={params0[5]:.2f}$'
    ]),
    transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props
)

def update_py(frame):
    wl = wavelengths_all[frame]
    fx = fluxes_all[frame]
    # update each line
    for i, line in enumerate(lines):
        line.set_xdata(wl)
        line.set_ydata(fx[:, i])
    # update text
    params = parameter_table[frame]
    text.set_text(
        '   '.join([
            rf'$\dot{{M}}_{{disk}}={params[0]:.2e}$',
            rf'$\dot{{M}}_{{wind}}={params[1]:.2e}$',
            rf'$d={params[2]:.2f}$',
            rf'$r_{{exp}}={params[3]:.2f}$',
            rf'$a_{{l}}={params[4]:.2e}$',
            rf'$a_{{exp}}={params[5]:.2f}$'
        ])
    )
    return lines + [text]

def slider_update(val):
    idx = int(val)
    global sample_selected, grid_slider_list, anim, no_spectra_text
    # if no active runs, pause animation and show message
    if not grid_slider_list:
        anim.running = False
        ax.clear()
        no_spectra_text = ax.text(
            0.5, 0.5,
            'No Available Spectra With These Settings',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=16
        )
        fig.canvas.draw_idle()
        return
    else:
        # remove previous "no spectra" message if present and restart once
        if no_spectra_text:
            no_spectra_text.remove()
            no_spectra_text = None
            anim.running = True
    # determine which inclinations to display for this run
    if sample_selected != 'full':
        display_incs = [inc for inc in inclinations
                        if run_list_labels[inc][idx] == sample_selected]
    else:
        display_incs = inclinations
    wl     = wavelengths_all[idx]
    fx_all = (dimensionless_grid_fluxes[idx]
              if continuum_normalised
              else fluxes_all[idx])
    ax.clear()
    # Re-add rest wavelength line after clearing axes
    ax.axvline(6562.819,
               color='lightgrey',
               linestyle='--',
               linewidth=1,
               label=rf'$\lambda_{0}$')
    # Compute maximum flux among the live animation lines
    # Avoid testing the truth value of a numpy array
    if len(display_incs) > 0:
        dynamic_max = max(
            np.max(fx_all[:, original_inclinations.index(inc)])
            for inc in display_incs
        )
    else:
        dynamic_max = 1.0
    # Compute maximum flux among any permanently added spectra
    if add_grid_plots:
        if continuum_normalised:
            perm_max = max(
                dimensionless_grid_fluxes[run_idx,:,inc_idx].max()
                for run_idx, inc_idx, *_ in add_grid_plots
            )
        else:
            perm_max = max(
                fluxes_all[run_idx,:,inc_idx].max()
                for run_idx, inc_idx, *_ in add_grid_plots
            )
        overall_max = max(dynamic_max, perm_max)
    else:
        overall_max = dynamic_max
    # Compute minimum for normalized case
    if continuum_normalised:
        # compute minimum among live animation lines
        if len(display_incs) > 0:
            dynamic_min = min(
                np.min(fx_all[:, original_inclinations.index(inc)])
                for inc in display_incs
            )
        else:
            dynamic_min = 0.0
        # compute minimum among permanent spectra, normalized
        if add_grid_plots:
            perm_min = min(
                dimensionless_grid_fluxes[run_idx,:,inc_idx].min()
                for run_idx, inc_idx, *_ in add_grid_plots
            )
        else:
            perm_min = dynamic_min
        overall_min = min(dynamic_min, perm_min)
        # set y-limits slightly beyond data range
        bottom = overall_min * 0.95
        top = overall_max * 1.05
        ax.set_xlim(6500, 6625)
        ax.set_ylim(bottom, top)
    else:
        # For unnormalized, set bottom as 95% of the global minimum for a single inclination,
        # or zero when displaying all inclinations.
        if len(display_incs) == 1:
            inc = display_incs[0]
            idx_inc_glob = original_inclinations.index(inc)
            global_min = np.min(fluxes_all[:, :, idx_inc_glob])
            bottom = global_min * 0.95
        else:
            bottom = 0
        ax.set_xlim(6500, 6625)
        ax.set_ylim(bottom, overall_max * 1.2)
    # Axis labels and y-axis formatting
    ax.set_xlabel('Wavelength (Å)')
    if continuum_normalised:
        ax.set_ylabel('Normalized Flux')
        # one decimal place, no scientific offset
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.offsetText.set_visible(False)
    else:
        ax.set_ylabel('Flux ($erg/s/cm^2/Å$)')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax.yaxis.set_offset_position('left')
        ax.yaxis.offsetText.set_x(0)
        ax.yaxis.offsetText.set_y(1.02)
    ax.set_title(f'H_α of CV for Run {run_number[idx]}')
    for i, inc in enumerate(display_incs):
        idx_inc = original_inclinations.index(inc)
        # lookup sample label for this run and inclination
        sample_label = run_list_labels[inc][idx]
        # unpack the single Line2D object from the list
        line2, = ax.plot(
            wl,
            fx_all[:, idx_inc],
            label=f'{inc}° ({sample_label.capitalize()})',
            color=colours[inc],
            linewidth=2,
        )
        line2.set_path_effects([
            pe.Stroke(linewidth=line2.get_linewidth()+1, foreground='black'),
            pe.Normal()
        ])
    params = parameter_table[idx]
    vals = [
        rf'$\dot{{M}}_{{disk}}={params[0]:.2e}$',
        rf'$\dot{{M}}_{{wind}}={params[1]:.2e}$',
        rf'$d={params[2]:.2f}$',
        rf'$⍺={params[3]:.2f}$',
        rf'$l={params[4]:.2e}$',
        rf'$β={params[5]:.2f}$'
    ]
    textstr = '   '.join(vals)
    props = dict(boxstyle='round',
                 facecolor='lightgrey',
                 alpha=0.5)
    ax.text(0.11, 1.07,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='bottom',
            bbox=props)
    # overlay any permanently stored spectra
    for run_idx, inc_idx, label_p, color_p, style_p in add_grid_plots:
        wl_p = wavelengths_all[run_idx]
        data_line = (dimensionless_grid_fluxes[run_idx,:,inc_idx]
                    if continuum_normalised
                    else fluxes_all[run_idx,:,inc_idx])
        line3, = ax.plot(
            wl_p,
            data_line,
            linestyle=style_p,
            linewidth=2,
            color=color_p,
            label=label_p
        )
        line3.set_path_effects([
            pe.Stroke(linewidth=line3.get_linewidth()+1, foreground='black'),
            pe.Normal()
        ])
    ax.legend(bbox_to_anchor=(1.02, 1.0),
              loc='upper left',
              ncol=1)
    fig.canvas.draw_idle()

def animation_step(frame):
    if anim.running:
        # current slider value
        curr = grid_slider.val
        # find its position in the active list
        try:
            pos = grid_slider_list.index(curr)
        except ValueError:
            pos = 0
        # advance to next active run
        next_pos = (pos + 1) % len(grid_slider_list)
        next_val = grid_slider_list[next_pos]
        # update slider and plot
        grid_slider.set_val(next_val)
        slider_update(next_val)

def play_pause(event):
    """Pause play button function to stop animation on command"""
    if anim.running:
        anim.running = False
        slider_update(grid_slider.val)
    else:
        anim.running = True

def left_button_func(_) -> None:
    """Left button to iterate the grid to the previous active run"""
    anim.running = False
    curr = int(grid_slider.val)
    try:
        pos = grid_slider_list.index(curr)
    except ValueError:
        pos = 0
    prev_pos = (pos - 1) % len(grid_slider_list)
    prev_val = grid_slider_list[prev_pos]
    grid_slider.set_val(prev_val)
    slider_update(prev_val)

def right_button_func(_) -> None:
    """Right button to iterate the grid to the next active run"""
    anim.running = False
    curr = int(grid_slider.val)
    try:
        pos = grid_slider_list.index(curr)
    except ValueError:
        pos = 0
    next_pos = (pos + 1) % len(grid_slider_list)
    next_val = grid_slider_list[next_pos]
    grid_slider.set_val(next_val)
    slider_update(next_val)

def animation_speed(event):
    """Slider function to change timestep of animation in ms"""
    anim._interval = animation_slider.val 

def inclination_angle(event):
    """Slider function to change the spectral inclination that's plotted"""
    global inclinations
    if inclination_slider.val == -1.0:
        inclinations = [20.0,45.0,60.0,72.5,85.0]
    else:
        inclinations = [inclination_slider.val]
    rebuild_active_runs()
    slider_update(grid_slider.val)

def slider_pause(event, *args, **kwargs):
    """Pauses/unpauses the animation on slider mouse click."""
    # Identifying the slider's location on the figure
    (xm,ym),(xM,yM) = grid_slider.label.clipbox.get_points()
    if xm < event.x < xM and ym < event.y < yM:
        anim.running = False # if clicking slider, pause



def freeze_parameter_update(label):
    """The function freezes the parameter checkbox selected by the user."""
    
    global grid_slider_list

    if label[:5] != 'param': # only to covert tickbox names to the dict keys
        key_list = list(parameter_labels)
        val_list = list(parameter_names)
        position = val_list.index(label) # finding the position of the label
        label = key_list[position] # finding the key of the label
    
    frozen[label] = not frozen[label] # toggle True/False from checkbox

    rebuild_active_runs()
    slider_update(grid_slider.val)


def freeze_slider_update(event):
    """This function updates the frozen slider parameter value for plotting."""
    for label in freeze_sliders.keys():
        if freeze_sliders[label].val != freeze_slider_history[label] and frozen[label]:
            # update history to current (new) slider value
            freeze_slider_history[label] = freeze_sliders[label].val
            break  # only one slider changes at a time
    # rebuild the active run list using all filters
    rebuild_active_runs()
    slider_update(grid_slider.val)

# Freezing parameters of the unique combinations with a checkbox
freeze_axis = fig.add_axes([0.02, 0.35, 0.185, 0.5]) # Checkbox shape
fig.text(0.06, 0.86, 'Tick Box to Fix a Parameter', fontsize=12, ha='left')
parameter_names = ['Disk Accretion Rate',
                    'Wind Mass Loss Rate',
                    'd Collimation',
                    '⍺ Wind Loss Rate exponent',
                    'l Acceleration Length Scale',
                    'β Acceleration Exponent']
parameter_labels = ['param1', 'param2', 'param3', 'param4', 'param5', 'param6']

parameter_checkboxes = CheckButtons(freeze_axis,labels=parameter_names)
parameter_checkboxes.on_clicked(freeze_parameter_update)

# Initialisation
freeze_sliders = {} # Slider Class for each frozen parameter
freeze_slider_history = {} # Dict recording previous slider values
# This is to avoid conflicting values on different sliders
freeze_slider_axis = [] # Adding frozen parameter sliders axes
frozen = {} # True/False dictionary for each parameter if frozen
frozen_store = {} # Dictionary to store frozen parameter indexes

# Sliders axes correctly positioned next to corresponding checkboxs, 
# enabled for a dynamic number of parameters emulated
for i in range(len(parameter_names)):
    freeze_slider_axis.append(fig.add_axes(
        [0.065, (0.815-(0.50/(len(parameter_names)+1))) - (0.50/(len(parameter_names)+1))*i, 0.09, 0.03]))

# Setting the slider for each possible parameter to be fixed.
for i in range(len(freeze_slider_axis)):
    # get the unique sorted values for parameter i
    unique_vals = sorted(set(parameter_table[:, i]))
    if i == 1:
        unique_vals = sorted(set([0.03,0.1,0.3]))
    freeze_sliders[f"param{i+1}"] = Slider(
        freeze_slider_axis[i],
        '',
        unique_vals[0],
        unique_vals[-1],
        valinit=unique_vals[0],
        valstep=unique_vals,
        initcolor='none',
        handle_style={'facecolor':'black', 'size':7}
    )
    frozen[f"param{i+1}"] = False # Start unfrozen, +1 to match params
    freeze_slider_history[f"param{i+1}"] = freeze_sliders[f"param{i+1}"].val
    freeze_sliders[f"param{i+1}"].on_changed(freeze_slider_update)

# draw red tick marks for each frozen‐parameter slider (limited to slider rail)
for s in freeze_sliders.values():
    # draw ticks spanning 40% to 60% of the axis height
    for step in s.valstep:
        s.ax.axvline(
            step,
            ymin=0.25,
            ymax=0.75,
            color='red',
            linewidth=1,
            zorder=10,
            clip_on=False,
            alpha=0.5
        )

# Slider and buttons
ax_slider = fig.add_axes([0.1, 0.05, 0.65, 0.03])
grid_slider = Slider(ax_slider,
                     'Run', 
                     0, 
                     len(run_number) - 1,
                     valinit=0,
                     valstep=grid_slider_list
                    )
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

# Animation slider 
animation_axis = fig.add_axes([0.67, 0.095, 0.08, 0.03])
animation_slider = Slider(animation_axis,
                          '<- Faster |Animation Speed| Slower ->  ',
                          10,
                          1000,
                          valinit=200,
                          valstep=10,
                          initcolor='none',
                          handle_style={'facecolor':'black'})
animation_slider.valtext.set_visible(False)
# If slider changes, call function to update plot to the new grid point.
animation_slider.on_changed(animation_speed)

# Inclination slider 
inclination_axis = fig.add_axes([0.41, 0.095, 0.08, 0.03])
inclination_slider = Slider(inclination_axis,
                          '<- Lower |Inclination (°)| Higher ->  ',
                          -1.0,
                          85.0,
                          valinit=-1.0,
                          valstep=[-1.0,20.0,45.0,60.0,72.5,85.0],
                          initcolor='none',
                          handle_style={'facecolor':'black'})
# If slider changes, call function to update plot to the new grid point.
inclination_slider.on_changed(inclination_angle)
# draw red tick marks for inclination slider
for step in inclination_slider.valstep:
    inclination_slider.ax.axvline(
        step,
        ymin=0.25,
        ymax=0.75,
        color='red',
        linewidth=1,
        zorder=10,
        clip_on=False,
        alpha=0.5
    )

# Sample slider 
labels = ['Full', 'Bronze', 'Silver', 'Gold']
idx2label = {i: lbl for i, lbl in enumerate(labels)}

def sample_choice(val):
    global sample_selected
    sample_selected = idx2label[int(val)].lower()
    sample_slider.valtext.set_text(idx2label[int(val)])
    rebuild_active_runs()
    slider_update(grid_slider.val)

sample_axis = fig.add_axes([0.065, 0.28, 0.09, 0.03])
sample_slider = Slider(
    sample_axis,
    '',
    0,
    len(labels) - 1,
    valinit=0,
    valstep=1,
    valfmt='%d',
    initcolor='none',
    handle_style={'facecolor':'black'}
)
sample_axis.set_xticks(list(idx2label.keys()))
sample_axis.set_xticklabels(labels)
sample_slider.valtext.set_text(idx2label[0])
# Add labeled text box above the slider
fig.text(
    0.065, 0.307,
    'Sirocco Sample',
    ha='left',
    va='bottom',
    fontsize=10
)
# If slider changes, call function to update plot to the new grid point.
sample_slider.on_changed(sample_choice)
# draw red tick marks for sample slider
for step in idx2label.keys():
    sample_slider.ax.axvline(
        step,
        ymin=0.25,
        ymax=0.75,
        color='red',
        linewidth=1,
        zorder=10,
        clip_on=False,
        alpha=0.5
    )

def add_grid_spectrum(event):
    """Capture and store the current grid spectra for all inclinations."""
    global add_grid_plots, add_plot_iter
    add_plot_iter += 1
    style = linestyles[(add_plot_iter - 1) % len(linestyles)]

    run_idx = int(grid_slider.val)
    for inc in inclinations:
        inc_idx = original_inclinations.index(inc)
        glabel = []
        for i, v in enumerate(parameter_table[run_idx]):
            # if v in scientific notation magnitudes above 1e4 or below 1e-3,
            if i == 0 or i == 1 or i == 4:
                glabel.append('{:.2e}'.format(v))
            else:
                glabel.append('{:.2f}'.format(v))
        # include sample label for this run & inclination
        sample_label = run_list_labels[inc][run_idx].capitalize()
        #glabel  = ['{:.2e}'.format(v) for v in parameter_table[run_idx]]
        label = f"{inc}° ({sample_label}): " + ", ".join(glabel)
        color = colours[inc]
        # store run index and inclination index instead of raw data
        add_grid_plots.append((run_idx, inc_idx, label, color, style))
    slider_update(run_idx)
    
def clear_grid_spectrum(event):
    """Clear all stored spectra."""
    global add_grid_plots
    add_grid_plots = []
    slider_update(int(grid_slider.val))

def normalise_grid_spectrum(event):
    """Toggle between raw and precomputed normalized flux."""
    global continuum_normalised
    continuum_normalised = not continuum_normalised
    slider_update(int(grid_slider.val))


 # label for the Add/Clear section
fig.text(
    0.06, 0.24,
    'Plot a Spectrum',
    ha='left',
    va='bottom',
    fontsize=12
)
# Plot a Spectrum – Add / Clear / Normalise buttons
add_spectrum_ax = fig.add_axes([0.06, 0.19, 0.05, 0.04])
add_btn = Button(add_spectrum_ax, 'Add', color='lightgrey', hovercolor='0.975')
add_btn.on_clicked(add_grid_spectrum)

clear_spectrum_ax = fig.add_axes([0.12, 0.19, 0.05, 0.04])
clear_btn = Button(clear_spectrum_ax, 'Clear', color='lightgrey', hovercolor='0.975')
clear_btn.on_clicked(clear_grid_spectrum)

normalise_spectrum_ax = fig.add_axes([0.04, 0.925, 0.15, 0.04])
norm_btn = Button(normalise_spectrum_ax, '(Un)/Normalise Spectra', color='lightgrey', hovercolor='0.975')
norm_btn.on_clicked(normalise_grid_spectrum)

# Mouse Events
fig.canvas.mpl_connect('button_press_event', slider_pause)

anim = FuncAnimation(fig, 
                     animation_step,
                     frames=len(run_number), 
                     interval=animation_slider.val , 
                     #event_source=Timer(interval=300)
                    )
anim.running = True
# Embed into PyScript environment
# html = anim.to_jshtml()
# container = document.getElementById("animation-container")
# container.innerHTML = html
# --- Annotate missing toolbar icons with labels and down‐arrows ---
tool_labels = ['Home', 'Back', 'Forward', 'Pan/Zoom', 'Zoom Rect']
# approximate x positions in figure fraction coordinates (adjust if needed)
xs = [0.398, 0.419, 0.441, 0.465, 0.487]
for x, lbl in zip(xs, tool_labels):
    fig.text(
        x, 0.02,
        f'{lbl} ↓',
        transform=fig.transFigure,
        ha='center',
        va='bottom',
        fontsize=8,
        color='black'
    )

plt.show()
# %%
