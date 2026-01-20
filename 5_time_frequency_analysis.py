import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")

data_path = "/Volumes/SSK Drive/Data/derivatives"   # Define data path
sub_ids = sorted(os.listdir(data_path))     # List and sort all subject folders

# Instantiate epoch lists to store single subject epochs in
alpha_power_subjects = []

# Define parameters for alpha band time-frequency analysis
tfr_freqs = np.arange(1, 41, 1)     # 1-40 Hz range
n_cycles_tfr = tfr_freqs / 2        # Number of cycles per frequency (controls time-frequency resolution)
method = "morlet"                   # Use Morlet wavelets for TFR computation
decim = 7                           # Downsampling factor to speed up processing
tmin, tmax = -0.8, 0.4              # Time window

# Selected subset of posterior electrodes based on literature
selected_channels = ["Pz", "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6", "O2", "PO8"] # Tarasi et al. + right side channels Limbach et al.
# selected_channels = None  # Activate if all channels should be used

all_epochs = list() # For concatenating all epochs later

# Iterate over all subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to preprocessed epoched file
    subject_data = os.path.join(data_path, subject_id, "epoching", f"{subject_id}_task-supratyp-epo.fif")

    # Load epoched data (already cropped to -1.0 to 0.6 s)
    epochs = mne.read_epochs(subject_data, preload=True).crop(-1.0, 0.6)

    # Identify group (HC or SZ) from subject number
    components = subject_id.split("_")[2]
    if components[-2] == "0":
        subject = components[-1]
    else:
        subject = components[-2:]
    epochs.metadata['subject'] = subject
    epochs.metadata['is_hc'] = epochs.metadata["subject"].astype(int) < 51
    del epochs.metadata["fileNum"]

    all_epochs.append(epochs)

# Concatenate all individuals subject epochs
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore")

# Compute TFR (no averaging yet)
epochs = epochs.pick(picks=selected_channels)
whole_tfr = epochs.compute_tfr(method=method, freqs=tfr_freqs, n_cycles=n_cycles_tfr, decim=decim, n_jobs=-1,
                               return_itc=False, average=False)

# Crop TFR to beautify
whole_tfr = whole_tfr.crop(-0.8, 0.4)

# --- PLOT TFR ---
# Plot tfr with analysis window, stim-onset and colorbar
figs = whole_tfr.average().plot(combine="mean", vlim=(0, None), fmin=5, show=False)
# Handle list or single figure
if isinstance(figs, list):
    fig = figs[0]
else:
    fig = figs

ax = fig.axes[0]
cbar_ax = fig.axes[-1]
ax.grid(False)
# Get pre-stimulus analysis window to mark in the TFR
rect = patches.Rectangle(
    (-0.5, 8), 0.4, 6,  # Width = 0.4 (timepoints in s), Height: = 6 (frequencies)
    linewidth=1.5,
    edgecolor="black",
    facecolor="none",
    linestyle="--"
)
ax.add_patch(rect)

# Add vertical line for stim onset
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.2)

# Add colorbar unit
cbar = fig.axes[-1]
cbar.set_ylabel("Power (μV²)", fontsize=14, rotation=90, labelpad=15)
ax.tick_params(axis='both', labelsize=14)
cbar_ax.tick_params(labelsize=14)
ax.set_xlabel(ax.get_xlabel(), fontsize=14)
ax.set_ylabel(ax.get_ylabel(), fontsize=14)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/TFR_all.svg")
plt.show()


# Compute TFR plots split by group (HC vs. SZ)
tfr_hc = whole_tfr["is_hc==True"]
tfr_sc = whole_tfr["is_hc==False"]
avrgd_tfr_hc = tfr_hc.average() # Average over all HC subjects
avrgd_tfr_sc = tfr_sc.average() # Average over all SZ subjects
avg_diff = avrgd_tfr_hc - avrgd_tfr_sc # Get difference HC - SZ
# Plot TFR split by group and plot difference
avg_diff.plot(combine="mean", title="Difference: TFR HC - TFR SC")
avrgd_tfr_hc.plot(combine="mean", title="TFR für HC Subjects")
avrgd_tfr_sc.plot(combine="mean", title="TFR für SC Subjects")


# --- COMPUTE TFR SPLIT BY DIFFICULTY (EASY vs. HARD)
# Relevant phase coherence values
valid_values = [17.5, 22.5]

# Get TFR for values of easy and hard trials only, exclude catch trials
valid_mask = whole_tfr.metadata["phaseCoherence"].isin(valid_values)
whole_tfr_clean = whole_tfr[valid_mask]
easy_mask = whole_tfr_clean.metadata["phaseCoherence"] == 22.5
hard_mask = whole_tfr_clean.metadata["phaseCoherence"] == 17.5
easy_idx = np.where(easy_mask)[0]
hard_idx = np.where(hard_mask)[0]
tfr_easy = whole_tfr_clean[easy_idx]
tfr_hard = whole_tfr_clean[hard_idx]

# Average over trials and get difference (easy - hard)
avrgd_tfr_easy = tfr_easy.average()
avrgd_tfr_hard = tfr_hard.average()
avg_difficulty_diff = avrgd_tfr_easy - avrgd_tfr_hard

# Plot TFR for easy and hard trials
avrgd_tfr_easy.plot(combine="mean", title="TFR for easy trials", vlim=(0, None))
avrgd_tfr_hard.plot(combine="mean", title="TFR for hard trials", vlim=(0, None))

# --- FINAL PLOT TFR DIFFERENCE OF EASY AND HARD TRIALS ---
figs = avg_difficulty_diff.plot(combine="mean",show=False)
# Handle list or single figure
if isinstance(figs, list):
    fig = figs[0]  # Nimm die erste Figure
else:
    fig = figs

ax = fig.axes[0]
cbar_ax = fig.axes[-1]
ax.grid(False)
# Get pre-stimulus analysis window to mark in the TFR
rect = patches.Rectangle(
    (-0.5, 8), 0.4, 6,  # Width = 0.4 (timepoints in s), Height: = 6 (frequencies)
    linewidth=1.5,
    edgecolor="black",
    facecolor="none",
    linestyle="--"
)
ax.add_patch(rect)

# Add vertical line for stim onset
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.2)

# Add colorbar unit
cbar = fig.axes[-1]
cbar.set_ylabel("Power (μV²)", fontsize=14, rotation=90, labelpad=15)
ax.tick_params(axis='both', labelsize=14)
cbar_ax.tick_params(labelsize=14)
ax.set_xlabel(ax.get_xlabel(), fontsize=14)
ax.set_ylabel(ax.get_ylabel(), fontsize=14)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/TFR_difference_difficulty.svg")
plt.show()