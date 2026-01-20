import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure proper visualization backend
plt.ion()
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")

data_path = "/Volumes/SSK Drive/Data/derivatives/"  # Define data path
sub_ids = sorted(os.listdir(data_path))     # List and sort all subject folders

# Define parameters for alpha band time-frequency analysis
alpha_freqs = np.arange(8, 15, 1)   # 8–14 Hz range
n_cycles = alpha_freqs / 2          # Number of cycles per frequency (controls time-frequency resolution)
method = "morlet"                   # Use Morlet wavelets for TFR computation
decim = 7                           # Downsampling factor to speed up processing
tmin, tmax = -0.5, -0.1             # Time window for pre-stimulus baseline power analysis

# Select a subset of posterior electrodes based on literature
selected_channels = ["Pz", "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6", "O2", "PO8"] # Tarasi et al. + right side channels Limbach et al.
# selected_channels = None # Activate if all channels should be used

# Loop over all subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Construct full file path to the subject's epoched data
    subject_data = os.path.join(data_path, subject_id, "epoching", f"{subject_id}_task-supratyp-epo.fif")

    # Load epoched data and crop to shorter time window
    epochs = mne.read_epochs(subject_data, preload=True).crop(-1.0, 0.6)

    # Compute TFR on each trial, not averaged (absolute power)
    epochs = epochs.pick(picks=selected_channels)
    alpha_subject = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1,
                                       return_itc=False, average=False)  # no average to compute absolute alpha power (not evoked)

    # Crop TFR results to the defined pre-stimulus interval
    alpha_subject = alpha_subject.crop(tmin, tmax)

    # Compute average alpha power per trial (over channels, freqs, time); result: one value per trial
    alpha_power_trials = alpha_subject.data.mean(axis=(1, 2, 3))

    # Add trial-wise alpha power to the epochs' metadata
    epochs.metadata["alpha_power"] = alpha_power_trials

    # Compute log-transformed alpha power (to normalize skewed power values)
    epochs.metadata['log_power'] = np.log(epochs.metadata['alpha_power'])

    # Compute the mean log(alpha power) across all trials (subject-level); result: one value per subject
    mean_log_power = epochs.metadata['log_power'].mean()
    epochs.metadata['log_power_mean'] = mean_log_power

    # Compute trial-wise deviations from the subject's average (centered log power)
    epochs.metadata['log_power_deviation'] = epochs.metadata['log_power'] - mean_log_power

    # Create a new subfolder to store alpha power-enhanced epochs and save epochs with alpha power
    try:
        os.makedirs(f"{data_path}/{subject_id}/alpha_power")
    except FileExistsError:
        print("EEG derivatives alpha_power directory already exists")
    epochs.save(f"{data_path}/{subject_id}/alpha_power/{subject_id}_task-supratyp-alpha-epo.fif",
                overwrite=True)
    del epochs
