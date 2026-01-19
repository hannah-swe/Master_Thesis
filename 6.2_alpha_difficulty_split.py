import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorsys

# Ensure proper visualization backend
plt.ion()
plt.style.use("fast")
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)


data_path = "/Volumes/SSK Drive/Data/derivatives"   # Define data path
sub_ids = sorted(os.listdir(data_path))     # List and sort all subject folders

# Instantiate epoch lists to store single subject epochs in
alpha_power_subjects = []
subject_alpha_hard = []
subject_alpha_easy = []

# Define parameters for alpha band time-frequency analysis
alpha_freqs = np.arange(8, 15, 1)   # 8–14 Hz range
n_cycles = alpha_freqs / 2          # Number of cycles per frequency (controls time-frequency resolution)
method = "morlet"                   # Use Morlet wavelets for TFR computation
decim = 7                           # Downsampling factor to speed up processing
tmin, tmax = -1.0, 1.5              # Time window

# Select a subset of posterior electrodes based on literature
selected_channels = ["Pz", "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6", "O2", "PO8"] # Tarasi et al. + right side channels Limbach et al.

all_epochs = list() # For concatenating all epochs later

# Loop over all subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to preprocessed epoched file
    subject_data = os.path.join(data_path, subject_id, "epoching", f"{subject_id}_task-supratyp-epo.fif")

    # Load subject's epochs and crop to time window
    epochs = mne.read_epochs(subject_data, preload=True).crop(-1.0, 1.5)

    # Filter epochs: include only trials with phase coherence < 85% (aka exclude catch trials)
    epochs = epochs[epochs.metadata["phaseCoherence"] < 85.00]

    # Rename and binarize 'difficulty' column based on 'phaseCoherence'
    epochs.metadata = epochs.metadata.rename(columns={"phaseCoherence": "difficulty"})
    epochs.metadata["difficulty"] = epochs.metadata["difficulty"].replace({22.5: 0, 17.5: 1})

    # Compute time-frequency representation (TFR) for each trial
    epochs = epochs.pick(picks=selected_channels)
    alpha_subject = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1,
                                           return_itc=False,
                                           average=False)  # no average to compute absolute alpha power (not evoked)

    # Create masks for each difficulty condition
    is_easy = epochs.metadata["difficulty"] == 0
    is_hard = epochs.metadata["difficulty"] == 1

    # Extract alpha power data for each condition
    data_easy = alpha_subject.data[is_easy]  # shape: (n_easy_trials, n_channels, n_freqs, n_times)
    data_hard = alpha_subject.data[is_hard]  # shape: (n_hard_trials, n_channels, n_freqs, n_times)

    # Compute average power over trials, channels, frequencies
    mean_alpha_easy = data_easy.mean(axis=(0, 1, 2))    # shape: (n_times,)
    mean_alpha_hard = data_hard.mean(axis=(0, 1, 2))    # shape: (n_times,)

    # Store subject-level alpha power per condition
    subject_alpha_easy.append(mean_alpha_easy)
    subject_alpha_hard.append(mean_alpha_hard)

    all_epochs.append(epochs)

# Concatenate all subject epochs
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore")

# Convert list to NumPy array: shape (n_subjects, n_times)
subject_alpha_easy = np.array(subject_alpha_easy)
subject_alpha_hard = np.array(subject_alpha_hard)

# Log-transform to normalize data
subject_alpha_easy = np.log(subject_alpha_easy)
subject_alpha_hard = np.log(subject_alpha_hard)

# Compute condition means and standard errors across subjects; shape: (n_times,)
mean_alpha_easy = subject_alpha_easy.mean(axis=0)
mean_alpha_hard = subject_alpha_hard.mean(axis=0)
sem_alpha_easy = subject_alpha_easy.std(axis=0, ddof=1) / np.sqrt(subject_alpha_easy.shape[0])
sem_alpha_hard = subject_alpha_hard.std(axis=0, ddof=1) / np.sqrt(subject_alpha_hard.shape[0])

# Create time axis for plotting
times = np.linspace(tmin, tmax, subject_alpha_hard.shape[1])

# --- Plot condition averages over time ith optional SEM shading ---
plt.figure(figsize=(10, 5))
plt.plot(times, mean_alpha_hard, label="Hard", linewidth=1, color="tomato")
# plt.fill_between(times, mean_alpha_hard - sem_alpha_hard, mean_alpha_hard + sem_alpha_hard, alpha=0.2, color="tomato")
plt.plot(times, mean_alpha_easy, label="Easy", linewidth=1, color="cornflowerblue")
# plt.fill_between(times, mean_alpha_easy - sem_alpha_easy, mean_alpha_easy + sem_alpha_easy, alpha=0.2, color="cornflowerblue")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Log Alpha Power (μV²)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(0, color="k", linewidth=1)
plt.grid(False)
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/alpha_over_time_difficulty_split.svg")
plt.show()

# --- Plot with subject-level lines ---
n_subjects = len(subject_alpha_easy)
assert n_subjects == len(subject_alpha_hard)
# Get n_subjects different colors
hues = np.linspace(0, 1, n_subjects, endpoint=False)
subject_colors = [colorsys.hls_to_rgb(h, 0.5, 1.0) for h in hues]
plt.figure(figsize=(10, 6))
# Plot individual subject curves in both conditions
for i, (easy_curve, hard_curve) in enumerate(zip(subject_alpha_easy, subject_alpha_hard)):
    base_color = subject_colors[i]
    plt.plot(times, easy_curve, color=base_color, alpha=0.3, linewidth=0.8) # Easy condition with less saturation
    plt.plot(times, hard_curve, color=base_color, alpha=0.9, linewidth=0.8) # Hard condition with higher saturation
# Plot group means in black (hard) and gray (easy)
plt.plot(times, mean_alpha_hard, label="Hard", color="black", linewidth=2.5)
plt.plot(times, mean_alpha_easy, label="Easy", color="gray", linewidth=2.5)
plt.xlabel("Time (s)")
plt.ylabel("Log Alpha Power (μV²)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/alpha_over_time_difficulty_split_subjectlines.png", dpi=400)
plt.tight_layout()
plt.show()
