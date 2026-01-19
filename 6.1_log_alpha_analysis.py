import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import sem, t
from mne.stats import permutation_cluster_test

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
subject_alpha_hc = []
subject_alpha_sz = []

# Define parameters for alpha band time-frequency analysis
alpha_freqs = np.arange(8, 15, 1)   # 8–14 Hz range
n_cycles = alpha_freqs / 2          # Number of cycles per frequency (controls time-frequency resolution)
method = "morlet"                   # Use Morlet wavelets for TFR computation
decim = 7                           # Downsampling factor to speed up processing
tmin, tmax = -1.0, 0.6              # Time window

# Select a subset of posterior electrodes based on literature
selected_channels = ["Pz", "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6", "O2", "PO8"] # Tarasi et al. + right side channels Limbach et al.
# selected_channels = None  # Activate if all channels should be used

all_epochs = list() # For concatenating all epochs later

# Loop over all subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to preprocessed epoched file
    subject_data = os.path.join(data_path, subject_id, "final_frame", f"{subject_id}_task-supratyp-alpha-demogr-epo.fif")

    # Load epoched data (already cropped to -1.0 to 0.6 s)
    epochs = mne.read_epochs(subject_data, preload=True).crop(-1.0, 0.6)

    # Identify group (HC or SZ) from metadata
    is_hc = epochs.metadata["is_hc"].iloc[0]

    # Compute trial-level TFR (no averaging yet)
    epochs = epochs.pick(picks=selected_channels)
    alpha_subject = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1,
                                           return_itc=False, average=False) # no average to compute absolute alpha power (not evoked)

    # Average across trials; result: subject-level average TFR
    absolute_alpha_subject = alpha_subject.average()

    # Extract mean power over all time, frequencies, and channels: shape = (n_times,)
    averaged_power_subject = absolute_alpha_subject.get_data(tmin=-1.0, tmax=0.6).mean(axis=(0,1))

    # Store results by group
    if is_hc:
        subject_alpha_hc.append(averaged_power_subject)
    else:
        subject_alpha_sz.append(averaged_power_subject)

    all_epochs.append(epochs)

# Concatenate all individuals subject epochs
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore")

# Convert group lists to NumPy arrays: shape = (n_subjects, n_times)
subject_alpha_hc = np.array(subject_alpha_hc)
subject_alpha_sz = np.array(subject_alpha_sz)

# Log-transform to normalize power values
subject_alpha_hc_log = np.log(subject_alpha_hc)
subject_alpha_sz_log = np.log(subject_alpha_sz)

# Calculate group-level means; shape = (n_timepoints)
mean_alpha_hc_log = subject_alpha_hc_log.mean(axis=0)
mean_alpha_sz_log = subject_alpha_sz_log.mean(axis=0)

# Calculate subject-wise averages: shape = (n_subjects,)
mean_subject_alpha_hc_log = subject_alpha_hc_log.mean(axis=1)
mean_subject_alpha_sz_log = subject_alpha_sz_log.mean(axis=1)

# Create time axis for plots (same length as number of timepoints)
times = np.linspace(tmin, tmax, subject_alpha_hc_log.shape[1])

# --- Plot individual and mean time courses ---
plt.figure(figsize=(10, 5))
# Plot individual subject curves
for subj_curve in subject_alpha_hc_log:
    plt.plot(times, subj_curve, color="teal", alpha=0.4, linewidth=1)  # HC Subjects
for subj_curve in subject_alpha_sz_log:
    plt.plot(times, subj_curve, color="darkorange", alpha=0.4, linewidth=1)  # SC Subjects
# Plot group means
plt.plot(times, mean_alpha_hc_log, label="HC (Mean)", color="teal", linewidth=3)
plt.plot(times, mean_alpha_sz_log, label="SZ (Mean)", color="darkorange", linewidth=3)
# Add labels and legend
plt.xlabel("Time (s)")
plt.ylabel("Log Alpha Power (μV²)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/log_alpha_over_time_split_none.png", dpi=400)
plt.show()

# Basic t-test on subject-level means
t_stat, p_value = stats.ttest_ind(mean_subject_alpha_hc_log, mean_subject_alpha_sz_log)
print(f"T-Statistik: {t_stat}, P-Wert: {p_value}")

# --- Non-parametric cluster-based permutation test for group difference (HC vs. SZ; time resolved)
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([subject_alpha_hc_log, subject_alpha_sz_log],
                                                                 out_type="mask",
                                                                 n_permutations=10000, seed=40,
                                                                 tail=0, n_jobs=-1)

# Print details about each cluster
for i_c, cluster in enumerate(clusters):
    sl = cluster[0]
    times_in_cluster = times[sl]
    t_values_in_cluster = T_obs[sl]
    p_val = cluster_p_values[i_c]

    print(f"Cluster {i_c+1}: {times_in_cluster[0]:.3f} s – {times_in_cluster[-1]:.3f} s "
          f"(p = {p_val:.4f}, max T = {t_values_in_cluster.max():.2f}, min T = {t_values_in_cluster.min():.2f})")

# Plot: Group means, CI, and T-values with significant clusters
# Group sizes
n_hc = subject_alpha_hc_log.shape[0]
n_sz = subject_alpha_sz_log.shape[0]
# Center group means for visualization
grand_mean = (mean_alpha_hc_log.mean() + mean_alpha_sz_log.mean()) / 2
mean_hc_centered = mean_alpha_hc_log - grand_mean
mean_sz_centered = mean_alpha_sz_log - grand_mean
diff = mean_hc_centered - mean_sz_centered

# Standard error and 95% CI
se_hc = sem(subject_alpha_hc_log, axis=0)
se_sz = sem(subject_alpha_sz_log, axis=0)
ci_mult_hc = t.ppf(0.975, df=n_hc - 1) # this ist CI, if SEM should be plotted ci_mult_hc = 1
ci_mult_sz = t.ppf(0.975, df=n_sz - 1) # this ist CI, if SEM should be plotted ci_mult_sz = 1
ci_hc = se_hc * ci_mult_hc
ci_sz = se_sz * ci_mult_sz

# Start plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis (Power)
ax1.plot(times, mean_hc_centered, label="HC with 95%-CI", color="teal", linestyle="--")
ax1.fill_between(times, mean_hc_centered - ci_hc, mean_hc_centered + ci_hc, color="teal", alpha=0.15)
ax1.plot(times, mean_sz_centered, label="SZ with 95%-CI", color="darkorange", linestyle="--")
ax1.fill_between(times, mean_sz_centered - ci_sz, mean_sz_centered + ci_sz, color="darkorange", alpha=0.15)
ax1.plot(times, diff, label="HC - SZ (diff)", color="black")
ax1.set_ylabel("Log Alpha Power (centered)", fontsize=14)
ax1.set_xlabel("Time (s)", fontsize=14)
ax1.set_axisbelow(True)
ax1.grid(False)

# Right axis (T-values)
ax2 = ax1.twinx()
ax2.plot(times, T_obs, color="gray", label="T-values")
ax2.set_ylabel("T-values", fontsize=14)
ax2.grid(False)

# Align y-axis ticks for T-values
yticks_left = ax1.get_yticks()
ymin_left, ymax_left = ax1.get_ylim()
ymin_right, ymax_right = min(T_obs), max(T_obs)
yticks_right = np.interp(yticks_left, (ymin_left, ymax_left), (ymin_right, ymax_right))
yticklabels_right = [f"{yt:.1f}" for yt in yticks_right]
yticklabels_right[0] = ""
yticklabels_right[-1] = ""
ax2.set_yticks(yticks_right)
ax2.set_yticklabels(yticklabels_right)
ax2.set_ylim(ymin_right, ymax_right)

# Highlight significant clusters
for i_c, c in enumerate(clusters):
    c = c[0]
    color = "red" if cluster_p_values[i_c] <= 0.05 else "gray"
    ax1.axvspan(times[c.start], times[c.stop - 1], color=color, alpha=0.3)

ax1.axhline(0, color='black', linewidth=1, alpha=0.6)
ax1.axvline(0, color='black', linewidth=1, alpha=0.6)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/permutation_cluster_test_no_legend.svg")
plt.show()
