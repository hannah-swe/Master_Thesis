import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mne.stats import permutation_cluster_1samp_test
from mne.stats import permutation_cluster_test
from scipy.stats import sem, t

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)

data_path = Path("/Volumes/SSK Drive/Data/derivatives") # Define data path
settings_path = "/Volumes/SSK Drive/Settings/"  # Define path for montage setting
sub_ids = sorted([d for d in os.listdir(data_path) if os.path.isdir(data_path / d)])    # List and sort all subject folders

# Define ERP time window and channel selection
tmin_erp, tmax_erp = -0.2, 1.0
selected_channels = None  # Set to list of channels to restrict, or None to include all

# Initialize containers for evoked data
subject_evokeds = []
subject_evokeds_easy = []   # easy trials
subject_evokeds_hard = []   # hard trials
subject_evokeds_diff = []   # difference of easy and hard trials
subject_evokeds_HC = []     # HC (healthy controls)
subject_evokeds_SZ = []     # SZ (schizophrenics)

# Load custom montage once
easycap_montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")

# Loop over all subjects
for subject_id in sub_ids[11:12]:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Define path to preprocessed epochs
    epochs_file = data_path / subject_id / "epoching" / f"{subject_id}_task-supratyp-epo.fif"
    if not epochs_file.exists():
        print(f"Warning: File not found for subject {subject_id}")
        continue

    # Load epochs and apply montage
    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs.set_montage(easycap_montage)

    # Determine group membership from subject ID
    subject_num = int(subject_id.split("_P")[-1])
    group = "HC" if subject_num < 51 else "SZ"

    # Determine channels to include (or use all if not specified)
    if selected_channels is None:
        picks = None
    else:
        picks = mne.pick_channels(epochs.info["ch_names"], include=selected_channels)

    # Check for the relevant event ("20")
    if "20" not in epochs.event_id:
        print(f"Warning: Condition '20' not found for subject {subject_id}")
        continue

    # Average all trials of event "20"
    epochs_condition = epochs["20"]
    evoked_all = epochs_condition.average()
    evoked_all.crop(tmin=tmin_erp, tmax=tmax_erp)
    subject_evokeds.append(evoked_all)

    # Save result split by group
    if group == "HC":
        subject_evokeds_HC.append(evoked_all)
    else:
        subject_evokeds_SZ.append(evoked_all)

    # Segment by task difficulty using metadata (phaseCoherence)
    meta = epochs_condition.metadata
    if meta is None or "phaseCoherence" not in meta.columns:
        print(f"Warning: Metadata with 'phaseCoherence' missing for subject {subject_id}")
        continue

    easy_epochs = epochs_condition[meta["phaseCoherence"] == 22.5]
    hard_epochs = epochs_condition[meta["phaseCoherence"] == 17.5]

    if len(easy_epochs) > 0:
        evoked_easy = easy_epochs.average()
        evoked_easy.crop(tmin=tmin_erp, tmax=tmax_erp)
        subject_evokeds_easy.append(evoked_easy)
    else:
        print(f"Warning: No easy trials for subject {subject_id}")

    if len(hard_epochs) > 0:
        evoked_hard = hard_epochs.average()
        evoked_hard.crop(tmin=tmin_erp, tmax=tmax_erp)
        subject_evokeds_hard.append(evoked_hard)
    else:
        print(f"Warning: No hard trials for subject {subject_id}")

    # Compute subject-wise difference waveform (easy - hard)
    if len(easy_epochs) > 0 and len(hard_epochs) > 0:
        evoked_diff = mne.combine_evoked([evoked_easy, evoked_hard], weights=[1, -1])
        evoked_diff.crop(tmin=tmin_erp, tmax=tmax_erp)
        subject_evokeds_diff.append(evoked_diff)

    # Optional: ERP-plot per subject
    # fig = evoked.plot(spatial_colors=True, show=False)
    # fig.suptitle(f"Visual stimulus onset - {subject_id}")
    # fig.savefig(f"ERP_{subject_id}.png")
    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()

# Check if we have any evoked data
if len(subject_evokeds) == 0:
    raise RuntimeError("No evoked data found for any subject.")

# Compute grand averages across all subjects or subgroups
grand_average = mne.grand_average(subject_evokeds).crop(tmin=tmin_erp, tmax=tmax_erp)
grand_average_easy = mne.grand_average(subject_evokeds_easy).crop(-0.2, 0.5) if subject_evokeds_easy else None
grand_average_hard = mne.grand_average(subject_evokeds_hard).crop(-0.2, 0.5) if subject_evokeds_hard else None
grand_average_diff = mne.grand_average(subject_evokeds_diff).crop(-0.2, 0.5) if subject_evokeds_diff else None
grand_average_HC = mne.grand_average(subject_evokeds_HC).crop(-0.2, 0.5)
grand_average_SZ = mne.grand_average(subject_evokeds_SZ).crop(-0.2, 0.5)

# --- PLOT ERP FOR ALL CHANNELS AVERAGED OVER ALL SUBJECTS ---
# Times for topomap plotting
times_to_plot = [0.12, 0.22, 0.36]
# Create the joint plot
fig = grand_average.plot_joint(times=times_to_plot, title=None)
fig.set_size_inches(12, 8)
# Add horizontal and vertical lines, change fontsize
for ax in fig.axes:
    if ax.get_xlabel() == "Time (s)":
        ax.axhline(0, color='k', linewidth=1.5) # horizontal line at 0 µV
        ax.axvline(0, color='k', linewidth=1.5) # vertical line at stimulus onset
    ax.tick_params(axis='both', labelsize=14)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
fig.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/ERPs/grand_average_plot3.pdf", dpi=800)

# Define region of interest (ROI) channels
roi_channels = ["Pz", "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6", "O2", "PO8"]

# Function to compute mean signal over ROI channels
def extract_mean_timecourse(evoked, channels):
    evoked_roi = evoked.copy().pick_channels(channels)
    return evoked.times, evoked_roi.data.mean(axis=0) * 1e6  # µV

# Plot group difference in ROI
times, mean_HC = extract_mean_timecourse(grand_average_HC, roi_channels)
_, mean_SZ = extract_mean_timecourse(grand_average_SZ, roi_channels)
plt.figure(figsize=(10, 5))
plt.plot(times, mean_HC, label="HC", color="teal")
plt.plot(times, mean_SZ, label="SZ", color="darkorange")
plt.axhline(0, color="k", linewidth=1)
plt.axvline(0, color="k", linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.tight_layout()
plt.show()

# Difference plot (HC - SZ)
plt.plot(times, mean_HC - mean_SZ, color="black")
plt.axhline(0, color="k", linewidth=1)
plt.axvline(0, color="k", linewidth=1)
plt.tight_layout()
plt.show()

# --- NON-PARAMETRIC CLUSTER BASED PERMUTATION TEST (HC vs. SZ)
# Prepare data for cluster-based permutation test
def extract_subject_means(evokeds, channels):
    data = []
    for evk in evokeds:
        roi = evk.copy().pick_channels(channels)
        data.append(roi.data.mean(axis=0))  # shape (n_times,)
    return np.array(data) * 1e6  # shape (n_subjects, n_times), in µV
X_HC = extract_subject_means(subject_evokeds_HC, roi_channels)
X_SZ = extract_subject_means(subject_evokeds_SZ, roi_channels)
X = [X_HC, X_SZ]  # shape (n_subjects, n_times)
times = subject_evokeds_HC[0].pick_channels(roi_channels).times

# Run cluster-based permutation test for group difference
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    X, n_permutations=10000, tail=0, n_jobs=1, out_type='mask'
)

# Print time range and stats for each cluster
for i_c, cluster in enumerate(clusters):
    sl = cluster[0]
    times_in_cluster = times[sl]
    t_values_in_cluster = T_obs[sl]
    p_val = cluster_p_values[i_c]

    print(f"Cluster {i_c+1}: {times_in_cluster[0]:.3f} s – {times_in_cluster[-1]:.3f} s "
          f"(p = {p_val:.4f}, max T = {t_values_in_cluster.max():.2f}, min T = {t_values_in_cluster.min():.2f})")

# Plot group differences with T-values and significant clusters
n_hc = X_HC.shape[0]
n_sz = X_SZ.shape[0]
mean_HC_plot = X_HC.mean(axis=0)  # shape (n_times)
mean_SZ_plot = X_SZ.mean(axis=0)  # shape (n_times)
grand_mean = (mean_HC_plot.mean() + mean_SZ_plot.mean()) / 2
# Centared group means
mean_hc_centered = mean_HC_plot - grand_mean
mean_sz_centered = mean_SZ_plot - grand_mean
diff = mean_hc_centered - mean_sz_centered
# Standard error and 95% CI
se_hc = sem(X_HC, axis=0)
se_sz = sem(X_SZ, axis=0)
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
ax1.set_ylabel("Amplitude centered (µV)", fontsize=14)
ax1.set_xlabel("Time (s)", fontsize=14)
ax1.set_axisbelow(True)
ax1.grid(False)

# Right axis (T-values)
ax2 = ax1.twinx()
ax2.plot(times, T_obs, color="gray", label="T-values")
ax2.set_ylabel("T-values", fontsize=14)
ax2.grid(False)

# Synchronize tick labels between axes
yticks_left = ax1.get_yticks()
ymin_left, ymax_left = ax1.get_ylim()
ymin_right, ymax_right = min(T_obs), max(T_obs)
yticks_right = np.interp(yticks_left, (ymin_left, ymax_left), (ymin_right, ymax_right))
yticklabels_right = [f"{yt:.1f}" for yt in yticks_right]
yticklabels_right[0] = ""  # unterstes Label ausblenden
yticklabels_right[-1] = ""  # oberstes Label ausblenden
ax2.set_yticks(yticks_right)
ax2.set_yticklabels(yticklabels_right)
ax2.set_ylim(ymin_right, ymax_right)

# Highlight significant clusters
for i_c, c in enumerate(clusters):
    c = c[0]
    color = "red" if cluster_p_values[i_c] <= 0.05 else "gray"
    ax1.axvspan(times[c.start], times[c.stop - 1], color=color, alpha=0.2)

ax1.axhline(0, color='black', linewidth=1, alpha=0.6)
ax1.axvline(0, color='black', linewidth=1, alpha=0.6)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/ERPs/permutation_cluster_test_HC_vs_SZ_no_legend.svg")
plt.show()


# ERP for easy and hard trials in ROI
fig = grand_average_easy.pick([roi_channels]).plot(spatial_colors=True)
for ax in fig.axes:
    if ax.get_xlabel() == "Time (s)":
        ax.axhline(0, color='k', linewidth=1)  # horizontal line at 0 µV
        ax.axvline(0, color='k', linewidth=1)  # vertical line at stimulus onset
fig = grand_average_hard.pick([roi_channels]).plot(spatial_colors=True)
for ax in fig.axes:
    if ax.get_xlabel() == "Time (s)":
        ax.axhline(0, color='k', linewidth=1)  # horizontal line at 0 µV
        ax.axvline(0, color='k', linewidth=1)  # vertical line at stimulus onset


# Difference between easy and hard trials
def extract_mean_timecourse(evoked, channels):
    evoked_roi = evoked.copy().pick_channels(channels)
    return evoked.times, evoked_roi.data.mean(axis=0) * 1e6  # µV
times, mean_easy = extract_mean_timecourse(grand_average_easy, roi_channels)
_, mean_hard = extract_mean_timecourse(grand_average_hard, roi_channels)
# Plot
plt.figure(figsize=(10, 5))
plt.plot(times, mean_hard, label="Hard", color="tomato")
plt.plot(times, mean_easy, label="Easy", color="cornflowerblue")
plt.plot(times, mean_easy - mean_hard, color="black")
plt.axhline(0, color="k", linewidth=1)
plt.axvline(0, color="k", linewidth=1)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Amplitude (µV)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plot with difference curve
plt.axhline(0, color="k", linewidth=1)
plt.axvline(0, color="k", linewidth=1)
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/ERPs/ERP_average_difference_easy_hard_both_plots.svg", dpi=400)
plt.show()


# --- NON-PARAMETRIC CLUSTER BASED PERMUTATION TEST (easy vs. hard)
# Prepare data for cluster-based permutation test
def extract_subject_means(evokeds, channels):
    data = []
    for evk in evokeds:
        roi = evk.copy().pick_channels(channels)
        data.append(roi.data.mean(axis=0))  # shape (n_times,)
    return np.array(data) * 1e6  # shape (n_subjects, n_times), in µV
X_easy = extract_subject_means(subject_evokeds_easy, roi_channels)
X_hard = extract_subject_means(subject_evokeds_hard, roi_channels)
# Difference matrix: easy - hard
X = X_easy - X_hard  # shape (n_subjects, n_times)
times = subject_evokeds_easy[0].times

# Run cluster-based permutation test for condition difference
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    X, n_permutations=1000, tail=0, n_jobs=1, out_type='mask'
)

# Plot difference and highlight significant clusters
plt.figure(figsize=(10, 5))
plt.plot(times, X.mean(axis=0), label='Difference (easy - hard)')
plt.axhline(0, color='black', linestyle='--')
# Signifikante Cluster einfärben
for i_c, c in enumerate(clusters):
    if cluster_p_values[i_c] < 0.05:
        plt.axvspan(times[c], times[c][-1], color='red', alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('µV')
plt.title('Cluster-based permutation test (easy vs hard)')
plt.legend()
plt.tight_layout()
plt.show()

