import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)


data_path = "/Volumes/SSK Drive/Data/derivatives/"  # Define data path
sub_ids = sorted(os.listdir(data_path))     # List and sort all subject folders

# Define parameters for alpha band time-frequency analysis
alpha_freqs = np.arange(8, 15, 1)   # 8–14 Hz range
n_cycles = alpha_freqs / 2          # Number of cycles per frequency (controls time-frequency resolution)
method = "morlet"                   # Use Morlet wavelets for TFR computation
decim = 7                           # Downsampling factor to speed up processing
tmin, tmax = -0.5, -0.1             # Time window

# Selected subset of posterior electrodes based on literature
selected_channels = ["Pz", "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6", "O2", "PO8"]    # Tarasi et al. + right side channels Limbach et al.
# selected_channels = None  # Activate if all channels should be used

all_epochs = list() # For concatenating all epochs later

# Iterate over subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to preprocessed epoched file
    subject_data = os.path.join(data_path, subject_id, "final_frame", f"{subject_id}_task-supratyp-alpha-demogr-epo.fif")

    # Load epoched data (already cropped to -1.0 to 0.6 s)
    epochs = mne.read_epochs(subject_data, preload=True).crop(-1.0, 0.6)

    all_epochs.append(epochs)

# Concatenate all individuals subject epochs
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore")
df_metadata = epochs.metadata

# Compute TFR (no averaging yet)
epochs = epochs.pick(picks=selected_channels)
whole_tfr = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1,
                               return_itc=False, average=False)

# Crop TFR to pre-stimulus time window
whole_tfr = whole_tfr.crop(tmin, tmax)

# Use metadata filtering to split data by group (HC vs. SZ)
tfr_hc = whole_tfr["is_hc==True"]
tfr_sz = whole_tfr["is_hc==False"]

# Compute average TFR for each group
avrgd_tfr_hc = tfr_hc.average()
avrgd_tfr_sz = tfr_sz.average()

# Compute difference between groups (HC - SZ)
avg_diff = avrgd_tfr_hc - avrgd_tfr_sz

# SPLIT TFR BY DECISION OUTCOME WITHIN EACH GROUP
# Filter and compute averages for each outcome category
tfr_hc_H = tfr_hc["dec_outcome=='H'"]
tfr_hc_FA = tfr_hc["dec_outcome=='FA'"]
tfr_hc_M = tfr_hc["dec_outcome=='M'"]
tfr_hc_CR = tfr_hc["dec_outcome=='CR'"]

tfr_sz_H = tfr_sz["dec_outcome=='H'"]
tfr_sz_FA = tfr_sz["dec_outcome=='FA'"]
tfr_sz_M = tfr_sz["dec_outcome=='M'"]
tfr_sz_CR = tfr_sz["dec_outcome=='CR'"]

# Compute average TFRs per outcome
avrgd_tfr_hc_H = tfr_hc_H.average()
avrgd_tfr_hc_FA = tfr_hc_FA.average()
avrgd_tfr_hc_M = tfr_hc_M.average()
avrgd_tfr_hc_CR = tfr_hc_CR.average()

avrgd_tfr_sz_H = tfr_sz_H.average()
avrgd_tfr_sz_FA = tfr_sz_FA.average()
avrgd_tfr_sz_M = tfr_sz_M.average()
avrgd_tfr_sz_CR = tfr_sz_CR.average()


# --- PLOT PROPORTION OF DECISION OUTCOMES PER GROUP ---
# Count how often each decision outcome occurred per subject
counts = df_metadata.groupby(['subject', 'Group', 'dec_outcome']).size().unstack(fill_value=0)
# Convert counts to proportions
proportions = counts.div(counts.sum(axis=1), axis=0)
# Get new dataframe for plotting
df_outcome = proportions.reset_index().melt(
    id_vars=['subject', 'Group'],
    value_vars=['H', 'M', 'CR', 'FA'],
    var_name='dec_outcome',
    value_name='proportion'
)
df_outcome['subject'] = df_outcome['subject'].astype(int)
df_outcome = df_outcome.sort_values('subject').reset_index(drop=True)

# Plot boxplots of decision outcome proportions per group
palette = {"HC": "teal", "SZ": "darkorange"}
# change order for x-axis; if order changed here, then also change x-axis labels at the end
outcome_order = ['H', 'M', 'CR', 'FA']
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_outcome, x="dec_outcome", y="proportion", hue="Group", palette=palette, legend=False)
plt.xlabel("")
plt.xticks(ticks=range(len(outcome_order)), labels=['Hits', 'Misses', 'Correct\nRejections', 'False\nAlarms'], fontsize=14)
plt.ylabel("Proportion", fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/proportion_dec_outcomes.svg")
plt.show()


# --- TOPOPLOTS OF ALPHA POWER FOR DECISION OUTCOMES PER GROUP ---
# Prepare data sequences for subplotting and set up titles
hc_tfrs = [avrgd_tfr_hc, avrgd_tfr_hc_H, avrgd_tfr_hc_FA, avrgd_tfr_hc_M, avrgd_tfr_hc_CR]
sz_tfrs = [avrgd_tfr_sz, avrgd_tfr_sz_H, avrgd_tfr_sz_FA, avrgd_tfr_sz_M, avrgd_tfr_sz_CR]
col_titles = ["All Conditions", "Hits", "False Alarms", "Misses", "Correct Rejections"]
row_labels = ["HC", "SZ"]

# Create figure with 2 rows (groups) x 5 columns (outcomes)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
im = None

# Add row labels (leftmost column, HC/SZ)
for ax, row_label in zip(axes[:, 0], row_labels):
    ax.set_ylabel(row_label, fontsize=18, labelpad=17, rotation=0, va='center')

# Plot topographies for HC (top row, cloumns 0-4)
for i, tfr in enumerate(hc_tfrs):
    tfr.plot_topomap(axes=axes[0, i], show=False, vlim=(None, 1.3e-09), colorbar=False)
    axes[0, i].set_title(col_titles[i], fontsize=18)

# Plot topographies for SZ (bottom row, cloumns 0-4)
for i, tfr in enumerate(sz_tfrs):
    tfr.plot_topomap(axes=axes[1, i], show=False, vlim=(None, 1.3e-09), colorbar=False)
    if im is None:
        im = axes[1, i].images[0]

# Add shared colorbar to the right
cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label("Power (µV²)", fontsize=18)
cbar.ax.tick_params(labelsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/prestim_dec_outcomes2.svg")
plt.show()


# --- SPAGHETTI PLOT: SUBJECT-WISE LOG ALPHA POWER PER OUTCOME ---
# Compute mean log power per subject, group, and decision outcome
subject_means = (df_metadata.groupby(['subject', 'Group', 'dec_outcome'], as_index=False).agg({'log_power': 'mean'}))

# Ensure outcome order is meaningful for x-axis; if order changed here, then also change x-axis labels at the end
outcome_order = ['H', 'FA', 'M', 'CR']
subject_means['dec_outcome'] = pd.Categorical(subject_means['dec_outcome'], categories=outcome_order, ordered=True)

# Plot subject-level lines and group means
plt.figure(figsize=(10, 6))
group_colors = {'HC': 'teal', 'SZ': 'darkorange'}

# Individual subject curves (low alpha for visibility)
for group in subject_means['Group'].unique():
    data_group = subject_means[subject_means['Group'] == group]
    sns.lineplot(data=data_group, x='dec_outcome', y='log_power', units='subject', estimator=None, lw=1,
                 alpha=0.3, color=group_colors[group], legend=False)

# Group-level mean lines
for group in subject_means['Group'].unique():
    data_group = subject_means[subject_means['Group'] == group]
    sns.lineplot(data=data_group, x='dec_outcome', y='log_power', estimator='mean', errorbar=None, lw=3,
                 color=group_colors[group], legend=False,)
plt.ylabel('Log Alpha Power (μV²)', fontsize=14)
plt.xlabel('')
plt.yticks(fontsize=14)
plt.xticks(ticks=range(len(outcome_order)), labels=['Hits', 'False Alarms', 'Misses', 'Correct\nRejections'], fontsize=14)
plt.grid(True, axis='x', alpha=0.4)
sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/prestim_dec_outcomes_lineplot.png", dpi=400)
plt.show()
