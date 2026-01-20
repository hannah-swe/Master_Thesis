import mne
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import numpy as np

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")
data_path = "/Volumes/SSK Drive/Data/derivatives"   # Define data path
sub_ids = sorted(os.listdir(data_path))     # List all subject folders
subject_dropped_epochs = [] # Initialize list to store subject-level epoch info

# --- Loop over each subject ---
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to subject's epoched data
    subject_data = os.path.join(data_path, subject_id, "epoching", f"{subject_id}_task-supratyp-epo.fif")

    # Load epochs, crop to specific time window (-1 to 0.6s)
    epochs = mne.read_epochs(subject_data, preload=True).crop(-1.0, 0.6)

    # Extract subject number and group info (HC if subject_num < 51)
    subject_num = subject_id.split("_")[2]
    subject_str = subject_num[1:]
    subject_int = int(subject_str)
    is_hc = subject_int < 51    # Boolean: True for HC

    # Count remaining and dropped epochs (assuming 288 total trials)
    n_epochs = len(epochs)
    n_epochs_dropped = 288 - n_epochs

    # Save subject data
    subject_dropped_epochs.append({
        'subject_id': subject_id,
        'subject': subject_int,
        'is_hc': is_hc,
        'n_epochs': n_epochs,
        'n_epochs_dropped': n_epochs_dropped
    })

# --- Convert results to a DataFrame ---
df_epochs = pd.DataFrame(subject_dropped_epochs)

# --- Grouped stats (mean and SD) of kept epochs for HC vs SZ ---
epoch_stats = df_epochs.groupby("is_hc")['n_epochs'].agg(['mean', 'std']).reset_index()

# --- Grouped stats for dropped epochs ---
dropped_epoch_stats = df_epochs.groupby("is_hc")['n_epochs_dropped'].agg(['mean', 'std']).reset_index()

# --- Grouped stats for percent dropped ---
df_epochs['percent_dropped'] = df_epochs['n_epochs_dropped'] / 288 * 100
percent_stats = df_epochs.groupby("is_hc")['percent_dropped'].agg(['mean', 'std']).reset_index()
print(percent_stats)

# --- Statistical test: Independent-sample t-test ---
hc_values = df_epochs[df_epochs['is_hc'] == True]['n_epochs_dropped']
sc_values = df_epochs[df_epochs['is_hc'] == False]['n_epochs_dropped']

# Welch's t-test (doesn't assume equal variance), one-sided (HC < SZ)
t_stat, p_value = ttest_ind(hc_values, sc_values, equal_var=False, alternative="less")

# Calculate degrees of freedom for Welch's t-test (Welch-Satterthwaite)
n1, n2 = len(hc_values), len(sc_values)
s1_sq = np.var(hc_values, ddof=1)
s2_sq = np.var(sc_values, ddof=1)
df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/((n1**2)*(n1 - 1)) + (s2_sq**2)/((n2**2)*(n2 - 1)))

# Print test result
print(f"t = {t_stat:.3f}, p = {p_value:.4f}, df = {df:.2f}")

# --- Plotting: Barplot of dropped epochs by group ---
df_epochs['Group'] = df_epochs['is_hc'].apply(lambda x: "HC" if x else "SZ")
palette = {True: "teal", False: "darkorange"}   # Colors for HC and SZ
# Start barplot
sns.barplot(data=df_epochs, x="Group", y="n_epochs_dropped", hue="is_hc", errorbar="se", palette=palette, legend=False)
plt.xlabel("")
plt.ylabel("Dropped EEG Epochs", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/data quality/dropped_epochs_barplot.svg")
plt.show()