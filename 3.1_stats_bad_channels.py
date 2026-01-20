import mne
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from bad_channels import bads
import seaborn as sns
import numpy as np

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")
# --- Load and process bad channel data for each subject ---
data_path = "/Volumes/SSK Drive/Data/raw"   # Path to the raw EEG data
sub_ids = sorted(os.listdir(data_path))     # Get a sorted list of subject directories
subject_bad_channels = []   # Container for subject-level bad channel data

for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Exclude specific subjects
        continue
    try:
        bad = bads[subject_id]  # Look up bad channels for this subject
    except KeyError:
        bad = None  # If not found, assume no bad channels

    # Extract numeric subject ID and determine group (HC or SZ)
    subject_num = subject_id.split("_")[2]
    subject_str = subject_num[1:]
    subject_int = int(subject_str)
    subject_bad_channels.append({
        'subject_id': subject_id,
        'subject': subject_int,
        'is_hc': subject_int < 51,  # Healthy controls have ID < 51
        'n_bad_channels': len(bad) if bad else 0,
        'bad_channels': bad if bad else []
    })

# Create a DataFrame from the list of subject data
df_bads = pd.DataFrame(subject_bad_channels)

# --- Summary statistics: percentage of subjects with bad channels ---
percentage = df_bads.groupby("is_hc").apply(
    lambda g: (g['n_bad_channels'] > 0).sum() / len(g) * 100
).reset_index(name="percent_with_bad_channels")

# --- Mean and standard deviation of bad channels per group ---
stats = df_bads.groupby("is_hc")['n_bad_channels'].agg(
    mean='mean',
    std='std'
).reset_index()

# --- Independent samples t-test: HC vs SZ ---
hc_values = df_bads[df_bads['is_hc'] == True]['n_bad_channels']
sc_values = df_bads[df_bads['is_hc'] == False]['n_bad_channels']

# Welch's t-test (assumes unequal variance), one-sided: HC < SZ
t_stat, p_value = ttest_ind(hc_values, sc_values, equal_var=False, alternative="less")

# Calculate Welch-Satterthwaite degrees of freedom
n1, n2 = len(hc_values), len(sc_values)
s1_sq = np.var(hc_values, ddof=1)
s2_sq = np.var(sc_values, ddof=1)
df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/((n1**2)*(n1 - 1)) + (s2_sq**2)/((n2**2)*(n2 - 1)))

# Print t-test result
print(f"t = {t_stat:.3f}, p = {p_value:.4f}, df = {df:.2f}")

# --- Visualization ---
# Add a readable group label column
df_bads['Group'] = df_bads['is_hc'].apply(lambda x: "HC" if x else "SZ")
# Define custom color palette
palette = {True: "teal", False: "darkorange"}
# Create a barplot showing the number of bad channels per group
sns.barplot(data=df_bads, x="Group", y="n_bad_channels", hue="is_hc", errorbar="se", palette=palette, legend=False)
plt.xlabel("")
plt.ylabel("Interpolated Channels", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/data quality/bad_channels_barplot.svg")
plt.show()
