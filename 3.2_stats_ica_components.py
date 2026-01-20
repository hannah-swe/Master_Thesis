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

data_path = "/Volumes/SSK Drive/Data/derivatives"   # Path to the raw EEG data
sub_ids = sorted(os.listdir(data_path))     # Get a sorted list of subject directories
subject_rejected_components = []    # Initialize list to collect ICA rejection data

# --- Loop through subjects and collect rejection data ---
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # subjects to be excluded
        continue

    # Try to get bad channels from dictionary
    try:
        bad = bads[subject_id]
    except KeyError:
        bad = None

    # Define path to ICA label file
    txt_file = f"{data_path}/{subject_id}/preprocessing/{subject_id}_task-supratyp_ica_labels.txt"

    # Try to read ICA component labels from file
    try:
        with open(txt_file, "r") as f:
            lines = f.readlines()
            components = [int(line.strip()) for line in lines if line.strip()]  # Convert to ints
            n_rejected = len(components)    # Count rejected components
    except FileNotFoundError:
        components = []
        n_rejected = 0  # Assume no components rejected if file not found

    # Extract subject number and define group (HC < 51)
    subject_num = subject_id.split("_")[2]
    subject_str = subject_num[1:]
    subject_int = int(subject_str)

    # Append subject data to list
    subject_rejected_components.append({
        "subject_id": subject_id,
        'subject': subject_int,
        'is_hc': subject_int < 51,  # Boolean: True = Healthy Control
        'n_bad_channels': len(bad) if bad else 0,
        "n_rejected_ica_components": n_rejected,
        "rejected_components": components
    })

# --- Convert list of dictionaries into a DataFrame ---
df_ica = pd.DataFrame(subject_rejected_components)

# --- Grouped statistics: mean & std of rejected components for HC and SZ ---
ica_stats = df_ica.groupby("is_hc")['n_rejected_ica_components'].agg(
    mean='mean',
    std='std'
).reset_index()

# --- Compute percentage of rejected ICA components (adjusted for bad channels) ---
total_components = 64 - df_ica['n_bad_channels']    # ICA run on remaining channels
df_ica['percent_rejected'] = df_ica['n_rejected_ica_components'] / total_components * 100

# --- Grouped stats on percentage of rejected components ---
ica_percentage_stats = df_ica.groupby("is_hc")['percent_rejected'].agg(
    mean='mean',
    std='std'
).reset_index()

# --- Independent-samples t-test: compare number of rejected components (HC vs SZ) ---
hc_values = df_ica[df_ica['is_hc'] == True]['n_rejected_ica_components']
sc_values = df_ica[df_ica['is_hc'] == False]['n_rejected_ica_components']

# Welch's t-test (unequal variance)
t_stat, p_value = ttest_ind(hc_values, sc_values, equal_var=False)

# Compute degrees of freedom for Welch's t-test (Welch-Satterthwaite)
n1, n2 = len(hc_values), len(sc_values)
s1_sq = np.var(hc_values, ddof=1)
s2_sq = np.var(sc_values, ddof=1)
df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/((n1**2)*(n1 - 1)) + (s2_sq**2)/((n2**2)*(n2 - 1)))

# Print the statistical result
print(f"t = {t_stat:.3f}, p = {p_value:.4f}, df = {df:.2f}")

# --- Visualization: Boxplot of rejected ICA components per group ---
df_ica['Group'] = df_ica['is_hc'].apply(lambda x: "HC" if x else "SZ")  # Label for plotting
palette = {True: "teal", False: "darkorange"}   # Custom colors for HC and SZ
# Create boxplot
sns.boxplot(data=df_ica, x="Group", y="n_rejected_ica_components", hue="is_hc", palette=palette, legend=False)
plt.xlabel("")
plt.ylabel("Rejected ICA components", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/data quality/rejected_components_boxplot.svg")
plt.show()