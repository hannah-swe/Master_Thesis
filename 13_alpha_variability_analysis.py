import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")

data_path = "/Volumes/SSK Drive/Data/derivatives"   # Define data path
sub_ids = sorted(os.listdir(data_path))     # List and sort all subject folders

all_epochs = list() # For concatenating all epochs later

# Iterate over subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to preprocessed epoched file
    subject_data = os.path.join(data_path, subject_id, "final_frame", f"{subject_id}_task-supratyp-alpha-demogr-epo.fif")

    # Load epoched data
    epochs = mne.read_epochs(subject_data, preload=True)

    all_epochs.append(epochs)

# Concatenate all individuals subject epochs and get dataframe for metadata
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore")
df_metadata = epochs.metadata

# Group by subject and get SD and mean of alpha power per subject
df_variability = (
    df_metadata
    .groupby('subject')
    .agg(
        # alpha power
        alpha_variability=('alpha_power', 'std'),
        alpha_power_mean=('alpha_power', 'mean'),
        # Group label
        Group=('Group', 'first')
    )
    .reset_index()
)

# Get Coefficent of Variation (CV = SD/mean) per subject
df_variability['alpha_cv'] = df_variability['alpha_variability'] / df_variability['alpha_power_mean']
df_variability['subject'] = df_variability['subject'].astype(int)
df_variability = df_variability.sort_values('subject').reset_index(drop=True)

# --- COMPARE ALPHA VARIABILITY BETWEEN HC AND SZ ---
# Split alpha variability by groups
alpha_var_hc = df_variability[df_variability['Group'] == 'HC']['alpha_variability']
alpha_var_sz = df_variability[df_variability['Group'] == 'SZ']['alpha_variability']

# t-Test for independant samples
t_stat, p_value = ttest_ind(alpha_var_hc, alpha_var_sz, equal_var=False)
# Sample sizes and variation per group
n1, n2 = len(alpha_var_hc), len(alpha_var_sz)
s1_sq = np.var(alpha_var_hc, ddof=1)
s2_sq = np.var(alpha_var_sz, ddof=1)
# Degrees of freedom (Welch-Satterthwaite)
df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/((n1**2)*(n1 - 1)) + (s2_sq**2)/((n2**2)*(n2 - 1)))
# Print results
print(f"t-Statistik: {t_stat:.3f}")
print(f"p-Wert: {p_value:.4f}")
print(f"Freiheitsgrade: {df:.2f}")

# --- COMPARE COEFFICIENT OF VARIATION BETWEEN HC AND SZ ---
# Split coefficient of variation (CV) by groups
cv_hc = df_variability[df_variability['Group'] == 'HC']['alpha_cv']
cv_sz = df_variability[df_variability['Group'] == 'SZ']['alpha_cv']

# t-Test for independant samples
t_stat, p_value = ttest_ind(cv_hc, cv_sz, equal_var=False)
# Sample sizes and variation per group
n1, n2 = len(cv_hc), len(cv_sz)
s1_sq = np.var(cv_hc, ddof=1)
s2_sq = np.var(cv_sz, ddof=1)
# Degrees of freedom (Welch-Satterthwaite)
df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/((n1**2)*(n1 - 1)) + (s2_sq**2)/((n2**2)*(n2 - 1)))
# Print results
print(f"t-Statistik: {t_stat:.3f}")
print(f"p-Wert: {p_value:.4f}")
print(f"Freiheitsgrade: {df:.2f}")


# --- HORIZONTAL PLOT OF VARIABILITY AND CV (VARIABLES ON X-AXIS) ---
# Define colors
palette = {"HC": "teal", "SZ": "darkorange"}
# Figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

variables = [
    ("alpha_variability", "Alpha Power Variability (μV²)", 0),
    ("alpha_cv", "Coefficent of Variation", 0)
]

# Function for asteriks of p-values
def pval_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# Boxplots
for i, (ax, (var, label, x_min)) in enumerate(zip(axes, variables)): #, x_max, xticks
    sns.boxplot(data=df_variability, x=var, hue="Group", palette=palette, ax=ax)
    ax.set_xlabel(label, fontsize=14)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.get_legend().remove()

    # Means for position of significance bar
    hc_vals = df_variability[df_variability["Group"] == 'HC'][var]
    sz_vals = df_variability[df_variability["Group"] == 'SZ'][var]
    hc_mean = hc_vals.mean()
    sz_mean = sz_vals.mean()

    # t-Test
    t_stat, p_val = ttest_ind(hc_vals, sz_vals, equal_var=False)
    print(f"{var}: t = {t_stat:.2f}, p = {p_val:.4f}")

    # Format asteriks
    stars = pval_to_stars(p_val)
    font_size = 13 if stars != 'n.s.' else 10
    y_offset = 0.01 if stars != 'n.s.' else -0.005

    # Position of significance bar
    y_pos = 0.5
    ax.plot([hc_mean, sz_mean], [y_pos, y_pos], color='black', linewidth=2)
    ax.text((hc_mean + sz_mean)/2, y_pos + y_offset, stars,
            ha='center', va='bottom', fontsize=font_size)

    # Group names as y-label in the first plot
    if i == 0:
        x_left, x_right = ax.get_xlim()
        ax.text(x_left - 0.1 * (x_right - x_left), 0.2, "SZ", ha='right', fontsize=14)
        ax.text(x_left - 0.1 * (x_right - x_left), -0.2, "HC", ha='right', fontsize=14)

sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/combined_boxplot_variability_cv.svg")
plt.show()


# --- VERTICAL PLOTS OF VARIABILITY AND CV (VARIABLES ON Y-AXIS) ---
palette = {"HC": "teal", "SZ": "darkorange"}

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3, 8))

variables = [
    ("alpha_variability", "Alpha Power Variability (μV²)"),
    ("alpha_cv", "Coefficient of Variation")
]

# Function for asteriks of p-values
def pval_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# Boxplots
for i, (ax, (var, label)) in enumerate(zip(axes, variables)):
    # Dummy-x for grouping
    df_plot = df_variability.copy()
    df_plot["x"] = 0    # shared x-position to center plots

    sns.boxplot(data=df_plot, x="x", y=var, hue="Group", palette=palette, dodge=True, ax=ax)
    ax.set_ylabel(label, fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.get_legend().remove()

    # Means for position of significance bar
    hc_vals = df_plot[df_plot["Group"] == 'HC'][var]
    sz_vals = df_plot[df_plot["Group"] == 'SZ'][var]
    hc_mean = hc_vals.mean()
    sz_mean = sz_vals.mean()
    t_stat, p_val = ttest_ind(hc_vals, sz_vals, equal_var=False)
    print(f"{var}: t = {t_stat:.2f}, p = {p_val:.4f}")

    # x-positions of boxes while dodge=True (common: -0.2 und +0.2)
    x_hc, x_sz = -0.2, 0.2
    x_bar, x_bar2 = -0.5, -0.5

    # Format asteriks
    stars = pval_to_stars(p_val)
    font_size = 13 if stars != 'n.s.' else 10
    x_text = x_bar + 0.07 if stars != 'n.s.' else x_bar + 0.05

    # Position of significance bar
    ax.plot([x_bar, x_bar2], [hc_mean, sz_mean], color='black', lw=2)
    y_text = (hc_mean + sz_mean) / 2
    ax.text(x_text, y_text, stars, ha='center', va='bottom', fontsize=font_size, rotation=90)

    # Group names as y-label in the first plot
    if i == 1:
        ax.text(x_hc, ax.get_ylim()[0] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "HC", ha='center', fontsize=14)
        ax.text(x_sz, ax.get_ylim()[0] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "SZ", ha='center', fontsize=14)

sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/alpha/combined_boxplot_variability_cv_vertical_rotated.svg")
plt.show()