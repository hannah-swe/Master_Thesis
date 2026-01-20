import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")
# Define data paths and read csv files
demogr_data_path = "/Volumes/SSK Drive/Data/compare_patients/demographics_SUPRATYP_CBBM_full_sample.csv"    # Demographical data
demogr_df = pd.read_csv(demogr_data_path)
meds_data_path = "/Volumes/SSK Drive/Data/clinical_data/medication_SUPRATYP_CBBM.xlsx"  # Medication data
meds_df = pd.read_excel(meds_data_path)
panss_data_path = "/Volumes/SSK Drive/Data/clinical_data/PANSS_SUPRATYP_CBBM.xlsx"  # PANSS interviews
panss_df = pd.read_excel(panss_data_path)
questionnaire_data_path = "/Volumes/SSK Drive/Data/compare_patients/questionnaire_SUPRATYP_CBBM_full_sample.csv"    # Questionnaire data
questionnaire_df = pd.read_csv(questionnaire_data_path)

# IDs of EEG subsample, N = 9
eeg_ids = [51, 53, 54, 61, 62, 66, 77, 85, 87]

# --- COMPARE DEMOGRAPHIC DATA ---
# List of excluded IDs
drop_ids = list(range(1, 38)) + [59, 71, 72, 74, 82, 83]
demogr_df = demogr_df[~demogr_df["cbbm_part_id"].isin(drop_ids)]
# Add column 'eeg': 1 = Subsample, 0 = Rest
demogr_df["eeg"] = demogr_df["cbbm_part_id"].isin(eeg_ids).astype(int)
demogr_df.loc[demogr_df["Bildungsgrad"] <= 7, "Bildungsgrad"] = np.nan
# Demopgraphic information of full sample (EEG and not EEG)
count_females = (demogr_df['Geschlecht'] == 'W').sum()
count_males = (demogr_df['Geschlecht'] == 'M').sum()
total = len(demogr_df)
# Sex
percent_females = (count_females / total) * 100
percent_males = (count_males / total) * 100
# Age
age_mean = demogr_df['Alter'].mean()
age_median = demogr_df['Alter'].median()
age_sd = demogr_df['Alter'].std()
age_min = demogr_df['Alter'].min()
age_max = demogr_df['Alter'].max()
# Years of education
education_mean = demogr_df['Bildungsgrad'].mean()
education_median = demogr_df['Bildungsgrad'].median()
education_sd = demogr_df['Bildungsgrad'].std()
education_min = demogr_df['Bildungsgrad'].min()
education_max = demogr_df['Bildungsgrad'].max()
# Print everything
print(f"Number females: {count_females} ({percent_females:.1f}%)")
print(f"Number of males: {count_males} ({percent_males:.1f}%)")
print(f"Age: M = {age_mean:.2f}, Median = {age_median:.2f}, SD = {age_sd:.2f}, Range = {age_min}–{age_max}")
print(f"Years of education: M = {education_mean: .2f}, Median = {education_median: .2f}, SD = {education_sd: .2f}, "
      f"Range = {education_min}-{education_max}")

# One-sample t-Test for age (one sample: compare eeg-subsample against the population (full sample) mean
mean_alter_full = demogr_df["Alter"].mean()
eeg_sub_alter = demogr_df[demogr_df["eeg"] == 1]["Alter"]
t_stat, p_val = ttest_1samp(eeg_sub_alter, popmean=mean_alter_full)
df_alter = len(eeg_sub_alter) - 1
print(f"Alter: t = {t_stat:.3f}, p = {p_val:.4f}, df = {df_alter} (M_Vergleichswert = {mean_alter_full:.2f})")

# One-sample t-Test for education (one sample: compare eeg-subsample against the population (full sample) mean
mean_bildung_full = demogr_df["Bildungsgrad"].mean()
eeg_sub_bildung = demogr_df[demogr_df["eeg"] == 1]["Bildungsgrad"]
t_stat_bildung, p_val_bildung = ttest_1samp(eeg_sub_bildung, popmean=mean_bildung_full)
df_bildung = len(eeg_sub_bildung) - 1
print(f"Bildungsgrad: t = {t_stat_bildung:.3f}, p = {p_val_bildung:.4f}, df = {df_bildung} "
      f"(M_Vergleichswert = {mean_bildung_full:.2f})")


# --- COMPARE MEDICATION DATA ---
meds_df = meds_df[~(meds_df['ID #'].isna() | (meds_df['ID #'] == '53a'))]
# Add column 'eeg': 1 = Subsample, 0 = Rest
meds_df["eeg"] = meds_df["ID #"].isin(eeg_ids).astype(int)
# Get medication information of patients with prescribed medication
filtered_non_zero = meds_df[meds_df['Unnamed: 3'] != 0].copy()
filtered_non_zero["Unnamed: 3"] = pd.to_numeric(filtered_non_zero["Unnamed: 3"], errors="coerce")
mean_value_non_zero = filtered_non_zero['Unnamed: 3'].mean()
median_value_non_zero = filtered_non_zero['Unnamed: 3'].median()
std_value_non_zero = filtered_non_zero['Unnamed: 3'].std()
meds_min = filtered_non_zero['Unnamed: 3'].min()
meds_max = filtered_non_zero['Unnamed: 3'].max()
count_non_zero = filtered_non_zero.shape[0]
# Print everything
print(f'Mittelwert (ohne 0): {mean_value_non_zero}')
print(f'Median (ohne 0): {median_value_non_zero}')
print(f'Standardabweichung (ohne 0): {std_value_non_zero}')
print(f'Anzahl der Zeilen (subjects) mit "Unnamed: 3" ungleich 0: {count_non_zero}')
print(f"Range = {meds_min}–{meds_max}")

# One-sample t-Test for medication (one sample: compare eeg-subsample against the population (full sample) mean
filtered_non_zero["Unnamed: 3"] = pd.to_numeric(filtered_non_zero["Unnamed: 3"], errors="coerce")
eeg_sub_meds = pd.to_numeric(
    filtered_non_zero[filtered_non_zero["eeg"] == 1]["Unnamed: 3"],
    errors="coerce"
)
mean_meds_full = filtered_non_zero["Unnamed: 3"].mean()
t_stat, p_val = ttest_1samp(eeg_sub_meds, popmean=mean_meds_full)
df_meds = len(eeg_sub_meds) - 1
print(f"Medication: t = {t_stat:.2f}, p = {p_val:.4f}, df = {df_meds} (M_full_sample = {mean_meds_full:.2f})")


# --- COMPARE PANSS SCORES ---
# Add column 'eeg': 1 = Subsample, 0 = Rest
panss_df["eeg"] = panss_df["subID"].isin(eeg_ids).astype(int)
# Get PANSS scores of full sample
panss_df["panss_positive_sum"] = panss_df[["P1", "P2", "P3", "P4", "P5", "P6", "P7"]].sum(axis=1)
panss_df["panss_negative_sum"] = panss_df[["N1", "N2", "N3", "N4", "N5", "N6", "N7"]].sum(axis=1)
# PANSS positive sum
positive_mean = panss_df['panss_positive_sum'].mean()
positive_median = panss_df['panss_positive_sum'].median()
positive_sd = panss_df['panss_positive_sum'].std()
positive_min = panss_df['panss_positive_sum'].min()
positive_max = panss_df['panss_positive_sum'].max()
# PANSS negative sum
negative_mean = panss_df['panss_negative_sum'].mean()
negative_median = panss_df['panss_negative_sum'].median()
negative_sd = panss_df['panss_negative_sum'].std()
negative_min = panss_df['panss_negative_sum'].min()
negative_max = panss_df['panss_negative_sum'].max()
# Print everything
print(f"PANSS positive sum: M = {positive_mean:.2f}, Median = {positive_median:.2f}, SD = {positive_sd:.2f}, "
      f"Range = {positive_min}–{positive_max}")
print(f"PANSS negative sum: M = {negative_mean:.2f}, Median = {negative_median:.2f}, SD = {negative_sd:.2f}, "
      f"Range = {negative_min}–{negative_max}")

# One-sample t-Test for PANSS positive sum (one sample: compare eeg-subsample against the population (full sample) mean
mean_panss_pos_full = panss_df["panss_positive_sum"].mean()
eeg_sub_panss_pos = panss_df[panss_df["eeg"] == 1]["panss_positive_sum"]
t_stat, p_val = ttest_1samp(eeg_sub_panss_pos, popmean=mean_panss_pos_full)
print(f"PANSS positive: t = {t_stat:.2f}, p = {p_val:.4f} (M_full_sample = {mean_panss_pos_full:.2f})")

# One-sample t-Test for PANSS negative sum (one sample: compare eeg-subsample against the population (full sample) mean
mean_panss_neg_full = panss_df["panss_negative_sum"].mean()
eeg_sub_panss_neg = panss_df[panss_df["eeg"] == 1]["panss_negative_sum"]
t_stat, p_val = ttest_1samp(eeg_sub_panss_neg, popmean=mean_panss_neg_full)
print(f"PANSS negative: t = {t_stat:.2f}, p = {p_val:.4f} (M_full_sample = {mean_panss_neg_full:.2f})")



# --- COMPARE QUESTIONNAIRE DATA ---
questionnaire_df = questionnaire_df[~questionnaire_df["CBBM_ID"].isin(drop_ids)]
# Add column 'eeg': 1 = Subsample, 0 = Rest
questionnaire_df["eeg"] = questionnaire_df["CBBM_ID"].isin(eeg_ids).astype(int)
# SPQ
SPQ_mean = questionnaire_df['SPQ_total'].mean()
SPQ_median = questionnaire_df['SPQ_total'].median()
SPQ_sd = questionnaire_df['SPQ_total'].std()
SPQ_min = questionnaire_df['SPQ_total'].min()
SPQ_max = questionnaire_df['SPQ_total'].max()
print(f"SPQ-B: M = {SPQ_mean:.2f}, Median = {SPQ_median:.2f}, SD = {SPQ_sd:.2f}, Range = {SPQ_min}–{SPQ_max}")
# LSHS
LSHS_mean = questionnaire_df['LSHS_total'].mean()
LSHS_median = questionnaire_df['LSHS_total'].median()
LSHS_sd = questionnaire_df['LSHS_total'].std()
LSHS_min = questionnaire_df['LSHS_total'].min()
LSHS_max = questionnaire_df['LSHS_total'].max()
print(f"LSHS-R: M = {LSHS_mean:.2f}, Median = {LSHS_median:.2f}, SD = {LSHS_sd:.2f}, Range = {LSHS_min}–{LSHS_max}")
# PDI
PDI_mean = questionnaire_df['PDI_total'].mean()
PDI_median = questionnaire_df['PDI_total'].median()
PDI_sd = questionnaire_df['PDI_total'].std()
PDI_min = questionnaire_df['PDI_total'].min()
PDI_max = questionnaire_df['PDI_total'].max()
print(f"PDI: M = {PDI_mean:.2f}, Median = {PDI_median:.2f}, SD = {PDI_sd:.2f}, Range = {PDI_min}–{PDI_max}")

# One-sample t-Test for SPQ (one sample: compare eeg-subsample against the population (full sample) mean
mean_SPQ_full = questionnaire_df["SPQ_total"].mean()
eeg_sub_SPQ = questionnaire_df[questionnaire_df["eeg"] == 1]["SPQ_total"]
t_stat, p_val = ttest_1samp(eeg_sub_SPQ, popmean=mean_SPQ_full)
print(f"SPQ: t = {t_stat:.2f}, p = {p_val:.4f} (M_full_sample = {mean_SPQ_full:.2f})")

# One-sample t-Test for LSHS (one sample: compare eeg-subsample against the population (full sample) mean
mean_LSHS_full = questionnaire_df["LSHS_total"].mean()
eeg_sub_LSHS = questionnaire_df[questionnaire_df["eeg"] == 1]["LSHS_total"]
t_stat, p_val = ttest_1samp(eeg_sub_LSHS, popmean=mean_LSHS_full)
print(f"LSHS: t = {t_stat:.2f}, p = {p_val:.4f} (M_full_sample = {mean_LSHS_full:.2f})")

# One-sample t-Test for PDI (one sample: compare eeg-subsample against the population (full sample) mean
mean_PDI_full = questionnaire_df["PDI_total"].mean()
eeg_sub_PDI = questionnaire_df[questionnaire_df["eeg"] == 1]["PDI_total"]
t_stat, p_val = ttest_1samp(eeg_sub_PDI, popmean=mean_PDI_full)
print(f"PDI: t = {t_stat:.2f}, p = {p_val:.4f} (M_full_sample = {mean_PDI_full:.2f})")


# --- PLOT OF PANSS DIFFERENCES WITH T-TEST ---
# Filter rows with eeg == 1
duplicate_rows = panss_df[panss_df['eeg'] == 1].copy()
# Add column for subsample
duplicate_rows['subsample'] = True
panss_df['subsample'] = False
# Get new dataframe for plot
df_plot_panss = pd.concat([panss_df, duplicate_rows], ignore_index=True)
# Define colors
palette = {False: "hotpink", True: "darkorange"}
# Figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Set up range of variables
variables = [
    ("panss_negative_sum", "PANSS Negative Sum", 6, 30, np.arange(7, 30, 5)),
    ("panss_positive_sum", "PANSS Positive Sum", 6, 30, np.arange(7, 30, 5))
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
for i, (ax, (var, label, x_min, x_max, xticks)) in enumerate(zip(axes, variables)):
    sns.boxplot(data=df_plot_panss, x=var, hue="subsample", palette=palette, ax=ax)
    ax.set_xlabel(label, fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.get_legend().remove()

    # Means for position of significance bar
    full_samp_vals = df_plot_panss[df_plot_panss["subsample"] == False][var]
    subsample_vals = df_plot_panss[df_plot_panss["subsample"] == True][var]
    full_samp_mean = full_samp_vals.mean()
    subsample_mean = subsample_vals.mean()

    # t-Test
    mean_panss_full = full_samp_vals.mean()
    t_stat, p_val = ttest_1samp(subsample_vals, popmean=mean_panss_full)
    print(f"{var}: t = {t_stat:.2f}, p = {p_val:.4f}")

    # Format asteriks
    stars = pval_to_stars(p_val)
    font_size = 13 if stars != 'n.s.' else 10
    y_offset = 0.01 if stars != 'n.s.' else -0.005

    # Position of significance bar
    y_pos = 0.5
    ax.plot([full_samp_mean, subsample_mean], [y_pos, y_pos], color='black', linewidth=2)
    ax.text((full_samp_mean + subsample_mean)/2, y_pos + y_offset, stars,
            ha='center', va='bottom', fontsize=font_size)

    # Group names as y-label in the first plot
    if i == 0:
        ax.text(x_min - 0.5,  0.2, "Subsample", va='center', ha='right', fontsize=14)
        ax.text(x_min - 0.5, -0.2, "Full Sample", va='center', ha='right', fontsize=14)

sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/patients/comparison_panss_both.svg")
plt.show()
