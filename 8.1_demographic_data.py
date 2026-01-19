import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)

# Define data paths and read csv files
demogr_data_path = "/Volumes/SSK Drive/Data/demographics/demographics_SUPRATYP_CBBM.csv"    # Demographical data
demogr_df = pd.read_csv(demogr_data_path)
meds_data_path = "/Volumes/SSK Drive/Data/clinical_data/medication_SUPRATYP_CBBM.xlsx"  # Medication data
meds_df = pd.read_excel(meds_data_path)
panss_data_path = "/Volumes/SSK Drive/Data/clinical_data/PANSS_SUPRATYP_CBBM.xlsx"  # PANSS interviews
panss_df = pd.read_excel(panss_data_path)
questionnaire_data_path = "/Volumes/SSK Drive/Data/demographics/questionnaire_SUPRATYP_CBBM.csv"    # Questionnaire data
questionnaire_df = pd.read_csv(questionnaire_data_path)

# --- DEMOGRAPHIC INFORMATION OF HC (healthy controls; index 0–34 (cbbm_part_id 1–37)) ---
HC = demogr_df.iloc[0:35]
# Sex
count_females = (HC['Geschlecht'] == 'W').sum()
count_males = (HC['Geschlecht'] == 'M').sum()
total = len(HC)
percent_females = (count_females / total) * 100
percent_males = (count_males / total) * 100
# Age
age_mean = HC['Alter'].mean()
age_median = HC['Alter'].median()
age_sd = HC['Alter'].std()
age_min = HC['Alter'].min()
age_max = HC['Alter'].max()
# Years of education
education_mean = HC['Bildungsgrad'].mean()
education_median = HC['Bildungsgrad'].median()
education_sd = HC['Bildungsgrad'].std()
education_min = HC['Bildungsgrad'].min()
education_max = HC['Bildungsgrad'].max()
# Print everything
print("Healthy Controls:")
print(f"Number females: {count_females} ({percent_females:.1f}%)")
print(f"Number of males: {count_males} ({percent_males:.1f}%)")
print(f"Age: M = {age_mean:.2f}, Median = {age_median:.2f}, SD = {age_sd:.2f}, Range = {age_min}–{age_max}")
print(f"Years of education: M = {education_mean: .2f}, Median = {education_median: .2f}, SD = {education_sd: .2f}, "
      f"Range = {education_min}-{education_max}")


# --- DEMOGRAPHIC INFORMATION OF SZ (patients; index 35–43 (cbbm_part_id > 38)) ---
SZ = demogr_df.iloc[35:44]
# Sex
count_females2 = (SZ['Geschlecht'] == 'W').sum()
count_males2 = (SZ['Geschlecht'] == 'M').sum()
total2 = len(SZ)
percent_females2 = (count_females2 / total2) * 100
percent_males2 = (count_males2 / total2) * 100
# Age
age_mean2 = SZ['Alter'].mean()
age_median2 = SZ['Alter'].median()
age_sd2 = SZ['Alter'].std()
age_min2 = SZ['Alter'].min()
age_max2 = SZ['Alter'].max()
# Years of education
education_mean2 = SZ['Bildungsgrad'].mean()
education_median2 = SZ['Bildungsgrad'].median()
education_sd2 = SZ['Bildungsgrad'].std()
education_min2 = SZ['Bildungsgrad'].min()
education_max2 = SZ['Bildungsgrad'].max()
# Print everything
print("\nPatients:")
print(f"Number of females: {count_females2} ({percent_females2:.1f}%)")
print(f"Number of males: {count_males2} ({percent_males2:.1f}%)")
print(f"Age: M = {age_mean2:.2f}, Median = {age_median2:.2f}, SD = {age_sd2:.2f}, Range = {age_min2}–{age_max2}")
print(f"Years of education: M = {education_mean2: .2f}, Median = {education_median2: .2f}, SD = {education_sd2: .2f}, Range = {education_min2}-{education_max2}")


# --- MEDICATION INFORMATION OF SZ ---
ids_to_keep = [51, 53, 54, 61, 62, 66, 77, 85, 87]  # Only keep patients of this sample
meds_df = meds_df[meds_df['ID #'].isin(ids_to_keep)]
# Filter for all patient with prescribed medication
filtered_non_zero = meds_df[meds_df['Unnamed: 3'] != 0]
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


# --- PANSS SCORES OF SZ ---
panss_df = panss_df.loc[panss_df['subID'].isin(ids_to_keep)].copy()
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
# Get correlation of PANSS scores
x = panss_df["panss_positive_sum"]
y = panss_df["panss_negative_sum"]
r, p_value = pearsonr(x, y)
print(f"Pearson r = {r:.3f}, p = {p_value:.4f}")
sns.regplot(data=panss_df, x='panss_negative_sum', y='panss_positive_sum')


# --- QUESTIONNAIRE DATA (SPQ, LSHS, PDI) OF HC AND SZ ---
questionnaire_df['is_hc'] = questionnaire_df["CBBM_ID"].astype(int) < 51
questionnaire_df['Group'] = questionnaire_df["is_hc"].apply(lambda x: "HC" if x else "SZ")
HC_quest = questionnaire_df.iloc[0:35] # Which integers are HC
SZ_quest = questionnaire_df.iloc[35:44] # Which integers are SZ
# SPQ HC
SPQ_mean = HC_quest['SPQ_total'].mean()
SPQ_median = HC_quest['SPQ_total'].median()
SPQ_sd = HC_quest['SPQ_total'].std()
SPQ_min = HC_quest['SPQ_total'].min()
SPQ_max = HC_quest['SPQ_total'].max()
print(f"HC SPQ-B: M = {SPQ_mean:.2f}, Median = {SPQ_median:.2f}, SD = {SPQ_sd:.2f}, Range = {SPQ_min}–{SPQ_max}")
# SPQ SZ
SPQ_mean2 = SZ_quest['SPQ_total'].mean()
SPQ_median2 = SZ_quest['SPQ_total'].median()
SPQ_sd2 = SZ_quest['SPQ_total'].std()
SPQ_min2 = SZ_quest['SPQ_total'].min()
SPQ_max2 = SZ_quest['SPQ_total'].max()
print(f"SZ SPQ-B: M = {SPQ_mean2:.2f}, Median = {SPQ_median2:.2f}, SD = {SPQ_sd2:.2f}, Range = {SPQ_min2}–{SPQ_max2}")
# LSHS HC
LSHS_mean = HC_quest['LSHS_total'].mean()
LSHS_median = HC_quest['LSHS_total'].median()
LSHS_sd = HC_quest['LSHS_total'].std()
LSHS_min = HC_quest['LSHS_total'].min()
LSHS_max = HC_quest['LSHS_total'].max()
print(f"HC LSHS-R: M = {LSHS_mean:.2f}, Median = {LSHS_median:.2f}, SD = {LSHS_sd:.2f}, Range = {LSHS_min}–{LSHS_max}")
# LSHS SZ
LSHS_mean2 = SZ_quest['LSHS_total'].mean()
LSHS_median2 = SZ_quest['LSHS_total'].median()
LSHS_sd2 = SZ_quest['LSHS_total'].std()
LSHS_min2 = SZ_quest['LSHS_total'].min()
LSHS_max2 = SZ_quest['LSHS_total'].max()
print(f"SZ LSHS-R: M = {LSHS_mean2:.2f}, Median = {LSHS_median2:.2f}, SD = {LSHS_sd2:.2f}, Range = {LSHS_min2}–{LSHS_max2}")
# PDI HC
PDI_mean = HC_quest['PDI_total'].mean()
PDI_median = HC_quest['PDI_total'].median()
PDI_sd = HC_quest['PDI_total'].std()
PDI_min = HC_quest['PDI_total'].min()
PDI_max = HC_quest['PDI_total'].max()
print(f"HC PDI: M = {PDI_mean:.2f}, Median = {PDI_median:.2f}, SD = {PDI_sd:.2f}, Range = {PDI_min}–{PDI_max}")
# PDI SZ
PDI_mean2 = SZ_quest['PDI_total'].mean()
PDI_median2 = SZ_quest['PDI_total'].median()
PDI_sd2 = SZ_quest['PDI_total'].std()
PDI_min2 = SZ_quest['PDI_total'].min()
PDI_max2 = SZ_quest['PDI_total'].max()
print(f"SZ PDI: M = {PDI_mean2:.2f}, Median = {PDI_median2:.2f}, SD = {PDI_sd2:.2f}, Range = {PDI_min2}–{PDI_max2}")


# --- PLOT QUESTIONNAIRE DATA WITH T-TEST FOR GROUP DIFFERENCES ---
palette = {"HC": "teal", "SZ": "darkorange"}

# Figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Set up range of variables
variables = [
    ("SPQ_total", "SPQ-B (total score)", -1, 19, np.arange(0, 19, 2)),
    ("PDI_total", "PDI (total score)", -1, 19, np.arange(0, 19, 2)),
    ("LSHS_total", "LSHS (total score)", -1, 42, np.arange(0, 41, 5))
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
    sns.boxplot(data=questionnaire_df, x=var, hue="Group", palette=palette, ax=ax)
    ax.set_xlabel(label, fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.get_legend().remove()

    # Means for position of significance bar
    hc_vals = questionnaire_df[questionnaire_df["Group"] == "HC"][var].dropna()
    sz_vals = questionnaire_df[questionnaire_df["Group"] == "SZ"][var].dropna()
    hc_mean = hc_vals.mean()
    sz_mean = sz_vals.mean()

    # t-Test
    t_stat, p_val = ttest_ind(hc_vals, sz_vals, equal_var=False)
    # Degrees of freedom (Welch-Satterthwaite)
    n1, n2 = len(hc_vals), len(sz_vals)
    s1, s2 = np.var(hc_vals, ddof=1), np.var(sz_vals, ddof=1)
    df = (s1 / n1 + s2 / n2) ** 2 / ((s1 ** 2) / ((n1 ** 2) * (n1 - 1)) + (s2 ** 2) / ((n2 ** 2) * (n2 - 1)))
    print(f"{var}: t = {t_stat:.2f}, p = {p_val:.4f}, df = {df:.2f}")

    # Position of significance bar
    y_pos = 0.5
    ax.plot([hc_mean, sz_mean], [y_pos, y_pos], color='black', linewidth=2)
    ax.text((hc_mean + sz_mean)/2, y_pos + 0.01, pval_to_stars(p_val),
            ha='center', va='bottom', fontsize=13)

    # Group names as y-label in the first plot
    if i == 0:
        ax.text(x_min - 0.5,  0.2, "SZ", va='center', ha='right', fontsize=14)
        ax.text(x_min - 0.5, -0.2, "HC", va='center', ha='right', fontsize=14)

sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/questionnaires/questionnaire_boxplot.svg")
plt.show()
