import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import pearsonr
from scipy.stats import chi2

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)


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

# Define simple effect coding for prior
class SimpleEffectContrast:
    def __init__(self, k):
        self.k = k

    def _simple_effect_contrast(self):
        # Define the contrast matrix
        contrast_matrix = {
            50: [-1 / k, -1 / k],
            33: [(k - 1) / k, -1 / k],
            66: [-1 / k, (k - 1) / k]
        }
        contrast_names = ["33-50", "66-50"]
        return contrast_matrix, contrast_names

# Apply contrast function
k = 3
contrast = SimpleEffectContrast(k)
contrast_matrix, contrast_names = contrast._simple_effect_contrast()

# Get contrast matrix
df_metadata['prior_str'] = df_metadata['prior'].astype(str)
df_metadata['prior_contrast_33_50'] = df_metadata['prior'].map(lambda x: contrast_matrix[x][0])
df_metadata['prior_contrast_66_50'] = df_metadata['prior'].map(lambda x: contrast_matrix[x][1])

# Exclude first and last block
df_metadata = df_metadata[~df_metadata['trial_number'].between(0, 31)]
df_metadata = df_metadata[~df_metadata['trial_number'].between(256, 287)]
print(df_metadata['trial_number'].isin(range(0, 32)).sum())
print(df_metadata['trial_number'].isin(range(256, 288)).sum())

# Make sure all categorical variables are set as categorical
df_metadata['difficulty'] = df_metadata['difficulty'].astype('category')
df_metadata['Group'] = df_metadata['Group'].astype('category')
df_metadata['dec_outcome'] = df_metadata['dec_outcome'].astype('category')
df_metadata['subject'] = df_metadata['subject'].astype('category')

# Exclude HC
df_patient = df_metadata[df_metadata['Group'] == 'SZ'].copy()

# Set up PANSS sums as float variable
df_patient['panss_positive_sum'] = df_patient['panss_positive_sum'].astype(float)
df_patient['panss_negative_sum'] = df_patient['panss_negative_sum'].astype(float)

# Center PANSS, trial number, response time, dropped epochs and chlorpromazine
df_patient['panss_positive_sum_c'] = df_patient['panss_positive_sum'] - df_patient['panss_positive_sum'].mean()
df_patient['panss_negative_sum_c'] = df_patient['panss_negative_sum'] - df_patient['panss_negative_sum'].mean()
df_patient['trial_number_c'] = df_patient['trial_number'] - df_patient['trial_number'].mean()
df_patient['response_time_c'] = df_patient['response_time'] - df_patient['response_time'].mean()
df_patient['dropped_epochs_c'] = df_patient['dropped_epochs'] - df_patient['dropped_epochs'].mean()
df_patient['chlorpromazine_equi_c'] = df_patient['chlorpromazine_equi'] - df_patient['chlorpromazine_equi'].mean()

# Caution: If the model should include previous difficulty regressors, trials where previous or 2 previous was a catch
# trial need to be excluded here
before = len(df_patient)
df_patient = df_patient[df_patient['difficulty_previous'] != -1.0]
after = len(df_patient)
deleted = before - after
print(f"{deleted} Zeilen wurden gelöscht.")
before = len(df_patient)
df_patient = df_patient[df_patient['difficulty_previous2'] != -1.0]
after = len(df_patient)
deleted = before - after
print(f"{deleted} Zeilen wurden gelöscht.")
df_patient['difficulty_previous'] = df_patient['difficulty_previous'].astype('category')
df_patient['difficulty_previous2'] = df_patient['difficulty_previous2'].astype('category')


# --- Linear mixed model ---
# Define model formular here (this is the formula of the final single-trial model)
model = MixedLM.from_formula(
    "log_power ~ prior_contrast_33_50 + prior_contrast_66_50 + difficulty * trial_number_c"
    "+ dropped_epochs_c + panss_positive_sum_c + panss_negative_sum_c + chlorpromazine_equi_c"
    "+ prior_contrast_33_50:panss_positive_sum_c + prior_contrast_33_50:panss_negative_sum_c"
    "+ prior_contrast_66_50:panss_positive_sum_c + prior_contrast_66_50:panss_negative_sum_c",
    data=df_patient,
    groups="subject"
)
result = model.fit()
pvalues = result.pvalues.round(4)
# Calculate ICC
var_subject = result.cov_re.iloc[0, 0]   # Variance of random intercepts
var_residual = result.scale              # Residual variance
icc = var_subject / (var_subject + var_residual)
# Calculations of Marginal and Conditional R² (Nakagawa & Schielzeth)
var_fixed = np.var(result.fittedvalues)
r2_marginal = var_fixed / (var_fixed + var_subject + var_residual)
r2_conditional = (var_fixed + var_subject) / (var_fixed + var_subject + var_residual)
# Print model output
print(result.summary())
print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.3f}")
print(f"Marginal R²: {r2_marginal:.3f}")
print(f"Conditional R²: {r2_conditional:.3f}")


# Create dataframe for subject means of log alpha power for HC (as comparison for plots)
df_means = df_metadata.groupby('subject', as_index=False).agg({
    'log_power': 'mean',
    'Group': 'first'
})
df_means['subject'] = df_means['subject'].astype(int)
df_means = df_means.sort_values('subject').reset_index(drop=True)
mean_hc = df_means[df_means['Group'] == 'HC']['log_power'].mean()

# --- PLOT PANSS SUM MAIN EFFECTS ---
# Create a pastel color palette with as many colors as subjects
unique_subjects = df_patient['subject'].unique()
palette = sns.color_palette("pastel", len(unique_subjects))
subject_colors = dict(zip(unique_subjects, palette))

# Two subplots with shared y-axis
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Left plot: PANSS negative sum
ax1 = axes[0]
for subject in unique_subjects:
    df_sub = df_patient[df_patient['subject'] == subject]
    ax1.scatter(df_sub['panss_negative_sum'], df_sub['log_power_mean'],
                color=subject_colors[subject], label=subject, s=40)
# Add regression line across all patients (not subject-specific)
sns.regplot(
    data=df_patient,
    x="panss_negative_sum",
    y="log_power_mean",
    ax=ax1,
    scatter=False,
    line_kws=dict(color="darkorange", linewidth=2)
)
ax1.set_xlabel("PANSS Negative Sum", fontsize=14)
ax1.set_ylabel("Mean Log Alpha Power (μV²)", fontsize=14)
ax1.grid(False)
sns.despine()

# Right plot: PANSS positive sum
ax2 = axes[1]
for subject in unique_subjects:
    df_sub = df_patient[df_patient['subject'] == subject]
    ax2.scatter(df_sub['panss_positive_sum'], df_sub['log_power_mean'],
                color=subject_colors[subject], s=40)
sns.regplot(
    data=df_patient,
    x="panss_positive_sum",
    y="log_power_mean",
    ax=ax2,
    scatter=False,
    line_kws=dict(color="darkorange", linewidth=2)
)
ax2.set_xlabel("PANSS Positive Sum", fontsize=14)
ax2.set_ylabel("")
ax2.grid(False)
sns.despine()

# Add line for HC mean in both plots
x_min_neg = df_patient['panss_negative_sum'].min()
x_max_neg = df_patient['panss_negative_sum'].max()
ax1.plot([x_min_neg, x_max_neg], [mean_hc, mean_hc],
         color='teal', linewidth=2)
ax1.tick_params(axis='both', labelsize=14)

x_min_pos = df_patient['panss_positive_sum'].min()
x_max_pos = df_patient['panss_positive_sum'].max()
ax2.plot([x_min_pos, x_max_pos], [mean_hc, mean_hc],
         color='teal', linewidth=2)
ax2.tick_params(axis='x', labelsize=14)

plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM Patients/panss_positive_negative_main_effect.svg")
plt.show()


# --- TABLE WITH LIKELIHOOD RATIO TEST AND BAYES FACTOR FOR ALL REGRESSORS ---
# Model formula of the final model
full_formula = ("log_power ~ prior_contrast_33_50 + prior_contrast_66_50 + difficulty * trial_number_c"
                "+ dropped_epochs + panss_positive_sum_c + panss_negative_sum_c + chlorpromazine_equi"
                "+ prior_contrast_33_50:panss_positive_sum_c + prior_contrast_33_50:panss_negative_sum_c"
                "+ prior_contrast_66_50:panss_positive_sum_c + prior_contrast_66_50:panss_negative_sum_c")
# All terms explicitly listed
all_terms = [
    "prior_contrast_33_50",
    "prior_contrast_66_50",
    "difficulty",
    "trial_number_c",
    "dropped_epochs",
    "panss_positive_sum_c",
    "panss_negative_sum_c",
    "chlorpromazine_equi",
    "difficulty:trial_number_c",
    "prior_contrast_33_50:panss_positive_sum_c",
    "prior_contrast_33_50:panss_negative_sum_c",
    "prior_contrast_66_50:panss_positive_sum_c",
    "prior_contrast_66_50:panss_negative_sum_c"
]
# Mapping of main effects and belonging interactions
interactions = {
    "prior_contrast_33_50": ["prior_contrast_33_50:panss_positive_sum_c", "prior_contrast_33_50:panss_negative_sum_c"],
    "prior_contrast_66_50": ["prior_contrast_66_50:panss_positive_sum_c", "prior_contrast_66_50:panss_negative_sum_c"],
    "difficulty": ["difficulty:trial_number_c"],
    "trial_number_c": ["difficulty:trial_number_c"],
    "panss_positive_sum_c": ["prior_contrast_33_50:panss_positive_sum_c", "prior_contrast_66_50:panss_positive_sum_c"],
    "panss_negative_sum_c": ["prior_contrast_33_50:panss_negative_sum_c", "prior_contrast_66_50:panss_negative_sum_c"]
}
# Fit full model (REML set to false for LRT)
model_full = MixedLM.from_formula(full_formula, data=df_patient, groups="subject")
result_full = model_full.fit(reml=False)
llf_full = result_full.llf
df_full = result_full.df_modelwc
# Collect results
lrt_results_patient = []
# Iterate over all variables to get LLR test result for all regressors
for var in all_terms:
    # Reduced model: remove regressor and all associated interactions
    reduced_terms = [t for t in all_terms if t != var]
    for inter in interactions.get(var, []):
        reduced_terms = [t for t in reduced_terms if t != inter]
    reduced_formula = "log_power ~ " + " + ".join(reduced_terms)
    # Fit reduced model
    try:
        model_reduced = MixedLM.from_formula(reduced_formula, data=df_patient, groups="subject")
        result_reduced = model_reduced.fit(reml=False)

        if not np.isfinite(result_reduced.llf):
            raise ValueError("Log-likelihood is not finite")

        # LLR
        lr_stat = 2 * (llf_full - result_reduced.llf)
        df_diff = df_full - result_reduced.df_modelwc
        p_val = chi2.sf(lr_stat, df_diff)
        # Bayes Factor approx. from BIC
        bic_full = result_full.bic
        bic_reduced = result_reduced.bic
        bf_10 = np.exp((bic_reduced - bic_full) / 2)  # BF zugunsten des Vollmodells

        lrt_results_patient.append({
            "Regressor_removed": var,
            "LLR_statistic": lr_stat,
            "df": df_diff,
            "p_value": p_val,
            "BayesFactor_approx": bf_10
        })
    except Exception as e:
        lrt_results_patient.append({
            "Regressor_removed": var,
            "LLR_statistic": None,
            "df": None,
            "p_value": None,
            "BayesFactor_approx": None,
            "Error": str(e)
        })
# Sort results in a dataframe
lrt_results_patient_df = pd.DataFrame(lrt_results_patient).sort_values("p_value", na_position='last')
