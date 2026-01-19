import os
import mne
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2

# Ensure proper visualization backend
plt.ion()
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

# Function for computation of SDT measures
def sdt_measures(hits, misses, fas, crs):
    # Get hit and false alarm rate
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    fa_rate = fas / (fas + crs) if (fas + crs) > 0 else np.nan
    # Log-linear correction to avoid Inf/NaN
    n_signal = hits + misses
    n_noise = fas + crs
    if hit_rate == 1:
        hit_rate -= 0.5 / n_signal
    if hit_rate == 0:
        hit_rate += 0.5 / n_signal
    if fa_rate == 1:
        fa_rate -= 0.5 / n_noise
    if fa_rate == 0:
        fa_rate += 0.5 / n_noise
    # Get z-scores of hit and false alarm rate
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(fa_rate)
    # Get SDT measures d-prime and decision criterion
    d_prime = z_hit - z_fa
    criterion = -0.5 * (z_hit + z_fa)
    return hit_rate, fa_rate, d_prime, criterion

# Function to apply SDT measures on data
def compute_sdt(group):
    counts = group['dec_outcome'].value_counts()
    hits = counts.get("H", 0)
    misses = counts.get("M", 0)
    fas = counts.get("FA", 0)
    crs = counts.get("CR", 0)

    hit_rate, fa_rate, d_prime, criterion = sdt_measures(hits, misses, fas, crs)
    mean_alpha = group['alpha_power'].mean()
    mean_log_alpha = group['log_power'].mean()
    group_label = group['Group'].iloc[0]
    choice_prob = (group['response'] == 1).mean() * 100
    dropped_epochs = group['dropped_epochs'].mean()

    return pd.Series({
        "hit_rate": hit_rate,
        "hit_rate_perc": hit_rate * 100,
        "false_alarm_rate": fa_rate,
        "false_alarm_rate_perc": fa_rate * 100,
        "sensitivity": d_prime,
        "criterion": criterion,
        "alpha_power": mean_alpha,
        "log_power": mean_log_alpha,
        "choice_prob": choice_prob,
        "Group": group_label,
        "dropped_epochs": dropped_epochs
    })

# Apply SDT Function per subject and prior condition
df_sdt = df_metadata.groupby(["subject", "prior"]).apply(compute_sdt).reset_index()

# Make sure all categorical variables are set as categorical
df_sdt['Group'] = df_sdt['Group'].astype('category')
df_sdt['subject'] = df_sdt['subject'].astype('category')

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
df_sdt['prior_str'] = df_sdt['prior'].astype(str)
df_sdt['prior_contrast_33_50'] = df_sdt['prior'].map(lambda x: contrast_matrix[x][0])
df_sdt['prior_contrast_66_50'] = df_sdt['prior'].map(lambda x: contrast_matrix[x][1])

# Center log power and dropped epochs
df_sdt['log_power_c'] = df_sdt['log_power'] - df_sdt['log_power'].mean()
df_sdt['dropped_epochs_c'] = df_sdt['dropped_epochs'] - df_sdt['dropped_epochs'].mean()


# --- LINEAR MIXED MODEL FOR SENSITIVITY ---
# Define model formular here (this is the formula of the final sensitivity model)
model_sensitivity = MixedLM.from_formula(
    "sensitivity ~ prior_contrast_33_50 * Group + prior_contrast_66_50 * Group + log_power_c * Group"
    "+ dropped_epochs_c",
    data=df_sdt,
    groups="subject"
)
result_sensitivity = model_sensitivity.fit()
pvalues_s = result_sensitivity.pvalues.round(4)
# Calculate ICC
var_subject = result_sensitivity.cov_re.iloc[0, 0]   # Variance of random intercepts
var_residual = result_sensitivity.scale              # Residual variance
icc = var_subject / (var_subject + var_residual)
# Calculations of Marginal and Conditional R² (Nakagawa & Schielzeth)
var_fixed = np.var(result_sensitivity.fittedvalues)
r2_marginal = var_fixed / (var_fixed + var_subject + var_residual)
r2_conditional = (var_fixed + var_subject) / (var_fixed + var_subject + var_residual)
# Print model output
print(result_sensitivity.summary())
print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.3f}")
print(f"Marginal R²: {r2_marginal:.3f}")
print(f"Conditional R²: {r2_conditional:.3f}")


# --- TABLE WITH LIKELIHOOD RATIO TEST AND BAYES FACTOR FOR ALL REGRESSORS OF SENSITIVITY MODEL ---
# Model formula of the final model
full_formula = "sensitivity ~ prior_contrast_33_50 * Group + prior_contrast_66_50 * Group + log_power * Group + dropped_epochs"
# All terms explicitly listed
all_terms = [
    "prior_contrast_33_50",
    "prior_contrast_66_50",
    "Group",
    "log_power",
    "dropped_epochs",
    "prior_contrast_33_50:Group",
    "prior_contrast_66_50:Group",
    "log_power:Group"
]
# Mapping of main effects and belonging interactions
interactions = {
    "prior_contrast_33_50": ["prior_contrast_33_50:Group"],
    "prior_contrast_66_50": ["prior_contrast_66_50:Group"],
    "log_power": ["log_power:Group"],
    "Group": ["prior_contrast_33_50:Group", "prior_contrast_66_50:Group", "log_power:Group"]
}
# Fit full model (REML set to false for LRT)
model_full = MixedLM.from_formula(full_formula, data=df_sdt, groups="subject")
result_full = model_full.fit(reml=False)
llf_full = result_full.llf
df_full = result_full.df_modelwc
# Collect results
lrt_results_s = []
# Iterate over all variables to get LLR test result for all regressors
for var in all_terms:
    # Reduced model: remove regressor and all associated interactions
    reduced_terms = [t for t in all_terms if t != var]
    for inter in interactions.get(var, []):
        reduced_terms = [t for t in reduced_terms if t != inter]
    reduced_formula = "sensitivity ~ " + " + ".join(reduced_terms)
    # Fit reduced model
    try:
        model_reduced = MixedLM.from_formula(reduced_formula, data=df_sdt, groups="subject")
        result_reduced = model_reduced.fit(reml=False)
        # LR-Test
        lr_stat = 2 * (llf_full - result_reduced.llf)
        df_diff = df_full - result_reduced.df_modelwc
        p_val = chi2.sf(lr_stat, df_diff)
        # Bayes Factor approx. from BIC
        bic_full = result_full.bic
        bic_reduced = result_reduced.bic
        bf_10 = np.exp((bic_reduced - bic_full) / 2)  # BF zugunsten des Vollmodells

        lrt_results_s.append({
            "Regressor_removed": var,
            "LLR_statistic": lr_stat,
            "df": df_diff,
            "p_value": p_val,
            "BayesFactor_approx": bf_10
        })
    except Exception as e:
        lrt_results_s.append({
            "Regressor_removed": var,
            "LLR_statistic": None,
            "df": None,
            "p_value": None,
            "BayesFactor_approx": None,
            "Error": str(e)
        })

# Sort results in a dataframe
lrt_results_s_df = pd.DataFrame(lrt_results_s).sort_values("p_value", na_position='last')


# --- LINEAR MIXED MODEL FOR CRITERION ---
# Define model formular here (this is the formula of the final criterion model)
model_criterion = MixedLM.from_formula(
    "criterion ~ prior_contrast_33_50 * Group + prior_contrast_66_50 * Group + log_power_c * Group"
    "+ dropped_epochs_c",
    data=df_sdt,
    groups="subject"
)
result_criterion = model_criterion.fit()
pvalues_c = (result_criterion.pvalues.round(4))
# Calculate ICC
var_subject = result_criterion.cov_re.iloc[0, 0]   # Variance of random intercepts
var_residual = result_criterion.scale              # Residual variance
icc = var_subject / (var_subject + var_residual)
# Calculations of Marginal and Conditional R² (Nakagawa & Schielzeth)
var_fixed = np.var(result_criterion.fittedvalues)
r2_marginal = var_fixed / (var_fixed + var_subject + var_residual)
r2_conditional = (var_fixed + var_subject) / (var_fixed + var_subject + var_residual)
# Print model output
print(result_criterion.summary())
print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.3f}")
print(f"Marginal R²: {r2_marginal:.3f}")
print(f"Conditional R²: {r2_conditional:.3f}")


# --- TABLE WITH LIKELIHOOD RATIO TEST AND BAYES FACTOR FOR ALL REGRESSORS OF CRITERION MODEL ---
full_formula = "criterion ~ prior_contrast_33_50 * Group + prior_contrast_66_50 * Group + log_power * Group + dropped_epochs"
# All terms explicitly listed
all_terms = [
    "prior_contrast_33_50",
    "prior_contrast_66_50",
    "Group",
    "log_power",
    "dropped_epochs",
    "prior_contrast_33_50:Group",
    "prior_contrast_66_50:Group",
    "log_power:Group"
]
# Mapping of main effects and belonging interactions
interactions = {
    "prior_contrast_33_50": ["prior_contrast_33_50:Group"],
    "prior_contrast_66_50": ["prior_contrast_66_50:Group"],
    "log_power": ["log_power:Group"],
    "Group": ["prior_contrast_33_50:Group", "prior_contrast_66_50:Group", "log_power:Group"]
}
# Fit full model (REML set to false for LRT)
model_full = MixedLM.from_formula(full_formula, data=df_sdt, groups="subject")
result_full = model_full.fit(reml=False)
llf_full = result_full.llf
df_full = result_full.df_modelwc
# Collect results
lrt_results_c = []
# Iterate over all variables to get LLR test result for all regressors
for var in all_terms:
    # Reduced model: remove regressor and all associated interactions
    reduced_terms = [t for t in all_terms if t != var]
    for inter in interactions.get(var, []):
        reduced_terms = [t for t in reduced_terms if t != inter]
    reduced_formula = "criterion ~ " + " + ".join(reduced_terms)
    # Fit reduced model
    try:
        model_reduced = MixedLM.from_formula(reduced_formula, data=df_sdt, groups="subject")
        result_reduced = model_reduced.fit(reml=False)
        # LR-Test
        lr_stat = 2 * (llf_full - result_reduced.llf)
        df_diff = df_full - result_reduced.df_modelwc
        p_val = chi2.sf(lr_stat, df_diff)
        # Bayes Factor approx. from BIC
        bic_full = result_full.bic
        bic_reduced = result_reduced.bic
        bf_10 = np.exp((bic_reduced - bic_full) / 2)  # BF zugunsten des Vollmodells

        lrt_results_c.append({
            "Regressor_removed": var,
            "LLR_statistic": lr_stat,
            "df": df_diff,
            "p_value": p_val,
            "BayesFactor_approx": bf_10
        })
    except Exception as e:
        lrt_results_c.append({
            "Regressor_removed": var,
            "LLR_statistic": None,
            "df": None,
            "p_value": None,
            "BayesFactor_approx": None,
            "Error": str(e)
        })

# Sort results in a dataframe
lrt_results_c_df = pd.DataFrame(lrt_results_c).sort_values("p_value", na_position='last')


# --- MODEL COMPARISONS BETWEEN ONE NULL MODEL AND ALTERNATIVE MODEL ---
# Model 1: null model with less terms
model_null = MixedLM.from_formula(
    "criterion ~ prior_contrast_33_50 * Group + prior_contrast_66_50 * Group + log_power * Group + dropped_epochs",
    data=df_sdt,
    groups="subject"
)
result_null = model_null.fit(reml=False)  # REML=False für LRT-Vergleich
# Model 2: alternative model with more complex structure
model_alt = MixedLM.from_formula(
    "criterion ~ prior_contrast_33_50 * Group + prior_contrast_66_50 * Group + log_power * Group + dropped_epochs"
    "+ prior_contrast_33_50:log_power + prior_contrast_66_50:log_power",
    data=df_sdt,
    groups="subject"
)
result_alt = model_alt.fit(reml=False)
# Likelihood Ratio Test (LRT)
llf_null = result_null.llf
llf_alt = result_alt.llf
lr_stat = 2 * (llf_alt - llf_null)
df_diff = result_alt.df_modelwc - result_null.df_modelwc
p_value = chi2.sf(lr_stat, df_diff)
# Get all results (model 1 and model 2 output, AIC, BIC and df of both models, LRT)
print(result_null.summary())
print(result_alt.summary())
print(f"AIC null model: {result_null.aic:.2f}")
print(f"BIC null model: {result_null.bic:.2f}")
print(f"AIC alternative model: {result_alt.aic:.2f}")
print(f"BIC alternative model: {result_alt.bic:.2f}")
print(f"\nLikelihood Ratio Test:\n  LR stat = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")


# --- PLOT EFFECT OF NUMBER OF DROPPED EPOCHS ON SENSITIVITY ---
palette = {"HC": "teal", "SZ": "darkorange"}    # Define color palette
figsize = (7, 5)
aspect_ratio = figsize[0] / figsize[1]
# Regression lines of both groups
g = sns.lmplot(data=df_sdt, x="dropped_epochs", y="sensitivity", hue="Group", palette=palette, markers="o",
               scatter_kws={"alpha": 0.6, "s": 50}, line_kws={"linewidth": 2}, height=figsize[1],
               aspect=aspect_ratio, legend=False)
# Regression line over all subjects
sns.regplot(data=df_sdt, x="dropped_epochs", y="sensitivity", scatter=False, color="black", line_kws={"linewidth": 2, "linestyle": "--"})
plt.grid(False)
plt.xlabel("Dropped EEG-Epochs", fontsize=14)
plt.ylabel("Sensitivity (d')", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM2/correlation_dropped_epochs_sensitivity.svg")
plt.show()