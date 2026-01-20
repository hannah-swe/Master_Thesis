import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2

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

# Center trial number, response time and dropped epochs
df_metadata['trial_number_centered'] = df_metadata['trial_number'] - df_metadata['trial_number'].mean()
df_metadata['response_time_centered'] = df_metadata['response_time'] - df_metadata['response_time'].mean()
df_metadata['dropped_epochs_centered'] = df_metadata['dropped_epochs'] - df_metadata['dropped_epochs'].mean()

# Caution: If the model should include previous difficulty regressors, trials where previous or 2 previous was a catch
# trial need to be excluded here
before = len(df_metadata)
df_metadata = df_metadata[df_metadata['difficulty_previous'] != -1.0]
after = len(df_metadata)
deleted = before - after
print(f"{deleted} Zeilen wurden gelöscht.")
before = len(df_metadata)
df_metadata = df_metadata[df_metadata['difficulty_previous2'] != -1.0]
after = len(df_metadata)
deleted = before - after
print(f"{deleted} Zeilen wurden gelöscht.")
df_metadata['difficulty_previous'] = df_metadata['difficulty_previous'].astype('category')
df_metadata['difficulty_previous2'] = df_metadata['difficulty_previous2'].astype('category')

# Caution: For model with previous confidence rating as regressor, trials without confidence rating in the previous
# trial need to be excluded. Use this dataframe for it.
variables = [
    "log_power", "prior_contrast_33_50", "prior_contrast_66_50", "Group", "difficulty", "trial_number",
    "dropped_epochs", "subject", "confidence_previous"
]
df_clean = df_metadata.dropna(subset=variables)


# --- Linear mixed model ---
# Define model formular here (this is the formula of the final single-trial model)
model = MixedLM.from_formula(
    "log_power ~ Group + prior_contrast_33_50 + prior_contrast_66_50 + dropped_epochs_centered + trial_number_centered"
    "+ difficulty + response_time + Group:prior_contrast_33_50 + Group:prior_contrast_66_50"
    "+ trial_number_centered:difficulty + trial_number_centered:Group",
    data=df_metadata,
    groups="subject",
)
result = model.fit()
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
pvalues = result.pvalues.round(4) # Get p-values rounded to 4 digtis
print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.3f}")
print(f"Marginal R²: {r2_marginal:.3f}")
print(f"Conditional R²: {r2_conditional:.3f}")


# --- MODEL COMPARISONS BETWEEN ONE NULL MODEL AND ALTERNATIVE MODEL ---
# Model 1: null model with less terms
model_null = MixedLM.from_formula(
    "log_power ~ Group + prior_contrast_33_50 + prior_contrast_66_50 + dropped_epochs_centered + trial_number_centered"
    "+ difficulty + response_time + Group:prior_contrast_33_50 + Group:prior_contrast_66_50"
    "+ trial_number_centered:difficulty + trial_number_centered:Group",
    data=df_clean,
    groups="subject",
)
result_null = model_null.fit(reml=False)  # REML=False for LRT-comparisons
# Model 2: alternative model with more complex structure
model_alt = MixedLM.from_formula(
    "log_power ~ Group + prior_contrast_33_50 + prior_contrast_66_50 + dropped_epochs_centered + trial_number_centered"
    "+ difficulty + response_time + confidence_previous + Group:prior_contrast_33_50 + Group:prior_contrast_66_50"
    "+ trial_number_centered:difficulty + trial_number_centered:Group",
    data=df_clean,
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
print(f"Degrees of freedom null model: {result_null.df_modelwc:.2f}")
print(f"AIC alternative model: {result_alt.aic:.2f}")
print(f"BIC alternative model: {result_alt.bic:.2f}")
print(f"Degrees of freedom alternative model: {result_alt.df_modelwc:.2f}")
print(f"\nLikelihood Ratio Test:\n  LR stat = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")
print("Converged:", result_null.converged)
print("Converged:", result_alt.converged)


# --- PLOTS ---
# 1. Stripplot to visualize randomisation issues of prior over trial_number/blocks; Caution: Do not exclude first and
# last block for this plot, because here the reason for exclusion should be visualized.
plt.figure(figsize=(12, 8))
sns.stripplot(data=df_metadata, x="prior", y="trial_number", alpha=0.5, color="tomato")
# Block boundaries (all 32 trials included 288)
block_ends = list(range(0, 289, 32))
block_starts = [0] + block_ends[:-1]
block_centers = [(start + end) / 2 for start, end in zip(block_starts, block_ends)]
# Vertical Lines + centered block numbers between lines
for i, (line_y, label_y) in enumerate(zip(block_ends, block_centers[1:]), start=1):
    plt.axhline(y=line_y, color='gray', linewidth=1, alpha=0.4)
    plt.text(
        x=plt.gca().get_xlim()[1] + 0.02,
        y=label_y,
        s=str(i),
        va='center',
        ha='left',
        fontsize=16,
        color='black'
    )
plt.axhline(y=288, color='gray', linewidth=1, alpha=0.4)
# Shaded background for block 1 (0-32) and block 9 (256-288) to visualize which blocks will be excluded
ax = plt.gca()
ax.axhspan(0, 32, facecolor='gray', alpha=0.3)
ax.axhspan(256, 288, facecolor='gray', alpha=0.3)
# Finish plot
plt.xlabel("Prior Category", fontsize=16)
plt.ylabel("Trial Number", fontsize=16)
plt.yticks(range(0, 289, 32), fontsize=16)  # Y-axis ticks: set to every 32 trials
prior_labels = {33: "P-", 50: "P=", 66: "P+"}   # Define prior labels
# Set new x-tick labels based on the dictionary
ax = plt.gca()
ax.set_xticklabels([prior_labels.get(int(tick.get_text()), tick.get_text()) for tick in ax.get_xticklabels()], fontsize=16)
# Format the right y-axis (block numbers)
ax2 = ax.twinx()  # Create a second y-axis sharing the same x-axis
ax2.set_yticks([])
# Position and layout of label "Block" on the right
ax2.yaxis.set_label_position("right")
ax2.yaxis.label.set_color('black')
ax2.set_ylabel("Block", labelpad=20, rotation=90, fontsize=16)
ax2.grid(False)
plt.grid(False)
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM1/prior_trialnumber.svg")
plt.show()


# 2. Spaghetti plot of log power across prior conditions with subject means, group means and grand average
df_metadata['prior'] = df_metadata['prior'].astype(str)
subject_means = (df_metadata.groupby(['subject', 'prior'], as_index=False).agg({'log_power': 'mean'}))
prior_order = ['33', '50', '66']
subject_means['prior'] = subject_means['prior'].astype(str)
subject_means['prior'] = pd.Categorical(subject_means['prior'], categories=prior_order, ordered=True)
subject_to_group = df_metadata[['subject', 'Group']].drop_duplicates()
subject_means = subject_means.merge(subject_to_group, on='subject', how='left')
group_colors = {'HC': 'teal', 'SZ': 'darkorange'}
plt.figure(figsize=(10, 6))
# Get subject lines
for group in subject_means['Group'].unique():
    data_group = subject_means[subject_means['Group'] == group]
    sns.lineplot(
        data=data_group,
        x='prior',
        y='log_power',
        units='subject',
        estimator=None,
        lw=1,
        alpha=0.4,
        color=group_colors[group],
        legend=False
    )
# Get group means
for group in subject_means['Group'].unique():
    data_group = subject_means[subject_means['Group'] == group]
    sns.lineplot(
        data=data_group,
        x='prior',
        y='log_power',
        estimator='mean',
        lw=3,
        color=group_colors[group],
        errorbar=None,
        label=group,
        legend=False
    )
plt.ylabel('Log Alpha Power (μV²)', fontsize=14)
plt.xlabel('Prior Category', fontsize=14)
plt.xticks(ticks=range(len(prior_order)), labels=['P-', 'P=', 'P+'], fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, axis='x', alpha=0.4)
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM1/lineplot_power_prior_interaction.svg")
plt.tight_layout()
plt.show()


# 3. Barplot of difficulty effect (deviation from subject mean in easy and hard trials)
plt.figure(figsize=(7, 6))
ax = sns.barplot(data=df_metadata, x="difficulty", y="log_power_deviation", color="gray", errorbar='se')
# Get log_power_deviation means per subject and difficulty condition
mean_by_subject = df_metadata.groupby(["subject", "difficulty"])["log_power_deviation"].mean().reset_index()
subject_groups = df_metadata[["subject", "Group"]].drop_duplicates()
mean_by_subject = mean_by_subject.merge(subject_groups, on="subject", how="left")
# Plot subject lines
for subject in mean_by_subject["subject"].unique():
    subject_data = mean_by_subject[mean_by_subject["subject"] == subject]
    group = subject_data["Group"].iloc[0]
    color = "teal" if group == "HC" else "darkorange"
    x_positions = subject_data["difficulty"]
    y_values = subject_data["log_power_deviation"]
    x_ticks = ax.get_xticks()
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    x_map = dict(zip(tick_labels, x_ticks))
    x_vals = [x_map[str(val)] for val in x_positions]
    plt.plot(x_vals, y_values, linestyle="--", linewidth=1, alpha=0.5, color=color)
difficulty_labels = {0.0: "Easy", 1.0: "Hard"}
ax.set_xticklabels([difficulty_labels.get(float(label.get_text()), label.get_text()) for label in ax.get_xticklabels()], fontsize=14)
plt.xlabel("")
plt.ylabel("Log Alpha Power Deviation (μV²)", fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM1/barplot_difficulty_main_effect.svg")
plt.show()


# 4. Lineplot of log_power over trial_number with fitted subject lines, group averages and grand average
group_colors = {'HC': 'teal', 'SZ': 'darkorange'}
plt.figure(figsize=(10, 6))
# Fitted line per subject
for subj in df_metadata['subject'].unique():
    df_sub = df_metadata[df_metadata['subject'] == subj]
    group = df_sub['Group'].iloc[0]
    color = group_colors.get(group)
    sns.regplot(
        data=df_sub,
        x="trial_number",
        y="log_power",
        scatter=False,
        ci=None,
        color=color,
        line_kws={'linewidth': 1, 'alpha': 0.4}
    )
# Fitted lines for group means
handles = []    # List for legend
for group, color in group_colors.items():
    df_group = df_metadata[df_metadata['Group'] == group]
    df_group_avg = df_group.groupby("trial_number")["log_power"].mean().reset_index()
    plot = sns.regplot(
        data=df_group_avg,
        x="trial_number",
        y="log_power",
        scatter=False,
        color=color,
        line_kws={'linewidth': 3}
    )
    handles.append(plt.Line2D([], [], color=color, linewidth=3, label=f'{group}'))  # Save information for label
# Fitted line for grand average
sns.regplot(
    data=df_metadata,
    x="trial_number",
    y="log_power",
    scatter=False,
    color="black",
    line_kws={'linewidth': 3, 'linestyle': '--'}
)
handles.append(plt.Line2D([], [], color='black', linewidth=3, label='Grand Average'))
# Finish plot
plt.xlabel("Trial Number", fontsize=14)
plt.ylabel("Log Alpha Power (μV²)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM1/lmplot_trial_number_subjects.svg")
plt.tight_layout()
plt.show()


# 5. Stripplot to check randomisation of difficulty over trial number
palette = {0: "cornflowerblue", 1: "tomato"}
df_metadata['x_dummy'] = "all_trials"
plt.figure(figsize=(6, 10))
sns.stripplot(data=df_metadata, x='x_dummy', y='trial_number', hue='difficulty',
              alpha=0.5, palette=palette, jitter=0.3, legend=False)
plt.xticks([])
plt.xlabel(None)
plt.ylabel('Trial Number', fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM1/difficulty_trialnumber.svg")
plt.show()


# --- TABLE WITH LIKELIHOOD RATIO TEST AND BAYES FACTOR FOR ALL REGRESSORS ---
# Model formula of the final model
full_formula = ("log_power ~ Group + prior_contrast_33_50 + prior_contrast_66_50 + dropped_epochs "
                "+ trial_number_centered + difficulty + response_time + Group:prior_contrast_33_50 "
                "+ Group:prior_contrast_66_50 + trial_number_centered:difficulty + trial_number_centered:Group")
# All terms explicitly listed
all_terms = [
    "Group",
    "prior_contrast_33_50",
    "prior_contrast_66_50",
    "dropped_epochs",
    "trial_number_centered",
    "difficulty",
    "response_time",
    "Group:prior_contrast_33_50",
    "Group:prior_contrast_66_50",
    "trial_number_centered:difficulty",
    "trial_number_centered:Group"
]
# Mapping of main effects and belonging interactions
interactions = {
    "Group": ["Group:prior_contrast_33_50", "Group:prior_contrast_66_50", "trial_number_centered:Group"],
    "prior_contrast_33_50": ["Group:prior_contrast_33_50"],
    "prior_contrast_66_50": ["Group:prior_contrast_66_50"],
    "trial_number_centered": ["trial_number_centered:difficulty", "trial_number_centered:Group"],
    "difficulty": ["trial_number_centered:difficulty"]
}
# Fit full model (REML set to false for LRT)
model_full = MixedLM.from_formula(full_formula, data=df_metadata, groups="subject")
result_full = model_full.fit(reml=False)
llf_full = result_full.llf
df_full = result_full.df_modelwc
# Collect results
results = []
# Iterate over all variables to get LLR test result for all regressors
for var in all_terms:
    # Reduced model: remove regressor and all associated interactions
    reduced_terms = [t for t in all_terms if t != var]
    for inter in interactions.get(var, []):
        reduced_terms = [t for t in reduced_terms if t != inter]
    reduced_formula = "log_power ~ " + " + ".join(reduced_terms)
    # Fit reduced model
    try:
        model_reduced = MixedLM.from_formula(reduced_formula, data=df_metadata, groups="subject")
        result_reduced = model_reduced.fit(reml=False)
        # LR-Test
        lr_stat = 2 * (llf_full - result_reduced.llf)
        df_diff = df_full - result_reduced.df_modelwc
        p_val = chi2.sf(lr_stat, df_diff)
        # Bayes Factor approx. from BIC
        bic_full = result_full.bic
        bic_reduced = result_reduced.bic
        bf_10 = np.exp((bic_reduced - bic_full) / 2)  # BF zugunsten des Vollmodells

        results.append({
            "Regressor_removed": var,
            "LLR_statistic": lr_stat,
            "df": df_diff,
            "p_value": p_val,
            "BayesFactor_approx": bf_10
        })
    except Exception as e:
        results.append({
            "Regressor_removed": var,
            "LLR_statistic": None,
            "df": None,
            "p_value": None,
            "BayesFactor_approx": None,
            "Error": str(e)
        })
# Sort results in a dataframe
results_df = pd.DataFrame(results).sort_values("p_value", na_position='last')
