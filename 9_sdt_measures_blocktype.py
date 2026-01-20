import os
import mne
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")
# Read csv file with all metadata information
df = pd.read_csv('/Volumes/SSK Drive/Data/supratyp-dataframe-all-subjects.csv')

# Define path to safe SDT dataframe in wide-format for ANOVA with jamovi
save_csv_path = "/Volumes/SSK Drive/Data/supratyp-block-type-dataframe-all-subjects_short.csv"

# Function for computation of SDT measures
def sdt_measures(hits, misses, fas, crs):
    # Get hit and false alarm rate
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    fa_rate = fas / (fas + crs) if (fas + crs) > 0 else np.nan
    # log-linear correction to avoid Inf/NaN
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
    accuracy = group['correct'].mean() * 100

    # All variables that will be added to new dataframe
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
        "accuracy": accuracy,
        "Group": group_label
    })

# Apply SDT Function per subject and prior condition
df_sdt = df.groupby(["subject", "prior"]).apply(compute_sdt).reset_index()

# Save the dataframe in wide format
df_wide = df_sdt.pivot(index=['subject', 'Group'], columns='prior', values=['choice_prob', 'accuracy'])
df_wide.columns = df_wide.columns.to_flat_index()
rename_dict = {
    ('choice_prob', 33): 'choice_prob_33',
    ('choice_prob', 50): 'choice_prob_50',
    ('choice_prob', 66): 'choice_prob_66',
    ('accuracy', 33): 'accuracy_33',
    ('accuracy', 50): 'accuracy_50',
    ('accuracy', 66): 'accuracy_66'
}
df_wide = df_wide.rename(columns=rename_dict)
df_wide = df_wide.reset_index()
df_wide.to_csv(save_csv_path, index=False)

# Set-ups for plots
palette = {"HC": "teal", "SZ": "darkorange"}    # Define colors for groups
prior_labels = {33: "P-", 50: "P=", 66: "P+"}   # Define labels for prior conditions


# --- STATS CHOICHE PROBABILITY ---
# Get descriptive statistics of choice probability
summary_stats = df_sdt.groupby(["Group", "prior"])["choice_prob"].agg(
    mean="mean",
    std="std",
    min="min",
    max="max"
).reset_index()
summary_stats= summary_stats.round(2)


# --- STATS ACCURACY ---
# Get descriptive statistics of accuracy
summary_stats_accuracy = df_sdt.groupby(["Group", "prior"])["accuracy"].agg(
    mean="mean",
    std="std",
    min="min",
    max="max"
).reset_index()
summary_stats_accuracy= summary_stats_accuracy.round(2)


# --- PLOT CHOICE PROBABILITY PER PRIOR CONDITIONS ---
plt.figure(figsize=(7, 5))
# Calculate individual means
subject_means = df_sdt.groupby(["subject", "prior", "Group"])["choice_prob"].mean().reset_index()
# Plot group means and SE
sns.pointplot(data=df_sdt, x="prior", y="choice_prob", hue="Group", dodge=0.3, markers="o", capsize=0.1,
              err_kws={'linewidth': 1}, errorbar="ci", palette=palette, legend=False)
# Plot individual subject means
sns.stripplot(data=subject_means, x="prior", y="choice_prob", hue="Group", dodge=True, alpha=0.7, marker="o", size=5,
              jitter=True, palette=palette, legend=False)
# Plot grand average
grand_avg = df_sdt.groupby("prior")["choice_prob"].mean().reset_index()
sns.pointplot(data=grand_avg, x="prior", y="choice_prob", color="black", markers="o",
              errorbar=None, zorder=10, alpha=0.8, linestyles="--", legend=False)
plt.xlabel("Prior Category", fontsize=14)
plt.ylabel("Choice Probability (%)", fontsize=14)
plt.ylim(0, 79)  # y-axis from 0% to 80%
plt.grid(False)
sns.despine()
ax = plt.gca()
ax.set_xticklabels([prior_labels[int(tick.get_text())] for tick in ax.get_xticklabels()]) # Change x-ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/choice_prob.svg")
plt.show()


# --- PLOT ACCURACY PER PRIOR CONDITIONS ---
plt.figure(figsize=(7, 5))
# Calculate individual means
subject_means = df_sdt.groupby(["subject", "prior", "Group"])["accuracy"].mean().reset_index()
# Plot group means and SE
sns.pointplot(data=df_sdt, x="prior", y="accuracy", hue="Group", dodge=0.3, markers="o",
              capsize=0.1, err_kws={'linewidth': 1}, errorbar="ci", palette=palette, legend=False)
# Plot individual subject means
sns.stripplot(data=subject_means, x="prior", y="accuracy", hue="Group", dodge=True, alpha=0.6, marker="o",
              size=5, jitter=True, palette=palette, legend=False)
# Plot grand average
grand_avg = df_sdt.groupby("prior")["accuracy"].mean().reset_index()
sns.pointplot(data=grand_avg, x="prior", y="accuracy", color="black", markers="o",
              errorbar=None, zorder=10, alpha=0.8, linestyles="--", legend=False)
plt.xlabel("Prior Category", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.grid(False)
sns.despine()
ax = plt.gca()
ax.set_xticklabels([prior_labels[int(tick.get_text())] for tick in ax.get_xticklabels()]) # Change x-ticks
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/accuracy.svg")
plt.show()


# --- PLOT HIT RATE PER PRIOR CONDITIONS ---
plt.figure(figsize=(7, 5))
# Calculate individual means
subject_means = df_sdt.groupby(["subject", "prior", "Group"])["hit_rate_perc"].mean().reset_index()
# Plot group means and SE
sns.pointplot(data=df_sdt, x="prior", y="hit_rate_perc", hue="Group", dodge=0.3, markers="o", capsize=0.1,
              err_kws={'linewidth': 1}, errorbar="ci", palette=palette)
# Plot individual subject means
sns.stripplot(data=subject_means, x="prior", y="hit_rate_perc", hue="Group", dodge=True, alpha=0.7, marker="o", size=5,
              jitter=True, palette=palette, legend=False)
plt.xlabel("Prior Category")
plt.ylabel("Hit Rate (%)")
plt.ylim(0, 89)  # y-axis from 0% to 90%
plt.grid(True, axis='y', alpha=0.7)
ax = plt.gca()
ax.set_xticklabels([prior_labels[int(tick.get_text())] for tick in ax.get_xticklabels()]) # Change x-ticks
plt.legend(title=None, loc='upper right')
plt.tight_layout()
plt.show()


# --- PLOT FALSE ALARM RATE PER PRIOR CONDITIONS ---
plt.figure(figsize=(7, 5))
# Calculate individual means
subject_means = df_sdt.groupby(["subject", "prior", "Group"])["false_alarm_rate_perc"].mean().reset_index()
# Plot group means and SE
sns.pointplot(data=df_sdt, x="prior", y="false_alarm_rate_perc", hue="Group", dodge=0.3, markers="o",
              capsize=0.1, err_kws={'linewidth': 1}, errorbar="ci", palette=palette)
# Plot individual subject means
sns.stripplot(data=subject_means, x="prior", y="false_alarm_rate_perc", hue="Group", dodge=True, alpha=0.7, marker="o",
              size=5, jitter=True, palette=palette, legend=False)
plt.xlabel("Prior Category")
plt.ylabel("False Alarm Rate (%)")
plt.ylim(0, 89)  # y-axis from 0% to 90%
plt.grid(True, axis='y', alpha=0.7)
ax = plt.gca()
ax.set_xticklabels([prior_labels[int(tick.get_text())] for tick in ax.get_xticklabels()]) # Change x-ticks
plt.legend(title=None, loc='upper right')
plt.tight_layout()
plt.show()


# --- PLOT SENSITIVITY PER PRIOR CONDITIONS ---
plt.figure(figsize=(7, 5))
# Calculate individual means
subject_means = df_sdt.groupby(["subject", "prior", "Group"])["sensitivity"].mean().reset_index()
# Plot group means and SE
sns.pointplot(data=df_sdt, x="prior", y="sensitivity", hue="Group", dodge=0.3, markers="o",
              capsize=0.1, err_kws={'linewidth': 1}, errorbar="ci", palette=palette, legend=False)
# Plot individual subject means
sns.stripplot(data=subject_means, x="prior", y="sensitivity", hue="Group", dodge=True, alpha=0.6, marker="o",
              size=5, jitter=True, palette=palette, legend=False)
# Plot grand average
grand_avg = df_sdt.groupby("prior")["sensitivity"].mean().reset_index()
sns.pointplot(data=grand_avg, x="prior", y="sensitivity", color="black", markers="o",
              errorbar=None, zorder=10, alpha=0.8, linestyles="--", legend=False)
plt.xlabel("Prior Category", fontsize=14)
plt.ylabel("Sensitivity (d')", fontsize=14)
plt.grid(False)
ax = plt.gca()
ax.set_xticklabels([prior_labels[int(tick.get_text())] for tick in ax.get_xticklabels()]) # Change x-ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/sensitivity.svg")
plt.show()


# --- PLOT CRITERION PER PRIOR CONDITIONS ---
plt.figure(figsize=(7, 5))
# Calculate individual means
subject_means = df_sdt.groupby(["subject", "prior", "Group"])["criterion"].mean().reset_index()
# Plot group means and SE
sns.pointplot(data=df_sdt, x="prior", y="criterion", hue="Group", dodge=0.3, markers="o",
              capsize=0.1, err_kws={'linewidth': 1}, errorbar="ci", palette=palette, legend=False)
# Plot individual subject means
sns.stripplot(data=subject_means, x="prior", y="criterion", hue="Group", dodge=True, alpha=0.6, marker="o",
              size=5, jitter=True, palette=palette, legend=False)
# Plot grand average
grand_avg = df_sdt.groupby("prior")["criterion"].mean().reset_index()
sns.pointplot(data=grand_avg, x="prior", y="criterion", color="black", markers="o",
              errorbar=None, zorder=10, alpha=0.8, linestyles="--", legend=False)
plt.xlabel("Prior Category", fontsize=14)
plt.ylabel("Decision Bias (criterion)", fontsize=14)
plt.grid(False)
ax = plt.gca()
ax.set_xticklabels([prior_labels[int(tick.get_text())] for tick in ax.get_xticklabels()]) # Change x-ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/criterion.svg")
plt.show()


# --- CORRELATION HIT RATE AND LOG POWER ---
# Calculate correlation between hit rate and log power split by group
correlations = {}
for group in ["HC", "SZ"]:
    sub_df = df_sdt[df_sdt["Group"] == group]
    r, p = pearsonr(sub_df["hit_rate"], sub_df["log_power"])
    correlations[group] = (r, p)
    print(f"{group}: Pearson r = {r:.2f}, p = {p:.4f}")
# Plot correlation between hit rate and log power
figsize = (7, 5)
aspect_ratio = figsize[0] / figsize[1]
# Regressionsplot with both groups
g = sns.lmplot(data=df_sdt, x="hit_rate", y="log_power", hue="Group", palette=palette, markers="o",
               scatter_kws={"alpha": 0.6, "s": 50}, line_kws={"linewidth": 2}, height=figsize[1],
               aspect=aspect_ratio, legend=False)
ax = g.ax
g.set_axis_labels("Hit Rate (%)", "Log Alpha Power (μV²)")
ax.set_ylim(-24, -19.5)
# Get text box with r and p values
x_text = 0.78
y_start = 0.96
spacing = 0.06
for i, group in enumerate(["HC", "SZ"]):
    r, p = correlations[group]
    y = y_start - i * spacing
    # Add color as legend
    ax.add_patch(Rectangle((x_text - 0.15, y - 0.01), 0.025, 0.03,
                           transform=ax.transAxes, facecolor=palette[group], edgecolor=palette[group]))
    # Add text
    ax.text(x_text - 0.11, y, f"{group}: r = {r:.2f}, p = {p:.4f}", transform=ax.transAxes, fontsize=11,
            color="black", verticalalignment='center')
plt.grid(True, alpha=0.4)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/correlation_hit_rate_log_power.png", dpi=400)
plt.tight_layout()
plt.show()


# --- CORRELATION FALSE ALARM RATE AND LOG POWER ---
# Calculate correlation between false alarm rate and log power split by group
correlations = {}
for group in ["HC", "SZ"]:
    sub_df = df_sdt[df_sdt["Group"] == group]
    r, p = pearsonr(sub_df["false_alarm_rate"], sub_df["log_power"])
    correlations[group] = (r, p)
    print(f"{group}: Pearson r = {r:.2f}, p = {p:.4f}")
# Plot correlation between false alarm rate and log power
figsize = (7, 5)
aspect_ratio = figsize[0] / figsize[1]
# Regressionsplot with both groups
g = sns.lmplot(data=df_sdt, x="false_alarm_rate", y="log_power", hue="Group", palette=palette, markers="o",
               scatter_kws={"alpha": 0.6, "s": 50}, line_kws={"linewidth": 2}, height=figsize[1],
               aspect=aspect_ratio, legend=False)
ax = g.ax
g.set_axis_labels("False Alarm Rate (%)", "Log Alpha Power (μV²)")
ax.set_ylim(-24, -19.5)
# Get text box with r and p values
x_text = 0.78
y_start = 0.96
spacing = 0.06
for i, group in enumerate(["HC", "SZ"]):
    r, p = correlations[group]
    y = y_start - i * spacing
    # Add color as legend
    ax.add_patch(Rectangle((x_text - 0.15, y - 0.01), 0.025, 0.03,
                           transform=ax.transAxes, facecolor=palette[group], edgecolor=palette[group]))
    # Add text
    ax.text(x_text - 0.11, y, f"{group}: r = {r:.2f}, p = {p:.4f}", transform=ax.transAxes, fontsize=11,
            color="black", verticalalignment='center')
plt.grid(True, alpha=0.4)
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/behav/correlation_false_alarm_rate_log_power.png", dpi=400)
plt.tight_layout()
plt.show()

# --- CORRELATION SENSITIVITY AND LOG POWER ---
# Calculate correlation between sensitivity and log power split by group
correlations = {}
for group in ["HC", "SZ"]:
    sub_df = df_sdt[df_sdt["Group"] == group]
    r, p = pearsonr(sub_df["sensitivity"], sub_df["log_power"])
    correlations[group] = (r, p)
    print(f"{group}: Pearson r = {r:.2f}, p = {p:.4f}")
# Calculate correlation between sensitivity and log power for all subjects
r_all, p_all = pearsonr(df_sdt["sensitivity"], df_sdt["log_power"])
correlations["All Subjects"] = (r_all, p_all)
print(f"All Subjects: Pearson r = {r_all: .2f}, p = {p_all: .4f}")
# Plot correlation between sensitivity and log power split by group with grand average
figsize = (7, 5)
aspect_ratio = figsize[0] / figsize[1]
# Regression lines of both groups (HC vs. SZ)
g = sns.lmplot(data=df_sdt, x="log_power", y="sensitivity", hue="Group", palette=palette, markers="o",
               scatter_kws={"alpha": 0.6, "s": 50}, line_kws={"linewidth": 2}, height=figsize[1],
               aspect=aspect_ratio, legend=False)
ax = g.ax
# Regression line over all subjects (grand average)
sns.regplot(data=df_sdt, x="log_power", y="sensitivity", scatter=False, color="black", ax=ax,
            line_kws={"linewidth": 2, "linestyle": "--"})
plt.grid(False)
plt.xlabel("Log Alpha Power (μV²)", fontsize=14)
plt.ylabel("Sensitivity (d')", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM2/correlation_power_sensitivity.svg")
plt.show()


# --- CORRELATION CRITERION AND LOG POWER ---
# Calculate correlation between criterion and log power split by group
correlations = {}
for group in ["HC", "SZ"]:
    sub_df = df_sdt[df_sdt["Group"] == group]
    r, p = pearsonr(sub_df["criterion"], sub_df["log_power"])
    correlations[group] = (r, p)
    print(f"{group}: Pearson r = {r:.2f}, p = {p:.4f}")
# Calculate correlation between criterion and log power for all subjects
r_all, p_all = pearsonr(df_sdt["criterion"], df_sdt["log_power"])
correlations["All Subjects"] = (r_all, p_all)
print(f"All Subjects: Pearson r = {r_all: .2f}, p = {p_all: .4f}")
# Plot correlation between criterion and log power split by group with grand average
figsize = (7, 5)
aspect_ratio = figsize[0] / figsize[1]
# Regression lines of both groups (HC vs. SZ)
g = sns.lmplot(data=df_sdt, x="log_power", y="criterion", hue="Group", palette=palette, markers="o",
               scatter_kws={"alpha": 0.6, "s": 50}, line_kws={"linewidth": 2}, height=figsize[1],
               aspect=aspect_ratio, legend=False)
ax = g.ax
# Regression line over all subjects (grand average)
sns.regplot(data=df_sdt, x="log_power", y="criterion", scatter=False, color="black", ax=ax,
            line_kws={"linewidth": 2, "linestyle": "--"})
plt.grid(False)
plt.xlabel("Log Alpha Power (μV²)", fontsize=14)
plt.ylabel("Decision Bias (criterion)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
sns.despine()
plt.savefig("/Users/hannahschewe/Documents/Uni/MA/Analysis/LMM3/correlation_power_criterion.svg")
plt.show()
