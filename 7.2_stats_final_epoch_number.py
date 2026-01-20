import mne
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Ensure proper MNE visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)

data_path = "/Volumes/SSK Drive/Data/derivatives" # Define data path
sub_ids = sorted(os.listdir(data_path))     # List and sort all subject folders
subject_epochs = [] # List to hold results for each subject

# --- Loop over all subject folders ---
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  #Ssubjects to be excluded
        continue

    # Build the path to the subject's preprocessed EEG epochs file
    subject_data = os.path.join(data_path, subject_id, "final_frame", f"{subject_id}_task-supratyp-alpha-demogr-epo.fif")

    # Load epochs using MNE, with preload=True to load data into memory
    epochs = mne.read_epochs(subject_data, preload=True)

    # Extract numerical subject ID
    subject_num = subject_id.split("_")[2]
    subject_str = subject_num[1:]
    subject_int = int(subject_str)
    is_hc = subject_int < 51    # Determine group: Healthy Control (HC) if ID < 51, otherwise patient (SC)

    # Get the number of retained epochs for this subject
    n_epochs = len(epochs)

    # Store subject-level data in a list of dictionaries
    subject_epochs.append({
        'subject_id': subject_id,
        'subject': subject_int,
        'is_hc': is_hc,
        'n_epochs': n_epochs
    })

# --- Convert list of dicts into a pandas DataFrame ---
df_epochs = pd.DataFrame(subject_epochs)

# --- Output overall descriptive statistics for number of epochs ---
print(df_epochs['n_epochs'].describe())  # Count, mean, std, min, max, etc.

# --- Compute group-level mean and standard deviation of epochs ---
epoch_stats = df_epochs.groupby("is_hc")['n_epochs'].agg(['mean', 'std']).reset_index()