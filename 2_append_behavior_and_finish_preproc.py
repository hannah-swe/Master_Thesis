import mne
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)

# Define epoching time window parameters
params = dict(epoch_tmin=-2, epoch_tmax=2)

# Define paths for settings and data
settings_path = "/Users/hannahschewe/Documents/Uni/MA/Analysis/settings/"
data_path = "/Volumes/SSK Drive/Data/derivatives"

# Get sorted list of subject IDs (folder names) in the derivatives data directory
sub_ids = sorted(os.listdir(data_path))

bad_subjects = []   # List to keep track of subjects with processing errors
event_id = {"20": 20}   # Define event ID dictionary for epoching (key "20" maps to event code 20)

# Iterate over a subset of subjects
for subject_id in sub_ids:
    try:
        # Extract subject number from folder name by splitting on underscores and parsing last parts
        components = subject_id.split("_")[2]
        if components[-2] == "0":
            subject = components[-1]
        else:
            subject = components[-2:]

        # Load preprocessed raw EEG data for the subject
        raw = mne.io.read_raw_fif(f"{data_path}/{subject_id}/preprocessing/{subject_id}_task-supratyp_raw.fif",
                                  preload=True)
        # Extract events and event IDs from annotations in the raw data
        events, _ = mne.events_from_annotations(raw)

        stim_code = event_id["20"]  # Get event code for stimulus "20"

        # Count how many times this stimulus event occurs in the data
        count = np.sum(events[:, 2] == stim_code)
        print(f'Anzahl der "Stimulus/S 20"-Events: {count}')
        # Create epochs around events of interest with specified time window, no baseline correction
        epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=params["epoch_tmin"],
                            tmax=params["epoch_tmax"], baseline=None, event_repeated="drop")

        # Select only epochs corresponding to stimulus "Stimulus/S 20"
        epochs = epochs["Stimulus/S 20"]

        # Load behavioral data from MATLAB .mat file corresponding to this subject's visual task
        mat_data_path_vis = f"/Users/hannahschewe/Documents/Uni/MA/Data/behav/sorted/P{subject}_vis_sorted_file.mat"
        mat_data_vis = scipy.io.loadmat(mat_data_path_vis)

        # Clean the loaded MATLAB data by removing metadata keys starting with "__"
        clean_mat_vis = dict()
        for k, v in mat_data_vis.items():
            if "__" not in k:
                clean_mat_vis[k] = v.flatten().tolist()

        # Remove unwanted keys that are not needed for analysis
        clean_mat_vis.pop("blockOrder_vis", None)
        clean_mat_vis.pop("corrAnswersTrain", None)

        # Convert cleaned behavioral data dictionary to a pandas DataFrame
        df_vis = pd.DataFrame(clean_mat_vis)

        # Reset index to create a 'trial_number' column for easier merging with epochs
        df_vis = df_vis.reset_index()
        df_vis = df_vis.rename(columns={'index': 'trial_number'})

        # Attach behavioral metadata to the epochs object
        epochs.metadata = df_vis

        # Define rejection criteria for bad epochs based on EEG signal amplitude
        flat = dict(eeg=1e-6)   # Reject epochs with flat signals below 1 microvolt
        reject = dict(eeg=250e-6)   # Reject epochs with peak-to-peak amplitude above 250 microvolts

        # Special handling for subject "8": mark channel "FC4" as bad and interpolate it
        if subject == "8":
            epochs.info["bads"] = ["FC4"]
            epochs.interpolate_bads()

        # Drop bad epochs based on rejection criteria and flat signals
        epochs_final = epochs.copy().drop_bad(reject=reject, flat=flat)

        # Create directory for epoch data if it doesn't exist
        try:
            os.makedirs(f"{data_path}/{subject_id}/epoching/")
        except FileExistsError:
            print("EEG derivatives epoching directory already exists")

        # Save the cleaned and epoched data to disk in FIF format
        epochs_final.save(f"{data_path}/{subject_id}/epoching/{subject_id}_task-supratyp-epo.fif",
                          overwrite=True)

        # Clean up variables to free memory
        del raw, epochs, epochs_final

    except:
        # If any error occurs during processing, add subject to bad_subjects list
        bad_subjects.append(subject_id)


# Save list of bad subjects to a text file for record keeping
with open("/Volumes/SSK Drive/Data/eeg/bad_subjects.txt", "w") as file:
    for item in bad_subjects:
        file.write(f"{item}\n")

# Load bad subjects list from file if it exists
try:
    with open("/Volumes/SSK Drive/Data/eeg/bad_subjects.txt", "r") as file:  # "r" for read mode
        bad_subjects = [line.strip() for line in file]
        print(bad_subjects)
except FileNotFoundError:
    print("File not found.")

# Handle bad subjects separately (reprocess or special handling)
for subject_id in sub_ids: # in bad_subjects
    # Extract subject number from folder name
    components = subject_id.split("_")[2]
    if components[-2] == "0":
        subject = components[-1]
    else:
        subject = components[-2:]

    # Load raw preprocessed data again
    raw = mne.io.read_raw_fif(f"{data_path}/{subject_id}/preprocessing/{subject_id}_task-supratyp_raw.fif",
                              preload=True)

    # Extract events and create epochs with merging repeated events
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=params["epoch_tmin"],
                        tmax=params["epoch_tmax"], baseline=None, event_repeated="merge")

    # Select epochs for event "20"
    epochs = epochs["20"]

    # Load behavioral data again for this subject
    mat_data_path_vis = f"/Users/hannahschewe/Documents/Uni/MA/Data/behav/sorted/P{subject}_vis_sorted_file.mat"
    mat_data_vis = scipy.io.loadmat(mat_data_path_vis)

    # Clean MATLAB data dictionary
    clean_mat_vis = dict()
    for k, v in mat_data_vis.items():
        if "__" not in k:
            clean_mat_vis[k] = v.flatten().tolist()

    # Remove unwanted keys
    clean_mat_vis.pop("blockOrder_vis", None)
    clean_mat_vis.pop("corrAnswersTrain", None)

    # Convert to DataFrame and reset index
    df_vis = pd.DataFrame(clean_mat_vis)
    df_vis = df_vis.reset_index()
    df_vis = df_vis.rename(columns={'index': 'trial_number'})

    # Attach behavioral metadata to epochs
    epochs.metadata = df_vis

    # Define rejection criteria again
    flat = dict(eeg=1e-6)
    reject=dict(eeg=250e-6)

    # Special handling for subject "8"
    if subject == "8":
        epochs.info["bads"] = ["FC4"]
        epochs.interpolate_bads()

    # Drop bad epochs
    epochs_final = epochs.copy().drop_bad(reject=reject, flat=flat)

    # Create epoching directory if needed
    try:
        os.makedirs(f"{data_path}/{subject_id}/epoching/")
    except FileExistsError:
        print("EEG derivatives epoching directory already exists")

    # Save final epochs
    epochs_final.save(f"{data_path}/{subject_id}/epoching/{subject_id}_task-supratyp-epo.fif",
                overwrite=True)

    # Clean up variables
    del raw, epochs, epochs_final
