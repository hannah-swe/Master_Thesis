import mne
import mne_icalabel
import os
import numpy as np
import matplotlib.pyplot as plt
from bad_channels import bads
plt.ion()
print("test")
# Set MNE logging level to INFO to get informative messages during processing
mne.set_log_level("INFO")

# Define preprocessing parameters in a dictionary for easy modification
params = dict(
    resampling_freq=250,  # Downsample frequency to reduce data size and speed up processing
    add_ref_channel="TP9",  # Reference channel to add before re-referencing
    ica_reject_threshold=0.9,  # Threshold for rejecting ICA components based on ICLabel probabilities
    highpass=0.1,  # High-pass filter cutoff frequency (Hz)
    lowpass=30,  # Low-pass filter cutoff frequency (Hz)
    epoch_tmin=-2,  # Start time of epochs relative to event (seconds)
    epoch_tmax=2  # End time of epochs relative to event (seconds)
)

# Paths to settings and raw data directories
settings_path = "/Users/hannahschewe/Documents/Uni/MA/Analysis/settings/"
data_path = "/Volumes/SSK Drive/Data/raw"

# List all subject IDs (folder names) in the raw data directory and sort them
sub_ids = sorted(os.listdir(data_path))

for subject_id in sub_ids[11:12]:
    # Uncomment to skip specific subjects if needed
    # if subject_id in ["SP_EEG_P0020"]:  # weird subjects
    #     continue

    print(f"Preprocessing subject {subject_id} ... ")

    # Construct full path to the subject's raw data folder
    raw_fp = os.path.join(data_path, subject_id)

    # Find all BrainVision header files (.vhdr) in the subject folder and sort them
    raw_filenames = sorted([x for x in os.listdir(raw_fp) if ".vhdr" in x])

    # Load each BrainVision raw file (if multiple, last one will be used)
    for raw_filename in raw_filenames:
        raw_brainvision = mne.io.read_raw_brainvision(raw_fp + "/" + raw_filename, preload=True)

    # Keep original raw data for backup
    raw_orig = raw_brainvision

    # Downsample the data
    raw = raw_orig.copy().resample(params['resampling_freq'])

    # Extract events and event IDs from annotations in the raw data
    events, event_id = mne.events_from_annotations(raw)

    # Get the event code for stimulus "Stimulus/S 20"
    stim_code = event_id["Stimulus/S 20"]

    # Count how many times this stimulus event occurs
    count = np.sum(events[:, 2] == stim_code)
    print(f'Anzahl der "Stimulus/S 20"-Events: {count}')  # Print count of stimulus events

    # Add a reference channel (e.g., TP9) to the data before re-referencing
    raw.add_reference_channels([params["add_ref_channel"]])

    # Load montage (electrode layout) and set it to the raw data
    montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
    raw.set_montage(montage)

    # Interpolate bad channels if any are specified for this subject
    try:
        bad = bads[subject_id]  # bads is a dictionary imported from bad_channels.py
    except KeyError:
        bad = None
    if bad:
        raw.info["bads"] = bad  # Mark bad channels in info
        raw.interpolate_bads()  # Interpolate bad channels

    # Set average reference for EEG data (common average referencing)
    raw.set_eeg_reference("average")

    # Filter the data between 1 and 100 Hz for ICA fitting (helps ICA separate sources better)
    raw_filt = raw.copy().filter(1, 100)

    # Initialize ICA object with Infomax algorithm and extended mode enabled
    ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))

    # Fit ICA decomposition on filtered data
    ica.fit(raw_filt)

    # Use ICLabel to automatically label ICA components (brain, eye, muscle, heart, line noise, other)
    ic_labels = mne_icalabel.label_components(raw_filt, ica, method="iclabel")

    # Identify ICA components to exclude based on label and probability threshold
    exclude_idx = [
        idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"]))
        if label not in ["brain", "other"] and prob > params["ica_reject_threshold"]
    ]
    print(f"Excluding these ICA components: {exclude_idx}")

    # Optionally, plot properties of excluded ICs for manual inspection (commented out)
    # ica.plot_properties(raw_filt, picks=exclude_idx)

    # Apply ICA correction to a copy of the original raw data, excluding bad components
    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)

    # Bandpass filter the reconstructed raw data between specified highpass and lowpass frequencies
    reconst_raw_filt = reconst_raw.copy().filter(params["highpass"], params["lowpass"])

    # Create directory for preprocessed data if it doesn't exist
    try:
        os.makedirs(f"{data_path}{subject_id}/derivatives/preprocessing")
    except FileExistsError:
        print("EEG derivatives preprocessing directory already exists")

    # Save the preprocessed raw data to disk in FIF format
    reconst_raw_filt.save(
        f"{data_path}{subject_id}/derivatives/preprocessing/{subject_id}_task-supratyp_raw.fif",
        overwrite=True
    )

    # Save the ICA solution to disk for later use or inspection
    ica.save(
        f"{data_path}{subject_id}/derivatives/preprocessing/{subject_id}_task-supratyp_ica.fif",
        overwrite=True
    )

    # Save the indices of excluded ICA components to a text file for record keeping
    with open(f"{data_path}{subject_id}/derivatives/preprocessing/{subject_id}_task-supratyp_ica_labels.txt", "w") as file:
        for item in exclude_idx:
            file.write(f"{item}\n")

    # Create epochs around events using the filtered, ICA-cleaned raw data
    epochs = mne.Epochs(
        reconst_raw_filt,
        events=events,
        event_id=event_id,
        preload=True,
        tmin=params["epoch_tmin"],
        tmax=params["epoch_tmax"],
        baseline=None,  # No baseline correction applied
        event_repeated="merge"  # Merge repeated events if any
    )

    # Create directory for epoch data if it doesn't exist
    try:
        os.makedirs(f"{data_path}{subject_id}/derivatives/epoching")
    except FileExistsError:
        print("EEG derivatives epoching directory already exists")

    # Save the epochs to disk in FIF format
    epochs.save(
        f"{data_path}{subject_id}/derivatives/epoching/{subject_id}_task-supratyp-epo_test.fif",
        overwrite=True
    )

    # Clean up variables to free memory
    del raw_orig, raw, raw_filt, reconst_raw_filt, reconst_raw, epochs, ica