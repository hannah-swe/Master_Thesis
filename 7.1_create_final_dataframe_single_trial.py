import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy

# Ensure proper visualization backend
plt.ion()
plt.style.use('fast')
os.environ.pop("MNE_QT_BACKEND", None)
mne.viz.set_browser_backend("matplotlib")
matplotlib.use("TkAgg", force=True)
print("test")
# Define data paths
data_path = "/Volumes/SSK Drive/Data/derivatives"   # Epoch files
demogr_data_path = "/Volumes/SSK Drive/Data/demographics/demographics_SUPRATYP_CBBM.csv"    # Demographical data
questionnaire_data_path = "/Volumes/SSK Drive/Data/demographics/questionnaire_SUPRATYP_CBBM.csv"    # Questionnaire data
panss_data_path = "/Volumes/SSK Drive/Data/clinical_data/PANSS_SUPRATYP_CBBM.xlsx"  # PANSS interviews
meds_data_path = "/Volumes/SSK Drive/Data/clinical_data/medication_SUPRATYP_CBBM.xlsx"  # Medication data
save_csv_path = "/Volumes/SSK Drive/Data/supratyp-dataframe-all-subjects.csv"   # Path to save final dataframe

sub_ids = sorted(os.listdir(data_path)) # List and sort all subject folders
all_epochs = list()     # For concatenating all epochs later

# Iterate over subjects
for subject_id in sub_ids:
    if subject_id in ["SP_EEG_P0020", "SP_EEG_P0058", "SP_EEG_P0065"]:  # Subjects to be excluded
        continue

    # Path to preprocessed epoched file
    subject_data = os.path.join(data_path, subject_id, "alpha_power", f"{subject_id}_task-supratyp-alpha-epo.fif")

    # Load epoched data
    epochs = mne.read_epochs(subject_data, preload=True)

    # Add subject number in metadata
    components = subject_id.split("_")[2]
    if components[-2] == "0":
        subject = components[-1]
    else:
        subject = components[-2:]
    epochs.metadata['subject'] = subject

    # Add group in metadata (is_hc = is healthy control)
    epochs.metadata['is_hc'] = epochs.metadata["subject"].astype(int) < 51
    epochs.metadata['Group'] = epochs.metadata["is_hc"].apply(lambda x: "HC" if x else "SZ")

    # Drop trials with no reaction and response times less than 150 ms
    n_before = len(epochs)
    epochs = epochs[epochs.metadata["respTimeDec_vis"] >= 0.150]
    n_after = len(epochs)
    n_dropped = n_before - n_after

    # Delete catch trials
    epochs = epochs[epochs.metadata["phaseCoherence"] < 85.00]

    # Write .txt file with number of dropped trials
    with open(f"{data_path}/{subject_id}/epoching/{subject_id}_number_epochs_dropped_reaction_time.txt", "w") as file:
        file.write(f"{n_dropped}")

    # Add number of dropped epochs to metadata
    n_epochs = len(epochs)
    n_epochs_dropped = 270 - n_epochs
    epochs.metadata['dropped_epochs'] = n_epochs_dropped

    # Add variable on prior information
    epochs.metadata['prior_information'] = epochs.metadata["percentagePrior"].isin([33, 66])

    # --- ADD DEMOGRAPHIC DATA ---
    # Load csv dataframe with demographical information
    demogr_df = pd.read_csv(demogr_data_path)
    demogr_df["cbbm_part_id"] = demogr_df["cbbm_part_id"].astype(str)
    epochs.metadata["subject"] = epochs.metadata["subject"].astype(str)

    # Merge metadata and demographic data
    merged_metadata = epochs.metadata.merge(demogr_df, how="left", left_on="subject", right_on="cbbm_part_id")
    epochs.metadata = merged_metadata

    # Delete all not needed variables
    epochs.metadata = epochs.metadata.drop(
        columns=["respTimeConf_vis", "responseConf_vis", "fileNum", "Task_Name", "cbbm_part_id", "Exp_Subject_Id",
                 "Schulabschluss", "Musikalitaet", "Alter_Musik", "Hoerhilfe", "Hoerverlust", "ADHS", "Legasthenie",
                 "Behandlung", "schizoaffektiveStoerung", "Sehschwche", "Sehschwche_Dauer", "Sehschwche_type",
                 "Sehschwche_type_andere", "Sehschwche_type_astigm", "Sehschwche_type_keine", "Sehschwche_type_kurz",
                 "Sehschwche_type_raum", "Sehschwche_type_weit", "Sehschwche_type_winkel",
                 "Sehschwche_type1", "Sehschwche_typeII", "completed", "end_time",
                 "rec_session_id", "session_name", "start_time"])

    # Rename variables for clarity
    epochs.metadata = epochs.metadata.rename(
        columns={"imageOrderExp_vis": "image_order", "respTimeDec_vis": "response_time", "responseDec_vis": "response",
                 "generalCategory": "stimuli_category", "percentagePrior": "prior", "phaseCoherence": "difficulty",
                 "Alter": "age", "Geschlecht": "sex", "Bildungsgrad": "education_years"})

    # Change values of sex
    epochs.metadata["sex"] = epochs.metadata["sex"].replace({"W": "female", "M": "male"})

    # Change values of difficulty (easy: 0; hard: 1)
    epochs.metadata["difficulty"] = epochs.metadata["difficulty"].replace({22.5: 0, 17.5: 1})

    # --- ADD QUESTIONNAIRE DATA ---
    # Load csv dataframe with questionnaire data
    question_df = pd.read_csv(questionnaire_data_path)
    question_df["CBBM_ID"] = question_df["CBBM_ID"].astype(str)
    epochs.metadata["subject"] = epochs.metadata["subject"].astype(str)

    # Merge metadata and questionnaire data
    merged_metadata = epochs.metadata.merge(question_df, how="left", left_on="subject", right_on="CBBM_ID")
    epochs.metadata = merged_metadata

    # Delete all not needed variables
    epochs.metadata = epochs.metadata.drop(
        columns=["CBBM_ID", "ID", "BFI_neurotic_sum", "BFI_neurotic_avg", "BFI_openness_sum", "BFI_openness_avg",
                 "BFI_agree_sum", "BFI_agree_avg", "BFI_extrov_sum", "BFI_extrov_avg", "BFI_conscient_sum",
                 "BFI_conscient_avg", "BFI_control", "SPQ_cogPercep", "SPQ_interpersonal", "SPQ_disorganized",
                 "SPQ_control", "PDI_distress_sum", "PDI_preoccupation_sum", "PDI_conviction_sum", "LSHS_control"])

    # --- ADD DIFFICULTY OF PREVIOUS TRIALS ---
    # Load behavioral data visual
    mat_data_path_vis = f"/Users/hannahschewe/Documents/Uni/MA/Data/behav/sorted/P{subject}_vis_sorted_file.mat"    # Data path to behavioral data
    mat_data_vis = scipy.io.loadmat(mat_data_path_vis)
    clean_mat_vis = dict()
    for k, v in mat_data_vis.items():
        if "__" not in k:
            clean_mat_vis[k] = v.flatten().tolist()
    clean_mat_vis.pop("blockOrder_vis", None)
    clean_mat_vis.pop("corrAnswersTrain", None)

    # Load dataframe
    df_difficulty = pd.DataFrame(clean_mat_vis)
    df_difficulty = df_difficulty.reset_index()  # Create 'index' column
    df_difficulty = df_difficulty.rename(columns={'index': 'trial_number'})  # Rename for clarity
    df_difficulty["difficulty"] = df_difficulty["phaseCoherence"].copy().replace({22.5: 0, 17.5: 1, 85.0: -1})

    # Delete variables
    df_difficulty = df_difficulty.drop(columns=["imageOrderExp_vis", "respTimeConf_vis", "respTimeDec_vis", "responseConf_vis",
                                  "responseDec_vis", "block", "fileNum", "generalCategory", "percentagePrior",
                                  "phaseCoherence"])

    # Add difficulty of the last trial
    df_prev_difficulty = df_difficulty.copy()
    df_prev_difficulty['trial_number'] = df_prev_difficulty['trial_number'] + 1
    df_prev_difficulty = df_prev_difficulty.rename(columns={'difficulty': 'difficulty_previous'})

    # Add difficulty of the penultimate trial
    df_prev2_difficulty = df_difficulty.copy()
    df_prev2_difficulty['trial_number'] = df_prev2_difficulty['trial_number'] + 2
    df_prev2_difficulty = df_prev2_difficulty.rename(columns={'difficulty': 'difficulty_previous2'})

    # Add both variables to metadata
    epochs.metadata = epochs.metadata.merge(df_prev_difficulty[['trial_number', 'difficulty_previous']],
                                            on='trial_number', how='left')
    epochs.metadata = epochs.metadata.merge(df_prev2_difficulty[['trial_number', 'difficulty_previous2']],
                                            on='trial_number', how='left')

    # --- ADD CONFIDENCE CHOICE OF PREVIOUS TRIAL ---
    # Load dataframe
    df_confidence = pd.DataFrame(clean_mat_vis)
    df_confidence = df_confidence.reset_index()  # Create 'index' column
    df_confidence = df_confidence.rename(columns={'index': 'trial_number'})  # Rename for clarity

    # Delete variables
    df_confidence = df_confidence.drop(columns=["imageOrderExp_vis", "respTimeConf_vis", "respTimeDec_vis",
                                  "responseDec_vis", "block", "fileNum", "generalCategory", "percentagePrior",
                                  "phaseCoherence"])

    # Add confidence of the last trial to metadata
    df_confidence['trial_number'] = df_confidence['trial_number'] + 1
    df_confidence = df_confidence.rename(columns={'responseConf_vis': 'confidence_previous'})
    epochs.metadata = epochs.metadata.merge(df_confidence[['trial_number', 'confidence_previous']],
                                            on='trial_number', how='left')

    # --- ADD CLINICAL DATA (PANSS AND MEDICATION DATA FOR SZ)M---
    # Load PANSS data
    panss_df = pd.read_excel(panss_data_path)
    panss_df["subID"] = panss_df["subID"].astype(str)
    epochs.metadata["subject"] = epochs.metadata["subject"].astype(str)

    # Create positive and negative sum scores
    panss_df["panss_positive_sum"] = panss_df[["P1", "P2", "P3", "P4", "P5", "P6", "P7"]].sum(axis=1)
    panss_df["panss_negative_sum"] = panss_df[["N1", "N2", "N3", "N4", "N5", "N6", "N7"]].sum(axis=1)
    merge_df = panss_df[["subID", "panss_positive_sum", "panss_negative_sum"]]

    # Merge metadata and PANSS data
    epochs.metadata = epochs.metadata.merge(merge_df, left_on="subject", right_on="subID", how="left")
    epochs.metadata.drop(columns=["subID"], inplace=True)

    # Load medication data
    meds_df = pd.read_excel(meds_data_path)
    meds_df["ID #"] = meds_df["ID #"].astype(str)
    epochs.metadata["subject"] = epochs.metadata["subject"].astype(str)

    # Merge metadata and medication data
    meds_df = meds_df[["ID #", "Unnamed: 3"]].rename(columns={"Unnamed: 3": "chlorpromazine_equi"})
    epochs.metadata = epochs.metadata.merge(meds_df, left_on="subject", right_on="ID #", how="left")
    epochs.metadata.drop(columns=["ID #"], inplace=True)

    # --- ANALYSE BEHAVIORAL DATA ---
    # Get variable for 2x2 decision outcome (hit, correct rejection, false alarm, miss)
    def create_dec_outcome(row):
        if row['response'] == 1 and row['stimuli_category'] == 1:
            return 'H'
        elif row['response'] == 2 and row['stimuli_category'] == 2:
            return 'CR'
        elif row['response'] == 1 and row['stimuli_category'] == 2:
            return 'FA'
        elif row['response'] == 2 and row['stimuli_category'] == 1:
            return 'M'
        else:
            return None
    epochs.metadata['dec_outcome'] = epochs.metadata.apply(create_dec_outcome, axis=1)

    # Get trial-wise performance (correct/incorrect)
    epochs.metadata['correct'] = epochs.metadata["dec_outcome"].isin(["H", "CR"])

    # Save as a new file with metadata including all needed information
    try:
        os.makedirs(f"{data_path}/{subject_id}/final_frame")
    except FileExistsError:
        print("EEG derivatives final_frame directory already exists")
    epochs.save(f"{data_path}/{subject_id}/final_frame/{subject_id}_task-supratyp-alpha-demogr-epo.fif",
                overwrite=True)

    all_epochs.append(epochs)

# Concatenate all subject epochs
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore")

# Save complete dataframe as .csv
df_metadata = epochs.metadata
df_metadata.to_csv(save_csv_path, index=False)
