import numpy as np
import os
import re
from scipy.io import loadmat

# Labels for each trial in every session
session1Labels = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
session2Labels = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
session3Labels = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

# Trial splits for testing and training
trialTestSess1 = [0, 1, 2, 3, 4, 5, 7, 15]
trialTestSess2 = [0, 1, 2, 3, 4, 5, 8, 14]
trialTestSess3 = [0, 1, 2, 3, 4, 5, 11, 15]

# Directory structure
base_dir = './SEED-IV/eeg_feature_smooth'

# Function to find all files for each session
def find_files(session_num):
    session_dir = os.path.join(base_dir, str(session_num))
    return [os.path.join(session_dir, f) for f in sorted(os.listdir(session_dir)) if f.endswith('.mat')]

# Function to create directory if it doesn't exist
def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_and_save_matrices(session_labels, test_trials, subject_number, session_number):
    # Find all .mat files for the session
    paths = find_files(session_number)

    train_list = []
    test_list = []
    train_labels = []  # Initialize a list to hold training labels
    test_labels = []   # Initialize a list to hold testing labels

    # Get test labels and prepare train labels
    trial_test_labels = [session_labels[i] for i in test_trials]
    trial_train_labels = [label for i, label in enumerate(session_labels) if i not in test_trials]

    

    # Iterate over the .mat files and create training and test sets
    for idx, path in enumerate(paths):
        procDataEEGExample = loadmat(path)
        
        # Check if the file is loaded correctly
        if not procDataEEGExample:
            #print(f"Error: File {path} could not be loaded.")
            continue
            
        keysProcDataEEG = [key for key in procDataEEGExample.keys() if key not in ['header', 'version', 'globals']]
        
        #print(f"Processing file: {path}")
        for t in range(24):  # Assuming there are 24 trials
            # Find keys matching 'de_LDS1' to 'de_LDS24'
            keys = [key for key in keysProcDataEEG if key.startswith('de_LDS') and int(re.search(r'\d+', key).group()) == t + 1]
            #print(f"Trial {t + 1} keys: {keys}")  # Debugging: print the keys for the trial
            
            for k in keys:
                dt = procDataEEGExample[k]

                # Check if the data exists
                if dt.size == 0:
                    #print(f"Warning: Data for trial {t + 1} in file {path} is empty.")
                    continue
                
                W = dt.shape[1] 
                #print(f"W (Trial {t + 1}): {W}")

                # Apply Min-Max normalization column-wise (feature-wise)
                trial_matrix = dt.transpose(0, 2, 1).reshape(62 * 5, W)
                print(trial_matrix.shape)
                # Check if the trial_matrix is empty before normalization
                if trial_matrix.size > 0:
                    min_value = trial_matrix.min(axis=0)  # Compute min across columns
                    max_value = trial_matrix.max(axis=0)  # Compute max across columns
                    range_vals = max_value - min_value
                    
                    # Check for zero range to avoid division by zero
                    range_vals[range_vals == 0] = 1  # Set zero ranges to 1 to avoid division by zero

                    normalized_matrix = (trial_matrix - min_value) / range_vals
                    data = normalized_matrix.T
                    print(data.shape)
                    # Check if the data is empty after normalization
                    if data.size == 0:
                        #print(f"Warning: Normalized data for trial {t + 1} in file {path} is empty after normalization.")
                        continue

                    if t in test_trials:
                        test_list.extend(data)  # Add all W trials to test list
                        print("test_list: ",len(test_list))
                        test_labels.extend([trial_test_labels[test_trials.index(t)]] * len(data))  # Add corresponding labels
                        #print(f"Test Trial {t + 1}: Added {len(data)} samples with label {trial_test_labels[test_trials.index(t)]}")
                    else:
                        train_list.extend(data)  # Add all W trials to train list
                        print("train_list: ",len(train_list))
                        train_labels.extend([trial_train_labels[trial_train_labels.index(session_labels[t])]] * len(data))  # Add corresponding labels
                        #print(f"Train Trial {t + 1}: Added {len(data)} samples with label {trial_train_labels[trial_train_labels.index(session_labels[t])]}")  # Fix this line to access the correct label
                else:
                    print(f"Warning: Trial matrix for trial {t + 1} in file {path} is empty.")
        
    # Convert to numpy arrays
    train_array = np.array(train_list)
    test_array = np.array(test_list)

    # Check if arrays are empty before saving
    if train_array.size == 0:
        print(f"Error: Train array for Subject {subject_number}, Session {session_number} is empty. Skipping saving.")
        return

    if test_array.size == 0:
        print(f"Error: Test array for Subject {subject_number}, Session {session_number} is empty. Skipping saving.")
        return

    # Ensure labels are consistent with the data
    train_labels_array = np.array(train_labels)
    test_labels_array = np.array(test_labels)
    print(train_array.size())
    print("\n")
    # Define paths for saving data
    train_de_path = './PARSE/DATA/SEED_IV/new/intra_session/train/de/{}_{}.npy'.format(subject_number, session_number)
    test_de_path = './PARSE/DATA/SEED_IV/new/intra_session/test/de/{}_{}.npy'.format(subject_number, session_number)
    train_label_path = './PARSE/DATA/SEED_IV/new/intra_session/train/label/{}_{}.npy'.format(subject_number, session_number)
    test_label_path = './PARSE/DATA/SEED_IV/new/intra_session/test/label/{}_{}.npy'.format(subject_number, session_number)

    
    # Create directories if they do not exist
    create_directory_if_not_exists(os.path.dirname(train_de_path))
    create_directory_if_not_exists(os.path.dirname(test_de_path))
    create_directory_if_not_exists(os.path.dirname(train_label_path))
    create_directory_if_not_exists(os.path.dirname(test_label_path))

    # Save the arrays
    np.save(train_de_path, train_array)
    np.save(test_de_path, test_array)
    np.save(train_label_path, train_labels_array)
    np.save(test_label_path, test_labels_array)

    #print(f"Data saved for subject {subject_number}, session {session_number}")

# Loop over all subjects and sessions
for subject_number in range(1, 16):  # Assuming subjects are 1 to 15
    create_and_save_matrices(session1Labels, trialTestSess1, subject_number, 1)
    create_and_save_matrices(session2Labels, trialTestSess2, subject_number, 2)
    create_and_save_matrices(session3Labels, trialTestSess3, subject_number, 3)