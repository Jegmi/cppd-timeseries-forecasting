"""
Helper functions (get_person_data, get_nan_imputed_array, pixel_to_torch) handle tasks like imputation and reorganization of data, which makes sense in a helper module.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
import tqdm
import datetime
from features.preprocessing import extract_hours, extract_weekdays, extract_days_since_2020, extract_days_since_2020_array

def trange(t_min, seq_length):
    """Generate a time sequence in minutes starting from t_min."""
    return np.array([t_min + datetime.timedelta(minutes=int(i + 1)) for i in np.arange(seq_length)])


def save_pkl(outputpath: Path, pkl_file: str, data, logger = None) -> None:
    """
    Save the provided data object as a pickle file.

    This function saves the provided data object to a specified output path with a given file name. 
    It ensures the directory exists and appends the .pkl suffix to the file name if it is not already present.
    Optionally, it logs the save operation.

    Args:
        outputpath (Path): The path to the directory where the pickle file should be saved.
        pkl_file (str): The name of the pickle file. If it doesn't already end with '.pkl', the suffix will be added.
        data: The data object to be pickled and saved.
        logger (logging.Logger, optional): An optional logger to log the save operation.

    Returns:
        None

    Example usage
        outputpath = Path(PROCESSED_PATH)
        save_pkl(outputpath, 'static_XY', static_XY, logger)
        
    """
    outputpath.mkdir(parents=True, exist_ok=True)
    
    # Ensure the file name ends with .pkl
    if not pkl_file.endswith('.pkl'):
        pkl_file += '.pkl'
        
    file_path = outputpath / pkl_file
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    
    if logger:
        logger.info(f"Saved to {file_path}")


def get_person_data(raw_data, p_idx):
    n_seq = len(raw_data[p_idx])
    assert n_seq, f'no data for p_idx {p_idx}'
    dfs = []
    for i in tqdm.tqdm(range(n_seq)):
        tlin, fitbit = raw_data[p_idx][i].values()
        seq = np.c_[
            extract_days_since_2020(tlin), 
            extract_weekdays(tlin), 
            extract_hours(tlin), 
            fitbit
        ]
        df = pd.DataFrame(seq, columns=['index','day','hour','act','hr'])
        dfs.append(df)
    return pd.concat(dfs)

def get_nan_imputed_array(dfs):
    hours = sorted(dfs.hour.unique())
    X_imputed = []
    for i in dfs['index'].unique():
        X_imputed.append([])
        day_index = dfs['index'] == i
        for hour in hours:
            x_imputed = np.nan        
            if hour in dfs[day_index]['hour'].values:
                x_imputed = dfs[day_index][dfs[day_index]['hour'] == hour]['act']                            
            X_imputed[-1].append(x_imputed)
    return np.array(X_imputed).astype(float)

def yield_subject_arrays(all_data):
    """
    Generator function to yield processed data for each subject from the input iterator.

    Args:
        all_data (iterable): An iterable containing data samples for multiple subjects.

    Yields:
        Tuple: A tuple containing hours, hours_y, weekdays, days, days_y, x, y for each subject.
    """

    to_arr = lambda lists : np.squeeze(np.array(lists))
    
    hours, hours_y, weekdays, days, days_y, x, y = [], [], [], [], [], [], []

    subj_id = None  # Initialize subject ID

    for sample in all_data:
        current_subj_id = int(np.unique(sample[4]))  # Get current subject ID

        if subj_id is None:  # If it's the first subject, set the ID
            subj_id = current_subj_id

        if current_subj_id != subj_id:  # If we encounter a new subject
            # Yield processed data for the previous subject
            yield (to_arr(hours), to_arr(hours_y), to_arr(weekdays), to_arr(days), to_arr(days_y), to_arr(x), to_arr(y))

            # Reset lists for the new subject
            hours, hours_y, weekdays, days, days_y, x, y = [], [], [], [], [], [], []
            subj_id = current_subj_id  # Update to the new subject ID

        # Process the current sample
        hours.append(extract_hours(sample[2]))
        hours_y.append(extract_hours(sample[3]))
        weekdays.append(extract_weekdays(sample[2]))
        days.append(extract_days_since_2020_array(sample[2]))
        days_y.append(extract_days_since_2020_array(sample[3]))
        x.append(sample[0])
        y.append(sample[1])

    # Yield the last subject's data if any
    if hours:  # Check if there are any processed hours for the last subject
        yield (to_arr(hours), to_arr(hours_y), to_arr(weekdays), to_arr(days), to_arr(days_y), to_arr(x), to_arr(y))