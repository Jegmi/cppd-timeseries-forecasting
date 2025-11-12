# Consider clean up: mv run_seq() -> run(), rm get_xy()

#import glob

import tqdm
#from matplotlib import pyplot as plt
import os
#import seaborn as sns
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, matthews_corrcoef,
    confusion_matrix
)


# ========== Evaluation ==========

def safe_pearsonr(x, y):
    """Return Pearson correlation or np.nan if undefined."""
    if len(x) < 2:
        return np.nan
    if np.all(x == x.iloc[0]) or np.all(y == y.iloc[0]):
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def regression_metrics(y_true, y_pred):
    """Compute regression metrics for numeric series (ignores NaNs)."""
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'corr': np.nan, 'n': 0}
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    corr = safe_pearsonr(yt, yp)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr, 'n': int(mask.sum())}

def thresholded_classification_metrics(y_true_bin, y_prob, threshold=0.5):
    """
    Compute classification metrics at a single threshold.
    y_true_bin: binary (0/1) Series
    y_prob: probability-like Series (0..1)
    Returns dict with sensitivity, specificity, precision, npv, percent_agreement, f1, mcc, support_pos, support_neg
    """
    mask = y_true_bin.notna() & y_prob.notna()
    if mask.sum() == 0:
        return {k: np.nan for k in ['sensitivity','specificity','precision','npv','percent_agreement','f1','mcc','tp','tn','fp','fn','n']}
    y_true = y_true_bin[mask].astype(int)
    y_prob = y_prob[mask].astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # confusion matrix: TN, FP, FN, TP
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except ValueError:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # metrics with safe divisions
    def safe_div(a, b):
        return (a / b) if b != 0 else np.nan

    sensitivity = safe_div(tp, tp + fn)   # recall, TPR
    specificity = safe_div(tn, tn + fp)   # TNR
    precision = safe_div(tp, tp + fp)     # PPV
    npv = safe_div(tn, tn + fn)           # NPV
    percent_agreement = safe_div(tp + tn, tp + tn + fp + fn)
    
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        f1 = np.nan
    
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'percent_agreement': percent_agreement,
        'f1': f1,
        'mcc': mcc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n': int(mask.sum())
    }

def classification_curve_metrics(y_true_bin, y_prob):
    """
    Compute ROC-AUC and PR-AUC (average precision). If not computable, return np.nan.
    """
    mask = y_true_bin.notna() & y_prob.notna()
    if mask.sum() == 0:
        return {'roc_auc': np.nan, 'pr_auc': np.nan, 'n': 0}
    y_true = y_true_bin[mask].astype(int)
    y_prob = y_prob[mask].astype(float)
    
    try:
        roc = roc_auc_score(y_true, y_prob) if (y_true.sum() > 0 and (y_true == 0).sum() > 0) else np.nan
    except Exception:
        roc = np.nan
    try:
        pr = average_precision_score(y_true, y_prob) if (y_true.sum() > 0 and (y_true == 0).sum() > 0) else np.nan
    except Exception:
        pr = np.nan
    return {'roc_auc': roc, 'pr_auc': pr, 'n': int(mask.sum())}


def evaluate_run(
    df,
    meta_cols=None,
    thresholds=[0.01,1/3], # SB, MVPA
    do_threshold_sweep=False,
    sweep_thresholds=None
):
    """
    Evaluate one run DataFrame and return 4 structured tables:
    1. metadata: Single row with run configuration
    2. regression: Metrics for activity and heart rate across sleep conditions
    3. classification_auc: ROC-AUC and PR-AUC metrics across conditions
    4. classification_threshold: Threshold-based metrics across conditions
    
    Returns
    -------
    dict with keys: 'metadata', 'regression', 'classification_auc', 'classification_threshold'
    """
    if meta_cols is None:
        meta_cols = ['valid_time', 'model_name', 'field', 'model', 'kernel_day', 
                     'kernel_hour', 'time_step', 'alpha_reg', 'forecasting_modality', 
                     'train_hours', 'valid_days', 'nan_days', 'total_days', 'pid']

    # ========== 1. METADATA TABLE ==========
    metadata = {}
    for col in meta_cols:
        metadata[col] = df.iloc[0][col] if col in df.columns else np.nan
    
    # Add sleep summary counts
    sleep_col = 'sleep'
    if sleep_col not in df.columns:
        raise KeyError(f"Expected column '{sleep_col}' in df")
    
    metadata['sleep_n_nan'] = int(df[sleep_col].isna().sum())
    metadata['sleep_n_true'] = int((df[sleep_col] == True).sum())
    metadata['sleep_n_false'] = int((df[sleep_col] == False).sum())
    metadata['n_total_rows'] = int(len(df))
    
    metadata_df = pd.DataFrame([metadata])

    # Helper to filter by sleep condition
    def subset(condition):
        if condition == 'all':
            return df
        elif condition == 'sleep_true':
            return df[df[sleep_col] == True]
        elif condition == 'sleep_false':
            return df[df[sleep_col] == False]
        else:
            raise ValueError(f"Unknown condition: {condition}")

    # Prepare classification targets
    df = df.copy()
    n_before = len(df)
    df = df[~df['activity_true'].isna()]
    n_after = len(df)
    
    if 'activity_pred' in df.columns:
        preds = pd.to_numeric(df['activity_pred'], errors='coerce')
        assert (preds.max() <= 1) & (preds.min() >= 0), 'predictions must be in [0, 1]'
    
    # Define binary classification targets
    df['absent_sedentary'] = (df['activity_true'] > 0).astype(float)
    df['mvpa'] = (df['activity_true'] > (1/3)).astype(float)
    
    metadata_df['n_before_dropna'] = n_before
    metadata_df['n_after_dropna'] = n_after

    # Sleep conditions to evaluate
    sleep_conditions = ['all', 'sleep_false', 'sleep_true']
    
    # ========== 2. REGRESSION TABLE ==========
    regression_rows = []
    tasks = {
        'activity': ('activity_true', 'activity_pred'),
        'heart': ('heart_true', 'heart_pred')
    }
    
    for task_name, (col_true, col_pred) in tasks.items():
        for cond in sleep_conditions:
            subdf = subset(cond)
            metrics = regression_metrics(
                subdf[col_true] if col_true in subdf.columns else pd.Series(dtype=float),
                subdf[col_pred] if col_pred in subdf.columns else pd.Series(dtype=float)
            )
            row = {'task': task_name, 'condition': cond}
            row.update(metrics)
            regression_rows.append(row)
    
    regression_df = pd.DataFrame(regression_rows)
    
    # ========== 3. CLASSIFICATION AUC TABLE ==========
    classification_auc_rows = []
    
    for target_name in ['absent_sedentary', 'mvpa']:
        for cond in sleep_conditions:
            subdf = subset(cond)
            metrics = classification_curve_metrics(
                subdf[target_name] if target_name in subdf else pd.Series(dtype=float),
                subdf['activity_pred'] if 'activity_pred' in subdf else pd.Series(dtype=float)
            )
            
            # Add label balance info
            if target_name in subdf.columns:
                mask = subdf[target_name].notna()
                n = int(mask.sum())
                if n > 0:
                    pos = int((subdf.loc[mask, target_name] == 1).sum())
                    balance = pos / n
                else:
                    balance = np.nan
            else:
                balance = np.nan
                n = 0
            
            row = {
                'target': target_name,
                'condition': cond,
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'n': metrics['n'],
                'balance': balance
            }
            classification_auc_rows.append(row)
    
    classification_auc_df = pd.DataFrame(classification_auc_rows)
    
    # ========== 4. CLASSIFICATION THRESHOLD TABLE ==========
    classification_threshold_rows = []
    
    for (target_name, threshold_for_target) in zip(['absent_sedentary', 'mvpa'], thresholds):
        for cond in sleep_conditions:
            subdf = subset(cond)
            metrics = thresholded_classification_metrics(
                subdf[target_name] if target_name in subdf else pd.Series(dtype=float),
                subdf['activity_pred'] if 'activity_pred' in subdf else pd.Series(dtype=float),
                threshold=threshold_for_target
            )
            row = {'target': target_name, 'condition': cond, 'threshold': threshold_for_target}
            row.update(metrics)
            classification_threshold_rows.append(row)
    
    classification_threshold_df = pd.DataFrame(classification_threshold_rows)
    
    # Optional: Add threshold sweep results
    if do_threshold_sweep:
        if sweep_thresholds is None:
            sweep_thresholds = np.linspace(0, 1, 101)
        
        sweep_rows = []
        for target_name in ['absent_sedentary', 'mvpa']:
            for thr in sweep_thresholds:
                for cond in sleep_conditions:
                    subdf = subset(cond)
                    metrics = thresholded_classification_metrics(
                        subdf[target_name] if target_name in subdf else pd.Series(dtype=float),
                        subdf['activity_pred'] if 'activity_pred' in subdf else pd.Series(dtype=float),
                        threshold=thr
                    )
                    # Only keep key metrics for sweep to reduce size
                    row = {
                        'target': target_name,
                        'condition': cond,
                        'threshold': thr,
                        'f1': metrics.get('f1', np.nan),
                        'mcc': metrics.get('mcc', np.nan),
                        'sensitivity': metrics.get('sensitivity', np.nan),
                        'specificity': metrics.get('specificity', np.nan)
                    }
                    sweep_rows.append(row)
        
        threshold_sweep_df = pd.DataFrame(sweep_rows)
        
        return {
            'metadata': metadata_df,
            'regression': regression_df,
            'classification_auc': classification_auc_df,
            'classification_threshold': classification_threshold_df,
            'threshold_sweep': threshold_sweep_df
        }
    
    return {
        'metadata': metadata_df,
        'regression': regression_df,
        'classification_auc': classification_auc_df,
        'classification_threshold': classification_threshold_df
    }


# ========== Preprocessing ==========
    
def trim_nan_rows(val):
    """Remove leading and trailing rows that contain only NaN values.
    
    Parameters
    ----------
    val : np.ndarray
        2D array with shape (days, hours) containing time series data.
    
    Returns
    -------
    np.ndarray
        Trimmed array with all-NaN rows removed from the beginning and end.
        Returns empty array with shape (0, n_cols) if all rows are NaN.
    """
    # Identify rows that are completely NaN
    all_nan_rows = np.all(np.isnan(val), axis=1)
    
    # Find the first and last index of a row that is not all-NaN
    valid_indices = np.where(~all_nan_rows)[0]
    if valid_indices.size > 0:
        start = valid_indices[0]
        end = valid_indices[-1] + 1  # slice is exclusive on the end
        val_trimmed = val[start:end]
    else:
        val_trimmed = val[0:0]  # empty slice if all rows are NaN
    return val_trimmed


def load_and_process_grid(p_idx, data_path, modality='activity', 
                          n_kernel_recent=5, kernel_day=3, verbose=0):
    """Load and preprocess 2D time series data for a single patient.
    
    Loads patient data, trims NaN rows, applies padding, and initializes
    temporal context by copying nighttime data from the previous day.
    
    Parameters
    ----------
    p_idx : int
        Patient index to load from the dataset.
    data_path : str
        Path to the .npz file containing the data.
    modality : str, optional
        Data modality to extract (default: 'activity').
    n_kernel_recent : int, optional
        Number of steps for horizontal padding/context (default: 5).
    kernel_day : int, optional
        Number of days for vertical padding/context (default: 3).
    verbose : int, optional
        Print ID and valid / nan days (default : 0).
    
    Returns
    -------
    arr2 : np.ndarray
        Processed 2D time series with shape (n_days, n_hours).
    valid_days : float
        Number of days with valid (non-NaN) data.
    nan_days : float
        Number of days with NaN data.
    """
    
    # Load data
    loaded = np.load(data_path, allow_pickle=True)
    metadata = pd.DataFrame.from_records(loaded['metadata']).set_index(
        'patient_count').loc[p_idx]
    steps_per_day = metadata['n_time_bins']
    
    # Select modality
    m_idx = np.where(loaded['modalities'] == modality)[0]
    timeseries_2d = loaded['data'][p_idx][:,:,m_idx]
    
    arr = np.squeeze(trim_nan_rows(timeseries_2d))
    
    # Calculate valid and NaN days
    valid_days = np.sum(~np.isnan(arr)) / steps_per_day
    nan_days = np.sum(np.isnan(arr)) / steps_per_day
    if verbose:
        print(f'patient idx: {p_idx}. Valid data {valid_days:.1f}d, '
              f'nan data: {nan_days:.1f}d, total days {len(arr)}')    
    
    # Padding for kernel initialization
    max_pad = n_kernel_recent
    mean_value = np.mean(arr[~np.isnan(arr)])
    arr2 = np.pad(arr.copy(), pad_width=[(max_pad, max_pad)], 
                  mode='constant', constant_values=mean_value)
    arr2 = arr2[:-max_pad, :-max_pad]
    
    # Initialize kernel based on the night
    arr2[1:, :max_pad] = arr2[:-1, -max_pad:]  # create past
    arr2 = arr2[max_pad - kernel_day:]
    
    return arr2.copy(), valid_days, nan_days


def get_grid(time_t, true_t):
    """
    Makes a 2D grid representation of time series values folded by day and minute.

    Args:
        time_t: Array-like timestamps (datetime64 format).
        true_t: Array-like corresponding values for each timestamp.

    Returns:
        A 2D NumPy array where rows represent days and columns represent minutes of the day.
    """
    # Step 1: Extract minutes, hours, and unique days
    time_t = pd.to_datetime(time_t)  # Ensure datetime format
    unique_days = time_t.dt.date  # Extract unique days
    num_days = len(np.unique(unique_days))  # Count unique days

    hours = time_t.dt.hour  # Extract hours (0-23)
    mins = time_t.dt.minute  # Extract minutes (0-59)
    min_indices = hours * 60 + mins  # Convert (hour, min) into a single index (0-1439)

    # Step 2: Map time_t into day indices
    day_indices = pd.factorize(unique_days)[0]  # Factorize days into indices (0, 1, ...)

    # Step 3: Create a 2D grid (num_days, 1440) initialized with NaN
    grid_true = np.full((num_days, 1440), np.nan)

    # Step 4: Populate the grid with true_t values
    for i in range(len(time_t)):
        grid_true[day_indices[i], min_indices[i]] = true_t[i]

    return grid_true


def combine_patient_data(data_path, time_col='datetime', modalities = [
                                                    "activities_minutesFairlyActive_minute",
                                                    "activities_minutesLightlyActive_minute",
                                                    "activities_minutesSedentary_minute",
                                                    "activities_minutesVeryActive_minute",
                                                    "heart_minute"
                                                ], sleep=True):
    """ Go through all subject-ids and combine modalities on time-stamp """
    

    # List all patient folders
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    # Dictionary to store results
    patient_data = {}
    
    for patient_id in tqdm.tqdm(patient_dirs):
        patient_folder = os.path.join(data_path, patient_id)
        
        dfs = []
        for modality in modalities:
            # Build expected filename
            filename = f"{patient_id}_{modality}.csv"
            filepath = os.path.join(patient_folder, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                except pd.errors.EmptyDataError as e: 
                    print(filepath, 'is empty. Skip.')
                    continue
                    
                # Rename 'value' column to modality name for merging
                if modality.startswith("activities_minutes"):
                    activity_name = modality.replace("activities_minutes", "").replace("_minute", "")
                else:
                    activity_name = modality.replace("_minute", "")
                    
                df = df.rename(columns={"value": activity_name})
                
                # Make sure datetime is in datetime format
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                dfs.append(df)
            else:
                print(f"Warning: Missing file {filepath}")

        # read sleep dataframe (encoded in intervals)
        intervals = None
        if sleep:       
            filename = f"{patient_id}_sleep_sleep_info.csv"
            filepath = os.path.join(patient_folder, filename)
            
            if os.path.exists(filepath):
                try:
                    sleep_df = pd.read_csv(filepath)[['startTime','endTime','isMainSleep']]
                    sleep_df = sleep_df[sleep_df.isMainSleep == True]
                    
                    if not sleep_df.empty:
                        t0 = pd.to_datetime(sleep_df['startTime'], format='mixed', errors='coerce')
                        t1 = pd.to_datetime(sleep_df['endTime'], format='mixed', errors='coerce')                
                        intervals = pd.IntervalIndex.from_arrays(t0, t1, closed='both')
                except pd.errors.EmptyDataError: 
                    print(f"{filepath} is empty. Skip.")
                    
        # Merge all dataframes on 'datetime'
        if dfs:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on='datetime', how='outer')
                        
            # Sort by time
            merged_df = merged_df.sort_values('datetime')
                        
            # add sleep column if we have intervals
            if intervals is not None:
                # Initialize all as False
                merged_df['sleep'] = False
                
                # For each sleep interval, mark all times within it as True
                for interval in intervals:
                    mask = (merged_df['datetime'] >= interval.left) & (merged_df['datetime'] <= interval.right)
                    merged_df.loc[mask, 'sleep'] = True
            else: 
                merged_df['sleep'] = np.nan
            
            patient_data[patient_id] = merged_df
        else:
            print(f"Warning: No data found for patient {patient_id}")

    return patient_data


def ts_data_to_grids(
    patient_data,
    processed_path = None,
    window = 60,
    nan_thres_per_window = 0.3 # below this, make window nan
):

    """
    Convert patient time series data into gridded representations and optionally save to disk.
    
    Processes Fitbit activity, heart rate, and sleep data by:
    1. Combining activity levels into a weighted activity score
    2. Converting minute-level data into day × time_bin grids
    3. Aggregating data into configurable time windows
    4. Handling missing data with threshold-based masking
    
    Parameters
    ----------
    patient_data : dict
        Dictionary mapping patient IDs to DataFrames. Each DataFrame must contain:
        - 'datetime': timestamp column
        - 'Sedentary', 'LightlyActive', 'FairlyActive', 'VeryActive': activity levels
        - 'heart': heart rate values
        - 'sleep': boolean sleep indicator
    
    processed_path : str or None, optional
        Path to save processed data. If None, data is not saved to disk.
        Default: None
    
    window : int, optional
        Time window size in minutes for aggregating data.
        Must evenly divide 1440 (minutes per day).
        Default: 60 (hourly bins)
    
    nan_thres_per_window : float, optional
        Minimum proportion of valid (non-NaN) values required per window.
        Windows with fewer valid values are set to NaN.
        Range: 0.0 (no threshold) to 1.0 (all values must be valid)
        Default: 0.3 (30% of values must be valid)
    
    Returns
    -------
    dict
        Dictionary with keys 'activity', 'heart', 'sleep', each containing a list of
        DataFrames (one per patient). Each DataFrame has shape (n_patients, 2 + n_time_bins)
        with columns: ['patient_count', 'patient_id', time_bin_0, time_bin_1, ...]
    
    Side Effects
    ------------
    If processed_path is provided, saves a compressed .npz file containing:
        - data: object array of 3D arrays (days, time_bins, 3_modalities) per patient
        - timestamps: object array of datetime arrays per patient
        - modalities: ['activity', 'heart', 'sleep']
        - metadata: list of dicts with patient info
        - window: window size in minutes
        - window_minutes: duplicate of window for clarity
    
    Notes
    -----
    - Activity is computed as weighted sum: 0*Sedentary + 0.33*Lightly + 0.67*Fairly + 1.0*Very
    - Activity values are clipped to [0.0, 1.0]
    - Heart rate and sleep values are not bounded
    - Patients missing required columns are skipped with a warning
    - Data is organized into 2D grids: rows = days, columns = time bins within day
    
    Examples
    --------
    >>> # Process data without saving
    >>> grids = ts_data_to_grids(patient_data, window=15)
    >>> 
    >>> # Process and save with 5-minute windows, strict NaN threshold
    >>> grids = ts_data_to_grids(
    ...     patient_data,
    ...     processed_path="/path/to/save",
    ...     window=5,
    ...     nan_thres_per_window=0.5
    ... )
    """

    
    # Activity boundaries
    x_max = 1.0
    x_min = 0.0
        
    required_cols = ['Sedentary', 'LightlyActive', 'FairlyActive', 'VeryActive', 'heart', 'sleep']
    
    subject_count = 0
    # Dictionary to hold lists for each modality
    modality_lists = {'activity': [], 'heart': [], 'sleep': []}
    
    for subj_id, df in tqdm.tqdm(patient_data.items()):
    
        missing_cols = [req for req in required_cols if req not in df.columns]
        if any(missing_cols):
            print(subj_id,f'Skip because of missing col(s): {missing_cols}')
            continue
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Drop time points with nan
        if nan_thres_per_window > 0:
            rows_with_nan = df.isnull().any(axis=1)            
            df.loc[rows_with_nan, required_cols] = np.nan
                    
        # Weighted sum of activity columns
        df['activity'] = (
            x_min * df['Sedentary'] +
            1/3 * df['LightlyActive'] +
            2/3 * df['FairlyActive'] +
            x_max * df['VeryActive']
        )
        df.reset_index(drop=True, inplace=True)
    
        # Create grids for each modality
        grids = {
            'activity': get_grid(df['datetime'], df['activity']),
            'heart': get_grid(df['datetime'], df['heart']),
            'sleep': get_grid(df['datetime'], df['sleep']),
        }
        
        # Process each modality
        processed_grids = {}
        for modality, grid in grids.items():
            days, mins = grid.shape
            
            ### From Minute Grid to Window-Grid
            if nan_thres_per_window > 0:
                grid_reshaped = grid.reshape(days, mins // window, window)
                
                # Count valid (non-nan) values per window
                valid_counts = np.sum(~np.isnan(grid_reshaped), axis=-1)
                
                # nan-mean per window
                averages = np.nanmean(grid_reshaped, axis=-1)            
                
                # mask windows below threshold
                threshold = window * nan_thres_per_window
                averages[valid_counts < threshold] = np.nan
                
                processed_grid = averages
            else:
                # This will introduce nans if there's a single nan in the window
                processed_grid = grid.reshape(days, mins // window, window).mean(axis=-1)
            
            # Apply bounds only to activity (not heart or sleep)
            if modality == 'activity':
                processed_grid = np.clip(processed_grid, x_min, x_max)
            
            # Convert to DataFrame and add metadata
            df_modality = pd.DataFrame(processed_grid)
            df_modality.insert(0, "patient_id", subj_id)
            df_modality.insert(0, "patient_count", subject_count)
            
            modality_lists[modality].append(df_modality)
        
        subject_count += 1
    

    # Save all patients in one file with timestamps
    if processed_path:
            
        prefix = 'revision_nan_' if nan_thres_per_window > 0 else 'revision_'
        os.makedirs(processed_path, exist_ok=True)
        print('... save with prefix', prefix)
        
        all_patients_data = []
        all_patients_timestamps = []
        metadata = []
        
        for patient_idx in range(subject_count):
            patient_id = modality_lists['activity'][patient_idx]['patient_id'].iloc[0]
            
            # Stack modalities into 3D array
            patient_data_3d = np.stack([
                modality_lists['activity'][patient_idx].iloc[:, 2:].values,
                modality_lists['heart'][patient_idx].iloc[:, 2:].values,
                modality_lists['sleep'][patient_idx].iloc[:, 2:].values
            ], axis=-1)  # Shape: (days, time_bins, 3)
            
            all_patients_data.append(patient_data_3d)
            
            # Extract original timestamps for this patient
            # Get the datetime values that correspond to this patient's data
            patient_df = patient_data[patient_id]
            timestamps = patient_df['datetime'].values
            all_patients_timestamps.append(timestamps)
            
            metadata.append({
                'patient_id': patient_id,
                'patient_count': patient_idx,
                'n_days': patient_data_3d.shape[0],
                'n_time_bins': patient_data_3d.shape[1]
            })

        artifact = f"{processed_path}/{prefix}all_patients_{window}.npz"
        np.savez_compressed(
            artifact,
            data=np.array(all_patients_data, dtype=object),  # Store as object array
            timestamps=np.array(all_patients_timestamps, dtype=object),
            modalities=['activity', 'heart', 'sleep'],
            metadata=metadata,
            window=window,
            window_minutes=window
        )

        print('... artifact created at',artifact)
    return modality_lists
