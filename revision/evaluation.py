"""
Parallel evaluation script for model predictions.

Usage:
    python evaluate_parallel.py --run 20251112_1611 --n-jobs 8
    python evaluate_parallel.py --run 20251112_1611 --n-jobs -1  # Use all cores
"""
import argparse
import glob
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

import warnings  # (optional, if you later add warning handling)
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, matthews_corrcoef,
    confusion_matrix
)
from utils import infer_meta_cols


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


# ========== Uncertainty Scores ========
# Assuming 90 CI are read out
# coverage < 0.90: Model is overconfident (P0 too small) → increase initial_uncertainty
# coverage > 0.90: Model is underconfident (P0 too large) → decrease initial_uncertainty
# coverage ≈ 0.90 but large width: Calibrated but not sharp → improve model features
# Ideal: coverage ≈ 0.90 AND small width


def coverage_metrics(y_true, y_pred, y_lower, y_upper, confidence=0.90):
    """
    Evaluate calibration of confidence intervals.
    
    Returns
    -------
    dict with:
        - coverage: Fraction of observations within CI (should be ~0.90 for 90% CI)
        - mean_width: Average width of confidence intervals
        - sharpness: Width normalized by prediction scale
        - coverage_bias: coverage - expected_coverage (should be ~0)
    """
    mask = y_true.notna() & y_pred.notna() & y_lower.notna() & y_upper.notna()
    if mask.sum() == 0:
        return {k: np.nan for k in ['coverage', 'mean_width', 'sharpness', 
                                     'coverage_bias', 'n']}
    
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    yl = y_lower[mask].astype(float)
    yu = y_upper[mask].astype(float)
    
    # Check if true value is within CI
    within_ci = (yt >= yl) & (yt <= yu)
    coverage = within_ci.mean()
    
    # Interval width statistics
    widths = yu - yl
    mean_width = widths.mean()
    
    # Sharpness: narrower intervals are better (if calibrated)
    # Normalize by data range to make comparable across tasks
    data_range = yt.max() - yt.min()
    sharpness = mean_width / data_range if data_range > 0 else np.nan
    
    # Coverage bias: how far from expected coverage
    coverage_bias = coverage - confidence
    
    return {
        'coverage': coverage,
        'mean_width': mean_width,
        'sharpness': sharpness,
        'coverage_bias': coverage_bias,
        'n': int(mask.sum())
    }

def interval_score(y_true, y_lower, y_upper, alpha=0.10):
    """
    Interval score: lower is better. Proper scoring rule for interval forecasts.
    
    α = 1 - confidence (e.g., 0.10 for 90% CI)
    
    Score = (upper - lower) + (2/α) * (lower - y) * 1{y < lower}
                             + (2/α) * (y - upper) * 1{y > upper}
    
    Reference: Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"
    """
    mask = y_true.notna() & y_lower.notna() & y_upper.notna()
    if mask.sum() == 0:
        return {'interval_score': np.nan, 'n': 0}
    
    yt = y_true[mask].astype(float).values
    yl = y_lower[mask].astype(float).values
    yu = y_upper[mask].astype(float).values
    
    width = yu - yl
    penalty_lower = (2 / alpha) * np.maximum(0, yl - yt)
    penalty_upper = (2 / alpha) * np.maximum(0, yt - yu)
    
    score = width + penalty_lower + penalty_upper
    
    return {
        'interval_score': score.mean(),
        'interval_score_std': score.std(),
        'n': int(mask.sum())
    }


def evaluate_run(
    df,
    meta_cols=None,
    thresholds=[0.01,1/3], # SB, MVPA
    do_threshold_sweep=False,
    sweep_thresholds=None,
    evaluate_uncertainty=True,
    burn_in_days = 10, # burn in time
):
    """
    Evaluate one run DataFrame and return 4 structured tables:
    1. metadata: Single row with run configuration
    2. regression: Metrics for activity and heart rate across sleep conditions
    3. classification_auc: ROC-AUC and PR-AUC metrics across conditions
    4. (optional) classification_threshold: Threshold-based metrics across conditions
    5. (optional) uncertainty_calibration
    
    Returns
    -------
    dict with keys: 'metadata', 'regression', 'classification_auc', 'classification_threshold'
    """

        
    # ========== 1. METADATA TABLE ==========
    if meta_cols is None:
        meta_cols = infer_meta_cols(df)

    metadata = {}
    for col in meta_cols:
        metadata[col] = df.iloc[0][col] if col in df.columns else np.nan
        
    # Add sleep summary counts
    sleep_col = 'sleep'
    if sleep_col not in df.columns:
        raise KeyError(f"Expected column '{sleep_col}' in df")

    # remove burn-in from predictions
    n_before_burnin = len(df)
    if 'time' in df.columns and burn_in_days>0:
        df = df[df['time'] >= burn_in_days].copy()
    
    metadata['n_before_burnin'] = n_before_burnin
    metadata['n_after_burnin'] = len(df)
    metadata['burn_in_days'] = burn_in_days    
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

    output = {
        'metadata': metadata_df,
        'regression': regression_df,
        'classification_auc': classification_auc_df,
        'classification_threshold': classification_threshold_df
    }
    
    # ==== 4.1 (Optional): Add threshold sweep results
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

        output.update({'threshold_sweep': threshold_sweep_df})

    # ========== 5. (optional) UNCERTAINTY CALIBRATION TABLE ==========
    if evaluate_uncertainty:
        uncertainty_rows = []
        
        # Check if CI columns exist
        has_activity_ci = ('activity_low' in df.columns and 
                          'activity_up' in df.columns)
        has_heart_ci = ('heart_low' in df.columns and 
                       'heart_up' in df.columns)
        
        tasks = []
        if has_activity_ci:
            tasks.append(('activity', 'activity_true', 'activity_pred', 
                         'activity_low', 'activity_up'))
        if has_heart_ci:
            tasks.append(('heart', 'heart_true', 'heart_pred', 
                         'heart_low', 'heart_up'))
        
        for task_name, col_true, col_pred, col_low, col_up in tasks:
            for cond in sleep_conditions:
                subdf = subset(cond)
                
                # Coverage metrics
                cov_metrics = coverage_metrics(
                    subdf[col_true],
                    subdf[col_pred],
                    subdf[col_low],
                    subdf[col_up],
                    confidence=0.90
                )
                
                # Interval score
                int_metrics = interval_score(
                    subdf[col_true],
                    subdf[col_low],
                    subdf[col_up],
                    alpha=0.10
                )
                
                row = {
                    'task': task_name,
                    'condition': cond,
                    **cov_metrics,
                    **int_metrics
                }
                uncertainty_rows.append(row)
        
        if uncertainty_rows:
            uncertainty_calibration_df = pd.DataFrame(uncertainty_rows)
            output['uncertainty_calibration'] = uncertainty_calibration_df
    

    
    return output


def process_single_file(path_tuple, runs_path, do_threshold_sweep=False):
    """
    Process a single prediction file and return metrics for all patients.
    
    Args:
        path_tuple: (index, file_path)
        runs_path: Base path for runs
        do_threshold_sweep: Whether to perform threshold sweep
        
    Returns:
        dict: Metrics organized by table name
    """
    i, path = path_tuple
    
    try:
        # Suppress warnings for this process
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found in 'y_true' and 'y_pred'",
                category=UserWarning,
                module="sklearn.metrics._classification"
            )
            
            # Load and prepare data
            df = pd.read_csv(path)       
            df['path'] = path # acts as primary key!
            df['run'] = i
            ordered = ['run'] + [c for c in df.columns if c not in ['run']]
            df = df[ordered]

            if 'burn_in_days' not in df.columns:
                df['burn_in_days'] = 40 # fix calibration run
            
            # evaluate uncertainty? Assume one model per run!
            models = df['model_name'].unique()
            burn_in_days = df['burn_in_days'].unique()
            assert len(burn_in_days)==1, f'only one burn_in_days per run allowed, but found {burn_in_days}'
            burn_in_days = burn_in_days[0]
            assert len(models) == 1, f'assuming one model per run only! but found {models}'            
            evaluate_uncertainty = 'rls' in models # only rls has prob output

            assert 'pid' in df, f'pid must be in df.columns = {df.columns}'
            
            # Evaluate each patient
            file_metrics = {}
            for pid in df.pid.unique():
                results = evaluate_run(
                    df[df.pid == pid], 
                    do_threshold_sweep=do_threshold_sweep,
                    evaluate_uncertainty=evaluate_uncertainty,
                    burn_in_days = burn_in_days                    
                )
                
                for name, table in results.items():
                    if name not in file_metrics:
                        file_metrics[name] = []
                    # add primary keys
                    table['path'] = path
                    table['run'] = i
                    table['pid'] = pid
                    file_metrics[name].append(table)
            
            return file_metrics
            
    except pd.errors.EmptyDataError:
        print(f"⚠️ Skipping empty file: {path}")
        return {}
    #except Exception as e:
    #   print(f"❌ Error processing {path}: {e}")
    #  return {}

def main():
    parser = argparse.ArgumentParser(
        description='Parallel evaluation of model predictions'
    )
    parser.add_argument(
        '--run',
        type=str,
        required=True,
        help='Run identifier (e.g., 20251112_1611)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='/sc/arion/projects/Clinical_Times_Series/cpp_data/runs',
        help='Base path for runs directory'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores, -2 for all but one)'
    )
    parser.add_argument(
        '--threshold-sweep',
        action='store_true',
        help='Perform threshold sweep (slower)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    runs_path = Path(args.base_path) / args.run
    predictions_path = runs_path / 'predictions'
    metrics_path = runs_path / 'results'
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Processing run: {args.run}")
    print(f"📁 Predictions path: {predictions_path}")
    print(f"💾 Metrics path: {metrics_path}")
    
    # Find all prediction files
    prediction_files = list(glob.glob(str(predictions_path / '*.csv')))
    
    if args.limit:
        prediction_files = prediction_files[:args.limit]
        print(f"⚠️ Limited to {args.limit} files for testing")
    
    print(f"📊 Found {len(prediction_files)} prediction files")
    
    if not prediction_files:
        print("❌ No prediction files found!")
        return
    
    # Determine number of workers
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs == -2:
        n_jobs = max(1, cpu_count() - 1)
    elif n_jobs <= 0:
        n_jobs = max(1, cpu_count() + n_jobs)
    
    print(f"🚀 Using {n_jobs} parallel workers")
    
    # Create enumerated paths for tracking
    indexed_paths = list(enumerate(prediction_files))
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_file,
        runs_path=str(runs_path),
        do_threshold_sweep=args.threshold_sweep
    )
    
    # Process files in parallel
    print("🔄 Processing files...")
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_func, indexed_paths),
            total=len(indexed_paths),
            desc="Files processed"
        ))

    # Combine tables across parameter settings
    combined_results = {} 
    for result in results:
        for name, tables in result.items():
            if name not in combined_results:
                combined_results[name] = []
            combined_results[name].extend(tables)
    
    # Save combined tables after concat 
    print("💾 Saving metrics...")
    for name, tables in combined_results.items():
        if tables:  # Only save if we have data
            file_name = metrics_path / f'{name}.csv'
            combined_df = pd.concat(tables, ignore_index=True)
            combined_df.to_csv(file_name, index=False)
            print(f"   ✓ Saved {len(combined_df)} rows to {file_name}")    
    
    print("✅ Done!")

if __name__ == '__main__':
    main()