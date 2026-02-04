import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_arima(
    processed_path,
    patient_id,    
    model_name='arima',
    forecasting_modality='a|a',
    prediction_horizon=60,
    verbose=0,
    min_days=20,
    clip=True, # to [0,1]
    burn_in_days=10,
    eval_all_dt=True,
    **kwargs
):
    """Run ARIMA/SARIMA model with train/test split and online updating.
    
    Parameters:
    -----------
    eval_all_dt : bool
        If True, evaluate at every timestep. If False, evaluate only at prediction horizons.
    **kwargs : dict
        ARIMA parameters (p, d, q for order)
        SARIMA parameters (P, D, Q, m for seasonal_order)
    """
    
    # PATTERN 1: Log all arguments (from run_base)
    args = {
        "processed_path": processed_path,
        "patient_id": patient_id,
        "model_name": model_name,
        "forecasting_modality": forecasting_modality,
        "prediction_horizon": prediction_horizon,
        "verbose": verbose,
        "min_days": min_days,
        "burn_in_days": burn_in_days,
        "eval_all_dt": eval_all_dt,
        **kwargs
    }
    if verbose:
        print(args)
    
    HEART_RATE_SCALE = 200
        
    # Validate inputs
    valid_modes = {'a|a'}
    if forecasting_modality not in valid_modes:
        raise ValueError(f"forecasting_modality must be one of {valid_modes}")
    
    # Load metadata
    loaded = np.load(processed_path, allow_pickle=True)
    metadata = pd.DataFrame.from_records(loaded['metadata']).set_index('patient_count').loc[patient_id]
    n_per_day = metadata['n_time_bins']
    dt = int(60 * 24 / n_per_day)
    n_forward = int(prediction_horizon / dt)
    
    if n_forward < 1:
        raise ValueError(f"prediction_horizon ({prediction_horizon}min) must be >= dt ({dt}min)")
    
    if verbose:
        print(f'Time resolution: {dt}min')
        print(f'Prediction horizon: {prediction_horizon}min ({n_forward} steps)')
        print(f'Burn-in period: {burn_in_days} days')
    
    # PATTERN 2: Use consistent data loading (from run_base)
    def trim_nan_rows(arr):
        """Remove leading/trailing all-NaN rows."""
        valid_rows = ~np.all(np.isnan(arr), axis=tuple(range(1, arr.ndim)))
        first_valid = np.argmax(valid_rows)
        last_valid = len(valid_rows) - np.argmax(valid_rows[::-1])
        return arr[first_valid:last_valid]
    
    # Load all modalities consistently
    data = {}
    for mod in ['activity', 'heart', 'sleep']:
        m_idx = np.where(loaded['modalities'] == mod)[0]
        timeseries_2d = loaded['data'][patient_id][:, :, m_idx]
        data[mod] = np.squeeze(trim_nan_rows(timeseries_2d))
    
    # Flatten to 1D
    act = data['activity'].flatten()
    heart = data['heart'].flatten()
    sleep = data['sleep'].flatten()
    
    # PATTERN 3: Calculate statistics consistently (from run_base)
    total_steps = len(act)
    steps_per_day = n_per_day
    days = total_steps / steps_per_day
    valid_days = np.sum(~np.isnan(act)) / steps_per_day
    nan_days = np.sum(np.isnan(act)) / steps_per_day
    
    if verbose:
        print(f'Valid days: {valid_days:.1f}, NaN days: {nan_days:.1f}, Total days: {days:.1f}')
    
    # Check minimum data requirement
    if valid_days < min_days:
        if verbose:
            print(f'Insufficient data: {valid_days:.1f} valid days (required: {min_days})')
        return [{
            'valid_days': valid_days,
            'nan_days': nan_days,
            'total_days': days,
            'min_days': min_days,
            'pid': patient_id
        }]
    
    # Normalize and impute
    heart_normalized = heart / HEART_RATE_SCALE
    n_train = burn_in_days * n_per_day
    
    heart_imputed = heart_normalized.copy()
    heart_imputed[np.isnan(heart_imputed)] = np.nanmean(heart_imputed[:n_train])
    
    act_imputed = act.copy()
    act_imputed[np.isnan(act_imputed)] = np.nanmean(act_imputed[:n_train])
    
    # PATTERN 4: Get ARIMA parameters with defaults
    p = kwargs.get('p', 1)
    d = kwargs.get('d', 0)
    q = kwargs.get('q', 0)
    
    # Get SARIMA seasonal parameters with defaults
    P = kwargs.get('P', 0)
    D = kwargs.get('D', 0)
    Q = kwargs.get('Q', 0)
    m = kwargs.get('m', 0)  # seasonal period (0 means no seasonality)
    
    # Construct seasonal order tuple
    seasonal_order = (P, D, Q, m)
    
    # Determine if using SARIMA or just ARIMA
    is_seasonal = any([P, D, Q, m])
    model_type = 'SARIMA' if is_seasonal else 'ARIMA'
    
    if verbose:
        print(f'Model type: {model_type}')
        print(f'Order: ({p}, {d}, {q})')
        if is_seasonal:
            print(f'Seasonal order: ({P}, {D}, {Q}, {m})')
    
    # Train initial models with error handling
    try:
        model_a = SARIMAX(act_imputed[:n_train], 
                         order=(p, d, q),
                         seasonal_order=seasonal_order).fit(disp=False)
        model_h = SARIMAX(heart_imputed[:n_train], 
                         order=(p, d, q),
                         seasonal_order=seasonal_order).fit(disp=False)
    except Exception as e:
        if verbose:
            print(f'{model_type} fitting failed: {e}')
        return [{
            'error': str(e),
            'valid_days': valid_days,
            'nan_days': nan_days,
            'total_days': days,
            'pid': patient_id
        }]
    
    # Results storage
    res = []
    
    # PATTERN 5: Support both evaluation modes (from run_base)
    if eval_all_dt:
        # Evaluate at every timestep
        eval_points = range(n_train, total_steps)
        eval_stride = 1
    else:
        # Evaluate only at prediction horizons
        eval_points = range(n_train, total_steps, n_forward)
        eval_stride = n_forward
    
    # Iterate test data
    for t in eval_points:
        t1 = min(t + n_forward, total_steps)  # Don't go past end
        actual_n_forward = t1 - t
        
        if actual_n_forward < 1:
            continue
        
        # Get true values
        true_a = act[t:t1]
        true_h = heart[t:t1]
        
        # Predict before updating
        try:
            pred_a = model_a.forecast(steps=actual_n_forward)
            pred_h = model_h.forecast(steps=actual_n_forward)
            if clip:
                pred_a = np.clip(pred_a, 0, 1)
                pred_h = np.clip(pred_h, 0, 1)

        except Exception as e:
            if verbose > 1:
                print(f'Prediction failed at t={t}: {e}')
            continue
        
        # PATTERN 6: Consistent time indexing (from run_base)
        for i in range(actual_n_forward):
            global_step = t + i
            day = global_step // steps_per_day
            hour = (global_step % steps_per_day) * (24 / steps_per_day)
            
            res.append({
                # Time indexing
                'global_step': global_step,
                'day': day,
                'hour': hour,
                'time': day + (global_step % steps_per_day) / steps_per_day,
                'is_train': False,
                
                # Activity predictions
                'activity_true': true_a[i],
                'activity_pred': pred_a[i] if i < len(pred_a) else np.nan,
                
                # Heart rate predictions (denormalized)
                'heart_true': true_h[i] * HEART_RATE_SCALE,
                'heart_pred': pred_h[i] * HEART_RATE_SCALE if i < len(pred_h) else np.nan,
                
                # Context
                'sleep': sleep[global_step],
                
                # Model configuration
                'model_name': model_name,
                'model_type': model_type,
                'n_pars_activity': p + q + 1 + d + P + Q + D,  # AR + MA + constant + diff + seasonal
                'n_pars_heart': p + q + 1 + d + P + Q + D,
                'p': p,
                'd': d,
                'q': q,
                'P': P,
                'D': D,
                'Q': Q,
                'm': m,
                
                # Prediction metadata
                'dt': dt,
                'prediction_horizon': prediction_horizon,
                'n_forward': n_forward,
                'prediction_index': i,
                'eval_all_dt': eval_all_dt,
                
                # Experimental design
                'forecasting_modality': forecasting_modality,
                
                # Data statistics
                'min_days': min_days,
                'burn_in_days': burn_in_days,
                'valid_days': valid_days,
                'nan_days': nan_days,
                'total_days': days,
                'pid': patient_id
            })
        
        # Update models with new observations (with error handling)
        try:
            if eval_stride == 1:
                # Single-step update
                model_a = model_a.append([act_imputed[t]], refit=False)
                model_h = model_h.append([heart_imputed[t]], refit=False)
            else:
                # Multi-step update
                model_a = model_a.append(act_imputed[t:t1], refit=False)
                model_h = model_h.append(heart_imputed[t:t1], refit=False)
        except Exception as e:
            if verbose > 1:
                print(f'Model update failed at t={t}: {e}')
            continue
    
    if verbose:
        print(f'Completed: {len(res)} time points in output')
        if verbose > 1 and len(res) > 0:
            res_df = pd.DataFrame(res)
            print(f'\nPrediction statistics:')
            print(f'  Activity RMSE: {np.sqrt(np.nanmean((res_df.activity_true - res_df.activity_pred)**2)):.4f}')
            print(f'  Heart Rate RMSE: {np.sqrt(np.nanmean((res_df.heart_true - res_df.heart_pred)**2)):.2f} bpm')
    
    return res

if __name__ == "__main__":
    data_path = "/sc/arion/projects/Clinical_Times_Series/cpp_data/final/processed/"
    
    # Example 1: Standard ARIMA (no seasonality)
    print("=" * 60)
    print("Running ARIMA(1,0,1)")
    print("=" * 60)
    results = run_arima(
        processed_path=f'{data_path}/case/revision_nan_all_patients_60.npz',
        patient_id=138,
        model_name="arima",
        forecasting_modality="a|a",
        prediction_horizon=60,
        verbose=2,
        min_days=20,
        burn_in_days=10,
        eval_all_dt=False,
        # ARIMA parameters
        p=1,
        d=0,
        q=1,
        # Seasonal parameters (defaults to 0,0,0,0 if not provided)
    )
    
    # Example 2: SARIMA with daily seasonality (24 periods if hourly data)
    print("\n" + "=" * 60)
    print("Running SARIMA(1,0,1)(1,0,1,24)")
    print("=" * 60)
    results_seasonal = run_arima(
        processed_path=f'{data_path}/case/revision_nan_all_patients_60.npz',
        patient_id=138,
        model_name="sarima",
        forecasting_modality="a|a",
        prediction_horizon=60,
        verbose=2,
        min_days=20,
        burn_in_days=10,
        eval_all_dt=False,
        # ARIMA parameters
        p=1,
        d=0,
        q=1,
        # Seasonal parameters (daily cycle)
        P=1,
        D=0,
        Q=1,
        m=24,  # 24 periods = daily seasonality for hourly data
    )
    
    # Quick visualization
    if results and len(results) > 0:
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score
        
        res_df = pd.DataFrame(results)
        
        # Filter out any error entries
        if 'activity_pred' in res_df.columns:
            # Remove NaN predictions
            valid = ~(res_df.activity_pred.isna() | res_df.activity_true.isna())
            res_df = res_df[valid]
            
            if len(res_df) > 0:
                true_a = res_df.activity_true.values
                pred_a = res_df.activity_pred.values
                
                rmse = np.sqrt(np.mean((pred_a - true_a)**2))
                r2 = r2_score(true_a, pred_a)
                
                print(f'\nFinal ARIMA Results:')
                print(f'  N predictions: {len(res_df)}')
                print(f'  Activity RMSE: {rmse:.4f}')
                print(f'  Activity R²: {r2:.4f}')
    
    if results_seasonal and len(results_seasonal) > 0:
        res_df_seasonal = pd.DataFrame(results_seasonal)
        
        if 'activity_pred' in res_df_seasonal.columns:
            valid = ~(res_df_seasonal.activity_pred.isna() | res_df_seasonal.activity_true.isna())
            res_df_seasonal = res_df_seasonal[valid]
            
            if len(res_df_seasonal) > 0:
                true_a = res_df_seasonal.activity_true.values
                pred_a = res_df_seasonal.activity_pred.values
                
                rmse = np.sqrt(np.mean((pred_a - true_a)**2))
                r2 = r2_score(true_a, pred_a)
                
                print(f'\nFinal SARIMA Results:')
                print(f'  N predictions: {len(res_df_seasonal)}')
                print(f'  Activity RMSE: {rmse:.4f}')
                print(f'  Activity R²: {r2:.4f}')