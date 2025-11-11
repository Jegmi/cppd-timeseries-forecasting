from utils import _get_combined_features, get_x, load_and_process_grid
import numpy as np
from scipy.stats import norm
from scipy.special import logit, expit
from sklearn.preprocessing import PolynomialFeatures


class RecursiveLeastSquares:
    """
    Unified Recursive Least Squares (RLS) with exponential forgetting and feature transforms.
    Supports both linear and logit-normal output spaces.
    
    References:
    - Ljung & Söderström (1983) "Theory and Practice of Recursive Identification"
    - Haykin (2002) "Adaptive Filter Theory"
    """
    
    def __init__(self, n_features, n_outputs=1, lambda_forget=0.99, 
                 initial_uncertainty=1000.0, output_space='linear',
                 clip_output=False, clip_range=(0, 1), epsilon=1e-6,
                 transform='linear', degree=2):
        """
        Args:
            n_features: Dimension of input features (before transformation)
            n_outputs: Number of outputs to predict
            lambda_forget: Forgetting factor in (0, 1]. 1 = no forgetting, <1 = adaptive
            initial_uncertainty: Initial covariance (large = uninformative prior)
            output_space: 'linear' or 'logit' for bounded [0,1] outputs
            clip_output: Whether to clip predictions to range (only for linear)
            clip_range: Tuple of (min, max) for clipping (only for linear)
            epsilon: Small value to avoid log(0) (only for logit)
            transform: 'linear' or 'poly2' (polynomial with interaction terms only)
            degree: Polynomial degree (only if transform='poly2')
        """
        self.n_features_input = n_features
        self.n_outputs = n_outputs
        self.lambda_forget = lambda_forget
        self._ess = 1 / (1 - lambda_forget) if lambda_forget < 1 else np.inf
        self.lambda_var = lambda_forget
        
        # Output space configuration
        self.output_space = output_space
        self.clip_output = clip_output if output_space == 'linear' else False
        self.clip_range = clip_range
        self.epsilon = epsilon
        
        # Feature transformation
        if transform == 'linear':
            self.feature_transform = None
            self.n_features = n_features
        else:  # poly2 or polynomial
            self.feature_transform = PolynomialFeatures(
                degree=degree,
                interaction_only=True,
                include_bias=False
            )
            dummy_X = np.zeros((1, n_features))
            self.feature_transform.fit(dummy_X)
            self.n_features = self.feature_transform.n_output_features_
        
        # Initialize parameters for transformed feature space
        self.w = np.zeros((n_outputs, self.n_features))
        self.P = [initial_uncertainty * np.eye(self.n_features) for _ in range(n_outputs)]
        
        # Track prediction uncertainty (innovation variance)
        self.sigma2 = np.ones(n_outputs)
        self.innovation_count = 0
        
        self.is_fitted = False

    def get_effective_sample_size(self):
        """Return current effective sample size."""
        return self._ess
    
    def _transform_y(self, y):
        """Transform y to working space (identity for linear, logit for logit space)."""
        if self.output_space == 'linear':
            return y
        else:  # logit
            y_clipped = np.clip(y, self.epsilon, 1 - self.epsilon)
            return logit(y_clipped)
    
    def _inverse_transform_y(self, y_transformed):
        """Transform from working space back to output space."""
        if self.output_space == 'linear':
            return y_transformed
        else:  # logit
            return expit(y_transformed)
    
    def _transform_features(self, x):
        """Apply feature transformation."""
        if self.feature_transform is None:
            return np.atleast_1d(x)
        else:
            return self.feature_transform.transform(np.atleast_1d(x).reshape(1, -1))[0]
    
    def update(self, x, y_true):
        """
        Online update with a single observation.
        
        Args:
            x: Feature vector, shape (n_features_input,)
            y_true: Target vector, shape (n_outputs,). Can contain NaN for missing values.
        """
        x_transformed = self._transform_features(x)
        y_true = np.atleast_1d(y_true)
        
        # Update each output dimension independently
        for k in range(self.n_outputs):
            if np.isnan(y_true[k]):
                # Missing observation: skip update, increase uncertainty
                self.P[k] = self.P[k] / self.lambda_forget
                continue
            
            # Transform y to working space
            y_working = self._transform_y(y_true[k])
            
            # Prediction error
            e = y_working - self.w[k] @ x_transformed
            
            # Kalman gain
            Px = self.P[k] @ x_transformed
            denominator = self.lambda_forget + x_transformed @ Px
            K = Px / denominator
            
            # Parameter update
            self.w[k] = self.w[k] + K * e
            
            # Covariance update (Joseph form for numerical stability)
            self.P[k] = (self.P[k] - np.outer(K, Px)) / self.lambda_forget
            
            # Update innovation variance
            self.sigma2[k] = self.lambda_var * self.sigma2[k] + (1-self.lambda_var) * e**2
        
        self.innovation_count += 1
        self.is_fitted = True
        return self
    
    def predict(self, X, return_std=False, return_intervals=False, confidence=0.95):
        """
        Predict outputs for input features.
        
        Args:
            X: Features, shape (n_samples, n_features_input)
            return_std: If True, also return prediction standard deviation
            return_intervals: If True, return confidence intervals
            confidence: Confidence level for intervals
            
        Returns:
            y_pred: Predictions, shape (n_samples, n_outputs)
            y_std: Standard deviations (if return_std=True)
            y_intervals: Dict with 'lower' and 'upper' (if return_intervals=True)
        """
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        y_pred = np.zeros((n_samples, self.n_outputs))
        y_std = np.zeros((n_samples, self.n_outputs)) if (return_std or return_intervals) else None
        y_lower = np.zeros((n_samples, self.n_outputs)) if return_intervals else None
        y_upper = np.zeros((n_samples, self.n_outputs)) if return_intervals else None
        
        z = norm.ppf((1 + confidence) / 2) if return_intervals else None
        
        for i in range(n_samples):
            x_transformed = self._transform_features(X[i])
            
            for k in range(self.n_outputs):
                # Predict in working space
                y_working = self.w[k] @ x_transformed
                
                # Transform to output space
                y_pred[i, k] = self._inverse_transform_y(y_working)
                
                # Compute uncertainty
                if return_std or return_intervals:
                    var_working = x_transformed @ self.P[k] @ x_transformed + self.sigma2[k]
                    std_working = np.sqrt(var_working)
                    
                    if self.output_space == 'linear':
                        # Linear space: uncertainty is direct
                        y_std[i, k] = std_working
                        
                        if return_intervals:
                            y_lower[i, k] = y_pred[i, k] - z * std_working
                            y_upper[i, k] = y_pred[i, k] + z * std_working
                    
                    else:  # logit space
                        if return_intervals:
                            # Proper intervals: transform from logit scale
                            y_lower[i, k] = self._inverse_transform_y(y_working - z * std_working)
                            y_upper[i, k] = self._inverse_transform_y(y_working + z * std_working)
                        
                        if return_std:
                            # Delta method approximation
                            mu = y_pred[i, k]
                            derivative = mu * (1 - mu)
                            y_std[i, k] = derivative * std_working
        
        # Clip predictions if requested (only for linear space)
        if self.clip_output:
            y_pred = np.clip(y_pred, self.clip_range[0], self.clip_range[1])
        
        if return_std:
            return y_pred, y_std
        if return_intervals:
            return y_pred, {'lower': y_lower, 'upper': y_upper}
        return y_pred


def get_model(model_name, n_features, n_outputs=1, clip_output=True, clip_range=(0, 1), **kwargs):
    """
    Factory function to create model instances with optional output clipping.
    
    Parameters
    ----------
    model_name : str
        String identifier for the model ('rls', 'rls_poly', 'rls_logit', 
        'arima', 'persistence', 'lstm').
    n_features : int
        Number of input features.
    n_outputs : int, optional
        Number of output targets (default: 1).
    clip_output : bool, optional
        Whether to clip model predictions (default: True).
    clip_range : tuple, optional
        Range for clipping predictions (default: (0, 1)).
    **kwargs : dict
        Additional model-specific parameters:
        - For RLS: lambda_forget, initial_uncertainty
        - For ARIMA: order (tuple), seasonal_order (tuple)
        - For LSTM: hidden_size, num_layers, learning_rate, seq_length
    
    Returns
    -------
    model
        Configured model object, optionally wrapped with clipping transformer.
    """
    
    model_name_lower = model_name.lower()
    
    # Recursive Least Squares variants
    if 'rls' in model_name_lower:
        output_space = 'logit' if 'logit' in model_name_lower else 'linear'
        transform = 'poly' if 'poly' in model_name_lower else 'linear'
        
        model = RecursiveLeastSquares(
            n_features=n_features,
            n_outputs=n_outputs,
            output_space=output_space,
            transform=transform,
            clip_output=clip_output,
            clip_range=clip_range,
            initial_uncertainty=kwargs.get('initial_uncertainty', 1000.0),
            lambda_forget=kwargs.get('lambda_forget', 1.0),
        )

    return model

def run_prob(processed_path, patient_id, kernel_day=2, kernel_hour=2, field='cross',
             model_name='rls', 
             verbose=0, min_days=30, forecasting_modality='a|a', prediction_horizon=60): #lambda_forget=0.99, initial_uncertainty=1000.0,
    """
    Online forecasting with Recursive Least Squares (RLS) or Logit-Normal RLS
    for activity and heart rate time series with automatic missing data handling.
    
    Parameters
    ----------
    processed_path : str
        Path to processed data file.
    patient_id : int
        Patient identifier.
    kernel_day : int, default=2
        Number of previous days in receptive field (time-delay embedding).
    kernel_hour : int, default=2
        Number of previous hours in receptive field (time-delay embedding).
    field : str, default='cross'
        Feature construction method. Options: 'cross', 'separate', 'concat'.
        Defines how historical observations are transformed into feature vectors.
    model_name : str, default='rls'
        Model type for online learning:
        - 'rls': Recursive Least Squares (Gaussian/Kalman filter) with output clipping
        - 'logit_rls': Logit-Normal RLS (Beta regression approximation for [0,1] outputs)
    lambda_forget : float, default=0.99
        Exponential forgetting factor in (0, 1]. Controls how quickly the model
        adapts to new patterns. λ=1 gives equal weight to all past data; λ<1
        gives exponentially decaying weights (effective memory ≈ 1/(1-λ) steps).
        Typical values: 0.95-0.995 for slow adaptation, 0.99-0.995 for moderate.
    initial_uncertainty : float, default=1000.0
        Initial parameter covariance (large values = uninformative prior).
        Controls how quickly the model learns from initial observations.
    verbose : int, default=0
        Verbosity level (0=silent, 1=basic info, 2=detailed).
    min_days : int, default=30
        Minimum valid days required to process patient data.
    forecasting_modality : str, default='a|a'
        Defines which signals are used to predict which targets:
        - 'ah|ah': Predict both activity and heart rate from both signals
        - 'a|ah': Predict activity from both, heart rate from heart rate only
        - 'a|a': Predict each signal from itself independently (no cross-modal)
        - 'a|h': Predict activity from heart rate, heart rate from itself
    prediction_horizon : int, default=60
        Prediction horizon in minutes. Determines how far ahead predictions are made.
        With time resolution of time_step minutes, the model predicts shift_steps
        values where shift_steps = prediction_horizon / time_step.
    
    Returns
    -------
    results : List[Dict] or int
        If successful, returns list of dictionaries with predictions and metadata.
        Returns -1 if insufficient data (valid_days < min_days).
        
        Each dictionary contains:
        - Predictions: activity_pred, heart_pred, activity_std, heart_std
        - Ground truth: activity_true, heart_true
        - Metadata: day, hour, time, model parameters, patient info
        
    Notes
    -----
    Model Framework:
        The approach implements online time series forecasting using state-space
        models with time-delay embedding (Takens, 1981; Akaike, 1974). The augmented
        state vector z_t = [y_t, y_{t-1}, ..., y_{t-T}]^T follows a linear dynamical
        system with companion matrix structure. Parameters are learned recursively
        via RLS (Ljung & Söderström, 1983), which is equivalent to Kalman filtering
        for linear Gaussian models (Durbin & Koopman, 2012).
    
    Missing Data:
        Missing observations are handled via the Kalman filter principle: when
        y_t is NaN, parameter updates are skipped but uncertainty (covariance P)
        increases, reflecting reduced confidence. Missing values are imputed with
        model predictions and used in subsequent feature construction.
    
    Memory and Computation:
        The online RLS algorithm maintains only current parameters (w, P, σ²),
        requiring O(d²) storage and O(d²) computation per update, where d is
        the feature dimension. No historical data storage is needed, implementing
        a true Markov process with the forgetting factor λ providing adaptive
        windowing equivalent to exponentially weighted moving average.
    
    Examples
    --------
    # Gaussian RLS with default settings
    >>> results = run_prob(data_path, pid=123, model_name='rls')
    
    # Logit-Normal RLS for proper [0,1] modeling
    >>> results = run_prob(data_path, pid=123, model_name='logit_rls')
    
    # Fast adaptation with shorter memory
    >>> results = run_prob(data_path, pid=123, lambda_forget=0.95)
    
    # Cross-modal prediction with 2-hour horizon
    >>> results = run_prob(data_path, pid=123, forecasting_modality='ah|ah',
    ...                    prediction_horizon=120)
    
    References
    ----------
    Akaike, H. (1974). Markovian representation of stochastic processes.
        Annals of the Institute of Statistical Mathematics, 26(1), 363-387.
    Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods.
        Oxford University Press.
    Ferrari, S., & Cribari-Neto, F. (2004). Beta regression for modelling rates
        and proportions. Journal of Applied Statistics, 31(7), 799-815.
    Ljung, L., & Söderström, T. (1983). Theory and Practice of Recursive
        Identification. MIT Press.
    Takens, F. (1981). Detecting strange attractors in turbulence. Lecture Notes
        in Mathematics, 898, 366-381.
    """

    HEART_RATE_SCALE = 200
    
    # Validate inputs
    valid_modes = {'ah|ah', 'a|ah', 'a|a', 'a|h'}
    if forecasting_modality not in valid_modes:
        raise ValueError(f"forecasting_modality must be one of {valid_modes}")
    
    valid_models = {'rls', 'logit_rls'}
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}")
    
    if not 0 < lambda_forget <= 1:
        raise ValueError(f"lambda_forget must be in (0, 1], got {lambda_forget}")
    
    if verbose:
        print(f'Model: {model_name} (λ={lambda_forget})')
        print(f'Forecasting mode: {forecasting_modality}')
        print(f'Field: {field}')    
        
    # Load data
    act, val_days, nan_days = load_and_process_grid(
        patient_id, processed_path, modality='activity', 
        kernel_hour=kernel_hour, kernel_day=kernel_day)
    heart, _, _ = load_and_process_grid(
        patient_id, processed_path, modality='heart', 
        kernel_hour=kernel_hour, kernel_day=kernel_day)
    sleep, _, _ = load_and_process_grid(
        patient_id, processed_path, modality='sleep', 
        kernel_hour=kernel_hour, kernel_day=kernel_day)

    # Data statistics
    days = act.shape[0] - kernel_day
    n_time_steps = act.shape[1] - kernel_hour  # Number of time steps per day
    time_step = int(60 * 24 / n_time_steps)  # Time resolution in minutes
    
    # Calculate prediction shift (how many time steps = prediction horizon)
    shift_steps = int(prediction_horizon / time_step)
    
    if shift_steps < 1:
        raise ValueError(f"prediction_horizon ({prediction_horizon}min) must be >= time_step ({time_step}min)")
    
    if verbose:
        print(f'Time resolution: {time_step}min')
        print(f'Prediction horizon: {prediction_horizon}min ({shift_steps} steps)')
        print(f'Valid days: {val_days:.1f}, NaN days: {nan_days:.1f}, Total days: {days}')
    
    # Remove subjects without sufficient data
    if val_days < min_days:
        if verbose:
            print(f'Insufficient data: {val_days:.1f} valid days (required: {min_days})')
        return -1
    
    # Normalize heart rate to [0, 1] range
    heart_normalized = heart / HEART_RATE_SCALE
    
    # Create working copies for imputation
    act_imputed = act.copy()
    heart_imputed = heart_normalized.copy()    

    # Results storage
    res = []
    val_count = 0
    
    for day in range(days):    
        for step in range(0, n_time_steps, shift_steps):
            
            if step + shift_steps > n_time_steps:
                continue
            
            # Extract features
            x_act = get_x(act_imputed, field, day, step, 
                         kernel_day, kernel_hour, shift_steps)
            x_heart = get_x(heart_imputed, field, day, step, 
                           kernel_day, kernel_hour, shift_steps)            

            
            # Get true values (may contain NaN)
            ya_true_seq = act[day + kernel_day, step + kernel_hour:step + kernel_hour + shift_steps]
            yh_true_seq = heart_normalized[day + kernel_day, step + kernel_hour:step + kernel_hour + shift_steps]            
            
            # Combine features based on forecasting modality
            xa, xh = _get_combined_features(x_act, x_heart, forecasting_modality)
            
            # Check validity
            all_valid = ~np.any(np.isnan(ya_true_seq))
            
            # Initialize on first valid sample
            if val_count == 0:
                if not all_valid:
                    # Impute missing with simple mean before model initialization
                    mean_a = np.nanmean(act_imputed)
                    mean_h = np.nanmean(heart_imputed)
                    for i in range(shift_steps):
                        if np.isnan(ya_true_seq[i]):
                            act_imputed[day + kernel_day, step + kernel_hour + i] = mean_a
                            heart_imputed[day + kernel_day, step + kernel_hour + i] = mean_h
                    continue
                else:
                    # First valid sample: initialize with single update
                    assert len(ax) == len(hx), 'activity and heart rate must have same length!'
                    model_a = get_model(model_name, n_features=feat_dim, n_outputs=shift_steps, **model_kwargs)
                    model_h = get_model(model_name, n_features=feat_dim, n_outputs=shift_steps, **model_kwargs)
                    
                    model_a.update(xa, ya_true_seq)
                    model_h.update(xh, yh_true_seq)
                    val_count = 1
                    continue
            
            # ========================================
            # PREDICT-UPDATE CYCLE (Online Learning)
            # ========================================
            
            # 1. PREDICT using current model with uncertainty
            #ya_pred_seq, ya_std_seq = model_a.predict(np.array([xa]), return_std=True)
            yh_pred_seq, yh_intervals = model_h.predict(np.array([xh]), 
                                                      return_intervals=True, 
                                                      confidence=0.95)
            
            ya_pred_seq, ya_intervals = model_a.predict(np.array([xa]), 
                                                        return_intervals=True, 
                                                        confidence=0.95)
            
            # Extract from batch dimension
            ya_pred_seq = ya_pred_seq[0]            
            yh_pred_seq = yh_pred_seq[0]
            for k in ['lower','upper']:                
                ya_intervals[k] = ya_intervals[k][0]
                yh_intervals[k] = yh_intervals[k][0]
            
            # 2. IMPUTE missing values with predictions
            for i in range(shift_steps):
                if np.isnan(ya_true_seq[i]):
                    act_imputed[day + kernel_day, step + kernel_hour + i] = ya_pred_seq[i]
                if np.isnan(yh_true_seq[i]):
                    heart_imputed[day + kernel_day, step + kernel_hour + i] = yh_pred_seq[i]
            
            # 3. UPDATE model with observation (even if partially missing!)
            # Critical: Model sees NaN values and updates uncertainty accordingly
            model_a.update(xa, ya_true_seq)
            model_h.update(xh, yh_true_seq)
            
            # Count valid samples
            if all_valid:
                val_count += 1
            
            # Store results for each predicted time point
            fac = time_step / 60.0  # Convert to hours
            ya_observed_seq = ya_true_seq.copy()
            yh_observed_seq = yh_true_seq.copy()
            
            for i in range(shift_steps):
                
                res.append({
                    # Time indexing                    
                    'hour': (step + i) * fac,
                    'day': day,
                    'time': day + (step + i) / n_time_steps,
                    'valid_time': val_count / n_time_steps,                    
                    
                    # Activity predictions and uncertainty
                    'activity_true': ya_observed_seq[i],
                    'activity_pred': ya_pred_seq[i],
                    'activity_low': ya_intervals['lower'][i],
                    'activity_up': ya_intervals['upper'][i],
                    
                    # Heart rate predictions and uncertainty (denormalized)
                    'heart_true': yh_observed_seq[i] * HEART_RATE_SCALE,
                    'heart_pred': yh_pred_seq[i] * HEART_RATE_SCALE,
                    'heart_low': yh_intervals['lower'][i] * HEART_RATE_SCALE,
                    'heart_up': yh_intervals['upper'][i] * HEART_RATE_SCALE,
                    
                    # Context
                    'sleep': sleep[day + kernel_day, step + kernel_hour + i],
                    
                    # Model configuration
                    'model_name': model_name,
                    'field': field,
                    'lambda_forget': lambda_forget,
                    'initial_uncertainty': initial_uncertainty,
                    'kernel_day': kernel_day,
                    'kernel_hour': kernel_hour,
                    
                    # Prediction metadata
                    'time_step': time_step,
                    'prediction_horizon': prediction_horizon,
                    'shift_steps': shift_steps,
                    'prediction_index': i,  # Which step in the prediction sequence
                    
                    # Experimental design
                    'forecasting_modality': forecasting_modality,
                    
                    # Data statistics
                    'valid_days': val_days,
                    'nan_days': nan_days,
                    'total_days': days,
                    'pid': patient_id
                })
        
        # Transfer imputed values across day boundary
        if day + kernel_day + 1 < act_imputed.shape[0]:
            act_imputed[day+kernel_day+1, :kernel_hour] = \
                act_imputed[day+kernel_day, -kernel_hour:]
            heart_imputed[day+kernel_day+1, :kernel_hour] = \
                heart_imputed[day+kernel_day, -kernel_hour:]

    if verbose:
        print(f'Completed: {len(res)} time points in output')
        if verbose > 1:
            # Show some statistics
            res_df = pd.DataFrame(res)
            print(f'\nPrediction statistics:')
            print(f'  Activity RMSE: {np.sqrt(np.nanmean((res_df.activity_true - res_df.activity_pred)**2)):.4f}')
            print(f'  Heart Rate RMSE: {np.sqrt(np.nanmean((res_df.heart_true - res_df.heart_pred)**2)):.2f} bpm')
            print(f'  Mean activity std: {np.nanmean(res_df.activity_std):.4f}')
            print(f'  Mean heart rate std: {np.nanmean(res_df.heart_std):.2f} bpm')
    
    return res