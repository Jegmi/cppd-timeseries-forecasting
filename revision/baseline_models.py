import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA as ARIMA_SM

class PersistenceModel:
    """Simple persistence/naive forecast baseline."""
    def __init__(self):
        self.last_value = None
        self.model_params = {}  # No hyperparameters
    
    def fit(self, X, y):
        """Fit model (persistence doesn't need training).
        
        X: (n_samples, seq_length, n_features) - not used
        y: (n_samples,) - we only care about the last value
        """
        if len(y) > 0:
            self.last_value = y[-1]
    
    def predict(self, X, n_steps=1):
        """Return last observed value for n_steps ahead.
        
        X: (n_samples, seq_length, n_features)
        Returns: (n_samples, n_steps)
        """
        if self.last_value is None:
            pred = np.full((X.shape[0], n_steps), 0.1)
        else:
            pred = np.full((X.shape[0], n_steps), self.last_value)
        
        return pred
                
    def get_n_params(self):
        return 0

    def get_n_states(self):
        return 1
    
    def get_model_params(self):
        return self.model_params.copy()



class ARIMAModel:
    def __init__(
        self,
        order=(1, 0, 1),
        clip_output=True,
        clip_range=(0, 1),
        **kwargs,
    ):
        self.order = order
        self.clip_output = clip_output
        self.clip_range = clip_range
        self.model = None
        self.fitted = None
        self.model_params = {"order": order}

    def fit(self, X, y):
        # ARIMA is univariate; ignore X
        self.model = ARIMA_SM(y, order=self.order)
        self.fitted = self.model.fit()

    def predict(self, X, n_steps=1):
        if self.fitted is None:
            raise RuntimeError("Model not fit")

        preds = self.fitted.forecast(steps=n_steps)
        if self.clip_output:
            preds = np.clip(preds, *self.clip_range)

        # match shape: (n_samples, n_steps)
        # ARIMA has no sequence conditioning, so repeat
        batch_size = X.shape[0]
        return np.tile(preds, (batch_size, 1))

    def get_n_params(self):
        return len(self.fitted.params) if self.fitted is not None else 0

    def get_n_states(self):
        """Return number of states in ARIMA model.
        
        ARIMA(p, d, q) has:
        - p AR (autoregressive) states
        - d levels of differencing (stored differences)
        - q MA (moving average) states (lagged errors)
        
        Total states = p + d + q
        """
        p, d, q = self.order
        return p + d + q        

    def get_model_params(self):
        return dict(self.model_params)


class RecurrentForecaster(nn.Module):
    def __init__(
        self,
        rnn_cls,
        n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        learning_rate=0.001,
        seq_length=30,
        n_epochs=50,
        batch_size=32,
        clip_output=True,
        clip_range=(0, 1),
        **layer_kwargs,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_output = clip_output
        self.clip_range = clip_range

        # recurrent layer is the only model-specific part
        self.rnn = rnn_cls(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            **layer_kwargs
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.model_params = dict(
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            seq_length=seq_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            dropout=dropout,
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

    def fit(self, X, y):
        if len(X) < self.seq_length:
            print(f"Warning: Not enough data ({len(X)} < {self.seq_length})")
            return

        X_train = torch.FloatTensor(X)
        y_train = torch.FloatTensor(y).unsqueeze(-1)
        n = len(X_train)

        self.train()
        for epoch in range(self.n_epochs):
            idx = np.random.permutation(n)
            total = 0
            nb = 0
            for s in range(0, n, self.batch_size):
                batch = idx[s:s+self.batch_size]
                bx, by = X_train[batch], y_train[batch]

                self.optimizer.zero_grad()
                pred = self(bx)
                loss = self.criterion(pred, by)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
                nb += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {total/nb:.6f}")

    def predict(self, X, n_steps=1):
        self.eval()
        X = torch.FloatTensor(X)
        out = []

        with torch.no_grad():
            for i in range(X.shape[0]):
                seq = X[i].unsqueeze(0)
                preds = []
                for _ in range(n_steps):
                    y = self(seq).item()
                    if self.clip_output:
                        y = np.clip(y, *self.clip_range)

                    preds.append(y)
                    next_input = torch.tensor([[[y]]], dtype=torch.float32)
                    seq = torch.cat([seq[:, 1:, :], next_input], dim=1)
                out.append(preds)

        return np.array(out)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_model_params(self):
        return dict(self.model_params)

    def get_n_states(self):
        """Return total number of hidden states across all layers.
        
        For RNN/GRU: num_layers * hidden_size
        For LSTM: num_layers * hidden_size * 2 (hidden state + cell state)
        """
        rnn_type = type(self.rnn).__name__
        base_states = self.num_layers * self.hidden_size
        
        if rnn_type == 'LSTM':
            # LSTM has both hidden state (h) and cell state (c)
            return base_states * 2
        else:
            # RNN and GRU only have hidden state (h)
            return base_states        


class LSTMModel(RecurrentForecaster):
    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, **kwargs)


class GRUModel(RecurrentForecaster):
    def __init__(self, **kwargs):
        super().__init__(nn.GRU, **kwargs)


class RNNModel(RecurrentForecaster):
    def __init__(self, **kwargs):
        super().__init__(nn.RNN, **kwargs)


def get_models(model_name, n_features=1, clip_output=True, clip_range=(0, 1), **kwargs):
    """Factory function for baseline models."""
    model_name_lower = model_name.lower()
    
    # Persistence baseline (naive forecast)
    if model_name_lower == 'persistence':      
        return PersistenceModel()

    elif model_name_lower == "arima":
        return ARIMAModel(
            order=(kwargs.get("order_p", 1),
                   kwargs.get("order_d", 0),
                   kwargs.get("order_q", 1),
                  ),
            clip_output=clip_output,
            clip_range=clip_range,
        )
    
    elif model_name_lower in ("lstm", "gru", "rnn"):
        cls_map = {
            "lstm": LSTMModel,
            "gru": GRUModel,
            "rnn": RNNModel,
        }
        cls = cls_map[model_name_lower]        
        seq_length = kwargs["seq_length_min"] // int(kwargs["window"])
    
        return cls(
            n_features=n_features,
            hidden_size=int(kwargs.get("hidden_size", 64)),
            num_layers=int(kwargs.get("num_layers", 2)),
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            seq_length=seq_length,
            clip_output=clip_output,
            clip_range=clip_range,
            n_epochs=int(kwargs.get("n_epochs", 50)),
            batch_size=int(kwargs.get("batch_size", 32)),
            dropout=float(kwargs.get("dropout", 0.0)),
        )


    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _get_combined_features(x_act, x_heart, forecasting_modality):
    """Combine features based on forecasting modality."""
    if forecasting_modality == 'a|a':
        xa = x_act.reshape(-1, 1) if x_act.ndim == 1 else x_act
        xh = x_heart.reshape(-1, 1) if x_heart.ndim == 1 else x_heart
    elif forecasting_modality == 'a|ah':
        xa = np.column_stack([x_act])
        xh = np.column_stack([x_act, x_heart])
    elif forecasting_modality == 'ah|ah':
        xa = np.column_stack([x_act, x_heart])
        xh = np.column_stack([x_act, x_heart])
    elif forecasting_modality == 'a|h':
        xa = x_heart.reshape(-1, 1) if x_heart.ndim == 1 else x_heart
        xh = x_heart.reshape(-1, 1) if x_heart.ndim == 1 else x_heart
    else:
        raise ValueError(f"Unknown forecasting modality: {forecasting_modality}")
    
    return xa, xh    

def run_base(
    processed_path,
    patient_id,
    model_name='lstm',
    eval_all_dt=True, # if false, eval only in shifts of prediction horizons
    forecasting_modality='a|a',
    prediction_horizon=60,
    verbose=0,
    min_days=0,
    burn_in_days=10,
    n_recent_kernel=30,  # Number of recent timesteps to use as context
    **kwargs
):
    """Run baseline model with train/test split and recursive prediction.
    
    Parameters:
    -----------
    processed_path : str
        Path to processed data
    patient_id : int
        Patient identifier
    model_name : str
        Model name ('lstm', 'persistence', etc.)
    forecasting_modality : str
        Input/output modality ('a|a', 'a|ah', 'ah|ah', 'a|h')
    prediction_horizon : int
        Prediction horizon in minutes
    verbose : int
        Verbosity level
    min_days : int
        Minimum days required
    burn_in_days : int
        Days to use for training
    n_recent_kernel : int
        Number of recent timesteps for context
    **kwargs : dict
        Additional model parameters (hidden_size, num_layers, learning_rate, etc.)
    """

    args = {
        "processed_path": processed_path,
        "patient_id": patient_id,
        "model_name": model_name,
        "forecasting_modality": forecasting_modality,
        "prediction_horizon": prediction_horizon,
        "verbose": verbose,
        "min_days": min_days,
        "burn_in_days": burn_in_days,
        "n_recent_kernel": n_recent_kernel,
        **kwargs
    }
    print(args)
    
    HEART_RATE_SCALE = 200
    
    # Validate inputs
    valid_modes = {'ah|ah', 'a|ah', 'a|a', 'a|h'}
    if forecasting_modality not in valid_modes:
        raise ValueError(f"forecasting_modality must be one of {valid_modes}")
        
    # Load data (patient_id serves as index)
    loaded = np.load(processed_path, allow_pickle=True)
    metadata = pd.DataFrame.from_records(loaded['metadata']).set_index('patient_count').loc[patient_id]
    n_per_day = metadata['n_time_bins']
    dt = int(60 * 24 / n_per_day)
    n_forward = int(prediction_horizon / dt)
    n_recent_kernel = int(kwargs.get('seq_length_min', 60)) // dt
    
    if n_forward < 1:
        raise ValueError(f"prediction_horizon ({prediction_horizon}min) must be >= dt ({dt}min)")
    
    if verbose:
        print(f'Time resolution: {dt}min')
        print(f'Prediction horizon: {prediction_horizon}min ({n_forward} steps)')
        print(f'Burn-in period: {burn_in_days} days')
        
    def trim_nan_rows(arr):
        """Remove leading/trailing all-NaN rows."""
        valid_rows = ~np.all(np.isnan(arr), axis=tuple(range(1, arr.ndim)))
        first_valid = np.argmax(valid_rows)
        last_valid = len(valid_rows) - np.argmax(valid_rows[::-1])
        return arr[first_valid:last_valid]
    
    # Select modalities
    data = {}
    for mod in ['activity', 'heart', 'sleep']:
        m_idx = np.where(loaded['modalities'] == mod)[0]
        timeseries_2d = loaded['data'][patient_id][:, :, m_idx]
        data[mod] = np.squeeze(trim_nan_rows(timeseries_2d))
    
    # Flatten to 1D
    act = data['activity'].flatten()
    heart = data['heart'].flatten()
    sleep = data['sleep'].flatten()
    
    total_steps = len(act)
    steps_per_day = n_per_day
    days = total_steps / steps_per_day
    
    # Calculate statistics
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
    
    # Normalize heart rate
    heart_normalized = heart / HEART_RATE_SCALE
    
    # ========================================
    # TRAIN/TEST SPLIT
    # ========================================
    train_steps = int(burn_in_days * steps_per_day)
    
    if train_steps >= total_steps:
        print(f"Error: burn_in_days ({burn_in_days}) exceeds total data")
        return []
    
    # Training data
    act_train = act[:train_steps].copy()
    heart_train = heart_normalized[:train_steps].copy()
    
    # Compute mean on training data for imputation
    train_mean_act = np.nanmean(act_train)
    train_mean_heart = np.nanmean(heart_train)
    
    if verbose:
        print(f'Training mean - Activity: {train_mean_act:.4f}, Heart: {train_mean_heart:.4f}')
    
    # Impute training data with mean
    act_train_imputed = np.where(np.isnan(act_train), train_mean_act, act_train)
    heart_train_imputed = np.where(np.isnan(heart_train), train_mean_heart, heart_train)
    
    # Create full imputed arrays (for prediction context)
    act_imputed = np.where(np.isnan(act), train_mean_act, act)
    heart_imputed = np.where(np.isnan(heart_normalized), train_mean_heart, heart_normalized)
    
    # ========================================
    # BUILD TRAINING SEQUENCES
    # ========================================
    X_train_a = []
    y_train_a = []
    X_train_h = []
    y_train_h = []

    # all possible pairs of (seq, target)
    for i in range(n_recent_kernel, train_steps):
        x_act_ctx = act_train_imputed[i-n_recent_kernel:i]
        x_heart_ctx = heart_train_imputed[i-n_recent_kernel:i]

        # TODO: shape? Column stack? Should be [n_samples, n_dim]
        xa, xh = _get_combined_features(x_act_ctx, x_heart_ctx, forecasting_modality)
        
        y_act = act_train_imputed[i]
        y_heart = heart_train_imputed[i]
        
        X_train_a.append(xa)
        y_train_a.append(y_act)
        X_train_h.append(xh)
        y_train_h.append(y_heart)
    
    X_train_a = np.array(X_train_a)
    y_train_a = np.array(y_train_a)
    X_train_h = np.array(X_train_h)
    y_train_h = np.array(y_train_h)
    
    # ========================================
    # TRAIN MODELS
    # ========================================
    n_features_a = X_train_a.shape[-1] if X_train_a.ndim > 1 else 1
    n_features_h = X_train_h.shape[-1] if X_train_h.ndim > 1 else 1
    
    if verbose:
        print(f'\nTraining activity model (n_features={n_features_a})...')
    model_a = get_models(model_name, n_features=n_features_a, **kwargs)    
    model_a.fit(X_train_a, y_train_a)
    
    if verbose:
        print(f'Training heart rate model (n_features={n_features_h})...')
    model_h = get_models(model_name, n_features=n_features_h, **kwargs)
    model_h.fit(X_train_h, y_train_h)

   # Get model hyperparameters
    model_params_a = model_a.get_model_params()
    model_params_h = model_h.get_model_params()
    
    if verbose:
        print(f'Model parameters - Activity: {model_params_a}, Heart: {model_params_h}')


    
    # ========================================
    # PREDICT ON ALL DATA
    # ========================================
    res = []

    # dense: t -> t+dt, sparse (like Kalman model) t -> t + dt*n_forward
    dt_eval = 1 if eval_all_dt else n_forward
    for t in range(n_recent_kernel, total_steps - n_forward + 1, dt_eval):
        day = t // steps_per_day
        step = t % steps_per_day
        
        # Determine train/test flag
        is_train = t < train_steps
        
        # Extract context:
        x_act_ctx = act_imputed[t-n_recent_kernel:t]
        x_heart_ctx = heart_imputed[t-n_recent_kernel:t]

        #WHY doesn't this break it?
        xa, xh = _get_combined_features(x_act_ctx, x_heart_ctx, forecasting_modality)
        
        # Predict n_forward steps recurrsively
        # augment shape: (seq_length, n_features) -> (n_samples=1, seq_length, n_features)
        ya_pred_seq = model_a.predict(xa[None], n_steps=n_forward)[0]
        yh_pred_seq = model_h.predict(xh[None], n_steps=n_forward)[0]
                
        # Store results for each prediction step
        for i in range(n_forward):
            target_idx = t + i
            if target_idx >= total_steps:
                break
            
            ya_true = act[target_idx]
            yh_true = heart_normalized[target_idx]
            
            fac = dt / 60.0
            
            res.append({
                # Time indexing (backward compatible)
                'hour': (step + i) * fac,
                'day': day,
                'time': day + (step + i) / steps_per_day,
                
                # Activity predictions
                'activity_true': ya_true,
                'activity_pred': ya_pred_seq[i],
                
                # Heart rate predictions (denormalized)
                'heart_true': yh_true * HEART_RATE_SCALE if not np.isnan(yh_true) else np.nan,
                'heart_pred': yh_pred_seq[i] * HEART_RATE_SCALE,
                
                # Context
                'sleep': sleep[target_idx],
                
                # Train/test flag (NEW)
                'is_train': is_train,
                
                # Model configuration
                'model_name': model_name,
                'n_pars_activity': model_a.get_n_params(),
                'n_pars_heart': model_h.get_n_params(),
                
                # Prediction metadata
                'dt': dt,
                'prediction_horizon': prediction_horizon,
                'n_forward': n_forward,
                'prediction_index': i,
                
                # Experimental design
                'forecasting_modality': forecasting_modality,
                'n_recent_kernel': n_recent_kernel,
                
                # Data statistics
                'min_days': min_days,
                'burn_in_days': burn_in_days,
                'valid_days': valid_days,
                'nan_days': nan_days,
                'total_days': days,
                'pid': patient_id
            })

            # read out hyperparams
            for pref, model_params in zip(['a','h'],[model_params_a, model_params_h]):
                for param_name, param_value in model_params.items():
                    res[-1][f'{pref}_{param_name}'] = param_value

    
    if verbose:
        print(f'\nCompleted: {len(res)} predictions')
        if verbose > 1:
            res_df = pd.DataFrame(res)
            
            # Separate train/test
            train_df = res_df[res_df['is_train']]
            test_df = res_df[~res_df['is_train']]
            
            print(f'\nTrain statistics (n={len(train_df)}):')
            if len(train_df) > 0:
                train_act_rmse = np.sqrt(np.nanmean((train_df.activity_true - train_df.activity_pred)**2))
                train_hr_rmse = np.sqrt(np.nanmean((train_df.heart_true - train_df.heart_pred)**2))
                print(f'  Activity RMSE: {train_act_rmse:.4f}')
                print(f'  Heart Rate RMSE: {train_hr_rmse:.2f} bpm')
            
            print(f'\nTest statistics (n={len(test_df)}):')
            if len(test_df) > 0:
                test_act_rmse = np.sqrt(np.nanmean((test_df.activity_true - test_df.activity_pred)**2))
                test_hr_rmse = np.sqrt(np.nanmean((test_df.heart_true - test_df.heart_pred)**2))
                print(f'  Activity RMSE: {test_act_rmse:.4f}')
                print(f'  Heart Rate RMSE: {test_hr_rmse:.2f} bpm')
    
    return res

if __name__ == "__main__":
    run_base(
        processed_path="/sc/arion/projects/Clinical_Times_Series/cpp_data/final/processed/case/revision_nan_all_patients_60.npz",
        patient_id=138,
        model_name="arima",
        forecasting_modality="a|a",
        prediction_horizon=60,
        verbose=1,
        min_days=20,
        burn_in_days=10,
        window=60,
        order_p=1,
        order_d=0,
        order_q=1,
        #n_recent_kernel=30,
        # kwargs below        
        #hidden_size=32,
        #num_layers=1,
        #learning_rate=0.001,
        #seq_length_min=120,
        #n_epochs=30,
        #batch_size=32,
        #dropout=0.1,
        #has_field=False,
    )
