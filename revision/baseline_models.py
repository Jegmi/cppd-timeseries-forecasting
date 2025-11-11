    
    # ARIMA baseline
    elif model_name_lower == 'arima':
        from statsmodels.tsa.arima.model import ARIMA
        
        class ARIMAWrapper:
            """Wrapper to make ARIMA compatible with online learning interface."""
            def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), 
                        clip_output=True, clip_range=(0, 1)):
                self.order = order
                self.seasonal_order = seasonal_order
                self.clip_output = clip_output
                self.clip_range = clip_range
                self.history = []
                self.fitted_model = None
            
            def predict(self, X):
                """Make prediction."""
                if len(self.history) < max(self.order[0], self.order[2]) + 1:
                    # Not enough history, return mean or zero
                    pred = np.zeros(X.shape[0])
                else:
                    try:
                        model = ARIMA(self.history, order=self.order, 
                                     seasonal_order=self.seasonal_order)
                        fitted = model.fit()
                        pred = fitted.forecast(steps=X.shape[0])
                    except:
                        pred = np.full(X.shape[0], np.mean(self.history[-10:]))
                
                if self.clip_output:
                    pred = np.clip(pred, self.clip_range[0], self.clip_range[1])
                return pred
            
            def update(self, X, y):
                """Update with new observations."""
                self.history.extend(y.flatten().tolist())
                # Keep history manageable
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]
        
        model = ARIMAWrapper(
            order=kwargs.get('order', (1, 0, 1)),
            seasonal_order=kwargs.get('seasonal_order', (0, 0, 0, 0)),
            clip_output=clip_output,
            clip_range=clip_range
        )
    
    # Persistence baseline (naive forecast)
    elif model_name_lower == 'persistence':
        class PersistenceModel:
            """Simple persistence/naive forecast baseline."""
            def __init__(self, clip_output=True, clip_range=(0, 1)):
                self.last_value = None
                self.clip_output = clip_output
                self.clip_range = clip_range
            
            def predict(self, X):
                """Return last observed value."""
                if self.last_value is None:
                    pred = np.full(X.shape[0], 0.5)  # Default to mid-range
                else:
                    pred = np.full(X.shape[0], self.last_value)
                
                if self.clip_output:
                    pred = np.clip(pred, self.clip_range[0], self.clip_range[1])
                return pred
            
            def update(self, X, y):
                """Update with most recent observation."""
                self.last_value = y[-1, 0] if y.ndim > 1 else y[-1]
        
        model = PersistenceModel(clip_output=clip_output, clip_range=clip_range)
    
    # LSTM baseline
    elif model_name_lower == 'lstm':
        import torch
        import torch.nn as nn
        
        class LSTMModel:
            """Simple LSTM for time series forecasting."""
            def __init__(self, n_features, n_outputs=1, hidden_size=64, 
                        num_layers=2, learning_rate=0.001, seq_length=10,
                        clip_output=True, clip_range=(0, 1)):
                self.n_features = n_features
                self.n_outputs = n_outputs
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.seq_length = seq_length
                self.clip_output = clip_output
                self.clip_range = clip_range
                
                # Build network
                self.lstm = nn.LSTM(n_features, hidden_size, num_layers, 
                                   batch_first=True)
                self.fc = nn.Linear(hidden_size, n_outputs)
                self.optimizer = torch.optim.Adam(
                    list(self.lstm.parameters()) + list(self.fc.parameters()),
                    lr=learning_rate
                )
                self.criterion = nn.MSELoss()
                
                self.history_X = []
                self.history_y = []
            
            def predict(self, X):
                """Make prediction."""
                if len(self.history_X) < self.seq_length:
                    # Not enough history
                    return np.full(X.shape[0], 0.5)
                
                self.lstm.eval()
                with torch.no_grad():
                    # Use last seq_length samples
                    seq = torch.FloatTensor(self.history_X[-self.seq_length:]).unsqueeze(0)
                    lstm_out, _ = self.lstm(seq)
                    pred = self.fc(lstm_out[:, -1, :])
                    pred = pred.numpy().flatten()
                
                if self.clip_output:
                    pred = np.clip(pred, self.clip_range[0], self.clip_range[1])
                
                return pred[:X.shape[0]]
            
            def update(self, X, y):
                """Update with new observations."""
                self.history_X.extend(X.tolist())
                self.history_y.extend(y.tolist())
                
                # Keep manageable history
                if len(self.history_X) > 1000:
                    self.history_X = self.history_X[-1000:]
                    self.history_y = self.history_y[-1000:]
                
                # Train on sequences
                if len(self.history_X) >= self.seq_length:
                    self.lstm.train()
                    for i in range(max(1, len(X) // 5)):  # Light training
                        idx = np.random.randint(0, len(self.history_X) - self.seq_length)
                        seq_X = torch.FloatTensor(
                            self.history_X[idx:idx+self.seq_length]
                        ).unsqueeze(0)
                        seq_y = torch.FloatTensor(
                            self.history_y[idx+self.seq_length:idx+self.seq_length+1]
                        )
                        
                        self.optimizer.zero_grad()
                        lstm_out, _ = self.lstm(seq_X)
                        pred = self.fc(lstm_out[:, -1, :])
                        loss = self.criterion(pred, seq_y)
                        loss.backward()
                        self.optimizer.step()
        
        model = LSTMModel(
            n_features=n_features,
            n_outputs=n_outputs,
            hidden_size=kwargs.get('hidden_size', 64),
            num_layers=kwargs.get('num_layers', 2),
            learning_rate=kwargs.get('learning_rate', 0.001),
            seq_length=kwargs.get('seq_length', 10),
            clip_output=clip_output,
            clip_range=clip_range
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    