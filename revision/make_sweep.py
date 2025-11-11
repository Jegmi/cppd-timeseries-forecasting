# generate_configs.py
import itertools
import pandas as pd
import numpy as np

# === Define parameter ranges ===
patients = list(range(159))
time_windows = [1, 5, 15, 30, 60]
forecasting_modalities = ['ah|ah', 'a|ah', 'a|a', 'a|h']
train_days = [1, 2, 3, 5, 10, None]  # None = infinite
alpha_regs = [0.001, 0.01, 0.1]
kernel_hours = [3, 6, 12]
kernel_days = [1, 3, 7]

# === Optional filtering logic (skip too-large sweeps) ===
configs = []
for p, w, fm, td, ar, kh, kd in itertools.product(
    patients, time_windows, forecasting_modalities, train_days, alpha_regs, kernel_hours, kernel_days
):
    # Example: only do sensitivity for a few patients and windows
    if p % 10 == 0 and w in [15, 60]:
        configs.append(dict(
            patient_id=p,
            window=w,
            forecasting_modality=fm,
            train_hours=(None if td is None else 24 * td),
            alpha_reg=ar,
            kernel_hour=kh,
            kernel_day=kd
        ))

# === Save ===
df = pd.DataFrame(configs)
df.to_csv("sweep_configs.csv", index=False)
print(f"Saved {len(df)} configurations to sweep_configs.csv")