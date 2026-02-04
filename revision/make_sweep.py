#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np
import yaml
import os
import shutil
from datetime import datetime
import argparse
from utils import count_pars

#BAD_PID = [ 1.,   2.,  11.,  13.,  21.,  22.,  26.,  39.,  49.,  53.,  80.,
#        93.,  94.,  95.,  98., 112., 113., 132., 133., 134., 141., 144.,
#       147., 155., 157.]

#job_index = [10, 100] # but probably it's every job

def compute_decay(T, window):
    return np.exp(- window / T)

def main(make_scripts: bool, n_pars_max: int = 10000, timestamp = None):


    param_rgs = BEST_PARAM_RGS['arima'] # gru, lstm, rls
    
    # === Define parameter ranges ===
    # LSTM
    """
    param_rgs = {
        "patient_id": ["calibration", "validation", "ctrl"],  #["ctrl","case"],
        "window": [60],
        "forecasting_modality": ["a|a"], #, "a|h", "a|ah", "ah|ah"],
        "hidden_size": [128],
        "num_layers": [1],
        "learning_rate": [1e-2, 5e-3, 1e-3],
        #"seq_length": [15, 30], # seq length
        "seq_length_min": [2400], #(np.array([40]) * 60).astype(int).tolist(), # avoid np.int64        
        "n_epochs": [50, 70, 90, 120],
        "batch_size": [32],
        "dropout": [0.1],
        "model_name" : ['gru','lstm'],
    }
    
    # RLS
    param_rgs = {
        "patient_id": ["ctrl","case"], # ["calibration"], # validation?
        "window": [60], #1, 5, 15, 30],
        "forecasting_modality": ["a|a","a|ah"], #  "a|h", 
        "T_relevant": (np.array([10,40]) * 24 * 60).tolist(),
        "initial_uncertainty": [5],
        "field": ["cross", "block", "hour", "day"],
        "kernel_minutes": [60],
        "kernel_day": [7],
        "diagonal_covariance": [False],
        "use_polynomial": [False],
        "use_logit_space": [False],
        "model_name" : 'rls',
    }

    # ARIMA

    param_rgs = {
        'patient_id': ['validation','ctrl'], # ["calibration"],  # "ctrl", "case"
        'p': [1],  # AR order (reduced for faster sweep)
        'd': [0],  # Differencing order
        'q': [0],  # MA order (reduced for faster sweep)
        'P': [1],  # Seasonal AR order
        'D': [0],  # Seasonal differencing (usually 0 or 1)
        'Q': [1],  # Seasonal MA order
        'm': [24],  # Seasonal period (0=no seasonality, 24=daily if hourly data)
        "window": [60],  # Time resolution in minutes
        'model_name': ['arima'],
        "forecasting_modality": ["a|a"],                    
        "eval_all_dt": [False],  # Evaluate only at prediction horizons (faster)
    }    
    """
    
    # === Fixed parameters ===
    fixed_params = {
        "min_days": 20,
        "prediction_horizon": 60,
        "has_field": 'field' in param_rgs,
        "burn_in_days": 10,
    }
    
    # === 2: Create Cartesian product grid ===
    keys = list(param_rgs.keys())
    values = list(param_rgs.values())
    
    configs = []
    dismissed = 0
    
    for combo in itertools.product(*values):
        row = dict(zip(keys, combo))
    
        # old filtering logic preserved        
        if 'field' in row and row['model_name']=='rls':
            row.update({"lambda_forget": compute_decay(row["T_relevant"], row["window"])})
            n_pars = count_pars(
                modality=row["forecasting_modality"],
                days=row["kernel_day"],
                minutes=row["kernel_minutes"],
                field=row["field"],
                poly_feat=row["use_polynomial"],
                time_res=row["window"],
                horizon=fixed_params["prediction_horizon"],
                diag=row["diagonal_covariance"]
            )
        else:            
            #print('only implemented pars count for -1')
            n_pars = -1
        
        below_max = n_pars < n_pars_max
            
        if below_max:
            cfg = {**row, **fixed_params}
            configs.append(cfg)
        else:
            dismissed += 1
            
    # === Save local CSV ===
    df = pd.DataFrame(configs)
    local_csv = "sweep_configs.csv"
    df.to_csv(local_csv, index=False)
    print(f"Generated {len(df)} configurations → {local_csv}")
    print(f"Not generated: {dismissed} with |pars| > {n_pars_max}")
    
    if not make_scripts:
        print("Skipping script generation (--make_scripts false).")
        return

    # === Prepare timestamped run folder ===
    base_run_path = "/sc/arion/projects/Clinical_Times_Series/cpp_data/runs/"
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    else:
        timestamp = str(timestamp)

    run_dir = os.path.join(base_run_path, timestamp)
    pred_dir = os.path.join(run_dir, "predictions")

    os.makedirs(pred_dir, exist_ok=True)

    # Copy sweep CSV to timestamp folder
    dest_csv = os.path.join(run_dir, os.path.basename(local_csv))
    shutil.copy2(local_csv, dest_csv)

    # Save parameter ranges        
    with open(os.path.join(run_dir, "param_rgs.yaml"), "w") as file:
        yaml.safe_dump({"param_rgs": param_rgs, "fixed_params": fixed_params}, file, sort_keys=False)
    
    # === Create run_sweep.lsf ===
    num_jobs = len(df)
    lsf_path = os.path.join(run_dir, "run_sweep.lsf")

    lsf_template = f"""#!/bin/bash
#BSUB -J "cpp_sweep[1-{num_jobs}]"
#BSUB -q premium
#BSUB -n 4
#BSUB -R "span[ptile=1]"
#BSUB -R affinity[core(4)]
#BSUB -R rusage[mem=16G]
#BSUB -W 12:00
#BSUB -P acc_Clinical_Times_Series
#BSUB -o logs/cpp_sweep.%J.%I.out
#BSUB -e logs/cpp_sweep.%J.%I.err

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

CONFIG_FILE="{dest_csv}"
ENV_PATH="/sc/arion/work/jegmij01/env_new/"
RUN_FILE="/hpc/users/jegmij01/jannes_setup/cpp_project/revision/run_config.py"

mkdir -p logs results

echo "LSB_JOBINDEX=${{LSB_JOBINDEX}}"
echo "Reading config file: ${{CONFIG_FILE}}"

conda run --no-capture-output -p ${{ENV_PATH}} python ${{RUN_FILE}} \\
    --config "${{CONFIG_FILE}}" \\
    --index ${{LSB_JOBINDEX}} \\
    --processed_root "/sc/arion/projects/Clinical_Times_Series/cpp_data/final/processed" \\
    --output_path "{pred_dir}"

echo "Done job ${{LSB_JOBINDEX}}"
"""

    with open(lsf_path, "w") as f:
        f.write(lsf_template)

    # === Create launch_jobs.sh in current working directory ===
    launch_path = os.path.join(os.getcwd(), "launch_jobs.sh")
    with open(launch_path, "w") as f:
        f.write(f"""#!/bin/bash
cd {run_dir}
export CUR="{run_dir}"
bsub < run_sweep.lsf
""")
    os.chmod(launch_path, 0o755)

    print(f"Created LSF file → {lsf_path}")
    print(f"Created launcher → {launch_path}")
    print(f"Run folder ready: $CUR → {run_dir} ")

def timestamp_type(x):
    if x is None:
        return None
    if x == "" or x.lower() == "none":
        return None
    if '_' in x:
        return x
    raise argparse.ArgumentTypeError("timestamp format, e.g. 20251112_1611")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sweep configs and optional LSF/launch scripts.")
    parser.add_argument("--make_scripts", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True,
                        help="If true (default), create timestamped folder and run scripts.")
    parser.add_argument(
        "--timestamp",
        type=timestamp_type,
        default=None,
        help="Optional numeric timestamp string."
    )
    args = parser.parse_args()
    main(args.make_scripts, timestamp=args.timestamp)
