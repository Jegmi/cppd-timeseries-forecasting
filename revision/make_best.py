#!/usr/bin/env python3
import pandas as pd
import yaml
import numpy as np
import os
import shutil
from datetime import datetime
import argparse

def compute_decay(T, window):
    return np.exp(- window / T)

WINDOW = 60

BEST_PARAMS = {
    "arima": {
        "model_name": "arima",
        "patient_id": "validation",
        "window": WINDOW,
        "forecasting_modality": "a|a",
        "p": 1,
        "d": 0,
        "q": 0,
        "P": 1,
        "D": 0,
        "Q": 1,
        "m": 24,
        "has_field": False,
        "eval_all_dt": False,
    },
    "rls": {
        "model_name": "rls",
        "patient_id": "validation",
        "window": WINDOW,
        "forecasting_modality": "a|a",
        "T_relevant": 40 * 24 * 60,
        "initial_uncertainty": 5,
        "field": "cross",
        "kernel_minutes": 60,
        "kernel_day": 7,
        "diagonal_covariance": False,
        "use_polynomial": False,
        "use_logit_space": False,
        "has_field": True,
        "eval_all_dt": False,
    },
    "lstm": {
        "model_name": "lstm",
        "patient_id": "validation",
        "window": WINDOW,
        "forecasting_modality": "a|a",
        "hidden_size": 128,
        "num_layers": 2,
        "learning_rate": 1e-3,        
        "n_epochs": 50,
        "dropout": 0.1,
        "batch_size": 32,
        'seq_length_min' : 1200, # minutes 20h
        "has_field": False,
        "eval_all_dt": False,
    },        
}
"""
"gru": {
        "model_name": "gru",
        "patient_id": "validation",
        "window": WINDOW,
        "forecasting_modality": "a|a",
        "hidden_size": 128,
        "num_layers": 1,
        "learning_rate": 1e-3,        
        "n_epochs": 50,
        "dropout": 0.1,
        'seq_length_min' : 2400, # minutes
        "batch_size": 32,
        "has_field": False,
        "eval_all_dt": False,
    }
"""

FIXED_PARAMS = {
    "min_days": 20,
    "prediction_horizon": 60,
    "burn_in_days": 10,
}


def main(timestamp=None):
    # Combine all best params into single list, running over both patient_id values
    configs = []
    for model_name, params in BEST_PARAMS.items():
        for patient_id in ['validation', 'ctrl']:
            cfg = {**params, **FIXED_PARAMS}
            cfg['patient_id'] = patient_id
                        
            if model_name == 'rls':
                cfg.update({"lambda_forget": compute_decay(cfg["T_relevant"], cfg["window"])})

            #if model_name == 'lstm':
            configs.append(cfg)
    
    # Save local CSV
    df = pd.DataFrame(configs)
    local_csv = "best_configs.csv"
    df.to_csv(local_csv, index=False)
    print(f"Generated {len(df)} best configurations → {local_csv}")
    
    # Prepare timestamped run folder
    base_run_path = "/sc/arion/projects/Clinical_Times_Series/cpp_data/runs/"
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    else:
        timestamp = str(timestamp)

    run_dir = os.path.join(base_run_path, timestamp)
    pred_dir = os.path.join(run_dir, "predictions")

    os.makedirs(pred_dir, exist_ok=True)

    # Copy CSV to timestamp folder
    dest_csv = os.path.join(run_dir, os.path.basename(local_csv))
    shutil.copy2(local_csv, dest_csv)

    # Save best parameters
    with open(os.path.join(run_dir, "best_params.yaml"), "w") as file:
        yaml.safe_dump({"best_params": BEST_PARAMS, "fixed_params": FIXED_PARAMS}, file, sort_keys=False)
    
    # Create run_best.lsf
    num_jobs = len(df)
    lsf_path = os.path.join(run_dir, "run_best.lsf")

    lsf_template = f"""#!/bin/bash
#BSUB -J "cpp_best[1-{num_jobs}]"
#BSUB -q premium
#BSUB -n 4
#BSUB -R "span[ptile=1]"
#BSUB -R affinity[core(4)]
#BSUB -R rusage[mem=16G]
#BSUB -W 12:00
#BSUB -P acc_Clinical_Times_Series
#BSUB -o logs/cpp_best.%J.%I.out
#BSUB -e logs/cpp_best.%J.%I.err

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

    # Create launch_jobs.sh in current working directory
    launch_path = os.path.join(os.getcwd(), "launch_jobs.sh")
    with open(launch_path, "w") as f:
        f.write(f"""#!/bin/bash
cd {run_dir}
{run_dir}
bsub < run_best.lsf
""")
    os.chmod(launch_path, 0o755)

    print(f"Created LSF file → {lsf_path}")
    print(f"Created launcher → {launch_path}")
    print(f"Run folder ready: $CUR → {run_dir}")


def timestamp_type(x):
    if x is None:
        return None
    if x == "" or x.lower() == "none":
        return None
    if '_' in x:
        return x
    raise argparse.ArgumentTypeError("timestamp format, e.g. 20251112_1611")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate best model configs and LSF/launch scripts.")
    parser.add_argument(
        "--timestamp",
        type=timestamp_type,
        default=None,
        help="Optional numeric timestamp string."
    )
    args = parser.parse_args()
    main(timestamp=args.timestamp)