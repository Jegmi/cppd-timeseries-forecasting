#!/usr/bin/env python3
import itertools
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
import argparse
from utils import count_pars

ratio90 = [125, 18, 109, 19, 44, 104, 43, 50, 153, 4, 45, 108, 91, 25, 156, 81, 52, 9, 138]

def compute_decay(T, window):
    return np.exp(- window / T)

def main(make_scripts: bool, n_pars_max: int = 10000, timestamp = None):
    # === Define parameter ranges ===
    patients = [-1] # ratio90 #list(range(159)) # will use ratio90
    time_windows = [60] # 1, 5, 15, 30,
    forecasting_modalities = ['a|a','a|h','a|ah','ah|ah'] # ['ah|ah', 'a|ah', 'a|a', 'a|h']
    T_relevants = np.array([10,40])*24*60 # np.array([1,2,3,4,5,6,8,10,15,20,25,30,35,40])  # 
    #initial_uncertainties = [round(5*(2/3)**n,2) for n in np.arange(12)] # 5 to 0.06
    initial_uncertainties = [5] # [round(5*(2/3)**n,2) for n in np.arange(12)] # 5 to 0.06
    # 0.05*np.array([100]) # [round(100 * (3/4)**n) for n in np.arange(12)]
    model_names = ['rls']    
    fields = ['cross','block','hour','day']
    K_mins = [5*60] # np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20])*60 # to minutes
    K_days = [5] # [1, 2, 3, 4, 5, 7, 10]
    
    # Binary model configuration flags
    use_diagonal = [True, False]
    use_poly = [False, True] # [True, False] #to enable
    use_logit = [False, True] # [True, False] #to enable
    
    # === Fixed parameters ===
    min_days = 20 # patient selection before
    prediction_horizon = 60 # min
    has_field = True
    burn_in_days = 40 # (should correspond to max(T_relevants))
        
    # === Generate combinations (with filtering) ===
    dismissed = 0
    configs = []
    for (p, w, fm, T, init_u, fld, km, kd, diag, poly, logit) in itertools.product(
        patients, time_windows, forecasting_modalities,
        T_relevants, initial_uncertainties, fields, 
        K_mins, K_days, use_diagonal, use_poly, use_logit
    ):
            
        lam = compute_decay(T, w)

        n_pars = count_pars(modality=fm, days=kd, minutes=km, field=fld, 
                            poly_feat=poly, time_res=w, 
                            horizon=prediction_horizon, diag=diag)        
                
        if n_pars < n_pars_max:
            configs.append(dict(
                patient_id=p,
                window=w,
                forecasting_modality=fm,
                lambda_forget=lam,
                T_relevant=T,
                initial_uncertainty=init_u,
                field=fld,
                has_field=has_field,
                min_days=min_days,
                prediction_horizon=prediction_horizon,
                kernel_minutes=km,
                kernel_day=kd,
                # Model configuration flags
                diagonal_covariance=diag,
                use_polynomial=poly,
                use_logit_space=logit,
                burn_in_days=burn_in_days
            ))
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
#BSUB -W 1:00
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
