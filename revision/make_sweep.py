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

def main(make_scripts: bool, n_pars_max: int = 1000):
    # === Define parameter ranges ===
    patients = [-1] # ratio90 #list(range(159)) # will use ratio90
    time_windows = [60] # 1, 5, 15, 30, 
    forecasting_modalities = ['ah|ah', 'a|ah', 'a|a', 'a|h']
    T_relevants = 24*60*np.array([1,2,3,4,5,6,8,10,15,20,25,30]) # convert days to minutes by 24*60
    initial_uncertainties = [100] #, 30, 10, 3, 1]
    model_names = ['rls'] #, 'rls_logit', 'rls_poly', 'rls_logit_poly']
    model_names = model_names + [m + '_diag' for m in model_names] # add diag    
    fields = ['cross'] #, 'block', 'hour', 'day']
    K_mins = [300] # np.array([1, 2, 3, 4, 6, 8, 12, 16, 20])*60 # to minutes
    K_days = [3] # [1, 2, 3, 5, 8]
    
    # === Fixed parameters ===
    min_days = 20 # patient selection before
    prediction_horizon = 60 # min
    has_field = True
        
    # === Generate combinations (with filtering) ===
    dismissed = 0
    configs = []
    for (p, w, fm, T, init_u, mname, fld, km, kd) in itertools.product(
        patients, time_windows, forecasting_modalities,
        T_relevants, initial_uncertainties, model_names,
        fields, K_mins, K_days
    ):
        
        lam = compute_decay(T, w)
        
        n_pars = count_pars(modality=fm, days=kd, minutes=km, field=fld, 
                            poly_feat='poly' in mname, time_res=w, 
                            horizon=prediction_horizon, diag='diag' in mname)
        
        if n_pars < n_pars_max:
            configs.append(dict(
                patient_id=p,
                window=w,
                forecasting_modality=fm,
                lambda_forget=lam,
                T_relevant=T,
                initial_uncertainty=init_u,
                model_name=mname,
                field=fld,
                has_field=has_field,
                min_days=min_days,
                prediction_horizon=prediction_horizon,
                kernel_minutes=km,
                kernel_day=kd
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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
#BSUB -R rusage[mem=64G]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sweep configs and optional LSF/launch scripts.")
    parser.add_argument("--make_scripts", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True,
                        help="If true (default), create timestamped folder and run scripts.")
    args = parser.parse_args()
    main(args.make_scripts)
