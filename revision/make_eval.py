#!/usr/bin/env python3
"""
Generate evaluation script for parallel processing of model predictions.

Usage:
    python make_eval.py --make_scripts  # Create LSF and launch scripts
    python make_eval.py                  # Just show what would be created
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

def main(make_scripts: bool):
    # === Define parameters ===    
    run = '20251222_1952' # '20251222_1325'
    base_path = '/sc/arion/projects/Clinical_Times_Series/cpp_data/runs'
    n_jobs = 16  # Number of parallel cores
    do_threshold_sweep = True
    
    # Derived paths
    assert isinstance(run, str), f'run must be str, including an underscore but found {run} of type {type(run)}'
    runs_path = Path(base_path) / run
    print(runs_path)
    predictions_path = runs_path / 'predictions'
    metrics_path = runs_path / 'results'
    
    # Script locations
    env_path = "/sc/arion/work/jegmij01/env_new/"
    eval_script = "/hpc/users/jegmij01/jannes_setup/cpp_project/revision/evaluation.py"
    
    print(f"📊 Evaluation configuration:")
    print(f"   Run: {run}")
    print(f"   Predictions: {predictions_path}")
    print(f"   Output: {metrics_path}")
    print(f"   Cores: {n_jobs}")
    print(f"   Threshold sweep: {do_threshold_sweep}")
    
    if not make_scripts:
        print("\n⚠️  Skipping script generation (--make_scripts not provided).")
        return
    
    # === Create LSF submission script ===
    lsf_path = runs_path / "make_eval.lsf"
    
    threshold_flag = "--threshold-sweep" if do_threshold_sweep else ""
    
    lsf_template = f"""#!/bin/bash
#BSUB -J "cpp_eval_{run}"
#BSUB -q premium
#BSUB -n {n_jobs}
#BSUB -R "span[ptile=1]"
#BSUB -R affinity[core({n_jobs})]
#BSUB -R rusage[mem=16G]
#BSUB -W 48:00
#BSUB -P acc_Clinical_Times_Series
#BSUB -o logs/eval_{run}.%J.out
#BSUB -e logs/eval_{run}.%J.err

export OMP_NUM_THREADS={n_jobs}
export OPENBLAS_NUM_THREADS={n_jobs}
export MKL_NUM_THREADS={n_jobs}

ENV_PATH="{env_path}"
EVAL_SCRIPT="{eval_script}"

mkdir -p logs
mkdir -p "{metrics_path}"

echo "Starting evaluation for run {run}"
echo "Using {n_jobs} cores"
echo "Output directory: {metrics_path}"

conda run --no-capture-output -p ${{ENV_PATH}} python ${{EVAL_SCRIPT}} \\
    --run {run} \\
    --base-path {base_path} \\
    --n-jobs {n_jobs} {threshold_flag}

echo "Evaluation complete!"
"""
    
    with open(lsf_path, "w") as f:
        f.write(lsf_template)
    
    print(f"\n✅ Created LSF file → {lsf_path}")
    
    # === Create launch script ===
    launch_path = Path.cwd() / "launch_jobs.sh"
    
    with open(launch_path, "w") as f:
        f.write(f"""#!/bin/bash
cd {runs_path}
export CUR="{runs_path}"
bsub < make_eval.lsf
""")
    
    os.chmod(launch_path, 0o755)
    
    print(f"✅ Created launcher → {launch_path}")
    print(f"\n🚀 Ready to launch! Run:")
    print(f"   bash {launch_path}")
    print(f"\nOr manually:")
    print(f"   cd {runs_path}")
    print(f"   bsub < make_eval.lsf")


if __name__ == "__main__":
    main(True)
