import argparse
import os
import pandas as pd
from functools import partial
import traceback

# Cohort definitions
PATIENT_COHORTS = {
    "calibration": [125, 18, 109, 19, 44, 104, 43, 50, 153, 4, 45, 108, 91, 25, 156, 81, 52, 9, 138],  # case, highest quality
    "ctrl": list(range(71)), # control group
    "case" : list(range(159))
}
PATIENT_COHORTS["validation"] = [i for i in range(159) if i not in PATIENT_COHORTS["calibration"]] # case, but not in calibration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="CSV file with sweep configs")
    p.add_argument("--index", required=True, type=int, help="1-based job index (LSF LSB_JOBINDEX)")
    p.add_argument("--processed_root", default="/sc/arion/projects/Clinical_Times_Series/cpp_data/final/processed",
                 help="root for processed files (optional)")
    p.add_argument("--output_path", default='/sc/arion/projects/Clinical_Times_Series/cpp_data/runs/', help="path for results (optional)")
    p.add_argument("--env-name", default=None, help="name/path for conda env (optional)")
    args = p.parse_args()
    
    # prep input of config
    df = pd.read_csv(args.config)
    if args.index < 1 or args.index > len(df):
        raise SystemExit(f"Index {args.index} out of range. Configs = {len(df)}")
    
    row = df.iloc[args.index - 1]  # convert 1-based LSF index to 0-based
    
    # Common parameters
    patient_id = row["patient_id"] # int for single patient, otherwise str 
    window = int(row["window"])
    forecasting_modality = str(row["forecasting_modality"])
    model_name = str(row.get("model_name", "rls"))
    has_field = bool(row["has_field"])
    min_days = int(row["min_days"])
    prediction_horizon = int(row.get("prediction_horizon", 60))
    burn_in_days = int(row.get("burn_in_days", 10)) # only used for eval
    verbose = 1
    
    # Resolve patient specification to list of IDs and data path
    try:
        patient_ids = [int(patient_id)]
        cohort_dir = 'case'
    except (ValueError, TypeError):
        patient_ids = PATIENT_COHORTS[patient_id]
        cohort_dir = 'ctrl' if patient_id == 'ctrl' else 'case'
    processed_path = os.path.join(args.processed_root, f"{cohort_dir}/revision_nan_all_patients_{window}.npz")
                
    # Define safe_mod early for output filename
    safe_mod = forecasting_modality.replace("|", "_")

    if has_field:
        field = row.get("field", "cross")
        lambda_forget = float(row["lambda_forget"])
        initial_uncertainty = float(row["initial_uncertainty"])
        n_kernel_recent = int(row["kernel_minutes"]/window)
        kernel_day = int(row["kernel_day"])
        
        # New binary flags
        diagonal_covariance = bool(row.get("diagonal_covariance", False))
        use_polynomial = bool(row.get("use_polynomial", False))
        use_logit_space = bool(row.get("use_logit_space", False))

        # load run file
        from prob_models import run_prob
        
        # Create partial function with all common arguments
        run_prob_configured = partial(
            run_prob,
            processed_path,
            kernel_day=kernel_day,
            n_kernel_recent=n_kernel_recent,
            field=field,
            model_name=model_name,
            lambda_forget=lambda_forget,
            initial_uncertainty=initial_uncertainty,
            diagonal_covariance=diagonal_covariance,
            use_polynomial=use_polynomial,
            use_logit_space=use_logit_space,
            verbose=verbose,
            min_days=min_days,
            forecasting_modality=forecasting_modality,
            prediction_horizon=prediction_horizon,
            burn_in_days=burn_in_days
        )

        # Determine cohort name for logging and output        

        print(f"==== Running config for Job {args.index}: {cohort_dir} ({len(patient_ids)} patient{'s' if len(patient_ids) > 1 else ''}) ====")
        print(row.to_dict())
        print("Processed path:", processed_path)
        print(f"Patient IDs: {patient_ids[:5]}{'...' if len(patient_ids) > 5 else ''}")
        print("========================")
        
        # init results
        success = []
        failure = []
                    
        for pid in patient_ids:
            print(f"--- Processing patient: {pid} ---")
            try:
                single = run_prob_configured(patient_id=pid)
        
                # if insufficient data
                if single[0]['valid_days'] < single[0]['min_days']:
                    print("insufficient data for pid:", pid)
                    failure.append({
                        "pid": pid,
                        "valid_days": single[0]['valid_days'],
                        "nan_days": single[0]['nan_days'],
                        "total_days": single[0]['total_days'],
                        "min_days": single[0]['min_days'],
                        "reason": "insufficient_data",
                        "job_index": args.index - 1,
                    })
                    continue
        
                # if success
                success.extend(single)

            # if unknown failure
            except Exception as e:
                tb = traceback.format_exc()
                print(f"!!! ERROR processing patient {pid} for job {args.index}: {e}")
                print(tb)
        
                failure.append({
                    "pid": pid,
                    "job_index": args.index - 1,
                    "error": str(e),
                    "traceback": tb,
                    "reason": "exception",
                })
       
        # Create output filename with binary flags (0/1)
        diag_flag = 1 if diagonal_covariance else 0
        poly_flag = 1 if use_polynomial else 0
        logit_flag = 1 if use_logit_space else 0
        
        output_file = (f"i{args.index - 1}_p{patient_id}_w{window}_{safe_mod}_"
                      f"m{model_name}_h{prediction_horizon}_f{field}_"
                      f"d{diag_flag}_poly{poly_flag}_logit{logit_flag}_burn{burn_in_days}.csv")
        
        # Make output directory
        out_dir = args.output_path
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, output_file)
        failed_path = os.path.join(out_dir, "failed.csv")
                
        # Save results
        if success:
            df = pd.DataFrame(success)
            df['cohort'] = cohort_dir
            df.to_csv(out_path, index=False)
            print(f"Saved {len(success)} results to: {out_path}")
        else:
            print(f"No results generated for job {args.index}. No file saved.")

        if failure:
            df_fail = pd.DataFrame(failure)
            df_fail['cohort'] = cohort_dir
            df_fail.to_csv(failed_path, index=False, mode='a',
                           header=not os.path.exists(failed_path))
            print(f"Saved {len(failure)} failures to: {failed_path}")
            

    else:  # baseline models
        print(f'Implement baseline models for job {args.index}')
        pass

if __name__ == "__main__":
    main()