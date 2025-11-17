import argparse
import os
import pandas as pd
from functools import partial

# List of patients to run when patient_id == -1
RATIO90_PATIENTS = [
    125, 18, 109, 19, 44, 104, 43, 50, 153, 4, 45, 108, 91, 25, 156, 81, 52, 9, 138
]

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
    patient_id = int(row["patient_id"]) 
    window = int(row["window"])
    forecasting_modality = str(row["forecasting_modality"])
    model_name = str(row.get("model_name", "rls"))
    has_field = bool(row["has_field"])
    min_days = int(row["min_days"])
    prediction_horizon = int(row.get("prediction_horizon", 60))
    burn_in_days = int(row.get("burn_in_days", 10)) # only used for eval
    verbose = 1

    # choose processed path depending on case/ctrl logic
    processed_path = os.path.join(args.processed_root, f"case/revision_nan_all_patients_{window}.npz")
    
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
        
        # This list will hold the final results
        final_results = []
        
        if patient_id == -1:
            # Run all patients in the RATIO90_PATIENTS list
            print(f"==== Running config for Job {args.index}: Patient ID -1 (processing {len(RATIO90_PATIENTS)} patients) ====")
            print(row.to_dict())
            print("Processed path:", processed_path)
            print("========================")
            
            for pid_from_list in RATIO90_PATIENTS:
                print(f"--- Processing patient: {pid_from_list} ---")
                try:
                    # Only need to pass patient_id now
                    single_patient_res = run_prob_configured(patient_id=pid_from_list)
                    if single_patient_res:
                        final_results.extend(single_patient_res)
                except Exception as e:
                    print(f"!!! ERROR processing patient {pid_from_list} for job {args.index}: {e}")
                    final_results.append({
                        "job_index": args.index - 1,
                        "config_patient_id": patient_id,
                        "processed_patient_id": pid_from_list,
                        "error": str(e)
                    })
            
            res = final_results

        else:
            # Run for single patient specified in config
            print(f"==== Running config for Job {args.index}: Patient ID {patient_id} (single) ====")
            print(row.to_dict())
            print("Processed path:", processed_path)
            print("========================")
            
            # Only need to pass patient_id now
            res = run_prob_configured(patient_id=patient_id)
        
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
        
        # Save results
        if res is not None and len(res) > 0:
            pd.DataFrame(res).to_csv(out_path, index=False)
            print("Saved:", out_path)
        else:
            print(f"No results generated for job {args.index}, patient_id {patient_id}. No file saved.")

    else:  # baseline models
        print(f'Implement baseline models for job {args.index}')
        pass

if __name__ == "__main__":
    main()