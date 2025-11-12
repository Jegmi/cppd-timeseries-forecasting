import argparse
import os
import pandas as pd

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
    
    # models and baselines:
    # This is the key variable from the config file
    patient_id = int(row["patient_id"]) 
    
    window = int(row["window"])
    forecasting_modality = str(row["forecasting_modality"])
    model_name = str(row.get("model_name", "rls"))
    has_field = bool(row["has_field"])
    min_days = int(row["min_days"])
    prediction_horizon = int(row.get("prediction_horizon", 60))
    verbose = 1

    # choose processed path depending on case/ctrl logic (adjust as needed)
    # If your CSV contains dataset column you can use that; here we assume 'case' processed naming
    processed_path = os.path.join(args.processed_root, f"case/revision_nan_all_patients_{window}.npz")
    
    # Define safe_mod early so it's available for the output filename
    safe_mod = forecasting_modality.replace("|", "_")

    if has_field:
        field = row.get("field", "cross")
        lambda_forget = float(row["lambda_forget"])
        initial_uncertainty = float(row["initial_uncertainty"])
        n_kernel_recent = int(row["kernel_minutes"]/window) # normalize
        kernel_day = int(row["kernel_day"])

        # load run file
        from prob_models import run_prob
        
        # This list will hold the final results, whether for one patient or many
        final_results = []
        
        # --- Option to run multiple patients:
        
        if patient_id == -1:
            # Special case: run all patients in the RATIO90_PATIENTS list
            print(f"==== Running config for Job {args.index}: Patient ID -1 (processing {len(RATIO90_PATIENTS)} patients) ====")
            print(row.to_dict())
            print("Processed path:", processed_path)
            print("========================")
            
            for pid_from_list in RATIO90_PATIENTS:
                print(f"--- Processing patient: {pid_from_list} ---")
                try:
                    # Call run_prob for the specific patient ID from the list
                    single_patient_res = run_prob(
                        processed_path,
                        patient_id=pid_from_list, # Use patient ID from loop
                        kernel_day=kernel_day,
                        n_kernel_recent=n_kernel_recent,
                        field=field,
                        model_name=model_name,
                        lambda_forget=lambda_forget,
                        initial_uncertainty=initial_uncertainty,
                        verbose=verbose,
                        min_days=min_days,
                        forecasting_modality=forecasting_modality,
                        prediction_horizon=prediction_horizon
                    )
                    # Concatenate results
                    if single_patient_res:
                        final_results.extend(single_patient_res)
                except Exception as e:
                    print(f"!!! ERROR processing patient {pid_from_list} for job {args.index}: {e}")
                    # Optionally add error info to results for easier debugging
                    final_results.append({
                        "job_index": args.index - 1,
                        "config_patient_id": patient_id,
                        "processed_patient_id": pid_from_list,
                        "error": str(e)
                    })
            
            # `final_results` now contains the concatenated results from all patients
            res = final_results

        else:
            # Original case: run for the single patient specified in the config
            print(f"==== Running config for Job {args.index}: Patient ID {patient_id} (single) ====")
            print(row.to_dict())
            print("Processed path:", processed_path)
            print("========================")
            
            res = run_prob(
                processed_path,
                patient_id=patient_id, # Use the specific patient_id from config
                kernel_day=kernel_day,
                n_kernel_recent=n_kernel_recent,
                field=field,
                model_name=model_name,
                lambda_forget=lambda_forget,
                initial_uncertainty=initial_uncertainty,
                verbose=verbose,
                min_days=min_days,
                forecasting_modality=forecasting_modality,
                prediction_horizon=prediction_horizon
            )
                    
        # The output filename uses the *original* patient_id from the config
        # (which will be -1 for the batch run, or a specific ID for a single run)
        output_file = f"i{args.index - 1}_p{patient_id}_w{window}_{safe_mod}_m{model_name}_h{prediction_horizon}_f{field}.csv"
        
        # Make output directory
        out_dir = args.output_path
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, output_file)
        
        # Save the results (which are either from one patient or concatenated from many)
        if res is not None and len(res) > 0:
            pd.DataFrame(res).to_csv(out_path, index=False)
            print("Saved:", out_path)
        else:
            print(f"No results generated for job {args.index}, patient_id {patient_id}. No file saved.")

    else: # baseline models
        print(f'Implement baseline models for job {args.index}')
        # Note: The original script would have crashed here.
        # This version will correctly do nothing if has_field is False.
        pass

if __name__ == "__main__":
    main()