# run_config.py
import argparse
import os
import pandas as pd
from utils import run_seq

def parse_train_hours(value):
    # value is either blank/NaN or an integer number of hours
    if pd.isna(value) or str(value).strip() == "":
        return None
    return int(value)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="CSV file with sweep configs")
    p.add_argument("--index", required=True, type=int, help="1-based job index (LSF LSB_JOBINDEX)")
    p.add_argument("--processed_root", default="/sc/arion/projects/Clinical_Times_Series/cpp_data/final/processed",
                   help="root for processed files (optional)")
    p.add_argument("--output_path", default='/sc/arion/projects/Clinical_Times_Series/cpp_data/runs/', help="path for results (optional)")
    p.add_argument("--env-name", default=None, help="name/path for conda env (optional)")
    args = p.parse_args()

    df = pd.read_csv(args.config)
    if args.index < 1 or args.index > len(df):
        raise SystemExit(f"Index {args.index} out of range. Configs = {len(df)}")
    
    row = df.iloc[args.index - 1]  # convert 1-based LSF index to 0-based
    patient_id = int(row["patient_id"])
    window = int(row["window"])
    forecasting_modality = str(row["forecasting_modality"])
    train_hours = parse_train_hours(row.get("train_hours", ""))
    alpha_reg = float(row["alpha_reg"])
    kernel_hour = int(row["kernel_hour"] * 60/window) # normalize
    kernel_day = int(row["kernel_day"])

    # choose processed path depending on case/ctrl logic (adjust as needed)
    # If your CSV contains dataset column you can use that; here we assume 'case' processed naming
    processed_path = os.path.join(args.processed_root,
                                  f"case/revision_nan_all_patients_{window}.npz")

    print("==== Running config ====")
    print(row.to_dict())
    print("Processed path:", processed_path)
    print("========================")
    
    # call run_seq() from utils
    res = run_seq(
        processed_path,
        patient_id=patient_id,
        kernel_day=kernel_day,
        kernel_hour=kernel_hour,
        field_model='cross-linear',
        alpha_reg=alpha_reg,
        train_hours=train_hours,
        verbose=1,
        min_days=20,
        forecasting_modality=forecasting_modality,
        prediction_horizon=60,
    )
    
    outdir = args.output_path
    os.makedirs(outdir, exist_ok=True)
    safe_mod = forecasting_modality.replace("|", "_")
    out_path = os.path.join(outdir,
                            f"p{patient_id}_w{window}_{safe_mod}_kh{int(kernel_hour)}-kd{int(kernel_day)}_a{alpha_reg}.csv")
    pd.DataFrame(res).to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
