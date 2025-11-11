#!/usr/bin/env python
"""
Process CPP time series data into gridded format.
Handles case and control datasets with configurable time windows.
"""

import argparse
from pathlib import Path
import sys
from utils import combine_patient_data, ts_data_to_grids

def main():
    parser = argparse.ArgumentParser(
        description='Process CPP patient time series data into grids'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing CPP data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['case', 'ctrl'],
        required=True,
        help='Dataset to process: case or ctrl'
    )
    parser.add_argument(
        '--window',
        type=int,
        required=True,
        help='Time window size in minutes (e.g., 1, 5, 15, 30, 60)'
    )
    parser.add_argument(
        '--nan_threshold',
        type=float,
        default=0.3,
        help='Minimum proportion of valid values per window (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    
    # Map dataset to correct path
    if args.dataset == 'ctrl':
        data_path = data_root / '2025-10-04/cppd-2__control/'
    elif args.dataset == 'case':
        data_path = data_root / '2025-06-30/cppd-2__case/'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Verify path exists
    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        sys.exit(1)
    
    print(f"Processing {args.dataset} dataset with {args.window}-minute windows")
    print(f"Data path: {data_path}")
    print(f"NaN threshold: {args.nan_threshold}")
    
    # Load patient data
    print("Loading patient data...")
    try:
        patient_data = combine_patient_data(data_path, sleep=True)
        print(f"Loaded {len(patient_data)} patients")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    # Process and save
    processed_path = data_root / f'processed/{args.dataset}/'
    print(f"Processing and saving to: {processed_path}")
    
    try:
        _ = ts_data_to_grids(
            patient_data=patient_data,
            processed_path=processed_path,
            window=args.window,
            nan_thres_per_window=args.nan_threshold
        )
        print(f"SUCCESS: Completed {args.dataset} with window={args.window}")
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()