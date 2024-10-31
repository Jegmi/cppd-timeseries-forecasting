from pathlib import Path
import os
import sys

DATA_PATH = Path('/sc/arion/work/jegmij01/patchtst/datasets/')

ROOT = DATA_PATH

# for data creation
OUT_PATH = Path(f'{ROOT}/private/cpp/20240924pkls/')

## contains 1, 5, 15 and 60 min: /cpp/20240820pkls/

# for plotting all the grids (original)
fig_path = Path(f'{ROOT}/../pic/data_grids/')

# Define the root path of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)