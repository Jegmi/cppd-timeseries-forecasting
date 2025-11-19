#!/bin/bash
cd /sc/arion/projects/Clinical_Times_Series/cpp_data/runs/20251118_1926
export CUR="/sc/arion/projects/Clinical_Times_Series/cpp_data/runs/20251118_1926"
bsub < make_eval.lsf
