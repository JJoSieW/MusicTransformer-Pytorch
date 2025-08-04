#!/bin/bash

python preprocess_midi.py \
-root 'data/maestro-v2-extraction_eval' \
-output_dir 'data/extracted_contour_100' \
--contour_extract_analysis


# --custom_dataset
# --contour_extract_analysis    action="store_true", help="Whether or not implement contour extract analysis.")









