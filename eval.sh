#!/bin/bash

TOP_DIR="." # Your path
python3 ${TOP_DIR}/modules/comet_dnn_eval.py \
    --flagfile=run_config.txt \
    --input_list=/Users/Jyx/Desktop/comet_stuff/input/input_files.txt