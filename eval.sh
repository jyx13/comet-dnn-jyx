#!/bin/bash

TOP_DIR="." # Your path
python3 ${TOP_DIR}/modules/comet_dnn_eval.py \
    --input_list=/Users/Jyx/Desktop/comet_stuff/input/input_files.txt \
    --flagfile=/Users/Jyx/Desktop/comet_stuff/trained_models/n_turns_only/00000001_parameters.txt \
    --model_dir=/Users/Jyx/Desktop/comet_stuff/trained_models/n_turns_only/00000001 \
    --checkpoint_num=4459