#!/bin/bash

TOP_DIR="." # Your path
python3 ${TOP_DIR}/modules/comet_dnn_eval.py \
    --flagfile=/Users/Jyx/Desktop/0008/00000001_parameters.txt \
    --input_list=/Users/Jyx/Desktop/comet_stuff/input/input_files.txt \
    --model_dir=/Users/Jyx/Desktop/0008/00000001 \
    --checkpoint_num=3888