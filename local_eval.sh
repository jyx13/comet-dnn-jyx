#!/bin/bash

TOP_DIR="." # Your path
python3 ${TOP_DIR}/modules/comet_dnn_eval.py \
    --input_list=/Users/Jyx/Desktop/comet_stuff/input/input_files.txt \
    --flagfile=local_train_config.txt \
    --checkpoint_num=0 \