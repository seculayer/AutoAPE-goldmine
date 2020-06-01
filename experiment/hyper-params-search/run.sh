#!/bin/bash
####
BASE_DIR=
EXEC_FILE="$0"
BASE_NAME=`basename "$EXEC_FILE"`
if [ "$EXEC_FILE" = "./$BASE_NAME" ] || [ "$EXEC_FILE" = "$BASE_NAME" ]; then
        BASE_DIR=`pwd`
else
        BASE_DIR=`echo "$EXEC_FILE" | sed 's/'"\/${BASE_NAME}"'$//'`
fi
####
PARAM_NAME="$1"

### export path
export PYTHONPATH=$PYTHONPATH:$BASE_DIR
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

### experiment run
nohup python3.5 -m hps.HPSearcher "$PARAM_NAME" 1>/dev/null 2>&1 &
