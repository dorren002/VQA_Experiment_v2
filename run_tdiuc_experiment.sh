#!/usr/bin/env bash

lr=2e-3
CONFIG=TDIUC_streaming
export PYTHONPATH=/home/qzhb/dorren/VQA_Experement/

DATA_ORDER=qtype
expt=${CONFIG}_${DATA_ORDER}_${lr}
MODE=limited_buffer
BUFFERSIZE=10000
NETWORK=mcan

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
--config_name ${CONFIG} \
--expt_name ${expt} \
--remind_features \
--icarl \
--rehearsal_mode ${MODE} \
--max_buffer_size ${BUFFERSIZE} \
--lr ${lr} &> ../logs/${expt}.log &

#DATA_ORDER=qtype # or qtype
#expt=${CONFIG}_${DATA_ORDER}_${lr}

#CUDA_VISIBLE_DEVICES=0 python -u vqa_experiments/vqa_trainer.py \
#--config_name ${CONFIG} \
#--expt_name ${expt} \
#--stream_with_rehearsal \
#--data_order ${DATA_ORDER} \
#--lr ${lr}
