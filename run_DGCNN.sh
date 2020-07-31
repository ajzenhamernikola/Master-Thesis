#!/bin/bash

# input arguments
DATA="CNF"
fold=${2-1}  # which fold as testing data

# general settings
gm=DGCNN  # model
gpu_or_cpu=cpu
GPU=0  # select the GPU number
CONV_SIZE="32-32-32-31"
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
bsize=1  # batch size, set to 50 or 100 to accelerate training
dropout=True

# dataset-specific settings
num_epochs=1
learning_rate=0.0001
test_number=1

CUDA_VISIBLE_DEVICES=${GPU} python -m src.models.DGCNN \
      -seed 1 \
      -data $DATA \
      -fold $fold \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -sortpooling_k $sortpooling_k \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $gpu_or_cpu \
      -dropout $dropout \
      -test_number ${test_number}
