#!/bin/bash

# input arguments
DATA="CNF"
model_output_dir="./out/models"
cnf_dir="./INSTANCES"
model=GCN
fold=1  # which fold as testing data

# general settings
gm=DGCNN  # model
gpu_or_cpu=cpu
GPU=0  # select the GPU number
CONV_SIZE="64-64"
sortpooling_k=0.55  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=31  # final dense layer's hidden size
bsize=2   # batch size, set to 50 or 100 to accelerate training
dropout=True

# dataset-specific settings
num_epochs=100
look_behind=10
learning_rate=0.0001
conv1d_activation="LeakyReLU(0.1)"

CODE=134
while [[ $CODE == 134 || $CODE == 137 ]]; do
    CUDA_VISIBLE_DEVICES=${GPU} python -m main \
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
      -look_behind=$look_behind \
      -model_output_dir=$model_output_dir \
      -model=$model \
      -cnf_dir=$cnf_dir \
      -conv1d_activation=$conv1d_activation
    CODE=$?
    echo "Process ended with code: $CODE"
done
