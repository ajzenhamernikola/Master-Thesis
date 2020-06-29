#!/usr/bin/bash

# Setup the variables
WD="./"
PRINT_CATEGORIES=false
PLOT_FILESIZES=false
PLOT_FILTERED_DATA=false
SATZILLA=false
EDGELIST=false
NODE2VEC=false

# Run code
python3 ./main.py \
    -wd $WD \
    -printCategories $PRINT_CATEGORIES \
    -plotFilesizes $PLOT_FILESIZES \
    -plotFilteredData $PLOT_FILTERED_DATA \
    -satzilla $SATZILLA \
    -edgelist $EDGELIST \
    -node2vec $NODE2VEC