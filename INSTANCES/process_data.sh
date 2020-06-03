#!/usr/bin/bash

# Setup the variables
WD="./"
PRINT_CATEGORIES=true
PLOT_FILESIZES=true
PLOT_FILTERED_DATA=true
SATZILLA=true
EDGELIST=true
NODE2VEC=true

# Run code
python3 ./main.py \
    -wd $WD \
    -printCategories $PRINT_CATEGORIES \
    -plotFilesizes $PLOT_FILESIZES \
    -plotFilteredData $PLOT_FILTERED_DATA \
    -satzilla $SATZILLA \
    -edgelist $EDGELIST \
    -node2vec $NODE2VEC