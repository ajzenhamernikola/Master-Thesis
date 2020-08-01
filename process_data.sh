#!/usr/bin/bash

# Setup the variables
WD="./INSTANCES/"
PRINT_CATEGORIES=false
PLOT_FILESIZES=false
PLOT_FILTERED_DATA=false
SATZILLA=false
EDGELIST=true
DGCNN=true

# Run code
python3 -m INSTANCES.main \
    -wd $WD \
    -printCategories $PRINT_CATEGORIES \
    -plotFilesizes $PLOT_FILESIZES \
    -plotFilteredData $PLOT_FILTERED_DATA \
    -satzilla $SATZILLA \
    -edgelist $EDGELIST \
    -dgcnn $DGCNN