#!/usr/bin/bash

# Setup the variables
SKIP_METADATA="--skip="${1-no}
FEATURES=./third-party/SATzilla2012_features/features
PWD=$(pwd)

# Script code
if [ "$SKIP_METADATA" == "--skip=no" ]; then 
    echo "Creating the metadata"
    python3 variables_clauses_distribution.py 
    python3 filter_data.py 
    python3 solver_data.py
fi 

echo "Generating SATzilla2012 features" 
FILES=$PWD/cnf/*
for file in $FILES
do
    if [[ -d $file ]]; then
        OUTPUT_DIR=$file/features/
        echo "Creating directory $OUTPUT_DIR"
        $(mkdir -p $OUTPUT_DIR)
        for cnf_file in $file/* 
        do 
            if [[ -f $cnf_file ]]; then 
                INPUT_FILENAME=$cnf_file
                OUTPUT_FILENAME=${cnf_file%.cnf}.features
                $FEATURES $INPUT_FILENAME $OUTPUT_FILENAME
                mv $OUTPUT_FILENAME $OUTPUT_DIR
            fi
        done 
    fi
done 
