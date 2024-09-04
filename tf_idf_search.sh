#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 [queryfile] [resultfile] [indexfile] [dictfile]"
    exit 1
fi

# Assign arguments to variables
arg1=$1
arg2=$2
arg3=$3
arg4=$4
python tf_idf_retrieval.py $1 $2 $3 $4

# bash tf_idf_search.sh cord19-trec_covid-queries retreval.txt simple.idx simple.dict
