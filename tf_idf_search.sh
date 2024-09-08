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

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

python tf_idf_retrieval.py $1 $2 $3 $4

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;
# bash tf_idf_search.sh cord19-trec_covid-queries retreval_simple.txt simple.idx simple.dict
# bash tf_idf_search.sh cord19-trec_covid-queries retreval_BPE.txt BPE.idx BPE.dict
# bash tf_idf_search.sh cord19-trec_covid-queries retreval_WPE.txt WPE.idx WPE.dict
