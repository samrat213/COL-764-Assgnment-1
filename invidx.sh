#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [coll-path] [indexfile] {0|1|2}"
    exit 1
fi

# Assign arguments to variables
arg1=$1
arg2=$2
arg3=$3

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

python invidx_cons.py $1 $2 $3

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;
#bash invidx.sh cord19-trec_covid-docs simple 0