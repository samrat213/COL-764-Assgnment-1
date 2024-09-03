if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [coll-path] [indexfile] {0|1|2}"
    exit 1
fi

# Assign arguments to variables
arg1=$1
arg2=$2
arg3=$3
python invidx_cons.py args1 args2 args3