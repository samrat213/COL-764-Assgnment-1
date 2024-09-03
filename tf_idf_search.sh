if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [queryfile] [resultfile] [indexfile] [dictfile]"
    exit 1
fi

# Assign arguments to variables
arg1=$1
arg2=$2
arg3=$3
arg4=$4
python invidx_cons.py args1 args2 args3 args4