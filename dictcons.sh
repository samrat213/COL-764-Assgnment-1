if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [coll-path] {0|1|2}"
    exit 1
fi

# Assign arguments to variables
arg1=$1
arg2=$2
python dict_cons.py args1 args2