# /bin/bash

DEFS=""
#DEFS="-D VA_TEST_DEV_ADDR"
DEFS="$DEFS -D TEST_MAIN"
DEFS="$DEFS -D constructor=deprecated"
DEFS="$DEFS -D destructor=deprecated"

DEBUG="-g"
#DEBUG=""

gcc -I /usr/local/cuda-10.0/include/ trace.c blkcpy.c cudawrt.c cudawblas.c $DEFS $DEBUG -pthread -ldl -lcuda -lnvidia-ml -o ./test

if [ -f test ] ; then
    ./test
    rm test
fi
