# /bin/bash

DEFS=""
#DEFS="-D VA_TEST_DEV_ADDR"
DEFS="$DEFS"

DEBUG="-g"
#DEBUG=""

function do_gcc_and_cp_libcudamaster() {
    #gcc -I /usr/local/cuda-10.0/include/ cudamaster.c $DEFS $DEBUG -fPIC -shared -ldl -lcuda -o ./libcudamaster.so
    gcc -I /usr/local/cuda-10.0/include/ cudatest.c $DEFS $DEBUG -pthread -lcuda -lnvidia-ml -o ./test
    if [ "$?" != "0" ]; then exit; fi
}


do_gcc_and_cp_libcudamaster
