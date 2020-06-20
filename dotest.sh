# /bin/bash



((cpmax = 0))
((cpmin = 0))
((cpsize = 0))
((argi = 0))
((argc = 3))

t_tail="0xfff"
t_crc="0"
t_testing="KIS_TESTING_RUN"
t_padding="-1"

if [ "$2" == "" ] ; then
    PY_RUN="main.py --epochs 1"
else
    PY_RUN="$2"
fi

case "$PY_RUN" in
    "main.py --epochs 1")
        R_END="Accuracy"
        R_OK="9665/10000"
        ;;
    "sum.py")
        R_END="T_END"
        R_OK="T_OK"
        ;;
esac

function do_print_kernel_cnt() {
    cp cuda-testing-rt.c cudart.c

    sed -i '/#define T_PRINT_KERNEL_CNT 0/d' cudart.c
    sed -i s/T_PRINT_KERNEL_CNT/1/ cudart.c

    sed -i s/T_CPSIZE/$cpsize/ cudart.c
    sed -i s/T_ARGI/$argi/ cudart.c
    sed -i s/T_ARGC/$argc/ cudart.c

    sed -i s/T_TAIL/$t_tail/ cudart.c
    sed -i s/T_CRC/$t_crc/ cudart.c
    sed -i s/T_TESTING/$t_testing/ cudart.c
    sed -i s/T_PADDING/$t_padding/ cudart.c

    run_test
}

function cp_cudart() {
    cp cuda-testing-rt.c cudart.c

    if [ "$1" != "" ] ; then
        sed -i s/cnt==0/$1/ cudart.c
    fi

    sed -i s/T_CPSIZE/$cpsize/ cudart.c
    sed -i s/T_ARGI/$argi/ cudart.c
    sed -i s/T_ARGC/$argc/ cudart.c

    sed -i s/T_TAIL/$t_tail/ cudart.c
    sed -i s/T_CRC/$t_crc/ cudart.c
    sed -i s/T_TESTING/$t_testing/ cudart.c
    sed -i s/T_PADDING/$t_padding/ cudart.c
}

function run_test() {
    LOG="$t_testing-$t_tail-$t_crc-$argi-$cpsize"
    echo min: $cpmin  cur: $cpsize  max: $cpmax padding: $t_padding
    if [ -f deal-rt.sh ] ; then
        cp cudart.c cuda-wrapper-rt.c
        do_gcc_and_cp_libcudart
        python $PY_RUN  1>$LOG.out  2>$LOG.err
        OUT=$(grep $R_END $LOG.out 2>/dev/null)
        if [ "$?" == "0" ]; then
            echo $OUT
            echo $OUT | grep "$R_OK" > /dev/null
            if [ "$?" == "0" ]; then
                return 0
            else
                return 1
            fi
        else
            ERR=$(grep RuntimeError $LOG.err 2>/dev/null)
            if [ "$?" == "0" ]; then
                echo $ERR
                return 2
            else 
                return 3
            fi
        fi
    fi
}

function test_argi_cpsize() {
    echo $t_testing $t_tail $t_crc $argc : $argi
    ((cpmin = 0))
    ((cpmax = 1024 * 1024))
    ((padding_cnt=3))
    PADDINGS="-1 0xc3 0"
    for ((cpsize=8; cpsize<=cpmax; cpsize=cpsize*2)); do
        ((maxcnt = 0))
        for t_padding in $(echo $PADDINGS) ; do 
            cp_cudart
            run_test
            case $? in
            0) # Successfully and accurately
                ((maxcnt = maxcnt+1))
                ;;
            1) # Successfully but inaccurately
                ((cpmin = cpsize))
                break
                ;;
            2) # Failed with RuntimeError
                ((cpmin = cpsize))
                break
                ;;
            3) # Failed with segment fault
                if ((cpsize <= 4096)) ; then
                    ((cpmin = cpsize))
                else
                    ((cpmax = cpsize))
                fi
                break
                ;;
            esac
        done
        if ((maxcnt == padding_cnt)) ; then
            ((cpmax = cpsize))
        fi
        if ((cpmax == cpsize)) ; then
            break
        fi
    done
    if ((cpmax <= 8)) ; then
        ((cpstep = 4))
    else 
        ((cpstep = 8))
    fi
    while ((cpmin + cpstep < cpmax)) ; do
        ((cpsize = ( (cpmax+cpmin) / 2 + cpstep - 1 ) / cpstep * cpstep))
        ((maxcnt = 0))
        for t_padding in $(echo $PADDINGS) ; do 
            cp_cudart
            run_test
            case $? in
            0) # Successfully and accurately
                ((maxcnt = maxcnt+1))
                ;;
            1) # Successfully but inaccurately
                ((cpmin = cpsize))
                break
                ;;
            2) # Failed with RuntimeError
                ((cpmin = cpsize))
                break
                ;;
            3) # Failed with segment fault
                ((cpmax = cpsize))
                break
                ;;
            esac
        done
        if ((maxcnt == padding_cnt)) ; then
            ((cpmax = cpsize))
        fi
    done

    while ((1)) ; do 
        ((padding_cnt=8))
        ((maxcnt=0))
        PADDINGS="0xf6 0x3 0x30 0xb 0xb0 0xf 0xf0 0x6f"
        ((cpsize = ( (cpmax+cpmin) / 2 + cpstep - 1 ) / cpstep * cpstep))
        TESTING_OUT="$t_testing $t_tail $t_crc $argi : $cpsize"
        for t_padding in $(echo $PADDINGS) ; do 
            cp_cudart
            run_test
            case $? in
            0) # Successfully and accurately
                ((maxcnt = maxcnt+1))
                ;;
            1) # Successfully but inaccurately
                break
                ;;
            2) # Failed with RuntimeError
                break
                ;;
            3) # Failed with segment fault
                break
                ;;
            esac
        done
        if ((maxcnt == padding_cnt)) ; then
            echo $TESTING_OUT >> output_of_testing.data
            break
        fi
        if (( cpsize == 4 )) ; then
            ((cpsize = 8))
        else
            ((cpsize = cpsize + 8))
        fi
    done
}

function test_func() {
    t_tail="$1"
    t_crc="$2"
    ((argc=$3))
    if [ "$4" != "" ] ; then
        ((argi=$4))
    fi
    for (( ; argi < argc; argi = argi + 1 )) ; do
        test_argi_cpsize
    done
}

function do_testing_cpsize() {
    t_testing="KIS_TESTING_CPSIZE"
    cat funcs.txt | while read line ; do
        test_func $line
    done
}

function print_devptr() {
    t_tail="$1"
    t_crc="$2"
    ((argc=$3))
    ((argi=$4))
    ((cpsize=$5))
    echo $t_tail $t_crc $argc $argi $cpsize
    cp_cudart
    run_test
    grep -e "^devp" -e "devptr" $LOG.out
}

function do_print_devptr() {
    t_testing="KIS_PRINT_DEVPTR"
    cat funcs.txt | while read line ; do
        print_devptr $line
    done
}

function test_argc() {
    t_tail="$1"
    t_crc="$2"
    ((n=$3))
    for ((argc = n - 1; argc > 0; argc--)) ; do
        echo $t_tail $t_crc $argc
        cp_cudart
        run_test
        if [ "$?" == "0" ] ; then
            ((n = argc))
        else 
            break
        fi
    done
    TESTING_OUT="$t_testing $t_tail $t_crc argc: $n"
    echo $TESTING_OUT >> output_of_argc.data
}

function do_testing_argc() {
    t_testing="KIS_TESTING_ARGC"
    cat funcs.txt | while read line ; do
        test_argc $line
    done
}

function do_testing_run() {
    t_testing="KIS_TESTING_RUN"
    cp_cudart
    run_test
}

function do_test_virt_addr() {
    t_testing="KIS_TESTING_RUN"
    cp_cudart "cnt>=0"
    cp cudart.c cuda-wrapper-rt.c
    ./deal-rt.sh
}

function do_test_maps() {
    (( tmin = 24, tmax = 27, tidx = -1, step = 4 ))
    while (( step != 0 )) ; do 
        echo "do test ($tmin, $tmax) ($tidx)"
        t_testing="KIS_TESTING_RUN"
        cp_cudart "cnt>=0"
        sed -i "s/T_ERRMIN/$tmin/g; s/T_ERRMAX/$tmax/g; s/T_ERRIDX/$tidx/g" cudart.c
        cp cudart.c cuda-wrapper-rt.c
        ./deal-rt.sh
        python p.py --epochs 1 1> out.log 2> out.err
        sleep 20
        grep RuntimeError out.err
        if [ "$?" == "0" ] ; then
            if (( step > 0 )) ; then
                (( step = -step ))
            fi
        else
            echo OK
            if (( step < 0 )) ; then
                (( step = -step ))
            fi
        fi
        (( step = step / 2 ))
        (( tmin = tmin + step))
    done
}

function do_deal() {
    t_testing="KIS_TESTING_RUN"
    cp_cudart 
    cp cudart.c cuda-wrapper-rt.c
    ./deal-rt.sh
}

case $1 in
  cpsize)
    do_testing_cpsize
    ;;
  count)
    do_print_kernel_cnt
    ;;
  devptr)
    do_print_devptr
    ;;
  argc)
    do_testing_argc
    ;;
  run)
    do_testing_run
    ;;
  virt)
    do_test_virt_addr
    ;;
  maps)
    do_test_maps
    ;;
  *)
    do_deal
    ;;
esac


