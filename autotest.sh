# /bin/bash

# VA_ENABLE_VIR_ADDR
# VA_TEST_DEV_ADDR
# VA_VTOR_PRINT

# KI_DISABLE_TRANS_ARGS
# KI_TEST_FUNC
# KI_TEST_ARGC
# KI_TEST_DEVPTR
# PRINT_MBLK_TOTAL
# PRINT_ONLY_FIRST_NEW_FUNC
# KI_BYPASS_NEW_FUNC_ARGS
# KI_BYPASS_ALL_FUNC

DEFS="-D SYNC_AND_HOLD -D VA_TEST_DEV_ADDR"
DEFS="$DEFS"

function do_gcc_and_cp_libcudart() {
    gcc -I /usr/local/cuda-10.0/include/ cudawrt.c trace.c blkcpy.c realloc.c $DEFS -fPIC -shared -g -lpthread -ldl -lcuda -o ./libcudart.so.10.0.130
    if [ "$?" != "0" ]; then exit; fi
    rm -f /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/libcudart.so.10.0.130
    cp ./libcudart.so.10.0.130 /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/
    rm -f /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so.10.0.130
    ln -s /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/libcudart.so.10.0.130 /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so.10.0.130
    rm -f /opt/conda/lib/libcudart.so.10.0.130
    ln -s /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/libcudart.so.10.0.130 /opt/conda/lib/libcudart.so.10.0.130

    gcc -I /usr/local/cuda-10.0/include/ cudawblas.c $DEFS -fPIC -shared -g -ldl -lcuda -o ./libcublas.so.10.0.130
    if [ "$?" != "0" ]; then exit; fi
    rm -f /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/libcublas.so.10.0.130
    cp ./libcublas.so.10.0.130 /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/
    rm -f /opt/conda/lib/libcublas.so.10.0.130
    ln -s /opt/conda/pkgs/cudatoolkit-10.0.130-0/lib/libcublas.so.10.0.130 /opt/conda/lib/libcublas.so.10.0.130
}

function do_build_and_run() {
    apply_funcs
    do_gcc_and_cp_libcudart 
    remove_funcs
    if [ "$PY_RUN" == "" ] ; then
        echo "Nothing to run"
        exit
    fi
    rm -f out.log out.err out.out
    python $PY_RUN 1> out.log 2>out.err
}

function set_tail_crc_argc() {
    t_tail="$1"
    t_crc="$2"
    if [ "$3" != "" ] ;  then 
        (( argc = $3 ))
        if (( argc <= 0 )) ; then return 1; fi
    fi
    if [ "$t_tail" == "" ] ; then return 1; fi
    if [ "$t_crc" == "" ]  ; then return 1; fi
    return 0
}

function do_find_new_func() {
    DEFS="-D PRINT_ONLY_FIRST_NEW_FUNC"
    do_build_and_run
    grep -e "^new-func:" out.log > out.out
    TAIL_CRC="$(grep -e "^new-func:.*tail:" out.out | awk '{print $4 " " $6}')"
    ARGC="$(for i in $(grep -e "^new-func: argi:" out.log | awk '{print $3}') ; do (( x = $i + 1 )) ; done ; echo $x)"
    set_tail_crc_argc $TAIL_CRC $ARGC
    RET="$?"
    if [ "$RET" != "0" ] ; then
        echo "Fail: New func not found" $TAIL_CRC $ARGC
        return 1
    else
        echo
        echo "Found a new func:" $t_tail $t_crc $argc
        echo 
    fi
    return 0
}

function do_test_func_argc() {
    for (( c = argc - 1; c > 0; c = c - 1 )); do
        DEFX="$DEFZ -D KI_TEST_ARGC -D T_ARGC=$c"
        DEFS="$DEFX -D T_ADDC=0 -D T_ARGI=0 -D T_CPSIZE=0 -D T_ADDV=0"
        do_build_and_run
        grep -e "RuntimeError" out.err > out.out
        if [ "$?" == "0" ] ; then break ; fi
        (( argc = c ))
    done
    echo "argc = $argc is OK for" $t_tail $t_crc
    return 0
}

function test_func_argi_cpsize() {
    echo -e "\ntest argi cpsize: $t_tail $t_crc $argc : $argi"
    ((cpmin = 0))
    ((cpmax = 1024 * 1024))
    DEFX="$DEFZ -D T_ADDC=0 -D T_ADDV=0 -D T_ARGC=$argc"
    for ((cpsize=8; cpsize<=cpmax; cpsize=cpsize*2)); do
        echo min: $cpmin max: $cpmax cpsize: $cpsize 
        DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=$cpsize"
        do_build_and_run
        grep -e "$R_END" out.log > out.out
        if [ "$?" == "0" ] ; then ((cpmax = cpsize)); break; fi
        ((cpmin = cpsize))
    done
    if ((cpmax <= 8)) ; then ((cpstep = 4)); else ((cpstep = 8)); fi
    while ((cpmin + cpstep < cpmax)) ; do
        ((cpsize = ( (cpmax+cpmin) / 2 + cpstep - 1 ) / cpstep * cpstep))
        echo min: $cpmin max: $cpmax cpsize: $cpsize 
        DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=$cpsize"
        do_build_and_run
        grep -e "$R_END" out.log > out.out
        if [ "$?" == "0" ] ; then ((cpmax = cpsize)); continue; fi
        ((cpmin = cpsize));
    done
    ((cpsize = ( (cpmax+cpmin) / 2 + cpstep - 1 ) / cpstep * cpstep))
    if (( cpsize == 4 )) ; then 
        echo "Check padding: $padding for cpsize: $cpsize"
        (( padding = -1 ))
        DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=4 -D KI_PADDING=$padding"
        do_build_and_run
        grep -e "$R_END" out.log > out.out
        if [ "$?" != "0" ] ; then
            cat out.err
            echo "Ooops! Check padding: $padding for cpsize: $cpsize"
            exit
        fi
    elif (( cpsize == 8 )) ; then
        echo "Check padding: $padding for cpsize: $cpsize"
        (( padding = -1 ))
        DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=4 -D KI_PADDING=$padding"
        do_build_and_run
        grep -e "$R_END" out.log > out.out
        if [ "$?" != "0" ] ; then
            if ((t_cpsize <= cpsize)) ; then
                if ((t_cpsize == 8)) ; then
                    addv[$addc]="$t_argi"
                    (( addc++ ))
                    echo "Warning: argi: $t_argi might be addr pointer!"
                fi
                ((t_argi = argi))
                ((t_cpsize = cpsize))
            fi
        fi
    elif ((t_cpsize < cpsize)) ; then 
        if ((t_cpsize > 8)) ; then
            echo "Ooops! t_($t_argi, $t_cpsize) -> ($argi, $cpsize)"
            exit
        elif ((t_cpsize == 8)) ; then
            addv[$addc]="$t_argi"
            (( addc++ ))
            echo "Warning: argi: $t_argi might be addr pointer!"
        fi
        ((t_argi = argi))
        ((t_cpsize = cpsize))
    fi
    echo "$t_tail $t_crc $argc : $argi, $cpsize ( $t_argi, $t_cpsize )"
}

function do_test_func_all_argi() {
    (( t_cpsize=0, t_argi=0, addc=0 ))
    for (( argi = 0; argi < argc; argi = argi + 1 )) ; do
        test_func_argi_cpsize
    done
    echo -e "\n$t_tail $t_crc $argc : ( $t_argi, $t_cpsize )\n"
}

function do_test_func_argi_cpsize() {
    if (( t_cpsize <= 8 )); then 
        (( cpsize=0, argi=0 ))
        return 0
    fi
    echo -e "\ndo_test_func_argi_cpsize $t_tail $t_crc $argc : ($t_argi, $t_cpsize)"
    DEFX="$DEFZ -D T_ADDC=0 -D T_ADDV=0 -D T_ARGC=$argc"
    (( cpsize=t_cpsize, argi=t_argi ))
    while ((1)) ; do 
        (( padding_cnt=12, maxcnt=0 ))
        PADDINGS="0 0xf6 0x3 0x30 0xb 0xb0 0xf 0xf0 0x6f 0x3c 0xc3 -1"
        for padding in $(echo $PADDINGS) ; do 
            DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=$cpsize -D KI_PADDING=$padding"
            echo "$t_tail $t_crc $argc : $argi, $cpsize ($padding)"
            do_build_and_run
            grep -e "$R_OK" out.log > out.out
            if [ "$?" != "0" ] ; then break; fi
            (( maxcnt=maxcnt+1 ))
        done
        if ((maxcnt == padding_cnt)) ; then
            ((t_cpsize = cpsize))
            break
        fi
        ((cpsize = cpsize + 8))
    done
    echo -e "\nThe cpsize of [$t_tail $t_crc $argc] is ( $argi, $cpsize )\n"
}

function do_test_func_devptr() {
    echo "do_test_func_devptr $t_tail $t_crc $argc : ( $argi, $cpsize )"
    DEFX="$DEFZ -D T_ADDC=0 -D T_ADDV=0 -D T_ARGC=$argc"
    DEFX="$DEFX -D VA_TEST_DEV_ADDR -D KI_TEST_DEVPTR"
    DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=$cpsize"
    do_build_and_run
    grep -e "devptr: arg" out.log > out.out
    if (( addc == 0 )) ; then
        sed -i '/devptr: args/d' out.out
    fi
    (( args_addc=$(grep -e "args" out.out | wc -l | awk '{print $1}') ))
    (( argi_addc=$(grep -e "argi" out.out | wc -l | awk '{print $1}') ))
    (( addc=(args_addc + argi_addc) ))
    if (( args_addc > 0 && argi_addc > 0 )) ; then
        echo "\nOoops! args_addc=$args_addc and argi_addc=$argi_addc\n"
        exit
    fi
    unset addv
    (( addc=0 ))
    for idx in $(cat out.out | awk '{print $3}') ; do 
        for (( k=0; k<addc; k++ )) ; do
            if [ "${addv[$k]}" == "$idx" ] ; then
                break
            fi
        done
        if (( k >= addc )) ; then
            echo "addv[$addc]=$idx"
            addv[$addc]="$idx"
            (( addc = addc + 1 ))
        fi
    done
    addv_str=$(echo "${addv[*]}" | sed 's/ /,/g')
    echo "    {NULL, $t_tail, 0, $t_crc, 0, $argc, $addc, $argi, $cpsize, {$addv_str}}, // Auto-generated" >> funcs.c
}

function do_test_func_deep_copy() {
    echo "do_test_func_deep_copy $t_tail $t_crc $argc : ( $argi, $cpsize ) {$addv_str}"
    DEFX="$DEFZ -D T_ADDC=0 -D T_ADDV=0 -D T_ARGC=$argc"
    DEFX="$DEFX -D KI_PARGS_DEEP_COPY -D T_ADDV=$addv_str -D PRINT_MBLK_TOTAL"
    DEFS="$DEFX -D T_ARGI=$argi -D T_CPSIZE=$cpsize"
    do_build_and_run
    grep -e "$R_END" out.log
    cat out.err
}

function do_run() {
    echo "Now Try Run under vaddr disabled"
    DEFS="$DEFS -D KI_DISABLE_TRANS_ARGS"
    do_build_and_run
    grep -e "$R_END" out.log
    cat out.err
}

function do_virt_run() {
    echo "Now Try Run under vaddr enabled"
    DEFS="$DEFS -D VA_ENABLE_VIR_ADDR" # -D PRINT_MBLK_TOTAL -D VA_VTOR_PRINT
    do_build_and_run
    grep -e "$R_END" out.log
    cat out.err
}

function do_pargs_deep_copy() {
    echo "Now Try Run with PARGS deep copy"
    DEFS="-D KI_PARGS_DEEP_COPY"
    do_build_and_run
    grep -e "$R_END" out.log
    cat out.err
    exit
}

function sort_funcs() {
    cp -f funcs.c f.c
    sed -i 's/0x..,/0x0&/g' funcs.c
    sed -i 's/0x00x/0x0/g' funcs.c
    sort funcs.c -o funcs.c
}

function apply_funcs() {
    sed -i '/TARGS_KI_AUTO_GENERATED_FUNC_INSERT_BELOW/r funcs.c' targs.c
}

function remove_funcs() {
    sed -i '/Auto-generated/d' targs.c
    return $?
}

function func_done() {
    grep -e "$t_tail, 0, $t_crc" funcs.c
    return $?
}

function jump_to_devptr() {
    t_tail="$1"
    t_crc="$2"
    argc="$3"
    argi="$4"
    cpsize="$5"
}

function do_auto_test() {
    if ((0)) ; then
        jump_to_devptr 0xfd0 2957903632 3 0 16
        DEFZ="-D KI_BYPASS_NEW_FUNC_ARGS -D KI_TEST_FUNC -D T_TAIL=$t_tail -D T_CRC=$t_crc"
        do_test_func_devptr
        exit
    fi
    sort_funcs
    while ((1)) ; do 
        do_find_new_func
        if [ "$?" == "1" ] ; then
            break
        fi
        DEFZ="-D KI_BYPASS_NEW_FUNC_ARGS -D KI_TEST_FUNC -D T_TAIL=$t_tail -D T_CRC=$t_crc"
        do_test_func_argc
        do_test_func_all_argi
        do_test_func_argi_cpsize
        do_test_func_devptr
        do_test_func_deep_copy
    done
    do_virt_run
}

case "$2" in
    "world")
        PY_RUN="world/main.py --cuda --epochs 1"
        R_END="End of training"
        R_OK="test ppl   233.70"
        ;;
    "sum.py")
        PY_RUN="sum.py"
        R_END="T_END"
        R_OK="T_OK"
        ;;
    "main.py")
        PY_RUN="main.py --epochs 1"
        R_END="Accuracy"
        R_OK="9665/10000"
        ;;
    "")
        PY_RUN=""
        ;;
    *)
        echo $*
        exit
        ;;
esac

case "$1" in 
    "run")
        do_run
        ;;
    "virt")
        do_virt_run
        ;;
    "deep")
        do_pargs_deep_copy
        ;;
    "auto")
        do_auto_test
        ;;
    "gcc")
        do_gcc_and_cp_libcudart
        ;;
    *)
        echo $*
        exit
        ;;
esac
