# /bin/bash

PIDS=$(ps -af | grep python | grep -v grep | awk '{print $2}')

for pid in $PIDS ; do
    kill -s SIGCONT $pid
    echo $pid
    exit
done

