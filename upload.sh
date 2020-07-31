# /bin/bash

if [ "$1" != "" ] ; then
    for f in $* ; do 
        sudo docker cp $f 720620440bcc:/workspace/rxy-wrapper/
    done
    exit
fi

sudo docker cp ./autotest.sh 720620440bcc:/workspace/rxy-wrapper/
sudo docker cp ./cudawrt.c 720620440bcc:/workspace/rxy-wrapper/
sudo docker cp ./cudawrt.h 720620440bcc:/workspace/rxy-wrapper/
#sudo docker cp ./vaddr.c 720620440bcc:/workspace/rxy-wrapper/
#sudo docker cp ./vaddr.h 720620440bcc:/workspace/rxy-wrapper/
#sudo docker cp ./targs.c 720620440bcc:/workspace/rxy-wrapper/
#sudo docker cp ./targs.h 720620440bcc:/workspace/rxy-wrapper/
sudo docker cp ./cudawblas.c 720620440bcc:/workspace/rxy-wrapper/
sudo docker cp ./trace.c 720620440bcc:/workspace/rxy-wrapper/
sudo docker cp ./cudaw.h 720620440bcc:/workspace/rxy-wrapper/
sudo docker cp ./ldsym.h 720620440bcc:/workspace/rxy-wrapper/
