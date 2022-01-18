#!/bin/bash

set -e
srcdir=$(dirname -- "$0"| xargs realpath)
srcdir=${srcdir/$(basename $srcdir)/}
scripts=${srcdir}/scripts

mk_pic_from_num() {

SCRIPT="import matplotlib 
matplotlib.use('agg') 
from pylab import *
figure(figsize=(10,2)) 
axis('off') 
text(0.4, 0.5, '$1', fontsize=40) 
savefig('num.png')" 

python -c "$SCRIPT"

}

T=${1:-0}
delay=10

currdir=$(pwd)
cd $srcdir

while true; 
do
    sleep $T # every 10 mins
    cd storage
    for f in trajectories_*; 
    do 
        vm=${f/trajectories_/visual_map_}
        tvn=${f/trajectories_/tvn_}
        tv=${f/trajectories_/tv_}
        n=$(echo ${f/trajectories_/}| sed -e's/0\+\(.*\).png/\1/')

        mk_pic_from_num $n

        convert $f $vm +append $tvn; 
        convert num.png $tvn -append $tv; 
    done
    N=$(ls trajectories_*| wc -l)
    [[ $N -lt 20 ]] && NN=1 || NN=$((N/20)) 
    if [[ ! -z "$(ls tv_*.png)" ]]; then
        convert -loop 0 -delay $delay $(ls tv_*.png|sort -n|awk NR%${NN}==0) tv.gif
    fi
    cd ..
    [[ -f "storage/tv.gif" ]] && mv storage/tv.gif tv.gif
done

cd $currdir
