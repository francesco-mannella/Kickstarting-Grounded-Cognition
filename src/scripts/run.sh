#!/bin/bash

set -e
srcdir=$(dirname -- "$0"| xargs realpath)
srcdir=${srcdir/$(basename $srcdir)/}
scripts=${srcdir}/scripts


clear() {
    rm imgs/* &> /dev/null 
    killall -9 ${scripts}/sync.sh &>/dev/null
    killall -9 ${scripts}/get_tv_movie.sh &>/dev/null
    killall -9 sleep &>/dev/null
}

MODE=${1:-complete}
PARAMS=${2:- -g}
ONLINE=${3:-0}
if [[ "$MODE" == complete ]]; then
    echo complete
    if [[ "$ONLINE" == 1 ]]; then
        echo online
        clear
        ${scripts}/sync.sh &> /dev/null & 
        ${scripts}/get_tv_movie.sh 600 &> /dev/null & 
    fi

    echo "$PARAMS"
    curdir=$(pwd)
    cd ${srcdir}
    python SMMain.py $PARAMS 1>out.log 2>err.log
    cd $currdir

elif [[ "$MODE" == clear ]]; then
    echo clear
    clear
fi
