#!/bin/bash

set -e
srcdir=$(dirname -- "$0"| xargs realpath)
srcdir=${srcdir/$(basename $srcdir)/}
scripts=${srcdir}/scripts

usage()
{
cat << EOF

usage: $0 options

This script runs a smc simulation with customized parameters

OPTIONS:
   -t --stime                 number of total time for the simulation (secs)
   -g --gpu                   use gpu
   -x --plots                 plot graphs
   -s --seed                  random seed (default 0)
   -p <param-string> 
     --params=<param-string>  parameters to be customized:
                              param-string: "<param>=<value>;<param>=<value>;..."    
   -o --online                sync graphs online
   -h --help                  show this help

EOF
}


TIME=
PARAMS=
GPU=
PLOTS=
SEED=0
ONLINE=0

# getopt
GOTEMP="$(getopt -o "gxoht:s:p:" -l "gpu,plots,online,help,time:,seed:,params:"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"


while true ;
do
    case "$1" in
        -t | --stime) 
            TIME="$2"
            shift 2;;
        -g | --gpu) 
            GPU="-g"
            shift;;
        -x | --plots) 
            PLOTS="-x"
            shift;;
        -s | --seed) 
            SEED="$2"
            shift 2;;
        -p | --params) 
            PARAMS="$2"
            shift 2;;
        -o | --online)
            ONLINE=1
            shift;;
        -h | --help)
            echo "on help"
            usage; exit;
            shift;
            break;;
        --) shift ; 
            break ;;
    esac
done

echo TIME=$TIME
echo PARAMS=$PARAMS
echo GPU=$GPU
echo PLOTS=$PLOTS
echo SEED=$SEED
echo ONLINE=$ONLINE

declare -A params 
name=sim
for par in $(echo "$PARAMS"| sed -e's/;/ /g'); do
    param_name=$(echo ${par} | sed -e"s/\([^ ]\+\)\s*=\s*\([^ ]\+\)/\1/")
    param_val=$(echo ${par} | sed -e"s/\([^ ]\+\)\s*=\s*\([^ ]\+\)/\2/")
    params[$param_name]=$param_val
    name=${name}_${param_name}_${param_val}
done

cd $srcdir    


for par in ${!params[@]}; do 
    sed -i -e "s/${par}\s*=.*/${par} = ${params[$par]}/" params.py
done

STIME="$([[ "$TIME" != "" ]] && echo "-t $TIME" || echo "")"
SEED="-s $SEED"

P="$SEED $GPU $PLOTS $STIME"
${scripts}/run.sh complete "$P" $ONLINE
