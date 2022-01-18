#!/bin/bash

set -e


usage()
{
cat << EOF

usage: $0 options

This script runs a battery of simulations 

OPTIONS:
   -p <pname:v1,v2,v3>              an array ov values  
   --param=<pname:v1,v2,v3>   
   -h --help                        show this help

EOF
}


declare -A PARAMS

# getopt
GOTEMP="$(getopt -o "h:p:" -l "help,param:"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"


while true ;
do
    case "$1" in
        -p | --param) 
            key=${2/:*/}
            vals=${2/*:/}
            PARAMS[$key]=$vals
            shift 2;;
        -h | --help)
            echo "on help"
            usage; exit;
            shift;
            break;;
        --) shift ; 
            break ;;
    esac
done

exit

#TODO read proper arrays of parameters

srcdir=$(dirname -- "$0"| xargs realpath)
srcdir=${srcdir/$(basename $srcdir)/}
scripts=${srcdir}/scripts
seeddir="$(echo ${srcdir} | sed 's/\/\([^\/]\+\)$//')/seeds"
mkdir -p $seeddir
n=6

seeds=($(for((i=0;i<n;i++)); do od -vAn -N2 -tu2 < /dev/urandom; done))
ess=(0.2 0.5 300 0.2 0.5 300)
pas=(5 5 5 9 9 9)


for((i=0;i<5;i++)); do
    seed=${seeds[$i]}
    es=${ess[$i]}
    pa=${pas[i]}
    tmpdir=$(mktemp -d)
    cd $tmpdir
    rsync --files-from=${srcdir}/code ${srcdir}/ ${tmpdir}/
    echo -n "running $seed ... in $tmpdir "
    scripts/schedule.sh -g -r -s $seed -p "epochs=300;explore_sigma=${es};predict_ampl=${pa};match_th=0.8"
    echo Done

    echo -n "Storing in  ${seeddir}/$seed "
    rsync -avz $tmpdir/ ${seeddir}/$seed/
    echo Done; echo

    cd $srcdir 
done
