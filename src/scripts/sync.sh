#!/bin/bash
set -e
srcdir=$(dirname -- "$0"| xargs realpath)
srcdir=${srcdir/$(basename $srcdir)/}
scripts=${srcdir}/scripts

while true; do
    rsync -avz --delete-excluded  $srcdir/  ~/public_html/src/ >/dev/null  2>&1 
done

