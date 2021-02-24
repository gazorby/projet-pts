#!/usr/bin/env sh

# Copy scripts to hosts

hostfile=${1-hostfile}

while IFS= read -r dest; do
    rsync -r -a ./node "pi@$(echo $dest | cut -d ' ' -f1):~/"
done < "$hostfile"
