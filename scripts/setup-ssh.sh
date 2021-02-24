#!/usr/bin/env sh

# Copy ssh keys to hosts listed in hostfile

machinefile=${1-machinefile}

copy() {
    while read -r line; do
        ssh-copy-id "pi@$(echo $line | cut -d ' ' -f1)"
    done < "$machinefile"
}

if [ -f ~/.ssh/id_rsa.pub ]; then
    copy
else
    ssh-keygen -t rsa
    copy
fi
