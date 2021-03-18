#!/usr/bin/env bash

# Copy ssh keys to hosts listed in hostfile

hostfile=${1-hostfile}
ssh_pwd=${2-raspberry}
    
copy(){
    while IFS= read -r line; do
        echo "$line"
        [ ! -z "$line" ] && sshpass -p "$ssh_pwd" ssh-copy-id -f -o StrictHostKeyChecking=no "pi@$(echo $line | cut -d ' ' -f1)"
    done < $hostfile
}
if [ -f ~/.ssh/id_rsa.pub ]; then
    copy
else
    ssh-keygen -q -t rsa -N '' <<< ""$'\n'"y" 2>&1 >/dev/null
    copy
fi