#!/usr/bin/env sh

# Copy ssh keys to hosts listed in hostfile

hostfile=${1-hostfile}
ssh_pwd=${2-raspberry}

copy() {
    while read -r line; do
        sshpass -p "$ssh_pwd" ssh-copy-id "pi@$(echo $line | cut -d ' ' -f1)"
    done < "$hostfile"
}

if [ -f ~/.ssh/id_rsa.pub ]; then
    copy
else
    ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa <<<y 2>&1 >/dev/null
    copy
fi
