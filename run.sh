#!/bin/bash

venv_name="venv"
vert_path=$PWD/$venv_name
first_installation=false
if [ ! -d "$vert_path" ]; then
    echo "Virtual environment will be created in this path: ${vert_path}"
    sleep 1
    #sudo apt-get update -y
    #sudo apt-get install -y python3-venv
    python3 -m venv $vert_path
    first_installation=true
fi

if [first_installation]; then
bash install_packages.sh
fi
source "${vert_path}/bin/activate"
python --version
python run.py "$@"


