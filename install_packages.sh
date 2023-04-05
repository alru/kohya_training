#!/bin/bash

install_xformers=true

pip install -r requirements.txt

repo_url="https://github.com/Linaqruf/kohya-trainer"


root_dir=$PWD
deps_dir=$root_dir/"deps"
training_dir=$root_dir/"LoRA"
pretrained_model=$root_dir/"pretrained_model"
vae_dir=$root_dir/"vae"
config_dir=$root_dir/"config"

mkdir $deps_dir
mkdir $training_dir
mkdir $pretrained_model
mkdir $vae_dir
mkdir $config_dir

git clone $repo_url

sudo apt -y update
sudo apt --fix-broken install -y
sudo apt install libunwind8-dev
sudo apt --fix-broken install



url="https://huggingface.co/Linaqruf/fast-repo/resolve/main/deb-libs.zip",
name="deb-libs.zip",
dst=$deps_dir

# getting liberary 
wget -q --show-progress ${url}
unzip $name -d $dst
echo "will install ubuntu dependencies now"
sudo dpkg -i ${dst}/*
rm $name
rm -r ${dst}

t4_available=$(nvidia-smi | grep "T4")
echo $t4_available

#check if variable is empty or no
if [ ! -z "$t4_available" ]; then
sed -i 's@cpu@cuda@' library/model_util.py
fi

pip -q install --upgrade -r requirements.txt

if [install_xformers]; then
echo "Will install xformers....."
pip -q install xformers==0.0.16 triton==2.0.0
fi


export LD_PRELOAD=libtcmalloc.so
export TF_CPP_MIN_LOG_LEVEL=3
export BITSANDBYTES_NOWELCOME=1  
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"
export SAFETENSORS_FAST_GPU=1

echo "Finished installing dependancies and packages........."