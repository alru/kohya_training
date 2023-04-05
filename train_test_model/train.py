import os
import config


repo_dir = config.repo_dir
sample_prompt = config.sample_prompt
config_file = config.config_file
dataset_config = config.dataset_config
accelerate_config = config.accelerate_config

def run():
    os.chdir(repo_dir)
    print("Will start training now")

    os.system(f"python train_network.py --sample_prompts={sample_prompt} --dataset_config={dataset_config} --config_file={config_file}")
    #In case you are using TPU, you can use next line 
    #os.system(f"accelerate launch --config_file={accelerate_config} --num_cpu_threads_per_process=1 train_network.py --sample_prompts={sample_prompt} --dataset_config={dataset_config} --config_file={config_file}")

