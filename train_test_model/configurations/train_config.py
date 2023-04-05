# @title ## 5.4. Training Config

import toml
import os
import config

root_dir = config.root_dir
repo_dir = config.repo_dir
config_dir = config.config_dir

v2 = config.v2
v_parameterization = config.v_parameterization
pretrained_model_name_or_path = config.pretrained_model_name_or_path
unet_lr = config.unet_lr
vae = config.vae
train_unet = config.train_unet
text_encoder_lr = config.text_encoder_lr
train_text_encoder = config.train_text_encoder
network_weight = config.network_weight
network_module = config.network_module
network_dim = config.network_dim
network_alpha = config.network_alpha
network_args = config.network_args
optimizer_type = config.optimizer_type
optimizer_args = config.optimizer_args
lr_scheduler = config.lr_scheduler
lr_warmup_steps = config.lr_warmup_steps
lr_scheduler_num_cycles = config.lr_scheduler_num_cycles
lr_scheduler_power = config.lr_scheduler_power
output_dir = config.output_dir
project_name = config.project_name

lowram = config.lowram
enable_sample_prompt = config.enable_sample_prompt
sampler = config.sampler 
noise_offset = config.noise_offset  
num_epochs = config.num_epochs
train_batch_size = config.train_batch_size
mixed_precision = config.mixed_precision
save_precision = config.save_precision
save_n_epochs_type = config.save_n_epochs_type
save_n_epochs_type_value = config.save_n_epochs_type_value
save_model_as = config.save_model_as
max_token_length = config.max_token_length
clip_skip = config.clip_skip
gradient_checkpointing = config.gradient_checkpointing
gradient_accumulation_steps = config.gradient_accumulation_steps
seed = config.seed
logging_dir = config.logging_dir
prior_loss_weight = config.prior_loss_weight


def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def run():
    os.chdir(repo_dir)

    sample_str = f"""
  masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt \
  --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry \
  --w 512 \
  --h 768 \
  --l 7 \
  --s 28    
"""

    config = {
    "model_arguments": {
        "v2": v2,
        "v_parameterization": v_parameterization
        if v2 and v_parameterization
        else False,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "vae": vae,
    },
    "additional_network_arguments": {
        "no_metadata": False,
        "unet_lr": float(unet_lr) if train_unet else None,
        "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
        "network_weights": network_weight,
        "network_module": network_module,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_args": network_args,
        "network_train_unet_only": True if train_unet and not train_text_encoder else False,
        "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
        "training_comment": None,
    },
    "optimizer_arguments": {
        "optimizer_type": optimizer_type,
        "learning_rate": unet_lr,
        "max_grad_norm": 1.0,
        "optimizer_args": eval(optimizer_args) if optimizer_args else None,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
    },
    "dataset_arguments": {
        "cache_latents": True,
        "debug_dataset": False,
    },
    "training_arguments": {
        "output_dir": output_dir,
        "output_name": project_name,
        "save_precision": save_precision,
        "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
        "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
        "save_last_n_epochs": None,
        "save_state": None,
        "save_last_n_epochs_state": None,
        "resume": None,
        "train_batch_size": train_batch_size,
        "max_token_length": 225,
        "mem_eff_attn": False,
        "xformers": True,
        "max_train_epochs": num_epochs,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "seed": seed if seed > 0 else None,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "clip_skip": clip_skip if not v2 else None,
        "logging_dir": logging_dir,
        "log_prefix": project_name,
        "noise_offset": noise_offset if noise_offset > 0 else None,
        "lowram": lowram,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps": None,
        "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
        "sample_sampler": sampler,
    },
    "dreambooth_arguments": {
        "prior_loss_weight": 1.0,
    },
    "saving_arguments": {
        "save_model_as": save_model_as
    },
}

    config_path = os.path.join(config_dir, "config_file.toml")
    prompt_path = os.path.join(config_dir, "sample_prompt.txt")

    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    config_str = toml.dumps(config)



    write_file(config_path, config_str)
    write_file(prompt_path, sample_str)
    
    print(config_str)

