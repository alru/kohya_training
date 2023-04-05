# @title ## 5.3. LoRA and Optimizer Config
import os
import config
#TODO get environment variable

network_category = config.network_category




conv_dim = config.conv_dim
conv_alpha = config.conv_alpha

network_dim = config.network_dim
network_alpha = config.network_alpha

network_weight = config.network_weight
network_module = config.network_module
network_args = config.network_args


optimizer_type = config.optimizer_type

optimizer_args = config.optimizer_args


train_unet = config.train_unet
unet_lr = config.unet_lr
train_text_encoder = config.train_text_encoder
text_encoder_lr = config.text_encoder_lr
lr_scheduler = config.lr_scheduler
lr_warmup_steps = config.lr_warmup_steps

lr_scheduler_num_cycles = config.lr_scheduler_num_cycles
lr_scheduler_power = config.lr_scheduler_power

def run(network_weight=network_weight):
    print("- LoRA Config:")
    print(f"  - Additional network category: {network_category}")
    print(f"  - Loading network module: {network_module}")
    if not network_category == "LoRA":
      print(f"  - network args: {network_args}")
    print(f"  - {network_module} linear_dim set to: {network_dim}")
    print(f"  - {network_module} linear_alpha set to: {network_alpha}")
    if not network_category == "LoRA":
      print(f"  - {network_module} conv_dim set to: {conv_dim}")
      print(f"  - {network_module} conv_alpha set to: {conv_alpha}")

    if not network_weight:
        print("  - No LoRA weight loaded.")
    else:
        if os.path.exists(network_weight):
            print(f"  - Loading LoRA weight: {network_weight}")
        else:
            print(f"  - {network_weight} does not exist.")
            network_weight = ""

    print("- Optimizer Config:")
    print(f"  - Using {optimizer_type} as Optimizer")
    if optimizer_args:
        print(f"  - Optimizer Args: {optimizer_args}")
    if train_unet and train_text_encoder:
        print("  - Train UNet and Text Encoder")
        print(f"    - UNet learning rate: {unet_lr}")
        print(f"    - Text encoder learning rate: {text_encoder_lr}")
    if train_unet and not train_text_encoder:
        print("  - Train UNet only")
        print(f"    - UNet learning rate: {unet_lr}")
    if train_text_encoder and not train_unet:
        print("  - Train Text Encoder only")
        print(f"    - Text encoder learning rate: {text_encoder_lr}")
    print(f"  - Learning rate warmup steps: {lr_warmup_steps}")
    print(f"  - Learning rate Scheduler: {lr_scheduler}")
    if lr_scheduler == "cosine_with_restarts":
        print(f"  - lr_scheduler_num_cycles: {lr_scheduler_num_cycles}")
    elif lr_scheduler == "polynomial":
        print(f"  - lr_scheduler_power: {lr_scheduler_power}")

