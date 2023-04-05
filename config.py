import os


#=============================== Options ============================
install_package = False
custom_caption = False

#=============================== Extra packages installation
# Please separate each package with a space
extra_packages = "accelerate tqdm torchvision" 

#=============================== Directories Configurations ==================================
root_dir = os.getcwd()
deps_dir = os.path.join(root_dir, "deps")
repo_dir = os.path.join(root_dir, "kohya-trainer")
training_dir = os.path.join(root_dir, "LoRA")
pretrained_model = os.path.join(root_dir, "pretrained_model")
vae_dir = os.path.join(root_dir, "vae")
config_dir = os.path.join(training_dir, "config")

# repo_dir
repo_url = "https://github.com/Linaqruf/kohya-trainer"
bitsandytes_main_py = "/usr/local/lib/python3.9/dist-packages/bitsandbytes/cuda_setup/main.py"
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
tools_dir = os.path.join(repo_dir, "tools")
finetune_dir = os.path.join(repo_dir, "finetune")

#=============================== Install dependencies configuration ==========================

branch = None  
install_xformers = True  
mount_drive = False  
verbose = False 


#=============================== Model Selection =============================================

#======= Pre-trained models----------
modelName = "Anything-v3-1"  #["Animefull-final-pruned", "Anything-v3-1", "AnyLoRA", "AnimePastelDream", "Chillout-mix", "OpenJourney-v4", "Stable-Diffusion-v1-5"]
v2ModelName = None  # ["stable-diffusion-2-1-base", "stable-diffusion-2-1-768v", "plat-diffusion-v1-3-1", "replicant-v1", "illuminati-diffusion-v1-0", "illuminati-diffusion-v1-1", "waifu-diffusion-1-4-anime-e2", "waifu-diffusion-1-5-e2", "waifu-diffusion-1-5-e2-aesthetic"]

#====== Custom models----------------
modelUrl = None  
#====== VAE--------------------------
vaeName =  "anime.vae.pt"  #["none", "anime.vae.pt", "waifudiffusion.vae.pt", "stablediffusion.vae.pt"]

hf_token = "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"

#======================================== Acquiring Data =======================================

train_data_dir = os.path.join(root_dir,"LoRA","train_data")  
reg_data_dir = os.path.join(root_dir,"LoRA","reg_data")  


zipfile_url = ""  
zipfile_name = "zipfile.zip"
unzip_to = ""

#======================================= Data Processing ========================================

#======== Data Cleaning------------------------
convert = False  
random_color = False 

#======== Data Annotation----------------------

#-----BLIB caption
batch_size_bilb = 8 #@param {type:'number'}
max_data_loader_n_workers_bilb = 2 #@param {type:'number'}
beam_search = True #@param {type:'boolean'}
min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}

#-----Waifu Diffusion Tagger
#@markdown [Waifu Diffusion 1.4 Tagger V2](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) is Danbooru-styled image classification developed by [SmilingWolf](https://github.com/SmilingWolf).  Can also be useful for general image tagging.
#@markdown Example: `1girl, solo, looking_at_viewer, short_hair, bangs, simple_background`. 
batch_size_waifu = 8 #@param {type:'number'}
max_data_loader_n_workers_waifu = 2 #@param {type:'number'}
model = "SmilingWolf/wd-v1-4-swinv2-tagger-v2" #@param ["SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
remove_underscores = True #@param {type:"boolean"} 
#@markdown Adjust threshold for better training results.
#@markdown - High threshold (e.g. `0.85`) for object/character training.
#@markdown - Low threshold (e.g. `0.35`) for general/style/environment training.
threshold = 0.35 #@param {type:"slider", min:0, max:1, step:0.05}

#-----Custom Caption
# @markdown Add or remove custom tags here.
# @markdown [Cheatsheet](https://rentry.org/kohyaminiguide#c-custom-tagscaption)
extension = ".txt"  # @param [".txt", ".caption"]
custom_tag = None  # @param {type:"string"}
# @markdown Enable this to append custom tags at the end of lines.
append = False  # @param {type:"boolean"}
# @markdown Enable this if you want to remove caption/tag instead
remove_tag = False  # @param {type:"boolean"}


#======================================= Training Configuration =====================================
#==============Model config--------------------
v2 = False  # @param {type:"boolean"}
v_parameterization = False  # @param {type:"boolean"}
project_name = None  
if not project_name:
    project_name = "last"
pretrained_model_name_or_path = os.path.join(root_dir,"pretrained_model","Anything-v3-1.safetensors")  # @param {type:"string"}
vae = None  
output_dir = os.path.join(root_dir,"LoRA","output")  # @param {'type':'string'}

# @markdown This will ignore `output_dir` defined above, and changed to `/$root_dir/drive/MyDrive/LoRA/output` by default
output_to_drive = False  # @param {'type':'boolean'}

#=============Dataset config-------------------
# @markdown ### Dreambooth Config
# @markdown This will only work for `one concept` training. You can still do multi-concept training but it will require more work. [Guide](https://rentry.org/kohyaminiguide#b-multi-concept-training)
train_repeats = 10  # @param {type:"number"}
reg_repeats = 1  # @param {type:"number"}
# @markdown There is a lot of misinformation about the activation word, so I have made it clear in this [Rentry](https://rentry.org/kohyaminiguide#a-activation-word).
instance_token = "mksks"  # @param {type:"string"}
class_token = "style"  # @param {type:"string"}
# @markdown Enable this option if you want to add an instance token to your caption files. This will function in the same way as the `4.2.3. Custom Caption/Tag (Optional)` cell.
add_token_to_caption = False  # @param {type:"boolean"}
# @markdown ### <br>General Config
resolution = 512  # @param {type:"slider", min:512, max:1024, step:128}
flip_aug = False  # @param {type:"boolean"}
caption_extension = ".txt"  # @param ["none", ".txt", ".caption"]
caption_dropout_rate = 0  # @param {type:"slider", min:0, max:1, step:0.05}
caption_dropout_every_n_epochs = 0  # @param {type:"number"}
keep_tokens = 0  # @param {type:"number"}
if add_token_to_caption and keep_tokens < 2:
    keep_tokens = 1
#============= LoRA and Optimizer Config---------
# @markdown ### LoRA Config:
network_category = "LoRA"  # @param ["LoRA", "LoCon", "LoCon_Lycoris", "LoHa"]

# @markdown Recommended values:

# @markdown | network_category | network_dim | network_alpha | conv_dim | conv_alpha |
# @markdown | :---: | :---: | :---: | :---: | :---: |
# @markdown | LoRA | 32 | 1 | - | - |
# @markdown | LoCon | 16 | 8 | 8 | 1 |
# @markdown | LoHa | 8 | 4 | 4 | 1 |

# @markdown - Currently, `dropout` and `cp_decomposition` is not available in this notebook.

# @markdown `conv_dim` and `conv_alpha` are needed to train `LoCon` and `LoHa`, skip it if you train normal `LoRA`. But remember, when in doubt, set `dim = alpha`.
conv_dim = 32  # @param {'type':'number'}
conv_alpha = 16  # @param {'type':'number'}
# @markdown It's recommended to not set `network_dim` and `network_alpha` higher than `64`, especially for LoHa.
# @markdown But if you want to train with higher dim/alpha so badly, try using higher learning rate. Because the model learning faster in higher dim.
network_dim = 32  # @param {'type':'number'}
network_alpha = 16  # @param {'type':'number'}
# @markdown You can specify this field for resume training.
network_weight = ""  # @param {'type':'string'}
network_module = "lycoris.kohya" if network_category in ["LoHa", "LoCon_Lycoris"] else "networks.lora"

network_args = "" if network_category == "LoRA" else [
    f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
    ]

if network_category == "LoHa":
  network_args.append("algo=loha")
elif network_category == "LoCon_Lycoris":
  network_args.append("algo=lora")

# @markdown ### <br>Optimizer Config:
# @markdown `AdamW8bit` was the old `--use_8bit_adam`.
optimizer_type = "AdamW8bit"  # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
# @markdown Additional arguments for optimizer, e.g: `["decouple=true","weight_decay=0.6"]`
optimizer_args = None  # @param {'type':'string'}
# @markdown Set `unet_lr` to `1.0` if you use `DAdaptation` optimizer, because it's a [free learning rate](https://github.com/facebookresearch/dadaptation) algorithm.
# @markdown However `text_encoder_lr = 1/2 * unet_lr` still applied, so you need to set `0.5` for `text_encoder_lr`.
# @markdown Also actually you don't need to specify `learning_rate` value if both `unet_lr` and `text_encoder_lr` are defined.
train_unet = True  # @param {'type':'boolean'}
unet_lr = 1e-4  # @param {'type':'number'}
train_text_encoder = True  # @param {'type':'boolean'}
text_encoder_lr = 5e-5  # @param {'type':'number'}
lr_scheduler = "constant"  # @param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
lr_warmup_steps = 0  # @param {'type':'number'}
# @markdown You can define `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial` in the field below.
lr_scheduler_num_cycles = 0  # @param {'type':'number'}
lr_scheduler_power = 0  # @param {'type':'number'}


#========== Train config---------------
lowram = True  # @param {type:"boolean"}
enable_sample_prompt = True  # @param {type:"boolean"}
sampler = "ddim"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
noise_offset = 0.0  # @param {type:"number"}
num_epochs = 10  # @param {type:"number"}
train_batch_size = 6  # @param {type:"number"}
mixed_precision = "fp16"  # @param ["no","fp16","bf16"] {allow-input: false}
save_precision = "fp16"  # @param ["float", "fp16", "bf16"] {allow-input: false}
save_n_epochs_type = "save_every_n_epochs"  # @param ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
save_n_epochs_type_value = 1  # @param {type:"number"}
save_model_as = "safetensors"  # @param ["ckpt", "pt", "safetensors"] {allow-input: false}
max_token_length = 225  # @param {type:"number"}
clip_skip = 2  # @param {type:"number"}
gradient_checkpointing = False  # @param {type:"boolean"}
gradient_accumulation_steps = 1  # @param {type:"number"}
seed = -1  # @param {type:"number"}
logging_dir = os.path.join(root_dir,"LoRA","logs")
prior_loss_weight = 1.0

#======================================= Started training configuration ---------------------
#@markdown Check your config here if you want to edit something: 
#@markdown - `sample_prompt` : /$root_dir/LoRA/config/sample_prompt.txt
#@markdown - `config_file` : /$root_dir/LoRA/config/config_file.toml
#@markdown - `dataset_config` : /$root_dir/LoRA/config/dataset_config.toml

#@markdown Generated sample can be seen here: /content/LoRA/output/sample

#@markdown You can import config from another session if you want.
sample_prompt = os.path.join(root_dir,"LoRA","config","sample_prompt.txt") #@param {type:'string'}
config_file = os.path.join(root_dir,"LoRA","config","config_file.toml") #@param {type:'string'}
dataset_config = os.path.join(root_dir,"LoRA","config","dataset_config.toml") #@param {type:'string'}

#======================================= Test Model ===========================================
# @markdown Now you can check if your LoRA trained properly.

# @markdown  If you used `clip_skip = 2` during training, the values of `lora_te_text_model_encoder_layers_11_*` will be `0.0`, this is normal. These layers are not trained at this value of `Clip Skip`.
no_verbose = True  # @param {type:"boolean"}