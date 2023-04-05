# @title ## 2.1. Download Available Model
import os
import config

root_dir = os.environ["root_dir"]
pretrained_model = os.environ["pretrained_model"]

os.chdir(root_dir)

installModels = []
installv2Models = []

# @markdown ### SD1.x model
modelUrl = [
    "",
    "https://huggingface.co/Linaqruf/personal-backup/resolve/main/models/animefull-final-pruned.ckpt",
    "https://huggingface.co/cag/anything-v3-1/resolve/main/anything-v3-1.safetensors",
    "https://huggingface.co/Lykon/AnyLoRA/resolve/main/AnyLoRA_noVae_fp16.safetensors",
    "https://huggingface.co/Lykon/AnimePastelDream/resolve/main/AnimePastelDream_Soft_noVae_fp16.safetensors",
    "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/chillout_mix-pruned.safetensors",
    "https://huggingface.co/prompthero/openjourney-v4/resolve/main/openjourney-v4.ckpt",
    "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/stable_diffusion_1_5-pruned.safetensors",
]
modelList = [
    "",
    "Animefull-final-pruned",
    "Anything-v3-1",
    "AnyLoRA",
    "AnimePastelDream",    
    "Chillout-mix",
    "OpenJourney-v4",
    "Stable-Diffusion-v1-5",
]
modelName = config.modelName 

# @markdown ### SD2.x model
v2ModelUrl = [
    "",
    "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors",
    "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors",
    "https://huggingface.co/p1atdev/pd-archive/resolve/main/plat-v1-3-1.safetensors",
    "https://huggingface.co/gsdf/Replicant-V1.0/resolve/main/Replicant-V1.0.safetensors",
    "https://huggingface.co/IlluminatiAI/Illuminati_Diffusion_v1.0/resolve/main/illuminati_diffusion_v1.0.safetensors",
    "https://huggingface.co/4eJIoBek/Illuminati-Diffusion-v1-1/resolve/main/illuminatiDiffusionV1_v11.safetensors",
    "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e2.ckpt",
    "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-fp32.safetensors",
    "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-aesthetic-fp32.safetensors",
]
v2ModelList = [
    "",
    "stable-diffusion-2-1-base",
    "stable-diffusion-2-1-768v",
    "plat-diffusion-v1-3-1",
    "replicant-v1",
    "illuminati-diffusion-v1-0",
    "illuminati-diffusion-v1-1",
    "waifu-diffusion-1-4-anime-e2",
    "waifu-diffusion-1-5-e2",
    "waifu-diffusion-1-5-e2-aesthetic",
]
v2ModelName = config.v2ModelName
if modelName:
    installModels.append((modelName, modelUrl[modelList.index(modelName)]))
if v2ModelName:
    installv2Models.append((v2ModelName, v2ModelUrl[v2ModelList.index(v2ModelName)]))


def install(checkpoint_name, url):
    ext = "ckpt" if url.endswith(".ckpt") else "safetensors"

    hf_token = "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"
    user_header = f'"Authorization: Bearer {hf_token}"'
    os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {pretrained_model} -o {checkpoint_name}.{ext} {url}")


def install_checkpoint():
    for model in installModels:
        install(model[0], model[1])
    for v2model in installv2Models:
        install(v2model[0], v2model[1])


