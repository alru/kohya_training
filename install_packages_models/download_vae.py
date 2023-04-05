# @title ## 2.3. Download Available VAE (Optional)
import os
import config

root_dir = os.environ["root_dir"]
vae_dir = os.environ["vae_dir"]
os.chdir(root_dir)

installVae = []
# @markdown Select one of the VAEs to download, select `none` for not download VAE:
vaeUrl = [
    "",
    "https://huggingface.co/Linaqruf/personal-backup/resolve/main/vae/animevae.pt",
    "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime.ckpt",
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt",
]
vaeList = ["none", "anime.vae.pt", "waifudiffusion.vae.pt", "stablediffusion.vae.pt"]
vaeName =  config.vaeName

installVae.append((vaeName, vaeUrl[vaeList.index(vaeName)]))


def install(vae_name, url):
    hf_token = "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"
    user_header = f'"Authorization: Bearer {hf_token}"'
    os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {vae_dir} -o {vae_name} {url}")


def install_vae():
    if vaeName != "none":
        for vae in installVae:
            install(vae[0], vae[1])
    else:
        pass


install_vae()