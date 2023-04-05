# @title ## 2.2. Download Custom Model
import os
import config
root_dir = config.root_dir
pretrained_model = config.pretrained_model
hf_token = config.hf_token
os.chdir(root_dir)

# @markdown ### Custom model
modelUrl = config.modelUrl


def install(url=modelUrl):
    if url:
        base_name = os.path.basename(url)

        if url.startswith("https://drive.google.com"):
            os.chdir(pretrained_model)
            os.system(f"gdown --fuzzy {url}")
        elif url.startswith("https://huggingface.co/"):
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")
            # @markdown Change this part with your own huggingface token if you need to download your private model
            user_header = f'"Authorization: Bearer {hf_token}"'
            os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {pretrained_model} -o {base_name} {url}")
        else:
            os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {pretrained_model} {url}")


install(url=modelUrl)