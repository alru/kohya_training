# @title ## 3.1. Locating Train Data Directory
# @markdown Define the location of your training data. This cell will also create a folder based on your input. Regularization Images is `optional` and can be skipped.
import os
import os
import shutil
from pathlib import Path
import config

root_dir = config.root_dir
training_dir = config.training_dir

train_data_dir = config.train_data_dir
reg_data_dir = config.reg_data_dir

# @markdown Specify this section if your dataset is in a `zip` file and has been uploaded somewhere. This will download your dataset and automatically extract it to the `train_data_dir` if the `unzip_to` is empty.
zipfile_url = config.zipfile_url
zipfile_name = config.zipfile_name
unzip_to = config.unzip_to

hf_token = "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"
user_header = f'"Authorization: Bearer {hf_token}"'

train_req_dir_dict = {"train_data_dir": config.train_data_dir,
                      "reg_data_dir":config.reg_data_dir }

for image_dir in train_req_dir_dict.keys():
    dir_value = train_req_dir_dict[image_dir]
    if dir_value:
        os.makedirs(dir_value, exist_ok=True)
        os.environ[image_dir] = str(dir_value)



print(f"Your train data directory : {train_data_dir}")
if reg_data_dir:
    print(f"Your reg data directory : {reg_data_dir}")




def download_dataset(url):
    if len(zipfile_url) > 1:
        if url.startswith(config.root_dir):
            os.system(f"unzip -j -o {url} -d {train_data_dir}")
        elif url.startswith("https://drive.google.com"):
            os.chdir(root_dir)
            os.system(f"gdown --fuzzy {url}")
        elif url.startswith("https://huggingface.co/"):
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")
            os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {root_dir} -o {zipfile_name} {url}")
        else:
            os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {root_dir} -o {zipfile_name} {url}")


def run(root_dir=root_dir, training_dir=training_dir, train_data_dir=train_data_dir,
        zipfile_url=zipfile_url, zipfile_name=zipfile_name,
        unzip_to=unzip_to, download_dataset=download_dataset):
    
    if len(unzip_to)>1:
        print("triggered unzip")
        os.makedirs(unzip_to, exist_ok=True)
    else:
        unzip_to = train_data_dir
    
    
    download_dataset(zipfile_url)

    os.chdir(root_dir)

    if len(zipfile_url)>1:
        if not zipfile_url.startswith(config.root_dir):
            os.system(f"unzip -j -o {root_dir}/{zipfile_name} -d {unzip_to}")
            os.remove(f"{root_dir}/{zipfile_name}")

    files_to_move = (
    "meta_cap.json",
    "meta_cap_dd.json",
    "meta_lat.json",
    "meta_clean.json",
)

    for filename in os.listdir(train_data_dir):
        file_path = os.path.join(train_data_dir, filename)
        if filename in files_to_move:
            if not os.path.exists(file_path):
                shutil.move(file_path, training_dir)
            else:
                os.remove(file_path)

