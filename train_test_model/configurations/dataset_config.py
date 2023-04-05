# @title ## 5.2. Dataset Config
import toml
import config
import os


root_dir = config.root_dir
train_data_dir = config.train_data_dir
reg_data_dir = config.reg_data_dir
config_dir = config.config_dir



train_repeats = config.train_repeats
reg_repeats = config.reg_repeats

instance_token = config.instance_token
class_token = config.class_token

add_token_to_caption = config.add_token_to_caption

resolution = config.resolution
flip_aug = config.flip_aug
caption_extension = config.caption_extension
caption_dropout_rate = config.caption_dropout_rate
caption_dropout_every_n_epochs = config.caption_dropout_every_n_epochs
keep_tokens = config.keep_tokens



if add_token_to_caption and keep_tokens < 2:
    keep_tokens = 1

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def add_tag(filename, tag):
    contents = read_file(filename)
    tag = ", ".join(tag.split())
    tag = tag.replace("_", " ")
    if tag in contents:
        return
    contents = tag + ", " + contents
    write_file(filename, contents)

def delete_tag(filename, tag):
    contents = read_file(filename)
    tag = ", ".join(tag.split())
    tag = tag.replace("_", " ")
    if tag not in contents:
        return
    contents = "".join([s.strip(", ") for s in contents.split(tag)])
    write_file(filename, contents)

def run():
    if caption_extension != "none":
        tag = f"{instance_token}_{class_token}" if 'class_token' in globals() else instance_token
        for filename in os.listdir(train_data_dir):
            if filename.endswith(caption_extension):
                file_path = os.path.join(train_data_dir, filename)
                if add_token_to_caption:
                    add_tag(file_path, tag)
                else:
                    delete_tag(file_path, tag)

    config = {
    "general": {
        "enable_bucket": True,
        "caption_extension": caption_extension,
        "shuffle_caption": True,
        "keep_tokens": keep_tokens,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
    },
    "datasets": [
        {
            "resolution": resolution,
            "min_bucket_reso": 320 if resolution > 640 else 256,
            "max_bucket_reso": 1280 if resolution > 640 else 1024,
            "caption_dropout_rate": caption_dropout_rate if caption_extension == ".caption" else 0,
            "caption_tag_dropout_rate": caption_dropout_rate if caption_extension == ".txt" else 0,
            "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
            "flip_aug": flip_aug,
            "color_aug": False,
            "face_crop_aug_range": None,
            "subsets": [
                {
                    "image_dir": train_data_dir,
                    "class_tokens": f"{instance_token} {class_token}" if 'class_token' in globals() else instance_token,
                    "num_repeats": train_repeats,
                },
                {
                    "is_reg": True,
                    "image_dir": reg_data_dir,
                    "class_tokens": class_token if 'class_token' in globals() else None,
                    "num_repeats": reg_repeats,
                },
            ],
        }
    ],
}

    config_str = toml.dumps(config)

    dataset_config = os.path.join(config_dir, "dataset_config.toml")

    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    config_str = toml.dumps(config)

    with open(dataset_config, "w") as f:
        f.write(config_str)

    print(config_str)