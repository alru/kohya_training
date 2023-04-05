# @title ## 6.2. Interrogating LoRA Weights
# @markdown Now you can check if your LoRA trained properly.
# @markdown  If you used `clip_skip = 2` during training, the values of `lora_te_text_model_encoder_layers_11_*` will be `0.0`, this is normal. These layers are not trained at this value of `Clip Skip`.
import os
import torch
import json
from safetensors.torch import load_file
from safetensors.torch import safe_open
import library.model_util as model_util
import config

network_weight = config.network_weight 
no_verbose = config.no_verbose

print("Loading LoRA weight:", network_weight)


def main(file=network_weight, verbose: bool = no_verbose):
    print("Will start testing now")
    if not verbose:
        sd = (
            load_file(file)
            if os.path.splitext(file)[1] == ".safetensors"
            else torch.load(file, map_location="cuda")
        )
        values = []

        keys = list(sd.keys())
        for key in keys:
            if "lora_up" in key or "lora_down" in key:
                values.append((key, sd[key]))
        print(f"number of LoRA modules: {len(values)}")

        for key, value in values:
            value = value.to(torch.float32)
            print(f"{key},{torch.mean(torch.abs(value))},{torch.min(torch.abs(value))}")

    if model_util.is_safetensors(file):
        with safe_open(file, framework="pt") as f:
            metadata = f.metadata()
        if metadata is not None:
            print(f"\nLoad metadata for: {file}")
            print(json.dumps(metadata, indent=4))
    else:
        print("No metadata saved, your model is not in safetensors format")

