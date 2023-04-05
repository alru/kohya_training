# @title ## 5.1. Model Config
import os
import config

v2 = config .v2 
v_parameterization = config.v_parameterization
project_name = config.project_name

pretrained_model_name_or_path = config.pretrained_model_name_or_path
vae = config.vae
output_dir = config.output_dir

# @markdown This will ignore `output_dir` defined above, and changed to `/content/drive/MyDrive/LoRA/output` by default
output_to_drive = config.output_to_drive  # @param {'type':'boolean'}



sample_dir = os.path.join(output_dir, "sample")
def run():
    for dir in [output_dir, sample_dir]:
        os.makedirs(dir, exist_ok=True)

    print("Project Name: ", project_name)
    print("Model Version: Stable Diffusion V1.x") if not v2 else ""
    print("Model Version: Stable Diffusion V2.x") if v2 and not v_parameterization else ""
    print("Model Version: Stable Diffusion V2.x 768v") if v2 and v_parameterization else ""
    print(
    "Pretrained Model Path: ", pretrained_model_name_or_path
) if pretrained_model_name_or_path else print("No Pretrained Model path specified.")
    print("VAE Path: ", vae) if vae else print("No VAE path specified.")
    print("Output Path: ", output_dir)

