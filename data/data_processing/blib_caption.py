#@title ### 4.2.1. BLIP Captioning
#@markdown [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) is a pre-training framework for unified vision-language understanding and generation, which achieves state-of-the-art results on a wide range of vision-language tasks.
#@markdown In short, it can be used as a tool for image captioning. Example: `astronaut riding a horse in space`. 
import os
import config 

finetune_dir = config.finetune_dir
train_data_dir = config.train_data_dir

batch_size = config.batch_size_bilb
max_data_loader_n_workers = config.max_data_loader_n_workers_bilb 
beam_search = config.beam_search
min_length = config.min_length
max_length = config.max_length

def run():
    os.chdir(finetune_dir) 
    print("will execute blib caption")
    os.system(f"""python make_captions.py \
    "{train_data_dir}" \
    --batch_size {batch_size} \
    {"--beam_search" if beam_search else ""} \
    --min_length {min_length} \
    --max_length {max_length} \
    --caption_extension .caption \
    --max_data_loader_n_workers {max_data_loader_n_workers}
        """)
    print("executed blib caption")

