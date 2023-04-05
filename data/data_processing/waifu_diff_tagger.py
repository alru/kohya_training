#@title ### 4.2.2. Waifu Diffusion 1.4 Tagger V2
import os
import config

root_dir = os.environ["root_dir"]
finetune_dir = os.environ["finetune_dir"]
train_data_dir = os.environ["train_data_dir"]
os.chdir(finetune_dir)


batch_size = config.batch_size_waifu
max_data_loader_n_workers = config.max_data_loader_n_workers_waifu 
model = config.model
remove_underscores = config.remove_underscores
threshold = config.threshold

def run():
    os.system(f"python tag_images_by_wd14_tagger.py {train_data_dir} --batch_size {batch_size} --repo_id {model} --thresh {threshold} --caption_extension .txt --max_data_loader_n_workers {max_data_loader_n_workers}0")

    if remove_underscores:
      for filename in os.listdir(train_data_dir):
          if filename.endswith('.txt'):
              filepath = os.path.join(train_data_dir, filename)
              with open(filepath, 'r') as f:
                  contents = f.read()
              contents = contents.replace('_', ' ')
              with open(filepath, 'w') as f:
                  f.write(contents)
