# @title ## 1.1. Install Dependencies
# @markdown Clone Kohya Trainer from GitHub and check for updates. Use textbox below if you want to checkout other branch or old commit. Leave it empty to stay the HEAD on main.  This will also install the required libraries.
import os
import zipfile
import shutil
import time
from subprocess import getoutput
# pip install accelerate
import config



# root_dir
root_dir = config.root_dir
deps_dir = config.deps_dir
repo_dir = config.repo_dir
training_dir = config.training_dir
pretrained_model = config.pretrained_model
vae_dir = config.vae_dir
config_dir = config.config_dir

# repo_dir
accelerate_config = config.accelerate_config
tools_dir = config.tools_dir
finetune_dir = config.finetune_dir

store_dict = {
    "root_dir": root_dir,
    "deps_dir": deps_dir,
    "repo_dir": repo_dir,
    "training_dir": training_dir,
    "pretrained_model": pretrained_model,
    "vae_dir": vae_dir,
    "accelerate_config": accelerate_config,
    "tools_dir": tools_dir,
    "finetune_dir": finetune_dir,
    "config_dir": config_dir,
}

for store in store_dict.keys():
    os.environ[store] = str(store_dict[store])

#def check_make_dir(dir_path):
#    if not os.path.exists(dir_path):
#        os.makedirs(dir_path)


repo_url = config.repo_url
bitsandytes_main_py = config.bitsandytes_main_py
branch = config.branch
install_xformers = config.install_xformers
mount_drive = config.mount_drive
verbose = config.verbose

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents


def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


def clone_repo(url):
    if not os.path.exists(repo_dir):
        os.chdir(root_dir)
        os.system(f"git clone {url}")
        print(f"cloned repo to: {os.getcwd()}")
    else:
        os.chdir(repo_dir)
        if branch:
            os.system(f"git pull origin {branch}") 
        else:
            os.system(f"git pull") 


def ubuntu_deps(url, name, dst):
    if not verbose:
        os.system(f"wget -q --show-progress {url}")
    else:
        os.system(f"wget --show-progress {url} -y")
    with zipfile.ZipFile(name, "r") as deps:
        deps.extractall(dst)
    print("will install ubuntu dependencies now")
    os.system(f"sudo dpkg -i {dst}/*")
    os.remove(name)
    shutil.rmtree(dst)


def install_dependencies():
    print("Will install dependencies now")
    s = getoutput('nvidia-smi')

    if 'T4' in s:
        os.system(f"sed -i 's@cpu@cuda@' library/model_util.py")

    print("will install packages now")
    if not verbose:
            os.system(f"pip -q install --upgrade -r requirements.txt")
    else:
        os.system(f"pip install --upgrade -r requirements.txt")

    
    if install_xformers:
        print("Will install xformers")
        if not verbose:
            os.system("pip -q install xformers==0.0.16 triton==2.0.0")
        else:
            os.system("pip install xformers==0.0.16 triton==2.0.0")
        print("Finished installing xformers")
    from accelerate.utils import write_basic_config

    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)


def remove_bitsandbytes_message(filename):
    print("Will remove bitsandbytes message")
    welcome_message = """
def evaluate_cuda_setup():
    print('')
    print('='*35 + 'BUG REPORT' + '='*35)
    print('Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues')
    print('For effortless bug reporting copy-paste your error into this form: https://docs.google.com/forms/d/e/1FAIpQLScPB8emS3Thkp66nvqwmjTEgxp8Y9ufuWTzFyr9kJ5AoI47dQ/viewform?usp=sf_link')
    print('='*80)"""

    new_welcome_message = """
def evaluate_cuda_setup():
    import os
    if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
        print('')
        print('=' * 35 + 'BUG REPORT' + '=' * 35)
        print('Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues')
        print('For effortless bug reporting copy-paste your error into this form: https://docs.google.com/forms/d/e/1FAIpQLScPB8emS3Thkp66nvqwmjTEgxp8Y9ufuWTzFyr9kJ5AoI47dQ/viewform?usp=sf_link')
        print('To hide this message, set the BITSANDBYTES_NOWELCOME variable like so: export BITSANDBYTES_NOWELCOME=1')
        print('=' * 80)"""
    print(f"Will read file now at: {filename}")
    contents = read_file(filename)
    print("Will replace content now")
    new_contents = contents.replace(welcome_message, new_welcome_message)
    write_file(filename, new_contents)
    print("Replaced file")


def run():
    os.chdir(root_dir)


    for dir in [
        deps_dir, 
        training_dir, 
        config_dir, 
        pretrained_model, 
        vae_dir
    ]:
        os.makedirs(dir, exist_ok=True)

    clone_repo(repo_url)

    if branch:
        os.chdir(repo_dir)
        status = os.system(f"git checkout {branch}")
        if status != 0:
            raise Exception("Failed to checkout branch or commit")

    os.chdir(repo_dir)
    
    os.system(f"sudo apt -y update {'-qq' if not verbose else ''}")
    os.system("sudo apt --fix-broken install -y")
    os.system(f"sudo apt install libunwind8-dev {'-qq' if not verbose else ''} -y")
    os.system("sudo apt --fix-broken install")
    
    ubuntu_deps(
        "https://huggingface.co/Linaqruf/fast-repo/resolve/main/deb-libs.zip",
        "deb-libs.zip",
        deps_dir,
    )
    os.system("sudo apt --fix-broken install -y")
    
    install_dependencies()
    print("Will sleep for 3 seconds now")
    time.sleep(3)
    #os.system("sudo apt --fix-broken install -y")

    #remove_bitsandbytes_message(bitsandytes_main_py)

    os.environ["LD_PRELOAD"] = "libtcmalloc.so"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"  
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

    print("Finished installing packages")