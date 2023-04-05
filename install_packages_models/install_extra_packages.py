import config
import os

def run():
    extra_packages = config.extra_packages
    os.system(f"pip install {extra_packages}")

