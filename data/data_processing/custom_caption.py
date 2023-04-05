# @title ### 4.2.3. Custom Caption/Tag (Optional)
import os
import config

root_dir = os.environ["root_dir"]
train_data_dir = os.environ["train_data_dir"]

os.chdir(root_dir)

extension = config.extension
custom_tag = config.custom_tag
append = config.append
remove_tag = config.remove_tag

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def add_tag(filename, tag, append):
    contents = read_file(filename)
    tag = ", ".join(tag.split())
    tag = tag.replace("_", " ")
    if tag in contents:
        return
    contents = contents.rstrip() + ", " + tag if append else tag + ", " + contents
    write_file(filename, contents)

def delete_tag(filename, tag):
    contents = read_file(filename)
    tag = ", ".join(tag.split())
    tag = tag.replace("_", " ")
    if tag not in contents:
        return
    contents = "".join([s.strip(", ") for s in contents.split(tag)])
    write_file(filename, contents)

tag = custom_tag

def run():
    os.chdir(root_dir)

    if not any(
    [filename.endswith(extension) for filename in os.listdir(train_data_dir)]
):
        for filename in os.listdir(train_data_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                open(
                os.path.join(train_data_dir, filename.split(".")[0] + extension),
                "w",
            ).close()

    if custom_tag:
        for filename in os.listdir(train_data_dir):
            if filename.endswith(extension):
                file_path = os.path.join(train_data_dir, filename)
                if custom_tag and not remove_tag:
                    add_tag(file_path, tag, append)
                if custom_tag and remove_tag:
                    delete_tag(file_path, tag)





