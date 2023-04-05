# @title ## 4.1. Data Cleaning
# @markdown ### Delete Unnecessary Files
import os
import random
import concurrent.futures
from tqdm import tqdm
from PIL import Image

root_dir = os.environ["root_dir"]
train_data_dir = os.environ["train_data_dir"]


test = os.listdir(train_data_dir)
# @markdown This section will delete unnecessary files and unsupported media such as `.mp4`, `.webm`, and `.gif`.

def run(train_data_dir=train_data_dir, test=test):
    os.chdir(root_dir)
    supported_types = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".caption",
    ".npz",
    ".txt",
    ".json",
]

    for item in test:
        file_ext = os.path.splitext(item)[1]
        if file_ext not in supported_types:
            print(f"Deleting file {item} from {train_data_dir}")
            os.remove(os.path.join(train_data_dir, item))

# @markdown ### <br> Convert Transparent Images
# @markdown This code will convert your transparent dataset with alpha channel (RGBA) to RGB and give it a white background.

    convert = False  # @param {type:"boolean"}
    random_color = False  # @param {type:"boolean"}

    batch_size = 32

    images = [
    image
    for image in os.listdir(train_data_dir)
    if image.endswith(".png") or image.endswith(".webp")
]
    background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


    def process_image(image_name):
        img = Image.open(f"{train_data_dir}/{image_name}")

        if img.mode in ("RGBA", "LA"):
            if random_color:
                background_color = random.choice(background_colors)
            else:
                background_color = (255, 255, 255)
            bg = Image.new("RGB", img.size, background_color)
            bg.paste(img, mask=img.split()[-1])

            if image_name.endswith(".webp"):
                bg = bg.convert("RGB")
                bg.save(f'{train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
                os.remove(f"{train_data_dir}/{image_name}")
                print(
                f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
            )
            else:
                bg.save(f"{train_data_dir}/{image_name}", "PNG")
                print(f" Converted image: {image_name}")
        else:
            if image_name.endswith(".webp"):
                img.save(f'{train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
                os.remove(f"{train_data_dir}/{image_name}")
                print(
                f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
            )
            else:
                img.save(f"{train_data_dir}/{image_name}", "PNG")


    num_batches = len(images) // batch_size + 1

    if convert:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in tqdm(range(num_batches)):
                start = i * batch_size
                end = start + batch_size
                batch = images[start:end]
                executor.map(process_image, batch)

        print("All images have been converted")

run(train_data_dir, test)