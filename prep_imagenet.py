### prep_imagenet.py
"""
This is a script to prepare the ImageNet dataset (http://image-net.org/download-images). 
We downloaded the validation sample from 2012 locally containing 50k images (7GB). 
To reduce the folder size, we resize the images to thumbnails with ~256x256 and convert them to JPG.
"""
from PIL import Image
from pathlib import Path

def resize_image(image_path: str, output_path: str, size: int):
    """ Resize an image and save it to the output path """
    im = Image.open(image_path)
    im.thumbnail((size, size), Image.ANTIALIAS)
    im.save(output_path, "jpeg", quality=90)


if __name__ == "__main__":
    input_path = Path("C:/Users/Admin/Downloads/ILSVRC2012_img_val/")
    output_path = Path("./data/imagenet")
    size = 256

    for file in input_path.iterdir():
        print(f"Converting {file.name}")
        resize_image(file, output_path / file.name, size)