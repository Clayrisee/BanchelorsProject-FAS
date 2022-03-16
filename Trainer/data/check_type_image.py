from typing import Tuple, Union
import os
# Reference : https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def img_has_allowed_extension(filename:str, extensions:Union[str, Tuple[str, ...]]) -> bool:
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def is_image_file(filename:str) -> bool:
    return img_has_allowed_extension(filename, IMG_EXTENSIONS)