from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import logging
from typing import Any, Mapping
import json

# def load_image(image, image_size, device):
#     raw_image = Image.open(str(image)).convert('RGB')
#     w, h = raw_image.size

#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     ])
#     image = transform(raw_image).unsqueeze(0)
#     # image = transform(raw_image).unsqueeze(0).to(device)
#     return image


class Logger:
    def __init__(self, filename):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, text):
        print(text)
        self.logger.info(text)


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def dummy(images, **kwargs):
    return images, False


def horz_stack(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    total_height = max(heights)

    new_im = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def vert_stack(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (total_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im