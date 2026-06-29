import os.path as osp
import sys
import base64
import io
import json
from PIL import Image
from openai import OpenAI

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.config import MLLM_CFG
from hms.common import may_create_folder, parent_folder, read_json, write_json, valid_str, hash_str

def resize_image(image, max_dimension):
    width, height = image.size

    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def encode_image(image):
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image = output.getvalue()
    return base64.b64encode(image).decode("utf-8")

class Mllm(object):
    def __init__(self, cache_dir):
        super().__init__()

        self.cache_dir = cache_dir
        may_create_folder(cache_dir)

        self.clients = {k: OpenAI(**v) for k, v in MLLM_CFG.items()}

    def chat_text(self, provider, model_name, system_prompt, user_prompt, task_name, use_cache=True, **kwargs):
        messages = list()
        if valid_str(system_prompt):
            messages.append({"role": "system", "content": system_prompt})
        if valid_str(user_prompt):
            messages.append({"role": "user", "content": user_prompt})

        cache_id = json.dumps(messages, sort_keys=True)
        cache_id = hash_str(cache_id)
        cache_path = osp.join(self.cache_dir, task_name, f"{cache_id}.json")
        if use_cache and osp.exists(cache_path):
            res = read_json(cache_path)
            return res

        responses = self.clients[provider].chat.completions.create(model=model_name, messages=messages, **kwargs)

        if hasattr(responses, "choices") and responses.choices is not None:
            response = responses.choices[0].message.content
        else:
            response = ""

        res = {
            "response": response,
            "provider": provider,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "task_name": task_name,
            "cache_path": cache_path,
        }
        res = {**res, **kwargs}

        if valid_str(response):
            may_create_folder(parent_folder(cache_path))
            write_json(cache_path, res)

        return res

    def chat_images(self, provider, model_name, system_prompt, user_prompt, images, task_name, image_max_dim=512, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if isinstance(images[0], str):
            images = [Image.open(ip) for ip in images]

        original_width, original_height = images[0].width, images[0].height

        images_base64 = list()
        for image in images:
            assert image.width == original_width and image.height == original_height
            if image.mode == "RGBA":
                bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(bg, image)
                image = image.convert("RGB")
            if image_max_dim is not None:
                image = resize_image(image, max_dimension=image_max_dim)
            image_base64 = encode_image(image)
            images_base64.append(image_base64)
        new_width, new_height = image.width, image.height

        messages = list()
        if valid_str(system_prompt):
            messages.append({"role": "system", "content": system_prompt})
        if valid_str(user_prompt) or len(images) > 0:
            ucontent = list()
            if valid_str(user_prompt):
                ucontent.append({"type": "text", "text": user_prompt})
            for image_base64 in images_base64:
                ucontent.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            messages.append({"role": "user", "content": ucontent})

        responses = self.clients[provider].chat.completions.create(model=model_name, messages=messages, **kwargs)

        if hasattr(responses, "choices") and responses.choices is not None:
            response = responses.choices[0].message.content
        else:
            response = ""

        res = {
            "response": response,
            "original_width": original_width,
            "original_height": original_height,
            "new_width": new_width,
            "new_height": new_height,
            "provider": provider,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "task_name": task_name,
        }
        res = {**res, **kwargs}

        return res
