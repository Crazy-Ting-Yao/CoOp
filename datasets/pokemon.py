import os
import os.path as osp
import sys
import json
import subprocess
from io import BytesIO

from PIL import Image
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing


def _load_hf_dataset(dataset_name, split="train"):
    """Load a HuggingFace dataset in a subprocess to avoid import conflicts."""
    hf_token = os.environ.get("HF_TOKEN", "")
    token_arg = f'token="{hf_token}"' if hf_token else ""

    script = f"""
import sys, os, json, base64
from io import BytesIO
os.environ['HF_TOKEN'] = '{hf_token}'
from datasets import load_dataset
ds = load_dataset("{dataset_name}", split="{split}", {token_arg})
data = []
for item in ds:
    row = {{}}
    for k, v in item.items():
        if k == 'image':
            buf = BytesIO()
            v.save(buf, format='PNG')
            row[k] = base64.b64encode(buf.getvalue()).decode()
        else:
            row[k] = v
    data.append(row)
print(json.dumps(data))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd="/",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to load HF dataset '{dataset_name}': {result.stderr}"
        )

    import base64
    data = json.loads(result.stdout)
    for item in data:
        if "image" in item:
            img_bytes = base64.b64decode(item["image"])
            item["image"] = Image.open(BytesIO(img_bytes))
    return data


@DATASET_REGISTRY.register()
class Pokemon(DatasetBase):
    """Pokemon type classification dataset from HuggingFace.

    Train: andyqmongo/IVL_CLS_pokemon_1_shot
    Test:  andyqmongo/pokemon_eval_standard
    """

    dataset_dir = "pokemon"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        mkdir_if_missing(self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots <= 1:
            train_name = "andyqmongo/IVL_CLS_pokemon_1_shot"
        else:
            train_name = "andyqmongo/IVL_CLS_pokemon_8_shot"

        print(f"Loading Pokemon from HuggingFace (train={train_name})...")
        train_raw = _load_hf_dataset(train_name, "train")
        test_raw = _load_hf_dataset("andyqmongo/pokemon_eval_standard", "test")

        classname_to_label = {}
        train_data = self._build_split(train_raw, classname_to_label, "train")
        test_data = self._build_split(test_raw, classname_to_label, "test")

        val_size = min(len(test_data) // 5, 50)
        val_data = test_data[:val_size]

        print(
            f"Pokemon: {len(classname_to_label)} classes, "
            f"{len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
        )

        super().__init__(train_x=train_data, val=val_data, test=test_data)

    def _build_split(self, raw_data, classname_to_label, prefix):
        items = []
        for idx, item in enumerate(raw_data):
            classname = self._get_classname(item)
            if classname not in classname_to_label:
                classname_to_label[classname] = len(classname_to_label)
            label = classname_to_label[classname]
            img_path = self._save_image(item, f"{prefix}_{idx}", classname)
            items.append(Datum(impath=img_path, label=label, classname=classname))
        return items

    @staticmethod
    def _get_classname(item):
        if "primary_type" in item and item["primary_type"]:
            return str(item["primary_type"]).lower().replace("_", " ")
        if "label" in item:
            return str(item["label"]).lower().replace("_", " ")
        return "unknown"

    def _save_image(self, item, prefix, classname):
        class_dir = osp.join(self.image_dir, classname.replace(" ", "_"))
        mkdir_if_missing(class_dir)
        img_path = osp.join(class_dir, f"{prefix}.jpg")
        if osp.exists(img_path):
            return img_path
        if "image" in item and isinstance(item["image"], Image.Image):
            item["image"].convert("RGB").save(img_path)
        return img_path
