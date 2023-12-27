import subprocess
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
import torch
import random
from sklearn.utils import shuffle
from os import listdir
from os.path import isfile, join
import pprint as pp


def load_labels(data_path, labels=None):
    if labels is None:
        labels = {
            "top": [],
            "bottom": [],
            "acc": [],
            "toy": [],
            "footwear": [],
            "case and bag": [],
            "digital": [],
            "food and drink": [],
            "home": [],
            "ppcp": [],
        }

    with open(data_path) as f:
        next(f)
        for _, line in enumerate(f):
            img_path, _, _, macro_group = line.strip().split(",")
            labels[macro_group].append(img_path)
    return labels


def rebalance_and_resize(d, elements_to_keep):
    for key, value in d.items():
        d[key] = shuffle(value, random_state=42)[:elements_to_keep]
    return d


def str_label_to_int(s):
    labels_dict = {
        "top": 0,
        "bottom": 1,
        "acc": 2,
        "toy": 3,
        "footwear": 4,
        "case and bag": 5,
        "digital": 6,
        "food and drink": 7,
        "home": 8,
        "ppcp": 9,
    }
    return labels_dict[s]


def dict_to_file(d, path):
    with open(path, "w") as w:
        w.write("img_path,class\n")
        for key, value in d.items():
            for el in value:
                w.write(f"{el},{key}\n")


def move_source_images_to_unified_folder(d, source_train, source_test, target_folder):
    # Get all the images we need to move
    all_pictures = set()
    for value in d.values():
        all_pictures = all_pictures.union(set(value))

    # Get all pictures in train folder
    train_images = [f for f in listdir(source_train) if isfile(join(source_train, f))]
    # Get all pictures in test folder
    test_images = [f for f in listdir(source_test) if isfile(join(source_test, f))]

    # Create dataset folder
    subprocess.call(f"mkdir {target_folder}", shell=True)

    # Now move only the pictures that we actually need to keep
    for image_path in train_images:
        if image_path in all_pictures:
            subprocess.call(
                f"cp {source_train}/{image_path} {target_folder}/", shell=True
            )

    for image_path in test_images:
        if image_path in all_pictures:
            subprocess.call(
                f"cp {source_test}/{image_path} {target_folder}/", shell=True
            )


if __name__ == "__main__":

    train_labels = load_labels("dataset/train.csv")
    all_labels = load_labels("dataset/test.csv", train_labels)
    # Old value 5750
    all_labels = rebalance_and_resize(all_labels, 2875)

    for key, value in all_labels.items():
        print(f"{key}-{len(value)}")

    dict_to_file(all_labels, "dataset_balanced/train_test_unified.csv")
    move_source_images_to_unified_folder(
        all_labels, "dataset/train", "dataset/test", "dataset_balanced/dataset"
    )

    print(
        len(
            [
                f
                for f in listdir("dataset_balanced/dataset")
                if isfile(join("dataset_balanced/dataset", f))
            ]
        )
    )
