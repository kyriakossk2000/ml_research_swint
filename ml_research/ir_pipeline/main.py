from collections import OrderedDict
from typing import List
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
import pprint as pp
from ranx import Qrels, Run
from ranx import evaluate
from tqdm import tqdm
import operator
from skimage.transform import resize
from skimage.feature import hog
from skimage.io import imread


def load_gallery_images(file_path, shape):
    images = []
    with open(file_path) as f:
        next(f)
        for _, line in enumerate(tqdm(f)):
            id, img, class_id = line.strip().split(",")
            images.append(
                {
                    "id": id,
                    "class_id": class_id,
                    "image": read_and_transform_img_hist(f"{img}", shape),
                }
            )

    return images


def load_query_images(file_path, shape):
    images = OrderedDict()
    with open(file_path) as f:
        next(f)
        for index, line in enumerate(tqdm(f)):
            _, img, _, _, _, _, rel = line.strip().split(",")
            if img not in images:
                images[img] = {
                    "image": read_and_transform_img_hist(f"{img}", shape),
                    "rel": [rel],
                }
            else:
                images[img]["rel"].append(rel)

    return images


def read_and_transform_img(path, shape):
    img = Image.open(path)
    transformed_img = np.asarray(resize_picture(img, shape))
    mean = [0.5371, 0.5066, 0.4895]
    std = [0.3237, 0.3203, 0.3216]
    transformed_img = transform_image(transformed_img, mean, std)
    return transformed_img


def read_and_transform_img_hist(path, shape):
    img = imread(path)
    resized_img = resize(img, shape)
    fd = hog(
        resized_img,
        orientations=30,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        multichannel=True,
    )
    return torch.from_numpy(fd)


def resize_picture(img, shape):
    custom_transform = transforms.Compose([transforms.Resize(shape)])
    return custom_transform(img)


def transform_image(img, mean, std):
    custom_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    return custom_transform(img)


if __name__ == "__main__":
    classes = [
        "Top",
        "Bottom",
        "Acc",
        "Toy",
        "Footwear",
        "Case & Bag",
        "Digital",
        "Food & Drink",
        "Home",
        "PPCPs",
    ]
    gallery_images = load_gallery_images(
        "/Users/alessandrospeggiorin/Desktop/university_of_edinburgh/MLP/3_coursework/cw3/mlpractical/ir_pipeline/gallery.csv",
        (224, 224),
    )
    query_images = load_query_images(
        "/Users/alessandrospeggiorin/Desktop/university_of_edinburgh/MLP/3_coursework/cw3/mlpractical/ir_pipeline/queries.csv",
        (224, 224),
    )

    new_query_images = {}
    for key, value in query_images.items():
        for gallery_image in gallery_images:
            if gallery_image["class_id"] in value["rel"]:
                if key not in new_query_images:
                    new_query_images[key] = {
                        "image": value["image"],
                        "rel": [gallery_image["id"]],
                    }
                else:
                    new_query_images[key]["rel"].append(gallery_image["id"])

    query_images = new_query_images.copy()

    # state = torch.load("vgg_16", map_location=torch.device("cpu"))
    state = torch.load(
        "/Users/alessandrospeggiorin/Desktop/exp/product10k_resnet_50_exp/saved_models/train_model_4",
        map_location=torch.device("cpu"),
    )
    new_params = OrderedDict()
    for key, value in state["network"].items():
        key = key.replace("model.", "")
        new_params[key] = value
    # Load model
    # VGG SPECIFIC
    # model = models.vgg16()
    # model.classifier[6] = torch.nn.Linear(4096, 10)
    # END VGG SPECIFIC
    # RESNET50 speific
    model = models.resnet50()
    feature_number = model.fc.in_features
    model.fc = torch.nn.Linear(feature_number, 10)

    # END RESNET50
    model.load_state_dict(new_params)
    # VGG
    # model.classifier = model.classifier[:-1]
    # RESNET
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    # query = np.reshape(
    #     query_images["queries/spicy-refreshing-bullfinch-of-science.jpeg"]["image"],
    #     newshape=(-1, 3, 224, 224),
    # )
    # query = np.reshape(
    #     read_and_transform_img(
    #         "queries/spicy-refreshing-bullfinch-of-science.jpeg", (224, 224)
    #     ),
    #     newshape=(-1, 3, 224, 224),
    # )
    # doc = np.reshape(
    #     read_and_transform_img(
    #         "gallery/optimal-meteoric-dove-of-radiance.jpg", (224, 224)
    #     ),
    #     newshape=(-1, 3, 224, 224),
    # )

    # query_embedding = model(query).data
    # doc_embedding = model(doc).data
    # # _, predicted = torch.max(out.data, 1)
    # # print(classes[predicted])

    # cos = torch.nn.CosineSimilarity(dim=1)
    # output = cos(query_embedding, doc_embedding)
    # print(output.item())
    #
    # Image embeddings
    for key, value in tqdm(query_images.items()):
        image = value["image"]
        # image = np.reshape(
        #     image,
        #     newshape=(-1, 3, 224, 224),
        # )
        # query_images[key]["e"] = model(image).data
        query_images[key]["e"] = image

    gallery_image_new = []
    for gallery_image in tqdm(gallery_images):
        gallery_image_image = gallery_image["image"]
        # gallery_image_image = np.reshape(
        #     gallery_image_image,
        #     newshape=(-1, 3, 224, 224),
        # )
        # gallery_image["e"] = model(gallery_image_image).data
        gallery_image["e"] = gallery_image_image
        gallery_image_new.append(gallery_image)

    cos = torch.nn.CosineSimilarity(dim=0)
    query_results = {}
    query_counter = 1
    for key, value in tqdm(query_images.items()):
        image = value["image"]

        # Get the query image embedding

        query_embedding = value["e"]
        scored_images = {}
        for gallery_image in tqdm(gallery_image_new):
            gallery_image_id = gallery_image["id"]
            gallery_image_embedding = gallery_image["e"]
            score = cos(query_embedding, gallery_image_embedding)
            scored_images[gallery_image_id] = score.item()

        query_results[f"q_{query_counter}"] = scored_images
        query_counter += 1

    # Generate qrels
    qrels = {}
    query_counter = 1
    key_query_mapping = {}
    for key, value in tqdm(query_images.items()):
        relevante_pics = value["rel"]
        rel_docs = {}
        for pic in relevante_pics:
            rel_docs[pic] = 1

        qrels[f"q_{query_counter}"] = rel_docs
        key_query_mapping[key] = f"q_{query_counter}"

        query_counter += 1

    c = 0
    for key, value in qrels.items():
        rel_dict = query_results[key]
        rel_dict = dict(
            sorted(rel_dict.items(), key=operator.itemgetter(1), reverse=True)
        )
        rel_keys = list(rel_dict.keys())[:100]
        for doc in value:
            if doc in rel_keys:
                c += 1
                break
    print(c / len(qrels))

    qrels = Qrels(qrels)

    run = Run(query_results)
    print(
        evaluate(
            qrels,
            run,
            [
                "map@10",
                "map@20",
                "map@30",
                "map@40",
                "map@50",
                "map@100",
                "map@1000",
                "precision@10",
                "precision@20",
                "precision@30",
                "precision@40",
                "precision@50",
                "precision@100",
                "precision@1000",
                "recall@10",
                "recall@20",
                "recall@30",
                "recall@40",
                "recall@50",
                "recall@100",
                "recall@1000",
                "f1",
                "mrr",
            ],
        )
    )
