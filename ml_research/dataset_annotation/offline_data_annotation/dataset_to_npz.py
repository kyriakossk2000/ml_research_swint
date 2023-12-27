import subprocess
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
import torch
import gc


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


def load_labels(data_path):
    labels = []
    with open(data_path) as f:
        next(f)
        for index, line in enumerate(f):
            _, macro_group = line.strip().split(",")
            labels.append(str_label_to_int(macro_group))
    return np.asarray(labels)


def load_images_from_folder(file_path, image_path, shape):
    images = []
    with open(file_path) as f:
        next(f)
        for index, line in enumerate(tqdm(f)):
            img, _ = line.strip().split(",")
            images.append(read_img(f"{image_path}/{img}", shape))

    return np.asarray(images)


def read_img(path, shape):
    img = Image.open(path)
    transformed_img = np.asarray(resize_picture(img, shape))
    img.close()
    return transformed_img


def resize_picture(img, shape):
    custom_transform = transforms.Compose([transforms.Resize(shape)])
    return custom_transform(img)


def transform_image(img, mean, std, data_augmentation=False):
    if not data_augmentation:
        custom_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ]
        )
    else:
        custom_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.3, 0.3),
                    shear=(-0.3, 0.3, -0.3, 0.3),
                ),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ]
        )

    return custom_transform(img).numpy()


def compute_mean_std(dataset):
    custom_transform = transforms.Compose([transforms.ToTensor()])
    transformed = torch.stack([custom_transform(img) for img in dataset], dim=3)

    mean = transformed.view(3, -1).mean(dim=1)
    std = transformed.view(3, -1).std(dim=1)
    return mean, std


def save_data_to_npz(data, labels, output_file):
    np.savez_compressed(output_file, inputs=data, targets=labels)


def save_data_to_npy(data, output_file):
    np.save(output_file, data)


def transform_dataset(data, mean, std, data_augmentation=False):
    results = []
    for i in range(data.shape[0]):
        results.append(
            transform_image(data[i, :, :, :], mean, std, data_augmentation=False)
        )
    if data_augmentation:
        for i in range(data.shape[0]):
            results.append(
                transform_image(data[i, :, :, :], mean, std, data_augmentation=True)
            )
    return np.asarray(results)


def randomize(data, labels):
    # Generate the permutation index array.
    permutation = np.random.RandomState(seed=42).permutation(data.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels


if __name__ == "__main__":

    print("Loading img train")
    data = load_images_from_folder(
        "dataset_balanced/train_test_unified.csv",
        "dataset_balanced/dataset",
        (224, 224),
    )
    print("Loading labels")
    labels = load_labels("dataset_balanced/train_test_unified.csv")

    print("Splitting")
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=1 - train_ratio, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        stratify=y_test,
        random_state=42,
    )
    print("Train labels distribution")
    print(np.unique(y_train, return_counts=True))
    print("Val labels distribution")
    print(np.unique(y_val, return_counts=True))
    print("Test labels distribution")
    print(np.unique(y_test, return_counts=True))
    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)
    print("Computing mean and std for dataset")
    mean, std = compute_mean_std(X_train)
    print("Mean:")
    print(mean)
    print("Std:")
    print(std)

    print("Transforming individual pictures")

    X_train = transform_dataset(X_train, mean, std, data_augmentation=True)
    y_train = np.concatenate((y_train, y_train), axis=None)
    X_train, y_train = randomize(X_train, y_train)

    print("Saving to file..")
    save_data_to_npy(X_train, f"product-10k-train_data.npy")
    save_data_to_npy(y_train, f"product-10k-train_labels.npy")
    print(X_train.shape)
    print(y_train.shape)

    del X_train
    del y_train
    gc.collect()

    X_val = transform_dataset(X_val, mean, std)
    save_data_to_npy(X_val, f"product-10k-valid_data.npy")
    save_data_to_npy(y_val, f"product-10k-valid_labels.npy")

    del X_val
    del y_val
    gc.collect()

    X_test = transform_dataset(X_test, mean, std)
    save_data_to_npy(X_test, f"product-10k-test_data.npy")
    save_data_to_npy(y_test, f"product-10k-test_labels.npy")
    print("Done")
