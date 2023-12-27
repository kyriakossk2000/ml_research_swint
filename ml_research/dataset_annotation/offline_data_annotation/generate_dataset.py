import subprocess


def load_training_data(training_data_path):
    training_data = []
    with open(training_data_path) as f:
        for line in f:
            image_path, class_label, group, macro_group = line.strip().split(",")
            training_data.append(
                {
                    "image_path": image_path,
                    "class_label": class_label,
                    "group": group,
                    "macro_group": macro_group,
                }
            )
    return training_data


def load_test_data(test_data_path):
    test_data = []
    with open(test_data_path) as f:
        for line in f:
            image_path, class_label, visibility, macro_group = line.strip().split(",")
            test_data.append(
                {
                    "image_path": image_path,
                    "class_label": class_label,
                    "visibility": visibility,
                    "macro_group": macro_group,
                }
            )
    return test_data


def filter_data_set(data, labels_to_keep):
    results = []
    for el in data:
        if el["macro_group"] in labels_to_keep:
            results.append(el)
    return results


def generate_training_file(data, file_path):
    with open(file_path, "w") as w:
        w.write("name,class,group,macro_group\n")
        for el in data:
            image_path = el["image_path"]
            class_label = el["class_label"]
            group = el["group"]
            macro_group = el["macro_group"]
            w.write(f"{image_path},{class_label},{group},{macro_group}\n")


def generate_test_file(data, file_path):
    with open(file_path, "w") as w:
        w.write("name,class,usage,macro_group\n")
        for el in data:
            image_path = el["image_path"]
            class_label = el["class_label"]
            visibility = el["visibility"]
            macro_group = el["macro_group"]
            w.write(f"{image_path},{class_label},{visibility},{macro_group}\n")


if __name__ == "__main__":
    CLASSES_TO_KEEP = [
        "top",
        "bottom",
        "acc",
        "toy",
        "footwear",
        "case and bag",
        "digital",
        "food and drink",
        "home",
        "ppcp",
    ]
    SOURCE_FOLDER_TRAIN = "../data/train"
    SOURCE_FOLDER_TEST = "../data/test"

    OUTPUT_DATASET_FOLDER = "dataset"
    OUTPUT_FOLDER_TRAIN = f"{OUTPUT_DATASET_FOLDER}/train"
    OUTPUT_FOLDER_TEST = f"{OUTPUT_DATASET_FOLDER}/test"

    training_data = load_training_data("../data/train_adjusted.csv")
    test_data = load_test_data("../data/test_kaggletest_adjusted.csv")

    training_data = filter_data_set(training_data, CLASSES_TO_KEEP)
    test_data = filter_data_set(test_data, CLASSES_TO_KEEP)

    subprocess.call(f"mkdir {OUTPUT_DATASET_FOLDER}", shell=True)
    subprocess.call(f"mkdir {OUTPUT_FOLDER_TRAIN}", shell=True)
    subprocess.call(f"mkdir {OUTPUT_FOLDER_TEST}", shell=True)

    generate_training_file(training_data, f"{OUTPUT_DATASET_FOLDER}/train.csv")
    generate_test_file(test_data, f"{OUTPUT_DATASET_FOLDER}/test.csv")

    # Copy the images to those folders
    for el in training_data:
        image_path = el["image_path"]
        subprocess.call(
            f"cp {SOURCE_FOLDER_TRAIN}/{image_path} {OUTPUT_FOLDER_TRAIN}/", shell=True
        )

    for el in test_data:
        image_path = el["image_path"]
        subprocess.call(
            f"cp {SOURCE_FOLDER_TEST}/{image_path} {OUTPUT_FOLDER_TEST}/", shell=True
        )
