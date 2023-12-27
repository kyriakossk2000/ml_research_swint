import numpy as np
import matplotlib.pyplot as plt
import splitfolders

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.metrics import f1_score

from tqdm import tqdm

if __name__ == '__main__':

    # Maybe use different method? depends on how we have the data stored
    # Split train data into train/valid
    splitfolders.ratio("marvel/train", output="marvel/split",
                       seed=1337, ratio=(0.8, 0.2), group_prefix=None, move=False)

    # Data augmentation transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load training and test sets
    train_ds = datasets.ImageFolder("data/split/train", transform=train_transform)
    valid_ds = datasets.ImageFolder("data/split/val", transform=valid_transform)
    test_ds = datasets.ImageFolder("data/test", transform=test_transform)

    print(f"Train data: Found {len(train_ds)} files belonging "
          f"to {len(train_ds.classes)} classes.")
    print(f"Validation data: Found {len(valid_ds)} files belonging "
          f"to {len(valid_ds.classes)} classes.")
    print(f"Test data: Found {len(test_ds)} files belonging "
          f"to {len(test_ds.classes)} classes.")

    # Enable GPU support if available and load data into iterable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        **kwargs
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=valid_ds.__len__(),
        shuffle=False,
        num_workers=8,
        **kwargs
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_ds.__len__(),
        shuffle=False,
        num_workers=8,
        **kwargs
    )
    # CLASSES TO BE CHANGED
    def view_image_batch():
        classes = {
            0: "Black Widow",
            1: "Captain America",
            2: "Doctor Strange",
            3: "Hulk",
            4: "Ironman",
            5: "Loki",
            6: "Spiderman",
            7: "Thanos"
        }

        rand_int = np.random.randint(2063)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        label = train_ds[rand_int][1]
        image_class = classes[label]

        image = train_ds[rand_int][0].permute(1, 2, 0).numpy()
        image = image * std + mean

        plt.axis("off")
        plt.title(f"{image_class}")
        plt.imshow(image)
        plt.show()


    view_image_batch()

    # Unnecessary
    def view_random_images():
        classes = {
            0: "Black Widow",
            1: "Captain America",
            2: "Doctor Strange",
            3: "Hulk",
            4: "Ironman",
            5: "Loki",
            6: "Spiderman",
            7: "Thanos"
        }

        figure = plt.figure(figsize=(8, 8))

        for i in range(1, 10):
            rand_int = np.random.randint(2063)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            label = train_ds[rand_int][1]
            image_class = classes[label]

            image = train_ds[rand_int][0].permute(1, 2, 0).numpy()
            image = image * std + mean

            figure.add_subplot(3, 3, i)
            plt.axis("off")
            plt.title(f"{image_class}")
            plt.imshow(image)
        plt.show()


    view_random_images()

    # Define model parameters
    EPOCHS = 24
    NUM_CLASSES = 8
    learn_rate = 0.0005

    # Load ResNet-18 with pre-trained weights
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze all but fully connected layer
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    # Adjust output to 8 classes
    feature_number = model.fc.in_features
    model.fc = nn.Linear(feature_number, NUM_CLASSES)

    model = model.to(device)

    # Use cross-entropy loss function and
    # stochastic gradient decent
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(
        lambda param: param.requires_grad,
        model.parameters()),
        learn_rate,
        momentum=0.9
    )

    loss_values_train = []
    loss_values_valid = []
    best_f1 = 0

    for epoch in range(EPOCHS):
        """ Train """
        model.train()

        train_loss = 0.0
        train_corrects = 0
        f1_scores = []

        # Transfer data to GPU if available
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            # Find loss
            loss = criterion(outputs, labels)
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()
            # Calculate loss
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            f1_scores.append(f1_score(labels.data, preds, average="weighted"))

        training_loss = train_loss / len(train_ds)
        loss_values_train.append(train_loss)
        accuracy = train_corrects / len(train_ds) * 100
        # Average of f1 score over batches
        f1 = np.mean(f1_scores)

        print(
            f"Epoch {epoch + 1} \n"
            f"-------- \n"
            f"Training Loss: {training_loss:.4f} \n"
            f"Accuracy: {accuracy:.4f}% \n"
            f"Average F1 Score: {f1} \n"
            f"---------------"
        )

        """ Validate """
        model.eval()

        with torch.no_grad():
            valid_loss = 0
            valid_corrects = 0

            # Transfer data to GPU if available
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                # Get predictions
                _, preds = torch.max(outputs, 1)
                # Find loss
                loss = criterion(outputs, labels)
                # Calculate loss
                valid_loss += loss.item() * inputs.size(0)
                valid_corrects += torch.sum(preds == labels.data)

        validation_loss = valid_loss / len(valid_ds)
        loss_values_valid.append(valid_loss)
        f1_val = f1_score(labels.data, preds, average="weighted")
        accuracy = valid_corrects / len(valid_ds) * 100

        if f1_val > best_f1:
            best_f1 = f1_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, "best-model-weights")

        print(
            f"Validation Loss: {validation_loss:.4f} \n"
            f"Accuracy: {accuracy:.4f}% \n"
            f"F1 Score: {f1_val}"
        )

    plt.plot(loss_values_train)
    plt.show()

    plt.plot(loss_values_valid)
    plt.show()

    # Test Model
    # With help from https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

    # Load saved weights and params:
    best_params = torch.load("best-model-weights")

    # Load Resnet with best weights
    model = models.resnet18()

    # Freeze all but fully connected layer
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    # Adjust output to 8 classes
    feature_number = model.fc.in_features
    model.fc = nn.Linear(feature_number, NUM_CLASSES)

    model.load_state_dict(best_params["model_state_dict"])
    model = model.to(device)

    """ Test """
    model.eval()

    with torch.no_grad():

        # Transfer data to GPU if available
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # Get predictions
            _, preds = torch.max(outputs, 1)

    f1_test = f1_score(labels.data, preds, average="weighted")
    print("Testing...")
    print(f"F1 Score: {f1_test}")



