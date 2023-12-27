import numpy as np

import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import BasicBlock, ConvolutionalNetwork, ResNet
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import VisionTransformer
from torchvision.models import ResNet18_Weights
import os
from tqdm import tqdm
from vit_pytorch.efficient import ViT   
import torch   
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms   
from torch.optim.lr_scheduler import StepLR   
from torch.utils.data import DataLoader, Dataset
from linformer import Linformer   


args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if args.dataset_name == 'emnist':
    train_data = data_providers.EMNISTDataProvider('train', batch_size=args.batch_size,
                                                   rng=rng,
                                                   flatten=False)  # initialize our rngs using the argument set seed
    val_data = data_providers.EMNISTDataProvider('valid', batch_size=args.batch_size,
                                                 rng=rng,
                                                 flatten=False)  # initialize our rngs using the argument set seed
    test_data = data_providers.EMNISTDataProvider('test', batch_size=args.batch_size,
                                                  rng=rng,
                                                  flatten=False)  # initialize our rngs using the argument set seed
    print(train_data)
    num_output_classes = train_data.num_classes

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7 #for learning rate scheduler 

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)


# loss function

criterion = nn.CrossEntropyLoss()

# optimizer

optimizer = optim.Adam(model.parameters(), lr=lr)

# scheduler

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#start training

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output, label)
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )