from collections import OrderedDict
import numpy as np

import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from hybrid_vit_resnet import HybridModel
from model_architectures import BasicBlock, ConvolutionalNetwork, ResNet
from models_evaluation import ModelsEvaluation

from resnet_improved import AttentionModule
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import VisionTransformer
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    EfficientNet_B5_Weights,
    VGG16_Weights,
    Swin_T_Weights,
)
import os

# from matplotlib import pyplot as plt
# import matplotlib

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

# if __name__ == "__main__":
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if args.dataset_name == "emnist":
    train_data = data_providers.EMNISTDataProvider(
        "train", batch_size=args.batch_size, rng=rng, flatten=False
    )  # initialize our rngs using the argument set seed

    val_data = data_providers.EMNISTDataProvider(
        "valid", batch_size=args.batch_size, rng=rng, flatten=False
    )  # initialize our rngs using the argument set seed
    test_data = data_providers.EMNISTDataProvider(
        "test", batch_size=args.batch_size, rng=rng, flatten=False
    )  # initialize our rngs using the argument set seed
    num_output_classes = train_data.num_classes
elif args.dataset_name == "cifar10":
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            Cutout(n_holes=1, length=14),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = data_providers.CIFAR10(
        root="data", set_name="train", download=True, transform=transform_train
    )
    train_data = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=4
    )

    valset = data_providers.CIFAR10(
        root="data", set_name="val", download=True, transform=transform_test
    )
    val_data = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=4
    )

    testset = data_providers.CIFAR10(
        root="data", set_name="test", download=True, transform=transform_test
    )
    test_data = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    num_output_classes = 10

elif args.dataset_name == "cifar100":
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            Cutout(n_holes=1, length=14),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = data_providers.CIFAR100(
        root="data", set_name="train", download=True, transform=transform_train
    )
    train_data = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=4
    )

    valset = data_providers.CIFAR100(
        root="data", set_name="val", download=True, transform=transform_test
    )
    val_data = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=4
    )

    testset = data_providers.CIFAR100(
        root="data", set_name="test", download=True, transform=transform_test
    )
    test_data = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4
    )

    num_output_classes = 47
elif args.dataset_name == "Product10k":
    train_data = data_providers.Products10kDataProvider(
        "test",
        batch_size=args.batch_size,
        rng=rng,
        flatten=False,
        shuffle_order=False,
        num_classes=10,
    )  # initialize our rngs using the argument set seed

    val_data = data_providers.Products10kDataProvider(
        "test",
        batch_size=args.batch_size,
        rng=rng,
        flatten=False,
        shuffle_order=False,
        num_classes=10,
    )  # initialize our rngs using the argument set seed
    test_data = data_providers.Products10kDataProvider(
        "test",
        batch_size=args.batch_size,
        rng=rng,
        flatten=False,
        shuffle_order=False,
        num_classes=10,
    )  # initialize our rngs using the argument set seed
    num_output_classes = test_data.num_classes


if args.model_to_load == "resnet18":
    print("Loading ResNet18")
    # RESNET
    # Load ResNet-18 with pre-trained weights
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    feature_number = model.fc.in_features
    model.fc = nn.Linear(feature_number, num_output_classes)
    # END RESNET
elif args.model_to_load == "resnet50":
    print("Loading ResNet50")
    # RESNET
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    feature_number = model.fc.in_features
    model.fc = nn.Linear(feature_number, num_output_classes)
    # END RESNET
elif args.model_to_load == "resnet50_multi_head":
    print("Loading ResNet50 with Multi Head")
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    feature_number = model.fc.in_features
    # Add Attention Layer to last layer
    # Freeze all layers except the last convolutional block
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    # END RESNET
    # Define attention layer and final classification layer
    attention_layer = AttentionModule(in_features=feature_number, hidden_dim=512)
    classification_layer = nn.Linear(attention_layer.out_features, num_output_classes)

    # Replace final layers with attention layer and classification layer
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Sequential(attention_layer, classification_layer)

elif args.model_to_load == "efficientnetb0":
    print("Loading EfficientNetB0")
    # EFFICIENTNET
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_output_classes)
    # END EFFICIENTNET
elif args.model_to_load == "efficientnetb3":
    print("Loading EfficientNetB3")
    # EFFICIENTNET
    weights = EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(in_features=1536, out_features=num_output_classes)
    # END EFFICIENTNET
elif args.model_to_load == "efficientnetb5":
    print("Loading EfficientNetB5")
    # EFFICIENTNET
    weights = EfficientNet_B5_Weights.DEFAULT
    model = models.efficientnet_b5(weights=weights)
    model.classifier[1] = nn.Linear(in_features=2048, out_features=num_output_classes)
elif args.model_to_load == "swin_t":
    print("Loading swin_t")
    # SWIN TINY TRANSFORMER
    weights = Swin_T_Weights.DEFAULT
    model = models.swin_t(weights=weights)
    print("Parameters: " + str(model.head.in_features))  # 768 in features
    model.head = nn.Linear(model.head.in_features, num_output_classes)
    # END SWIN TINY TRANSFORMER

elif args.model_to_load == "hybrid_resnet_swin_t":
    print("Loading hybrid_resnet_swin_t")
    # HYBRID RESNET SWIN TINY TRANSFORMER
    # Load the Swin Tiny Transformer model
    weights = Swin_T_Weights.DEFAULT
    swin_model = models.swin_t(weights=weights)
    swin_model.head = torch.nn.Linear(in_features=swin_model.head.in_features, out_features=num_output_classes)

    # Define ResNet backbone
    weights = ResNet18_Weights.DEFAULT
    resnet_backbone = models.resnet18(weights=weights)

    model = HybridModel(resnet_backbone, swin_model, num_output_classes)
    # END HYBRID RESNET SWIN TINY TRANSFORMER
elif args.model_to_load == "vit_b_16":
    print("Loading vit_b_16")
    # TRANSFORMER
    weights = ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    heads_layers = OrderedDict()
    heads_layers["head"] = nn.Linear(768, num_output_classes)
    model.heads = nn.Sequential(heads_layers)
    # END TRANSFORMER
elif args.model_to_load == "vit_b_32":
    print("Loading vit_b_32")
    # TRANSFORMER
    weights = ViT_B_32_Weights.DEFAULT
    model = models.vit_b_32(weights=weights)
    heads_layers = OrderedDict()
    heads_layers["head"] = nn.Linear(768, num_output_classes)
    model.heads = nn.Sequential(heads_layers)
    # END TRANSFORMER
elif args.model_to_load == "vit_l_16":
    print("Loading vit_l_16")
    # TRANSFORMER
    weights = ViT_L_16_Weights.DEFAULT
    model = models.vit_l_16(weights=weights)
    heads_layers = OrderedDict()
    heads_layers["head"] = nn.Linear(1024, num_output_classes)
    model.heads = nn.Sequential(heads_layers)
    # END TRANSFORMER
elif args.model_to_load == "vgg_16":
    print("Loading vgg_16")
    # VGG
    weights = VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model.classifier[6] = torch.nn.Linear(4096, num_output_classes)
    # END VGG
else:
    model = (
        ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
            input_shape=(
                args.batch_size,
                args.image_num_channels,
                args.image_height,
                args.image_height,
            ),
            dim_reduction_type=args.dim_reduction_type,
            num_filters=args.num_filters,
            num_layers=args.num_layers,
            use_bias=False,
            num_output_classes=num_output_classes,
        )
    )

if args.eval_mode_on == False:
    conv_experiment = ExperimentBuilder(
        network_model=model,
        use_gpu=args.use_gpu,
        experiment_name=args.experiment_name,
        num_epochs=args.num_epochs,
        weight_decay_coefficient=args.weight_decay_coefficient,
        continue_from_epoch=args.continue_from_epoch,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        learning_rate=args.lr,
    )  # build an experiment object

    (
        experiment_metrics,
        test_metrics,
    ) = conv_experiment.run_experiment()  # run experiment and return experiment metrics

else:
    filename_checkpoint = args.eval_model_checkpoint
    name_model = args.eval_model_name
    saved_model = torch.load(filename_checkpoint, map_location=torch.device("cpu"))
    eval_experiment = ModelsEvaluation(
        saved_model=saved_model,
        name_model=name_model,
        experiment_name=args.experiment_name,
        test_data=test_data,
    )


# Print graph for training and validation accuracy and loss
# train_acc = experiment_metrics["train_acc"]
# train_loss = experiment_metrics["train_loss"]
# val_acc = experiment_metrics["val_acc"]
# val_loss = experiment_metrics["val_loss"]

# epochs = range(len(train_acc))

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)

# first_ax = ax.plot(epochs, train_acc, label="Training Acc.")
# second_ax = ax.plot(epochs, val_acc, label="Val. Acc.")

# ax.set_xlabel("Epochs")
# ax.set_ylabel("Accuracy")
# ax.set_title(f"{args.experiment_name} Performance")

# ax2 = ax.twinx()
# third_ax = ax2.plot(epochs, train_loss, label="Training Loss", color="#FE1100")
# fourth_ax = ax2.plot(epochs, val_loss, label="Val. Loss.", color="#C4C417")
# ax2.set_ylabel("Loss")
# ax2.grid(True)

# lns = first_ax + second_ax + third_ax + fourth_ax
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=5)

# ax.set_yticks(
#     np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax2.get_yticks()))
# )
# ax2.set_yticks(
#     np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks()))
# )

# fig.tight_layout()

# graph_path = os.path.abspath(
#     os.path.join(os.path.abspath(args.experiment_name), "result_outputs")
# )
# fig.savefig(f"{graph_path}/model_performance_graph.pdf")
