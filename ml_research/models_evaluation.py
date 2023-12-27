from collections import OrderedDict
import pickle
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class ModelsEvaluation():
    def __init__(self, saved_model, name_model, experiment_name, test_data):
        super(ModelsEvaluation, self).__init__()
        self.experiment_name = experiment_name
        self.model = saved_model
        self.name_model=name_model
        self.test_data = test_data

        x_test, y_test = self.test_data.inputs, self.test_data.targets
        y_test = list(y_test)
        
        new_params = OrderedDict()
        for key, value in saved_model["network"].items():
            key = key.replace("model.", "")
            new_params[key] = value
        # Cases for each model
        # VGG-16
        if name_model=="vgg_16":
            self.model = models.vgg16()
            self.model.classifier[6] = torch.nn.Linear(4096, 10)
        # RESNET-18
        elif name_model == "resnet18":
            self.model = models.resnet18()
            self.model.fc = torch.nn.Linear(in_features=512, out_features=10)
        # EfficientNetB0
        elif name_model == "efficientnetb0":
            self.model = models.efficientnet_b0()
            self.model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10)
        # EfficientNetB3
        elif name_model == "efficientnetb3":
            self.model = model = models.efficientnet_b3()
            self.model.classifier[1] = torch.nn.Linear(in_features=1536, out_features=10)
        # EfficientNetB5
        elif name_model == "efficientnetb5":
            self.model = model = models.efficientnet_b5()
            self.model.classifier[1] = torch.nn.Linear(in_features=2048, out_features=10)
        # RESNET-50
        elif name_model == "resnet50":
            self.model = models.resnet50()
            self.model.fc = torch.nn.Linear(in_features=2048, out_features=10)
        # Swin Tiny ViT
        elif name_model == "swin_t":
            self.model = models.swin_t()
            self.model.head = torch.nn.Linear(768, 10)
        # ViT BASE 16
        elif name_model == "vit_b_16":
            print("heeeree")
            self.model = models.vit_b_16()
            heads_layers = OrderedDict()
            heads_layers["head"] = torch.nn.Linear(768, 10)
            self.model.heads = torch.nn.Sequential(heads_layers)
        # ViT Large 16
        elif name_model == "vit_l_16":
            print("hhe")
        # ViT Base 32
        elif name_model == "vit_b_32":
            print("hhe")
        else:
            raise Exception("No correct model name!")
        
        self.model.load_state_dict(new_params)
        self.model.eval()

        if self.file_for_predictions('evaluation_results\predictions_' + self.experiment_name +'.pkl'):
            with open('evaluation_results\predictions_' + self.experiment_name +'.pkl', 'rb') as f:
                y_test_pred = pickle.load(f)
        else:
            y_test_pred = self.make_prediction(inputs=x_test)

        # Get classification report
        self.save_results(y_test, y_test_pred)
        
        # Get Confusion Matrix
        self.save_confusion_matrix(y_test, y_test_pred)

    def file_for_predictions(self, fn:str) -> bool:
        try:
            open(fn, "r")
            return True
        except:
            print("File does not appear to exist! Making predictions...")
            return False


    # Saves classification Report in csv file with metrices Precision, Recall, F1, Support, Accuracy
    def save_results(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        out_report = pd.DataFrame(report).transpose()
        out_report.to_csv('evaluation_results\classification_report_' + self.experiment_name + '.csv', index= True)

    # Draws a multi-class Confusion Matrix
    def save_confusion_matrix(self, y_test, y_pred):

        classes = ['Top', 'Bottom', 'Acc', 'Toy', 'Footwear', 'Case & Bag', 'Digital', 'Food & Drink', 'Home', 'PPCPs']

        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(cm,
                            index = classes, 
                            columns = classes)
        #Plotting the confusion matrix
        plt.figure(figsize=(12,10))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Values')
        plt.xlabel('Predicted Values')
        plt.savefig('evaluation_results\confusion_matrix_' + self.experiment_name + '.png')

    def split(self):
        x_test, y_test ,_, _, _, = self.test_data
        return x_test, y_test

    def make_prediction(self, inputs):
        predictions = []
        for i in tqdm(range(len(inputs))):
            input = torch.from_numpy(inputs[i,:,:,:])
            input = np.reshape(input, newshape=(-1, 3, 224, 224))
            out = self.model(input)
            _, y_pred = torch.max(out.data, 1)
            predictions.append(y_pred.item())
        with open('evaluation_results\predictions_' + self.experiment_name +'.pkl', 'wb') as f:
            pickle.dump(predictions, f)
        return predictions