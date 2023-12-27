
import torch
import torch.nn as nn

"""
This code defines a new AttentionModule module that can be added to the ResNet50 
architecture in place of the original final fully connected layer. The attention 
layer calculates attention weights for each feature based on learned weights, and 
applies these weights to the input features to obtain a weighted sum. This helps 
the model focus on the most important features for each classification task.

"""

class AttentionModule(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_features),
            nn.Sigmoid()
        )

        self.out_features = in_features
        
    def forward(self, x):
        # Calculate attention weights for each feature
        weights = self.attention(x)

        # Apply attention weights to input features
        out = x * weights
        
        return out