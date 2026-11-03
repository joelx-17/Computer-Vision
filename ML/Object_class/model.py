import torch
from torch import nn

#Create the NeuralNetwork
class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(3,32,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4096,128),
        nn.ReLU(),
        nn.Linear(128,10),
        nn.ReLU()
    )
  def forward(self,x):
    y = self.conv(x)
    y = self.classifier(y)

    return y


