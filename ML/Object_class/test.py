from model import CNN
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import cv2
import matplotlib.pyplot as plt
from classify import classify

transform = transforms.ToTensor()

device = torch.device("cpu")

#Test data 
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_data = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)

model = CNN().to(device)

model.load_state_dict(torch.load("models/model.pth", map_location=device,weights_only=True))

model.eval()


#import img
img = cv2.resize(cv2.cvtColor(cv2.imread("img/deer.jpg"),cv2.COLOR_BGR2RGB),(32,32))
img = torch.from_numpy(img)
img = img.permute(2,0,1)
img = img.float() / 255.0
print(img.shape)
with torch.no_grad():
  correct=0
  total=0
  image=img
  image = image.to(device)
  lab=4
  label = torch.tensor([4], dtype=torch.long)
  label = label.to(device)
  output = model(image.unsqueeze(0))
  pred = output.argmax(dim=1)
  #total+=label.size(0)
  #correct+=(pred==label).float().sum().item()
  plt.imshow(image.permute(1, 2, 0).cpu().numpy())
  plt.title(f'Predicted class: {classify(pred.item())} | Actual class: {classify(lab)}')

  plt.axis('off')
  plt.show()
 