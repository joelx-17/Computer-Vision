from model import CNN
import torch
from torch import optim, nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_data = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

device = torch.device("cpu")

model = CNN().to(device)
#model.load_state_dict(torch.load("models/model.pth", map_location=device,weights_only=True))
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

iters=20

for iter in range(iters):
  model.train()
  total = 0
  correct=0
  total_loss=0
  for image, label in train_data:
    image=image.to(device)
    label=label.to(device)
    output = model(image)

    predictions = output.argmax(dim=1)
    optimizer.zero_grad()
    loss = loss_criterion(output, label)

    loss.backward()
    optimizer.step()

    total_loss+=loss.item()
    total += label.size(0)
    correct += (predictions==label).float().sum().item()
  print(f"Iteration: {iter+1},Total Loss: {total_loss}, Training Accuracy: {100*correct/total:.4f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
print("Model has been saved")