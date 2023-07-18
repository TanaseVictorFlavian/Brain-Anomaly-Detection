import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pandas import read_csv
import csv
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.rn = models.resnet50(weights=None)
        self.rn.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rn.fc = nn.Linear(self.rn.fc.in_features, 2)

    def forward(self, x):
        x = self.rn(x)
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.rn = models.resnet34(weights=None)
        self.rn.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rn.fc = nn.Linear(self.rn.fc.in_features, 2)

    def forward(self, x):
        x = self.rn(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.rn = models.resnet18(weights=None)
        self.rn.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rn.fc = nn.Linear(self.rn.fc.in_features, 2)

    def forward(self, x):
        x = self.rn(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def preprocessImages(path_to_directory='unibuc-brain-ad/data/data'):
    return np.array([[np.array(Image.open(os.path.join(path_to_directory, f)).convert('L'), dtype=np.float32)] for f in sorted(os.listdir(path_to_directory))])


data = preprocessImages()

trainingData = data[:15000]
validationData = data[15000:17000]
testData = data[17000:]
# citim label-urile cu read_csv din libraria pandas
trainingLabels = np.array(
    read_csv("unibuc-brain-ad/data/train_labels.txt")['class'].tolist())
validationLabels = np.array(
    read_csv("unibuc-brain-ad/data/validation_labels.txt")['class'].tolist())


trainDataSet = CustomDataset(trainingData, trainingLabels)
validationDataSet = CustomDataset(validationData, validationLabels)
testDataSet = CustomDataset(testData, 5149 * [0])

trainDataLoader = DataLoader(trainDataSet, batch_size=32)
validationDataLoader = DataLoader(validationDataSet, batch_size=32)
testDataLoader = DataLoader(testDataSet, batch_size=32)


gpuCheck = torch.cuda.is_available()
device = 'cuda' if gpuCheck else 'cpu'
print(device)

#hyperparameters

in_channels = 1
batch_size = 32
num_classes = 1
lr = 1e-3
num_epochs = 30
lfn = nn.CrossEntropyLoss()

for i in range(3):

    if i == 0:
        modelName = "ResNet18"
        RNModel = ResNet18()
    elif i == 1:
        modelName = "ResNet34"
        RNModel = ResNet34()
    else:
        modelName = "ResNet50"
        RNModel = ResNet50()

    print(modelName)

    optimizer = torch.optim.Adam(RNModel.parameters(), lr=lr)
    RNModel = RNModel.to(device)

    for epoch in range(num_epochs):
        print(f"###### EPOCH {epoch + 1}######")
        for batch_idx, (trainImageBatch, trainLabelsBatch) in enumerate(trainDataLoader):
            trainImageBatch = trainImageBatch.to(device)
            trainLabelsBatch = trainLabelsBatch.to(dtype=torch.long).to(device)

            # pasul forward
            innerPrediction = RNModel(trainImageBatch)

            #calculam loss-ul pe predictii
            loss = lfn(innerPrediction, trainLabelsBatch)

            #pasul backward -> ne intoarcem si intarim legaturile corespunzatoare raspunsurilor corecte
            optimizer.zero_grad()
            loss.backward()

            #gradient descent
            optimizer.step()

            # if(batch_idx + 1) % 100 == 0:
        loss = loss.item()
        print(f"Loss: {loss:>7f}")

    torch.save(RNModel.state_dict(), f'{modelName}.pth')
    validationResults = []

    correct = 0
    RNModel.eval()

    with torch.no_grad():
        for x, y in validationDataLoader:
            x = x.to(device=device)
            y = y.to(device)

            scores = RNModel(x)
            _, predictions = scores.max(1)

            validationResults.extend(predictions.cpu().tolist())

            correct += (predictions == y).sum()

        print(f'acc = {correct / 2000:.7f}')

    print(f1_score(validationLabels, validationResults))