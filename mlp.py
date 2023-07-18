from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv
from sklearn.metrics import f1_score
from PIL import Image
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class MLPNetwork(nn.Module):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(224*224, 512, dtype=torch.float32)
        self.second_layer = nn.Linear(512, 512, dtype=torch.float32)
        self.output_layer = nn.Linear(512, 1, dtype=torch.float32)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.output_layer(x)
        return x

def preprocessImages(path_to_directory=r'unibuc-brain-ad\data\data'):
    return np.array([np.array(Image.open(os.path.join(path_to_directory, f)).convert('L')) for f in sorted(os.listdir(path_to_directory))])

data = preprocessImages()
# data1 = np.array([np.reshape(img, (224*224)) for img in data[: 22149 // 2]])
# data2 = np.array([np.reshape(img, (224*224)) for img in data[22149 // 2:]])
# data = np.concatenate((data1, data2))

trainingData = data[:15000]
validationData = data[15000:17000]
testData = data[17000:]
# citim label-urile cu read_csv din libraria pandas
trainingLabels = np.array(
    read_csv(r"unibuc-brain-ad/data/train_labels.txt")['class'].tolist(), dtype=np.uint8)
validationLabels = np.array(
    read_csv(r"unibuc-brain-ad/data/validation_labels.txt")['class'].tolist(), dtype=np.uint8)


class MLPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


trainDataSet = MLPDataset(trainingData, trainingLabels)
validationDataSet = MLPDataset(validationData, validationLabels)

trainDataLoader = DataLoader(validationDataSet, batch_size=512)
validationDataLoader = DataLoader(validationDataSet, batch_size=512)

MLPModel = MLPNetwork() 

lfn = nn.CrossEntropyLoss() # setam loss functionul ca fiind cross-entropy-ul
optimizer = torch.optim.SGD(MLPModel.parameters(), lr=1e-2)


epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLPModel = MLPModel.to(device)
MLPModel.train(True)

for i in range(epochs):
    for trainImageBatch, trainLabelsBatch in trainDataLoader:
        print(trainLabelsBatch)
        print(trainImageBatch)

        trainImageBatch = trainImageBatch.to(device)
        trainLabelsBatch = trainLabelsBatch.to(dtype=torch.long).to(device)

        # pasul forward
        innerPrediction = MLPModel(trainImageBatch)
        loss = lfn(innerPrediction, trainLabelsBatch)

        #pasul backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            loss = loss.item()
            print(f"###### EPOCH {i+1} ######, \n loss: {loss:>7f}")
print(f"Last loss : {loss.item():>7f}")
