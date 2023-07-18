import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pandas import read_csv
import csv
from PIL import Image
import os
import torch.nn.functional as nnfunc
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


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


def preprocessImages(path_to_directory='/kaggle/input/unibuc-brain-ad/data/data'):
    return np.array([[np.array(Image.open(os.path.join(path_to_directory, f)).convert('L'), dtype=np.float32)] for f in sorted(os.listdir(path_to_directory))])


data = preprocessImages()

trainingData = data[:15000]
validationData = data[15000:17000]
testData = data[17000:]
# citim label-urile cu read_csv din libraria pandas
trainingLabels = np.array(
    read_csv("/kaggle/input/unibuc-brain-ad/data/train_labels.txt")['class'].tolist())
validationLabels = np.array(
    read_csv("/kaggle/input/unibuc-brain-ad/data/validation_labels.txt")['class'].tolist())

trainDataSet = CustomDataset(trainingData, trainingLabels)
validationDataSet = CustomDataset(validationData, validationLabels)
testDataSet = CustomDataset(testData, 5149 * [0])

trainDataLoader = DataLoader(trainDataSet, batch_size=32)
validationDataLoader = DataLoader(validationDataSet, batch_size=32)
testDataLoader = DataLoader(testDataSet, batch_size=32)

gpuCheck = torch.cuda.is_available()
device = 'cuda' if gpuCheck else 'cpu'

in_channels = 1
batch_size = 32
num_classes = 1
lr = 0.0001
num_epochs = 4

RN18 = ResNet18().to(device)
lfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(RN18.parameters(), lr=lr)

RN18.train()
for epoch in range(num_epochs):
    print(f"###### EPOCH {epoch + 1} ######")
    for batch_idx, (trainImageBatch, trainLabelsBatch) in enumerate(trainDataLoader):
        trainImageBatch = trainImageBatch.to(device)
        trainLabelsBatch = trainLabelsBatch.to(dtype=torch.long).to(device)

        # pasul forward -> efectuam convolutiile
        innerPrediction = RN18(trainImageBatch)

        #calculam loss-ul pe predictii
        loss = lfn(innerPrediction, trainLabelsBatch)

        #pasul backward -> ne intoarcem si intarim legaturile corespunzatoare raspunsurilor corecte
        optimizer.zero_grad()
        loss.backward()

        #gradient descent
        optimizer.step()

        if(batch_idx + 1) % 100 == 0:
            loss = loss.item()
            print(f"Batch {batch_idx + 1} : loss: {loss:>7f}")

validationResults = []

# functie pt calcularea acuratetii
def acc(loader, model):
    correct = 0
    model.eval()

    with torch.no_grad():

        for valData, valLabels in loader:
            # incarcam datele pe gpu
            valData = valData.to(device)
            valLabels = valLabels.to(device)

            # facem predictiile
            scores = RN18(valData)
            _, prd = scores.max(1)

            validationResults.extend(prd.cpu().tolist())

            correct += (prd == valLabels).sum()

        print(f'acc = {correct / 2000:.7f}')


print(acc(validationDataLoader, RN18))
f1 = f1_score(validationLabels, validationResults)
print(f1)

labels = []

# functie pentru a face predictiile fara etichete de validare
def test(loader, model):
    model.eval()

    with torch.no_grad():
        for testData, _ in loader:
            # incarcam datele pe gpu
            testData = testData.to(device)

            # facem predictia 

            scores = RN18(testData)
            _, prd = scores.max(1)

            # salvam predictiile in labels
            labels.extend(prd.cpu().tolist())


test(testDataLoader, RN18)
ids = ['0' + str(nr) for nr in range(17001, 22150)]

csvRows = list(zip(ids, labels))

with open('/kaggle/working/submission.csv', 'w', newline='') as f:
    writerObj = csv.writer(f)
    writerObj.writerow(['id', 'class'])
    writerObj.writerows(csvRows)
