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
        # apelam constructoul din clasa de baza
        super(ResNet18, self).__init__()

        # selectam resnet18 care nu este pre-antrenat
        self.rn = models.resnet18(weights=None)

        # modificam input_channels in 1 deoarece imaginile noastre sunt grayscaled
        # default este 3, pentru input-uri RGB
        self.rn.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # stratul fully connected
        # 2 reprezinta numarul de clase posibile
        self.rn.fc = nn.Linear(self.rn.fc.in_features, 2)

    def forward(self, x):
        x = self.rn(x)
        return x
# clasa pentru crearea de dataseturi
# avem nevoie de dataseturi pentru aincarca datele in tensor-uri
# pentru a utiliza gpu-ul


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# testam functionalitatea retelei


def test():
    net = ResNet18()
    x = torch.randn(5, 1, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


test()


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

# testam daca gpu-ul este disponibil
gpuCheck = torch.cuda.is_available()
device = 'cuda' if gpuCheck else 'cpu'


# testam diferiti hyperparametri
def trialAndError(num_epochs=20, batch_sizes=[32, 64, 128], learning_rates=[0.00008, 0.001, 0.005, 0.0001, 0.00012, 0.00015]):
    for bs in batch_sizes:

        trainDataSet = CustomDataset(trainingData, trainingLabels)
        validationDataSet = CustomDataset(validationData, validationLabels)
        testDataSet = CustomDataset(testData, 5149 * [0])

        trainDataLoader = DataLoader(trainDataSet, batch_size=bs)
        validationDataLoader = DataLoader(validationDataSet, batch_size=bs)
        testDataLoader = DataLoader(testDataSet, batch_size=bs)

        for lr in learning_rates:
            # instantiem reteaua
            RN18 = ResNet18()
            # definim optimizer-ul
            optimizer = torch.optim.Adam(RN18.parameters(), lr=lr)
            # punem reteaua pe gpu
            RN18.to(device)
            # activam state-ul de train
            RN18.train()
            # la fiecare epoca testam peformanta si afisam parametrii
            for epoch in range(num_epochs):
                for batch_idx, (trainImageBatch, trainLabelsBatch) in enumerate(trainDataLoader):
                    trainImageBatch = trainImageBatch.to(device)
                    trainLabelsBatch = trainLabelsBatch.to(
                        dtype=torch.long).to(device)

                    # pasul forward -> efectuam convolutiile
                    innerPrediction = RN18(trainImageBatch)

                    #calculam loss-ul pe predictii
                    loss = lfn(innerPrediction, trainLabelsBatch)

                    #pasul backward -> ne intoarcem si intarim legaturile corespunzatoare raspunsurilor corecte
                    optimizer.zero_grad()
                    loss.backward()

                    #gradient descent
                    optimizer.step()

                loss = loss.item()
                print(f"Loss: {loss:>7f}")
                correct = 0
                validationResults = []
                # punem modelul in state-ul de evaluare
                RN18.eval()
                with torch.no_grad():
                    for valData, valLabels in validationDataLoader:
                        # incarcam datele pe gpu
                        
                        valData = valData.to(device)
                        valLabels = valLabels.to(device)
                        
                        # facem predictia 
                        scores = RN18(valData)
                        _, prd = scores.max(1)
                        # salvam predictiile 
                        validationResults.extend(prd.cpu().tolist())
                        # correct -> counter pentru predictiile corecte
                        correct += (prd == valLabels).sum()

                    print(
                        f"Batch_size = {bs}\nLearning Rate = {lr}\nEpoch : {epoch + 1}")
                    print(f'Acc = {correct / 2000:.7f}')

                    f1 = f1_score(validationLabels, validationResults)
                    print(f"f1_score : {f1}\n################")
