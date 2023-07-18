from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv
from sklearn.metrics import f1_score

from PIL import Image
import os


# avem 255 de valori posibile pentru fiecare pixel
# impartim range-ul de valori in mai multe intervale
# practic, intervalele vor reprezenta clase de echivalenta pentru pixeli
# calculam P(c|X) pentru fiecare clasă c (c∈ [1, num_classes]), iar eticheta finală
# este dată de clasa cu probabilitatea cea mai mare

# incarcam png-urile din data in numpy array-uri
# os.listdir nu ne garanteaza ca imaginile vor fi procesate in ordine
# cum noi avem label-urile in ordine asta duce la invatarea gresita a clasificatorului
# cum numele pozelor sunt numere crescatoare, vom sorta dupa nume lista rezultata


def preprocessImages(path_to_directory='/kaggle/input/unibuc-brain-ad/data/data'):
    return [np.array(Image.open(os.path.join(path_to_directory, f)).convert('L')) for f in sorted(os.listdir(path_to_directory))]

# incarcam in data imaginile
# le dam reshape in 2 chunk-uri pt a evita cazul in care nu avem destul RAM
# apoi concatenam chunk-urile


data = preprocessImages()
data1 = np.array([np.reshape(img, (224*224)) for img in data[: 22149 // 2]])
data2 = np.array([np.reshape(img, (224*224)) for img in data[22149 // 2:]])
data = np.concatenate((data1, data2))

# partitionam imaginile exact ca in cerinta

trainingData = data[:15000]
validationData = data[15000:17000]
testData = data[17000:]


# incarcam etichetele imaginilor in numpy array-uri
trainingLabels = np.array(
    read_csv("/kaggle/input/unibuc-brain-ad/data/train_labels.txt")['class'].tolist())
validationLabels = np.array(
    read_csv("/kaggle/input/unibuc-brain-ad/data/validation_labels.txt")['class'].tolist())


# definim modelul

nBayesModel = MultinomialNB()

# impartim in chunkuri datele de training si etichetele pt ca RAM-ul sa nu devina o problema
# si facem partial fit, adica antrenam secvential modelul pe chunk-urile de imagini asociate cu etichetele respective

for partial, labelsChunk in zip(np.array_split(trainingData, 15000 // 8), np.array_split(trainingLabels, 15000 // 8)):
    # antrenarea partiala a modelului
    nBayesModel.partial_fit(partial, labelsChunk, [0,1])

# predictia pe datele de validare
validationResults = nBayesModel.predict(validationData)

# scorul f1 pe datele de validare
f1 = f1_score(validationLabels, validationResults)

# in ids numerotam etichetele din fisierul de output
# in labels se vor afla rezultatele predictiei noastre pe setul de test
labels = nBayesModel.predict(testData)
ids = ['0' + str(nr) for nr in range(17001, 22150)]


# formatam fisierul de output, sa aiba ca headere id si class, apoi pe fiecare coloana
# valorile respective din structurile determinate anterior

csvRows = list(zip(ids,labels))

with open ('/kaggle/working/submission.csv', 'w', newline='') as f:
    writerObj = csv.writer(f)
    writerObj.writerow(['id', 'class'])
    writerObj.writerows(csvRows)