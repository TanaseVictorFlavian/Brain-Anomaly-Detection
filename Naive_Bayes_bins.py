from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv
from sklearn.metrics import f1_score, confusion_matrix


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


# functia createBins va imparti intervalul in n - 1 intervale
def createBins(n):
    bins = np.linspace(start=0, stop=255, num=n)
    return bins

# folosim digitize pentru a incadra pixelii intr-o clasa
# clasele sunt date de nr de bin-uri - 1
# clasa e reprezentata de intervalul in care se va incadra pixelul


def valuesToBins(x, bins):
    return np.digitize(x, bins)

# partitionam imaginile exact ca in cerinta
trainingData = data[:15000]
validationData = data[15000:17000]
testData = data[17000:]

# incarcam etichetele imaginilor in numpy array-ul

trainingLabels = np.array(read_csv(
    "/kaggle/input/unibuc-brain-ad/data/train_labels.txt")['class'].tolist())
validationLabels = np.array(read_csv(
    "/kaggle/input/unibuc-brain-ad/data/validation_labels.txt")['class'].tolist())

# incercam o lista de parametri pentu numarul de binuri
# la fiecare iteratie cream un model nou, cu un nr diferit de binuri
# salvam scorurl f1 si acuratetea fiecarui model
# salvam la fiecare pas si modelul
bin_list = [3, 4, 5, 6, 7, 9]
f1s = []
accs = []
for bin in bin_list:
    bins = createBins(bin)

    # initializam modelul

    nBayesModel = MultinomialNB()

    # pentru ca nu avem destula memorie sa incarcam toate datele in RAM
    # impartim in 4 chunkuri datele de training si etichetele

    for partial, labelsChunk in zip(np.array_split(trainingData, 15000 // 8), np.array_split(trainingLabels, 15000 // 8)):
        partialDigitized = valuesToBins(partial, bins)
        nBayesModel.partial_fit(partialDigitized, labelsChunk, [0, 1])

    digitizedValidationData = valuesToBins(validationData, bins)
    results = nBayesModel.predict(digitizedValidationData)
    f1 = f1_score(results, validationLabels)
    score = nBayesModel.score(digitizedValidationData, validationLabels)
    f1s.append(f1)
    accs.append(score)

    # matricea de confuzie pt model


# afisam graficul pentru scorul f1
plt.plot(bin_list, f1s)
plt.xlabel('number of bins')
plt.ylabel('f1_score')
plt.title('Naive Bayes f1 score by number of bins')

# afisam graficul pentru acuratete
plt.plot(bin_list, accs)
plt.xlabel('number of bins')
plt.ylabel('accs_score')
plt.title('Naive Bayes accuaracy by number of bins')

# cream modelul cu 4 binuri


bins = createBins(4)

#initializam modelul

nBayesModel = MultinomialNB()


for partial, labelsChunk in zip(np.array_split(trainingData, 15000 // 8), np.array_split(trainingLabels, 15000 // 8)):
    # facem digitize pe chunk-ul de date
    # si facem partial_fit pe model cu chunk-urile de date digitized si chunk-ul de label-uri
    partialDigitized = valuesToBins(partial, bins)
    nBayesModel.partial_fit(partialDigitized, labelsChunk, [0, 1])


# facem digitize pe datele de test si facem predictia
# pe datele de test

digitizedTestData = valuesToBins(testData, bins)
labels = nBayesModel.predict(digitizedTestData)
ids = ['0' + str(nr) for nr in range(17001, 22150)]

# formatam fisierul de output, sa aiba ca headere id si class, apoi pe fiecare coloana
# valorile respective din structurile determinate anterior

csvRows = list(zip(ids, labels))

with open('/kaggle/working/submission.csv', 'w', newline='') as f:
    writerObj = csv.writer(f)
    writerObj.writerow(['id', 'class'])
    writerObj.writerows(csvRows)
