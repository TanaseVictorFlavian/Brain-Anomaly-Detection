import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os


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

# instantiem scalerul
scaler = StandardScaler()

# facem fit la scaler pe data
scaler.fit(data)

#apoi scalam datele
scaledData = scaler.transform(data)

# plotam un chunk din datele scalate
plt.plot(scaledData[:100], color='green')
plt.xlabel('Data Point')
plt.ylabel('Scaled Value')
plt.title('Scaled Data Plot')
plt.show()

# partitionam imaginile exact ca in cerinta
trainingData = scaledData[:15000]
validationData = scaledData[15000:17000]
testData = scaledData[17000:]

# incarcam etichetele imaginilor in numpy array-uri

trainingLabels = np.array(read_csv(
    "/kaggle/input/unibuc-brain-ad/data/train_labels.txt")['class'].tolist())
validationLabels = np.array(read_csv(
    "/kaggle/input/unibuc-brain-ad/data/validation_labels.txt")['class'].tolist())

# in f1s si scores vom tine rezultatele evaluarile modelelor
f1s = []
scores = []

# valorile pt numarul de vecini care urmeaza sa fie incercati
numberOfNeighbours = [3, 4, 5, 7, 9, 12, 20]

# la fiecare iteratie cream un model nou cu numarul de vecini diferit
# il evaluam si salvam rezultatul

for nr in numberOfNeighbours:
    knn = KNeighborsClassifier(n_neighbors=nr)

    # antrenarea modelului
    knn.fit(trainingData, trainingLabels)
    
    # predictia pe datele de validare
    result = knn.predict(validationData)

    # calcularea acuratetii
    score = knn.score(validationData, validationLabels)

    # calcularea scorului f1
    f1 = f1_score(result, validationLabels, average="binary")

    f1s.append(f1)
    scores.append(score)

# plotam scorurile f1

plt.plot(numberOfNeighbours, f1s)
plt.xlabel('number of neighbours')
plt.ylabel('f1_score')
plt.title('KNN f1 score by number of neighbours')

# plotam acuratetile 

plt.plot(numberOfNeighbours, scores)
plt.xlabel('number of neighbours')
plt.ylabel('accs_score')
plt.title('Knn accuaracy by number of neighbours')
