from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv
from sklearn.metrics import f1_score
from PIL import Image
import os
from sklearn.model_selection import GridSearchCV


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

# incarcam etichetele imaginilor in numpy array-uri

trainingData = data[:15000]
validationData = data[15000:17000]
testData = data[17000:]

# citim label-urile cu read_csv din libraria pandas

trainingLabels = np.array(
    read_csv("/kaggle/input/unibuc-brain-ad/data/train_labels.txt")['class'].tolist())
validationLabels = np.array(
    read_csv("/kaggle/input/unibuc-brain-ad/data/validation_labels.txt")['class'].tolist())

# instantiem modelul
RF_clf = RandomForestClassifier()

# paramGrid va lua o lista de dictionare
# cheile reprezinta hyperparametrii clasificatorului
# valorile cheilor reprezinta valorile pe care le vom incerca pentru hyperparametrii respectivi
# pentru fiecare dictionar se vor antrena x * y * z * ... * t modele
# unde x y z ... t reprezinta numarul de valori pt fiecare hyperparametru
# in total nr_modele_per_dictionar1 + nr_modele_per_dictionar2 + ... nr_modele_per_dictionarN

paramGrid = [
    {'n_estimators': [100, 200, 300], 'criterion': [
        "gini", "entropy"], 'n_jobs': [-1]},
    {'bootstrap': [False, True], 'criterion': [
        "gini", "entropy"], 'n_jobs': [-1]},
]

# pentru fiecare dictionar se vor antrena x * y * z * ... * t modele
# unde x y z ... t reprezinta numarul de valori pt fiecare hyperparametru
# in total nr_modele_per_dictionar1 + nr_modele_per_dictionar2 + ... nr_modele_per_dictionarN
# grid search primeste ca parametru clasificatorul grid-ul,
# strategia de cross-validation si metrica dupa care se evalueaza modelul

gridSearch = GridSearchCV(RF_clf, paramGrid, cv=5, scoring='f1')

# incepem cautarea efectiva
gridSearch.fit(trainingData, trainingLabels)

# afisam parametrii cei mai buni in urma cautarii
print(gridSearch.best_params_)

# salvam rezultatele cautarii
mean_test_scores = gridSearch.cv_results_
print(mean_test_scores)

# selectam rezultatele care ne intereseaza
f1_scores = gridSearch.cv_results_['mean_test_score']
params = gridSearch.cv_results_['params']
scores = list(zip(f1_scores, params))
scores.sort(key=lambda x: x[0], reverse=True)

for f1_score, param in scores:
    print("f1 score: {:.3f}, parameters: {}".format(f1_score, param))

params_list = [str(params) for params, score in scores]
f1_scores = [score for params, score in scores]

# plotam f1 si parametrii pentru care s-a obtinut scorul

y_ticks = np.arange(0.45, 0.5 + 0.005, 0.005).tolist()
plt.figure(figsize=(12, 11))
plt.bar(range(len(params_list)), f1_scores)
plt.xticks(range(len(params_list)), params_list, rotation=90)
plt.ylabel("F1 Score")
plt.xlabel("Hyperparameters")
plt.title("Grid Search Results")
plt.ylim(0.45, 0.5)
plt.yticks(f1_scores)
plt.tight_layout()
plt.show()


RF_clf = RandomForestClassifier(
    bootstrap=False, n_estimators=500, criterion="gini", n_jobs=-1)
RF_clf.fit(trainingData, trainingLabels)

# testam pe datele de validare
labels = RF_clf.predict(validationData)
f1 = f1_score(validationLabels, validationResults)

# si facem predictia pe datele de test
labels = RF_clf.predict(testData)

ids = ['0' + str(nr) for nr in range(17001, 22150)]
csvRows = list(zip(ids, labels))

with open('/kaggle/working/submission.csv', 'w', newline='') as f:
    writerObj = csv.writer(f)
    writerObj.writerow(['id', 'class'])
    writerObj.writerows(csvRows)
