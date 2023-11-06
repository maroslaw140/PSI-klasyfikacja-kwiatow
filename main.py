import csv
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

import wykresy
# wykresy.RysujWykresPlatka()
# wykresy.RysujWykresKielicha()
# wykresy.RysujHistogramDlugosciKielicha()
# wykresy.RysujHistogramSzerokosciKielicha()
# wykresy.RysujHistogramDlugosciPlatka()
# wykresy.RysujHistogramSzerokosciPlatka()

with open('Iris.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)

    species_data = {}

    for data in reader:
        species = data[5]

        if species not in species_data:
            species_data[species] = []

        species_data[species].append(data)

characteristics_train = []
species_train = []

characteristics_test = []
species_test = []

test_set = 0.2

for species, data_list in species_data.items():

    c_train, c_test, s_train, s_test = train_test_split(data_list, [species] * len(data_list), test_size=test_set, random_state=42)

    characteristics_train.extend(c_train)
    species_train.extend(s_train)

    characteristics_test.extend(c_test)
    species_test.extend(s_test)

data_train = list(zip(characteristics_train, species_train))
shuffle(data_train)
characteristics_train, species_train = zip(*data_train)

data_test = list(zip(characteristics_test, species_test))
shuffle(data_test)
characteristics_test, species_test = zip(*data_test)

characteristics_train = [row[:0] + row[0 + 1:] for row in characteristics_train]
characteristics_train = [row[:4] + row[4 + 1:] for row in characteristics_train]
characteristics_train = [[float(value) for value in row] for row in characteristics_train]

characteristics_test = [row[:0] + row[0 + 1:] for row in characteristics_test]
characteristics_test = [row[:4] + row[4 + 1:] for row in characteristics_test]
characteristics_test = [[float(value) for value in row] for row in characteristics_test]

accuracy = 0
precision = 0
recall = 0

scores = 0

# K-Nearest Neighbors

n_neighbors = 0
while accuracy < 1:
    n_neighbors += 1

    knn = KNeighborsClassifier(n_neighbors)

    knn.fit(characteristics_train, species_train)

    species_pred = knn.predict(characteristics_test)

    accuracy = accuracy_score(species_test, species_pred)
    precision = precision_score(species_test, species_pred, average='micro')
    recall = recall_score(species_test, species_pred, average='micro')

    scores = cross_val_score(knn, characteristics_train, species_train, cv=5, scoring='accuracy')

print("KNN: Dla", n_neighbors, "sąsiadów dokładność klasyfikacji:", accuracy, "Precyzja:", precision, "Czułość:", recall)
print("KNN: Dokładność walidacji krzyżowej: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))


# SVM

for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):

    svm_model = SVC(kernel=kernel, C=1, probability=True)

    svm_model.fit(characteristics_train, species_train)

    species_pred = svm_model.predict(characteristics_test)

    accuracy = accuracy_score(species_test, species_pred)
    precision = precision_score(species_test, species_pred, average='micro')
    recall = recall_score(species_test, species_pred, average='micro')

    scores = cross_val_score(svm_model, characteristics_train, species_train, cv=5, scoring='accuracy')

    print("SVM", kernel, ": Dokładność klasyfikacji:", accuracy, "Precyzja:", precision, "Czułość:", recall)
    print("SVM", kernel, ": Dokładność walidacji krzyżowej: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))

    results = permutation_importance(svm_model, characteristics_test, species_test, n_repeats=30, random_state=42)
    importance = results.importances_mean
    # wykresy.WaznoscCech(importance, ('SVM-' + kernel))


# Random Forests

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(characteristics_train, species_train)

species_pred = rf_model.predict(characteristics_test)

accuracy = accuracy_score(species_test, species_pred)
precision = precision_score(species_test, species_pred, average='micro')
recall = recall_score(species_test, species_pred, average='micro')

scores = cross_val_score(rf_model, characteristics_train, species_train, cv=5, scoring='accuracy')

print("RF: Dokładność klasyfikacji:", accuracy, "Precyzja:", precision, "Czułość:", recall)
print("RF: Dokładność walidacji krzyżowej: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))

importance = rf_model.feature_importances_
# wykresy.WaznoscCech(importance, 'Random Forest')
