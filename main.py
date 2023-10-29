import csv

from sklearn.model_selection import train_test_split
from random import shuffle

test_set = 0.2
trening_set = 1 - test_set

# Wczytaj dane z pliku CSV i podziel je na kategorie
with open('Iris.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Pomijamy nagłówek

    species_data = {}  # Słownik do przechowywania danych dla każdego gatunku

    for data in reader:
        species = data[5]

        if species not in species_data:
            species_data[species] = []

        species_data[species].append(data)

characteristics_train = []
species_train = []

characteristics_test = []
species_test = []

for species, data_list in species_data.items():

    # Podziel dane dla każdego gatunku
    c_train, c_test, s_train, s_test = train_test_split(data_list, [species] * len(data_list), test_size=test_set, random_state=42)

    # Dodaj dane do zbiorów treningowych i testowych
    characteristics_train.extend(c_train)
    species_train.extend(s_train)

    characteristics_test.extend(c_test)
    species_test.extend(s_test)

# Wymieszaj dane treningowe
data_train = list(zip(characteristics_train, species_train))
shuffle(data_train)
characteristics_train, species_train = zip(*data_train)

# Wymieszaj dane testowe
data_test = list(zip(characteristics_test, species_test))
shuffle(data_test)
characteristics_test, species_test = zip(*data_test)
