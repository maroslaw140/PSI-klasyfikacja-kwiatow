import csv
import matplotlib.pyplot as plt
import seaborn
import pandas

from sklearn.model_selection import train_test_split
from random import shuffle


test_set = 0.2
trening_set = 1 - test_set
data = []

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

    data.extend(data_list)

# Wymieszaj dane treningowe
data_train = list(zip(characteristics_train, species_train))
shuffle(data_train)
characteristics_train, species_train = zip(*data_train)

# Wymieszaj dane testowe
data_test = list(zip(characteristics_test, species_test))
shuffle(data_test)
characteristics_test, species_test = zip(*data_test)

data_frame = pandas.read_csv('Iris.csv', usecols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# Wykres punktowy dla długości i szerokości płatka
seaborn.scatterplot(x="PetalLengthCm", y="PetalWidthCm", hue="Species", data=data_frame)
plt.title("Wykres punktowy dla długości i szerokości płatka")
# plt.savefig('./wykresy/Wykres-punktowy-platka.png')
# plt.show()

# Wykres punktowy dla długości i szerokości działki kielicha
seaborn.scatterplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=data_frame)
plt.title("Wykres punktowy dla długości i szerokości działki kielicha")
# plt.savefig('./wykresy/Wykres-punktowy-kielicha.png')
# plt.show()

# Tworzenie histogramu długości działki kielicha dla każdego gatunku
for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['SepalLengthCm'], kde=True, label=species)

    plt.title("Histogram długości działki kielicha dla różnych gatunków")
    plt.xlabel("Długość działki kielicha (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    # plt.savefig('./wykresy/Histogram-dlugosci-kielicha-' + species + '.png')
    # plt.show()

for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['SepalLengthCm'], kde=True, label=species)

plt.title("Histogram długości działki kielicha dla różnych gatunków")
plt.xlabel("Długość działki kielicha (cm)")
plt.ylabel("Liczba próbek")
plt.legend()
# plt.savefig('./wykresy/Histogram-dlugosci-kielicha.png')
# plt.show()
# KONIEC Tworzenie histogramu długości działki kielicha dla każdego gatunku


# Tworzenie histogramu szerokości działki kielicha dla każdego gatunku
for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['SepalWidthCm'], kde=True, label=species)

    plt.title("Histogram szerokości działki kielicha dla różnych gatunków")
    plt.xlabel("Szerokość działki kielicha (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    # plt.savefig('./wykresy/Histogram-szerokosci-kielicha-' + species + '.png')
    # plt.show()

for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['SepalWidthCm'], kde=True, label=species)

plt.title("Histogram długości działki kielicha dla różnych gatunków")
plt.xlabel("Szerokość działki kielicha (cm)")
plt.ylabel("Liczba próbek")
plt.legend()
# plt.savefig('./wykresy/Histogram-szerokosci-kielicha.png')
# plt.show()
# KONIEC Tworzenie histogramu szerokości działki kielicha dla każdego gatunku


# Tworzenie histogramu długości płatków dla każdego gatunku
for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['PetalLengthCm'], kde=True, label=species)

    plt.title("Histogram długości płatka dla różnych gatunków")
    plt.xlabel("Długość płatka (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    # plt.savefig('./wykresy/Histogram-dlugosci-platka-' + species + '.png')
    # plt.show()

for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['PetalLengthCm'], kde=True, label=species)

plt.title("Histogram długości płatka dla różnych gatunków")
plt.xlabel("Długość płatka (cm)")
plt.ylabel("Liczba próbek")
plt.legend()
# plt.savefig('./wykresy/Histogram-dlugosci-platka.png')
# plt.show()
# KONIEC Tworzenie histogramu długości płatków dla każdego gatunku

# Tworzenie histogramu szerokości płatków dla każdego gatunku
for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['PetalWidthCm'], kde=True, label=species)

    plt.title("Histogram szerokości płatka dla różnych gatunków")
    plt.xlabel("Szerokość płatka (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    # plt.savefig('./wykresy/Histogram-szerokosci-platka-'+species+'.png')
    # plt.show()

for species in data_frame['Species'].unique():
    species_data = data_frame[data_frame['Species'] == species]

    seaborn.histplot(species_data['PetalWidthCm'], kde=True, label=species)

plt.title("Histogram szerokości płatka dla różnych gatunków")
plt.xlabel("Szerokość płatka (cm)")
plt.ylabel("Liczba próbek")
plt.legend()
# plt.savefig('./wykresy/Histogram-szerokosci-platka.png')
# plt.show()
# KONIEC Tworzenie histogramu szerokości płatków dla każdego gatunku