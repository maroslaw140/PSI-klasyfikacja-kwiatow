import pandas
import seaborn
from matplotlib import pyplot as plt

data_frame = pandas.read_csv('Iris.csv', usecols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# Wykres punktowy dla długości i szerokości płatka
def RysujWykresPlatka():
    seaborn.scatterplot(x="PetalLengthCm", y="PetalWidthCm", hue="Species", data=data_frame)
    plt.title("Wykres punktowy dla długości i szerokości płatka")
    plt.savefig('./wykresy/Wykres-punktowy-platka.png')
    plt.show()

# Wykres punktowy dla długości i szerokości działki kielicha
def RysujWykresKielicha():
    seaborn.scatterplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=data_frame)
    plt.title("Wykres punktowy dla długości i szerokości działki kielicha")
    plt.savefig('./wykresy/Wykres-punktowy-kielicha.png')
    plt.show()

# Tworzenie histogramu długości działki kielicha dla każdego gatunku
def RysujHistogramDlugosciKielicha():
    for species in data_frame['Species'].unique():
        species_data = data_frame[data_frame['Species'] == species]

        seaborn.histplot(species_data['SepalLengthCm'], kde=True, label=species)

    plt.title("Histogram długości działki kielicha dla różnych gatunków")
    plt.xlabel("Długość działki kielicha (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.savefig('./wykresy/Histogram-dlugosci-kielicha.png')
    plt.show()
# KONIEC Tworzenie histogramu długości działki kielicha dla każdego gatunku


# Tworzenie histogramu szerokości działki kielicha dla każdego gatunku
def RysujHistogramSzerokosciKielicha():
    for species in data_frame['Species'].unique():
        species_data = data_frame[data_frame['Species'] == species]

        seaborn.histplot(species_data['SepalWidthCm'], kde=True, label=species)

    plt.title("Histogram szerokości działki kielicha dla różnych gatunków")
    plt.xlabel("Szerokość działki kielicha (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.savefig('./wykresy/Histogram-szerokosci-kielicha.png')
    plt.show()
# KONIEC Tworzenie histogramu szerokości działki kielicha dla każdego gatunku


# Tworzenie histogramu długości płatków dla każdego gatunku
def RysujHistogramDlugosciPlatka():
    for species in data_frame['Species'].unique():
        species_data = data_frame[data_frame['Species'] == species]

        seaborn.histplot(species_data['PetalLengthCm'], kde=True, label=species)

    plt.title("Histogram długości płatka dla różnych gatunków")
    plt.xlabel("Długość płatka (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.savefig('./wykresy/Histogram-dlugosci-platka.png')
    plt.show()
# KONIEC Tworzenie histogramu długości płatków dla każdego gatunku

# Tworzenie histogramu szerokości płatków dla każdego gatunku
def RysujHistogramSzerokosciPlatka():
    for species in data_frame['Species'].unique():
        species_data = data_frame[data_frame['Species'] == species]

        seaborn.histplot(species_data['PetalWidthCm'], kde=True, label=species)

    plt.title("Histogram szerokości płatka dla różnych gatunków")
    plt.xlabel("Szerokość płatka (cm)")
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.savefig('./wykresy/Histogram-szerokosci-platka.png')
    plt.show()
# KONIEC Tworzenie histogramu szerokości płatków dla każdego gatunku

# Narysuj krzywą ROC
def RysujKrzywaROC(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc='lower right')
    plt.show()

