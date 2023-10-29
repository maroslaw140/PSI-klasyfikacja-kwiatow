import csv

with open('Iris.csv', 'r') as plik:
    czytacz = csv.reader(plik)

    for dane in czytacz:
        print(dane[0])
