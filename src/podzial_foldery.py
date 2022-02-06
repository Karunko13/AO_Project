
import csv
import os
import shutil

with open('dog.csv', mode='r') as inp:
    reader = csv.reader(inp)
    dictBreed = {rows[0]: rows[1] for rows in reader}

with open('dog.csv', mode='r') as inp:
    reader = csv.reader(inp)
    dictImageNumber = {rows[2]: rows[1] for rows in reader}

keyList = list(dictBreed.keys())
valueList = list(dictBreed.values())

keyList2 = list(dictImageNumber.keys())  # numery obrazkow
valueList2 = list(dictImageNumber.values())  # nazwa rasy
parent_dir = "" # folder glowny bazy zdjec

y = len(dictImageNumber)

for i in range(1, y):
    number = keyList2[i]  # aktualnie przechowywany numer obrazka
    name = dictImageNumber[number]  # aktualnie przechowywana nazwa
    path = os.path.join(parent_dir, name)
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    source_path = "" + number # sciezka do oryginalnej bazy zdjec 
    destination_path = ""+name # sciezka do nowej bazy zdjec 
    try:
        x = (shutil.move(source_path,destination_path))
    except OSError as error:
        print(error)
