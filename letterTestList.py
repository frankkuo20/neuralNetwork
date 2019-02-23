import csv
import numpy as np
from LetterNN import LetterNN, REAL_LABEL_LISTS


def getCSVResult(csvName):
    print('=' * 40)
    print(csvName)
    file = open(csvName, 'r', newline='')
    fileReader = csv.reader(file)
    labelLists = [
        'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
    ]


    input_x = []
    for row in fileReader:
        temp = []
        for col in row:
            temp.append(float(col))
        input_x.append(temp)
    input_x = np.array(input_x)
    # input_x = input_x.astype(np.uint8)
    # input_x = input_x.reshape()



    labels = np.zeros((len(labelLists), len(REAL_LABEL_LISTS)))

    for index, label in enumerate(labels):
        col = REAL_LABEL_LISTS.index(labelLists[index])
        labels[index, col] = 1
    input_y = labels

    acc = letterNN.getPredAccuracy(input_x, input_y)
    print(acc)


if __name__ == '__main__':

    letterNN = LetterNN()
    # csvName = 'all_test.csv'

    savePath1 = "ann0508/HW3-Test2-toT-different_T.csv"
    savePath2 = "ann0508/HW3-Test2-toT-noise_T.csv"
    savePath3 = 'ann0508/HW3-Test1-toT-incomplete_T.csv'

    savePathLists = [
        savePath1,
        savePath2,
        savePath3
    ]
    for savePath in savePathLists:
        getCSVResult(savePath)
    # csvName = 'all_ii.csv'
