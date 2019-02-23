import csv

from LetterNN import LetterNN

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
    i = 0
    totalNum = 0
    trueNum = 0
    falseNum = 0
    print('real, pred')
    for row in fileReader:
        result = letterNN.getPred([row])
        real = labelLists[i]
        if result == real:
            resultMSg = 'same'
            trueNum += 1
        else:
            resultMSg = 'not same'
            falseNum += 1

        print('{}, {}...{}'.format(real, result, resultMSg))
        i += 1
        totalNum += 1

    print('=' * 40)
    acc = trueNum / totalNum * 100.0
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
