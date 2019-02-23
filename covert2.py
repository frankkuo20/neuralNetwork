import csv
import glob

if __name__ == '__main__':
    DATASET_PATH = 'datasetCsv_ii'
    DATASET_CSV_PATH = 'datasetCsv_ii'
    FILE_ALL_NAME = 'all_ii'

    originPaths = glob.glob('{}/*'.format(DATASET_CSV_PATH))
    fileAll = open('{}.csv'.format(FILE_ALL_NAME), 'w', newline='')
    fileAllWriter = csv.writer(fileAll)

    for originPath in originPaths:
        print(originPath)
        tempList = []
        file = open(originPath, 'r')
        fileCsv = csv.reader(file)
        for row in fileCsv:
            for col in row:
                tempList.append(str(col))

        fileAllWriter.writerows([tempList])

    fileAll.close()
