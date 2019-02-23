import csv
import glob
import re

import cv2


def readFile(path):
    # 0 is black
    # 255 is white
    img = cv2.imread(path, 0)

    thresh = 127
    im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw = cv2.subtract(255, im_bw)
    im_bw2 = cv2.divide(255, im_bw)
    # print(im_bw2)
    # cv2.imshow('image', im_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    subPath = re.findall('\w.png', path)[0]
    subPath = subPath.split('.')[0]
    newPath = '{}/{}.csv'.format(DATASET_CSV_PATH, subPath)
    file = open(newPath, 'w', newline='')
    csvWriter = csv.writer(file)
    for line in im_bw2:
        csvWriter.writerows([[str(i) for i in line]])
    return im_bw2

DATASET_PATH = 'dataset'
DATASET_CSV_PATH = 'datasetCsv'
FILE_ALL_NAME = 'all'

# DATASET_PATH = 'dataset_test'
# DATASET_CSV_PATH = 'datasetCsv_test'
# FILE_ALL_NAME = 'all_test'


if __name__ == '__main__':
    originPaths = glob.glob('{}/*'.format(DATASET_PATH))

    fileAll = open('{}.csv'.format(FILE_ALL_NAME), 'w', newline='')
    fileAllWriter = csv.writer(fileAll)

    for originPath in originPaths:
        originImg = readFile(originPath)
        tempList = []
        for row in originImg:
            for col in row:
                tempList.append(str(col))

        fileAllWriter.writerows([tempList])

    fileAll.close()

