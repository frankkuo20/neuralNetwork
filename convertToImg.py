import csv

import pandas as pd
import cv2

import numpy as np
if __name__ == '__main__':
    path = 'all_me.csv'
    saveFolder = 'outputImg/me'
    # saveFolder = 'ann0508Img/HW3-Test1-toT-incomplete_T'
    # path = "ann0508/HW3-Test2-toT-different_T.csv"
    # path = "ann0508/HW3-Test2-toT-noise_T.csv"
    # path = 'ann0508/HW3-Test1-toT-incomplete_T.csv'

    df = pd.read_csv(path, header=None)
    dataList = df.iloc[:, :].values
    labelLists = [
        'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',

        # 'A2', 'B2', 'C2', 'D2', 'E2',
        # 'F2', 'G2', 'H2', 'I2', 'J2',
        # 'K2', 'L2', 'M2', 'N2', 'O2',
        # 'P2', 'Q2', 'R2', 'S2', 'T2',
        # 'U2', 'V2', 'W2', 'X2', 'Y2', 'Z2',
    ]

    for data, label in zip(dataList, labelLists):
        temp = data.reshape((11, 9))
        # temp = 1 - temp
        # temp = temp*255
        temp = np.array(temp)
        temp = temp.astype(np.uint8)
        print(temp)

        # img = cv2.imread(temp, 0)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('messigray.png', img)
        print('{}/{}.png'.format(saveFolder, label))
        cv2.imwrite('{}/{}.png'.format(saveFolder, label), temp)

