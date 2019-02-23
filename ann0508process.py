import numpy as np


if __name__ == '__main__':
    # path = 'ann0508/HW3-Test3-toT-different.csv'
    # path = 'ann0508/HW3-Test2-toT-noise.csv'
    path = 'ann0508/HW3-Test1-toT-incomplete.csv'
    # HW3 - Test1 - toT - incomplete.csv
    my_data = np.genfromtxt(path, delimiter=',')
    print(my_data.T)
    # savePath = "ann0508/HW3-Test2-toT-different_T.csv"
    # savePath = "ann0508/HW3-Test2-toT-noise_T.csv"
    savePath = 'ann0508/HW3-Test1-toT-incomplete_T.csv'
    np.savetxt(savePath, my_data.T, delimiter=",")
    # my_data.T
    # path = 'ann0508/HW3-Test3-toT-different.csv'