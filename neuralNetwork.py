from decimal import Decimal

class Neural:
    def __init__(self, weightList, bias, learningRate):
        self.weightList = weightList  # 初始權重
        self.bias = bias
        self.learningRate = learningRate
        self.reRun = False

    def passFunc(self, xList, t):
        '''

        :param xList: [x1, x2]
        :param t: 標準答案
        :return:
        '''
        for index, x in enumerate(xList):
            print('x{} = {}, '.format(str(index+1), x), end='')
        print('t = {}'.format(t))
        # w11 * x1 + w21 * x2 - bias
        netY = 0
        for i in range(len(xList)):
            netY += xList[i] * self.weightList[i]
        netY -= self.bias

        # if net y >0 then 1
        # if net y <= 0 then 0
        y = 0
        if netY > 0:
            y = 1

        # 標準答案
        loss = t - y
        if loss == 0:
            print('通過')
            print('============================================')

            return
        else:  # loss != 0
            deltaWeightList = []
            for x in xList:
                dw = learningRate * loss * x
                deltaWeightList.append(dw)
            deltaBias = -1 * learningRate * loss

            self.weightList = [round(w + dw, 3) for w, dw in zip(self.weightList, deltaWeightList)]

            self.bias += deltaBias
            self.bias = round(self.bias, 3)

            for index, weight in enumerate(self.weightList):
                print('w{} = {}, '.format(str(index+1), weight), end='')
            print('bias = {}'.format(self.bias))
            print('-再跑一次')
            self.passFunc(xList, t)

            self.reRun = True


    def getResult(self):
        print()
        print('結果為')
        for index, weight in enumerate(self.weightList):
            print('w{} = {}, '.format(str(index+1), weight), end='')
        print('bias = {}'.format(self.bias))


if __name__ == '__main__':
    # 第一題
    weightList = [1.4, 1.5]  # 初始權重
    bias = 1.0
    learningRate = 0.2
    xLists = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 輸入值
    tList = [0, 0, 0, 1]  # 標準答案

    # 第二題
    weightList = [0.2, 0.5]  # 初始權重
    bias = 1.0
    learningRate = 0.2
    xLists = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 輸入值
    tList = [0, 1, 1, 1]  # 標準答案


    # 第三題
    weightList = [0.2, 0.5]  # 初始權重
    bias = 1.0
    learningRate = 0.2
    xLists = [[0, 0], [0, 1], [0, 1], [1, 1]]  # 輸入值
    tList = [0, 1, 1, 0]  # 標準答案


    #  or opera
    # weightList = [1, 0.5]  # 初始權重
    # bias = 0.5
    # learningRate = 0.1
    # xLists = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 輸入值
    # tList = [0, 1, 1, 1]  # 標準答案

    # and opera
    # weightList = [0.5, 0.5]  # 初始權重
    # bias = 1
    # learningRate = 0.1
    # xLists = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 輸入值
    # tList = [0, 0, 0, 1]  # 標準答案


    neural = Neural(weightList, bias, learningRate)

    neural.reRun = True

    while neural.reRun:
        neural.reRun = False
        print()
        print('===============重新跑一次====================')
        for index, xList in enumerate(xLists):
            neural.passFunc(xList, tList[index])

    neural.getResult()

