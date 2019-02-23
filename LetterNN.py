import csv
import time

import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


INPUT_SIZE = 9 * 11
LABEL_CNT = 26

REAL_LABEL_LISTS = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]


class LetterNN:
    inputList_x = ''
    inputList_y = ''
    batchSize = 26

    def __init__(self):
        x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        y_ = tf.placeholder(tf.float32, [None, LABEL_CNT])  # right answer
        # one
        W_1 = weight_variable([INPUT_SIZE, INPUT_SIZE])
        b_1 = bias_variable([INPUT_SIZE])

        W_2 = weight_variable([INPUT_SIZE, INPUT_SIZE])
        b_2 = bias_variable([INPUT_SIZE])

        W_3 = weight_variable([INPUT_SIZE, LABEL_CNT])
        b_3 = bias_variable([LABEL_CNT])

        result_1 = tf.sigmoid(tf.matmul(x, W_1) + b_1)

        result_2 = tf.sigmoid(tf.matmul(result_1, W_2) + b_2)

        result_3 = tf.sigmoid(tf.matmul(result_2, W_3) + b_3)
        y = tf.nn.softmax(result_3)

        # y = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

        # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

        self.x = x
        self.y = y
        self.y_ = y_
        self.init = init
        self.train_step = train_step
        self.accuracy = accuracy

    def get_train_data(self, inputList_x, inputList_y):
        batchSize = self.batchSize
        totalNum = len(inputList_x)
        batchSizeNum = totalNum // batchSize
        inputList_x_batch = []
        inputList_y_batch = []
        for i in range(batchSizeNum):
            tempStart = batchSize * i
            tempEnd = batchSize * (i + 1)
            inputList_x_batch.append(inputList_x[tempStart:tempEnd])
            inputList_y_batch.append(inputList_y[tempStart:tempEnd])

        if totalNum % batchSize != 0:
            tempStart = batchSize * batchSizeNum
            inputList_x_batch.append(inputList_x[tempStart:])
            inputList_y_batch.append(inputList_y[tempStart:])
        return inputList_x_batch, inputList_y_batch

    def train(self):
        init = self.init
        train_step = self.train_step
        accuracy = self.accuracy
        x = self.x
        y_ = self.y_
        inputList_x = self.inputList_x
        inputList_y = self.inputList_y

        inputList_x_batch, inputList_y_batch = self.get_train_data(inputList_x, inputList_y)

        save_step = 10
        max_step = 10000  # 最大迭代次数
        step = 0
        saver = tf.train.Saver()  # 用来保存模型的
        epoch = 5

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

            sess.run(init)
            accList = []
            stepList = []
            differentPath = "ann0508/HW3-Test2-toT-different_T.csv"
            noisePath = "ann0508/HW3-Test2-toT-noise_T.csv"
            incompletePath = 'ann0508/HW3-Test1-toT-incomplete_T.csv'
            differentList = []
            noiseList = []
            incompleteList = []

            differentListAcc = []
            noiseListAcc = []
            incompleteListAcc = []

            while step < max_step:
                for j in range(len(inputList_x_batch)):
                    step += 1

                    input_x = inputList_x_batch[j]

                    input_y = inputList_y_batch[j]
                    # 训练。训练时dropout层要有值。
                    sess.run(train_step, feed_dict={x: input_x, y_: input_y})

                    if step % epoch == 0:  # step
                        # 输出当前batch的精度。预测时keep的取值均为1
                        acc = sess.run(accuracy, feed_dict={x: input_x, y_: input_y})
                        print('%s accuracy is %.2f' % (step, acc))

                    if step % 50 == 0:  # step
                        stepList.append(step)

                        acc = sess.run(accuracy, feed_dict={x: input_x, y_: input_y})
                        acc = acc*100
                        accList.append(acc)

                        input_x_test, input_y_test = self.readTestCsv(differentPath)
                        acc = sess.run(accuracy, feed_dict={x: input_x_test, y_: input_y_test})
                        acc = acc*100
                        differentList.append(acc)

                        input_x_test, input_y_test = self.readTestCsv(noisePath)
                        acc = sess.run(accuracy, feed_dict={x: input_x_test, y_: input_y_test})
                        acc = acc * 100
                        noiseList.append(acc)

                        input_x_test, input_y_test = self.readTestCsv(incompletePath)
                        acc = sess.run(accuracy, feed_dict={x: input_x_test, y_: input_y_test})
                        acc = acc * 100
                        incompleteList.append(acc)

                    if step % 1000 == 0:  # step
                        input_x_test, input_y_test = self.readTestCsv(differentPath)
                        acc = sess.run(accuracy, feed_dict={x: input_x_test, y_: input_y_test})
                        acc = acc * 10000
                        acc = int(acc)
                        acc = acc / 100
                        differentListAcc.append(acc)

                        input_x_test, input_y_test = self.readTestCsv(noisePath)
                        acc = sess.run(accuracy, feed_dict={x: input_x_test, y_: input_y_test})
                        acc = acc * 10000
                        acc = int(acc)
                        acc = acc / 100
                        noiseListAcc.append(acc)

                        input_x_test, input_y_test = self.readTestCsv(incompletePath)
                        acc = sess.run(accuracy, feed_dict={x: input_x_test, y_: input_y_test})
                        acc = acc * 10000
                        acc = int(acc)
                        acc = acc / 100
                        incompleteListAcc.append(acc)
                        # if step % save_step == 0:
                    #     # 保存当前模型
                    #     save_path = saver.save(sess, './trainModel/graph.ckpt', global_step=step)
                    #     print("save graph to %s" % save_path)

            saver.save(sess, './trainModel/graph.ckpt', global_step=step)

            red_patch = mpatches.Patch(color='blue', label='train')
            red_patch2 = mpatches.Patch(color='red', label='different')
            red_patch3 = mpatches.Patch(color='orange', label='noise')
            red_patch4 = mpatches.Patch(color='green', label='incomplete')
            plt.legend(handles=[red_patch, red_patch2, red_patch3, red_patch4])

            plt.plot(stepList, accList, color='b')
            plt.plot(stepList, differentList, color='r')
            plt.plot(stepList, noiseList, color='orange')
            plt.plot(stepList, incompleteList, color='green')
            plt.ylabel('accuracy')
            plt.xlabel('train step')
            plt.show()
            print('='*40)
            print(differentListAcc)
            print('='*40)
            print(noiseListAcc)
            print('='*40)
            print(incompleteListAcc)
        print("training is done")


    def getPred(self, testinput_x):
        x = self.x
        y = self.y

        labelLists = REAL_LABEL_LISTS
        sess = tf.Session()

        saver = tf.train.Saver()
        # tf.reset_default_graph()
        saver.restore(sess, tf.train.latest_checkpoint('trainModel'))
        y = sess.run(y, feed_dict={x: testinput_x})
        yNum = y.argmax()
        return labelLists[yNum]

    def getPredAccuracy(self, input_x, input_y):
        x = self.x
        y_ = self.y_
        accuracy = self.accuracy

        sess = tf.Session()
        saver = tf.train.Saver()
        # tf.reset_default_graph()
        saver.restore(sess, tf.train.latest_checkpoint('trainModel'))

        acc = sess.run(accuracy, feed_dict={x: input_x, y_: input_y})
        acc = acc*100
        return acc


    def readTestCsv(self, csvName):
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

        labels = np.zeros((len(labelLists), len(REAL_LABEL_LISTS)))

        for index, label in enumerate(labels):
            col = REAL_LABEL_LISTS.index(labelLists[index])
            labels[index, col] = 1
        input_y = labels
        return input_x, input_y

if __name__ == '__main__':
    df = pd.read_csv('all.csv', header=None)
    dataList = df.iloc[:, :].values
    # 真實答案
    labelLists = [
        'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',

        'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
    ]
    # 假設

    labels = np.zeros((len(labelLists), len(REAL_LABEL_LISTS)))

    for index, label in enumerate(labels):
        col = REAL_LABEL_LISTS.index(labelLists[index])
        labels[index, col] = 1
    print(len(labels))
    letterNN = LetterNN()
    letterNN.inputList_x = dataList
    letterNN.inputList_y = labels

    start_time = time.time()
    letterNN.train()
    # letterNN.test()
    print("--- %s seconds ---" % (time.time() - start_time))
