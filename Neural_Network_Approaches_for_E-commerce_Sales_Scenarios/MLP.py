import numpy as np
import math
from scipy.special import expit
import matplotlib.pyplot as plt

# 定义并初始化神经网络
class Nerual_Network(object):
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, epoch):
        """
        :param inputnodes: 输入层结点数
        :param hiddennodes: 隐藏层结点数
        :param outputnodes: 输出层结点数
        :param learningrate: 学习率
        :param epoch: 迭代次数
        """
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        # 输入层与隐藏层权重矩阵初始化
        self.w1 = np.random.randn(self.hiddennodes, self.inputnodes) * 0.01
        # 隐藏层与输出层权重矩阵初始化
        self.w2 = np.random.randn(self.outputnodes, self.hiddennodes) * 0.01
        # 构建第一层常量矩阵12 by 1 matrix
        self.b1 = np.zeros((12, 1))
        # 构建第二层常量矩阵 4 by 1 matrix
        self.b2 = np.zeros((4, 1))
        # 定义迭代次数
        self.epoch = epoch
        # 定义损失
        self.losses = []
    # 定义激活函数
    def softmax(self, x):
        """
        :param x: 输入数据
        :return:返回softmax激活函数值
        """
        return expit(x)

    def tanh(self, x):
        """
        :param x: 输入数据
        :return: 返回tanh激活函数值
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # 定义前向传播函数
    def forward_propagation(self, input_data, weight_matrix, b):
        """
        :param input_data: 输入数据
        :param weight_matrix: 权重矩阵
        :param b: 偏置
        :return: 激活函数后输出的活性值
        """
        z = np.add(np.dot(weight_matrix, input_data), b)
        return z, self.softmax(z)

    # 定义反向传播函数
    def back_propagation(self, a, z, da, weight_matrix, b):
        dz = da * (z * (1 - z))
        weight_matrix -= self.learningrate * np.dot(dz, a.T) / 60000
        b -= self.learningrate * np.sum(dz, axis=1, keepdims=True) / 60000
        da_n = np.dot(weight_matrix.T, da)
        return da_n

    #模型训练
    def train(self, input_data, label_data):
        m = input_data.shape[1]  # 样本数量
        for item in range(self.epoch):
            # 损失
            cost = 0

            for i in range(m):
                # 前向传播
                z1, a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
                z2, a2 = self.forward_propagation(a1, self.w2, self.b2)

                # 计算损失（交叉熵）
                cost += -np.sum(label_data[i] * np.log(a2) + (1 - label_data[i]) * np.log(1 - a2)) / m

                # 计算da[2]
                dz2 = a2 - label_data[i].reshape(-1, 1)
                dz1 = np.dot(self.w2.T, dz2) * a1 * (1.0 - a1)

                # 反向传播过程
                self.w2 -= self.learningrate * np.dot(dz2, a1.T)
                self.b2 -= self.learningrate * dz2

                self.w1 -= self.learningrate * np.dot(dz1, (input_data[:, i].reshape(-1, 1)).T)
                self.b1 -= self.learningrate * dz1

            self.losses.append(cost)

    # 预测函数
    def predict(self, input_data, label):
        m = input_data.shape[1]  # 样本数量
        precision = 0
        for i in range(m):
            z1, a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
            z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
            if np.argmax(a2) == label[i]:
                precision += 1
        print("准确率：%.4f" % (100 * precision / m) + "%")


    # 损失函数可视化
    def visualize_loss(self):
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, self.epoch + 1), self.losses)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('损失函数随迭代次数的变化')
        plt.grid(True)
        plt.show()
if __name__ == '__main__':
    print("这是神经网络模块")