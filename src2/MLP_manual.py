import torch
import math
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MLP_manual:
    def __init__(self, input_layer_length, hidden_layer_1_length, hidden_layer_2_length, output_layer_length):
        # initialization
        self.input_layer_length = input_layer_length
        self.hidden_layer_1_length = hidden_layer_1_length
        self.hidden_layer_2_length = hidden_layer_2_length
        self.output_layer_length = output_layer_length
        #W random matrix
        self.w1 = torch.rand(hidden_layer_1_length, self.input_layer_length).requires_grad_(True)
        self.w2 = torch.rand(hidden_layer_2_length, self.hidden_layer_1_length).requires_grad_(True)
        self.w3 = torch.rand(output_layer_length, self.hidden_layer_2_length).requires_grad_(True)
        #偏移矩阵 初始为0
        self.b1 = torch.tensor([[0.0] * self.hidden_layer_1_length]).T.requires_grad_(True)
        self.b2 = torch.tensor([[0.0] * self.hidden_layer_2_length]).T.requires_grad_(True)
        self.b3 = torch.tensor([[0.0] * self.output_layer_length]).T.requires_grad_(True)

    def layer_IO(self, train_data):
        self.i_l = torch.tensor([train_data]).T.float()
        self.l1 = self.w1 @ self.i_l + self.b1
        for i in range(self.hidden_layer_1_length):
            self.l1[i, 0] = sigmoid(self.l1[i, 0])
        self.l2 = self.w2 @ self.l1 + self.b2
        for i in range(self.hidden_layer_2_length):
            self.l2[i, 0] = sigmoid(self.l2[i, 0])
        self.o_l = self.w3 @ self.l2 + self.b3
        #s3
        s3 = 0
        for i in range(self.output_layer_length):
            self.o_l[i, 0] = math.exp(self.o_l[i, 0])
            s3 += self.o_l[i, 0]
        self.o_l = self.o_l/s3

    def mat_update(self, train_label, lr):
        loss = 0.0
        for i in range(self.output_layer_length):
            if i == train_label:
                loss = -math.log(self.o_l[i, 0])
        b3grad = [0]*self.output_layer_length
        for i in range(self.output_layer_length):
            if i == train_label:
                b3grad[i] = self.o_l[i, 0] - 1
            else:
                b3grad[i] = self.o_l[i, 0]
        b3grad = torch.tensor([b3grad]).T
        w3grad = b3grad@self.l2.T
        b2grad = self.w3.T@b3grad
        for i in range(self.hidden_layer_2_length):
            b2grad[i, 0] = b2grad[i, 0]*d_sigmoid(self.l2[i, 0])
        w2grad = b2grad*self.l1.T
        b1grad = self.w2.T@b2grad
        for i in range(self.hidden_layer_1_length):
            b1grad[i, 0] = b1grad[i, 0]*d_sigmoid(self.l1[i, 0])
        w1grad = b1grad*self.i_l.T
        #update
        self.b1 = self.b1-lr*b1grad
        self.b2 = self.b2-lr*b2grad
        self.b3 = self.b3-lr*b3grad
        self.w1 = self.w1-lr*w1grad
        self.w2 = self.w2-lr*w2grad
        self.w3 = self.w3-lr*w3grad
        return loss

    def fit(self, train_data, train_label, train_times, lr):
        err = []
        #对每一条数据进行迭代
        for i in range(train_times):
            tmp_err = 0.0
            for j in range(len(train_label)):
                self.layer_IO(train_data[j])
                tmp_err += self.mat_update(train_label[j], lr)
            err.append(tmp_err/len(train_label))
        plt.plot(err)
        plt.show()

def main():
    mmlp = MLP_manual(5, 4, 4, 3)
    train_data = np.random.rand(300, 5)
    train_label = np.random.randint(0, 3, (300, 1))
    mmlp.fit(train_data, train_label, 300, 0.005)

main()
