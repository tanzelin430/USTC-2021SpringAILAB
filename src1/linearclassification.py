from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import numpy as np



# 实现线性回归的类
class LinearClassification:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.000005, Lambda=0.001, epochs=1000):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs

    def fit(self, train_features, train_labels):
        ''''
        需要你实现的部分
        '''
        first_c = np.ones(train_features.shape[0])
        first_c = first_c.reshape(-1, 1)
        first_c = np.matrix(first_c)
        x = np.c_[first_c, train_features]
        print(x)
        print(x.shape)
        y = np.matrix(train_labels)
        print(y.shape)
        w = np.zeros([x.shape[1], 1])
        print(w.shape)
        fit_ep = self.epochs
        while fit_ep > 0:
            grad = -2 * x.T @ y + 2 * x.T @ x @ w + 2 * self.Lambda * w
            w = w - self.lr * grad
            fit_ep = fit_ep - 1
        self.w = w

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''

    def predict(self, test_features):
        ''''
        需要你实现的部分
        '''
        first_c = np.ones(test_features.shape[0])
        first_c = first_c.reshape(-1, 1)
        first_c = np.matrix(first_c)
        x = np.c_[first_c, test_features]
        return np.around(x @ self.w)

def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
