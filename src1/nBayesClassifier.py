import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def cal_for_mean_and_sd(self, feature):
        mean = np.average(feature)
        sd = np.sqrt(np.var(feature))
        return (mean, sd)

    def N_distribution(self, mean, sd, x):
        return (1 / (math.sqrt(2 * math.pi) * sd)) * math.exp(-0.5 * (math.pow(x - mean, 2)) / (math.pow(sd, 2)))

    def fit(self, traindata, trainlabel, featuretype):
        '''
        需要你实现的部分
        '''
        pc_count = np.array([0, 0, 0, 0])
        pc_distinct_count = np.zeros([4, 4])
        print(pc_distinct_count)
        pc_consis_count = {}
        pc_consistent_count = {}
        for i in range(1, 4):
            for j in range(1, 8):
                pc_consistent_count[(i, j)] = 0
                pc_consis_count[(i,j)] = 0
        for i in range(traindata.shape[0]):
            pc_count[trainlabel[i]] += 1
            # 所属分类与数据一一对应统计
            pc_distinct_count[trainlabel[i], int(traindata[i][0])] += 1
            print(4)
            for j in range(1, 8):
                print(trainlabel[i])
                print(2333)
                print(pc_consis_count[(int(trainlabel[i]),j)])
                if pc_consis_count[(int(trainlabel[i]),j)] == 0:
                    pc_consistent_count[(int(trainlabel[i]),j)] = np.array(traindata[i][j])
                    pc_consis_count[(int(trainlabel[i]), j)] += 1

                else: pc_consistent_count[(int(trainlabel[i]), j)] = np.append(pc_consistent_count[(int(trainlabel[i]), j)],
                                                                    traindata[i][j])

        # calculating PC
        for i in range(1, 4):
            self.Pc[i] = pc_count[i] / (pc_count[1] + pc_count[2] + pc_count[3]);
        for i in range(1, 4):
            for j in range(0, 8):
                if j == 0:
                    for k in range(1, 4):
                        self.Pxc[(i, 0, k)] = (pc_distinct_count[i, k]) / pc_count[i]
                # 第i类第j个属性
                else:
                    self.Pxc[(i, j)] = self.cal_for_mean_and_sd(pc_consistent_count[(i, j)])

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        '''
        需要你实现的部分
        '''
        result = []
        for i in range(features.shape[0]):
            max_probability_for_i = 0
            predict_category = 0
            # 不同的类
            for j in range(1, 4):
                # 离散属性
                probabiity = self.Pc[j] * self.Pxc[j, 0, features[i][0]]
                for k in range(1, 8):
                    # 连续属性
                    (mean, sd) = self.Pxc[(j, k)]
                    probabiity *= self.N_distribution(mean, sd, features[i][k])

                if probabiity > max_probability_for_i:
                    max_probability_for_i = probabiity
                    predict_category = j

            result.append(predict_category)
        result = np.array(result).reshape(features.shape[0], 1)
        return result


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()