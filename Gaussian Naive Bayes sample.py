#coding: utf-8
'''
有些特征可能是连续型变量，比如说人的身高，物体的长度，这些特征可以转换成离散型的值，比如如果身高在160cm以下，特征值为1；
在160cm和170cm之间，特征值为2；在170cm之上，特征值为3。也可以这样转换，将身高转换为3个特征，分别是f1、f2、f3，
如果身高是160cm以下，这三个特征的值分别是1、0、0，若身高在170cm之上，这三个特征的值分别是0、0、1。
不过这些方式都不够细腻，高斯模型可以解决这个问题。高斯模型假设这些一个特征的所有属于某个类别的观测值符合高斯分布
'''

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np

#example 1
iris = datasets.load_iris()
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

#example 2
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
#训练模型
clf.fit(X, Y)
#打印预测值
print(clf.predict([[-0.8, -1]]))
clf_pf = GaussianNB()
#训练模型with sample_weight=np.unique(Y)
clf_pf.partial_fit(X, Y, np.unique(Y))
#打印预测值
print(clf_pf.predict([[-0.8, -1]]))
#打印预测为不同类的概率,从小到大排列，下面的结果为：[为1的概率，为2的概率]。
print clf_pf.predict_proba([[-0.8, -1]])
print clf_pf.predict_log_proba([[-0.8, -1]])
#打印正确率
print '正确率：',clf_pf.score([[0.8, 1], [-0.8, -1]], [1, 2])