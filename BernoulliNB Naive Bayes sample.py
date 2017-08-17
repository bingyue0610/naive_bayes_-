#coding:utf-8
'''
伯努利模型中，对于一个样本来说，其特征用的是全局的特征。
也主要用户离散特征分类，和MultinomialNB的区别是：MultinomialNB以出现的次数为特征值，
BernoulliNB为二进制或布尔型特性
在伯努利模型中，每个特征的取值是布尔型的，即true和false，或者1和0。在文本分类中，就是一个特征有没有在一个文档中出现。
如果特征值xi值为1,那么
P(xi|yk)=P(xi=1|yk)
如果特征值xi值为0,那么
P(xi|yk)=1−P(xi=1|yk)
这意味着，“没有某个特征”也是一个特征。

可调参数：
alpha：浮点型，可选项，默认1.0，添加拉普拉修/Lidstone平滑参数
fit_prior：布尔型，可选项，默认True，表示是否学习先验概率，参数为False表示所有类标记具有相同的先验概率
class_prior：类似数组，数组大小为(n_classes,)，默认None，类先验概率
binarize：将数据特征二值化的阈值
'''

from sklearn.naive_bayes import BernoulliNB
import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
clf = BernoulliNB()
clf.fit(X, Y)
print(clf.predict(X[2:3]))