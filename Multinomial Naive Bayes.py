#coding: utf-8
'''
该模型常用于文本分类，特征是单词，值是单词的出现次数。
P(xi|yk)=Nykxi+αNyk+αn
其中，Nykxi是类别yk下特征xi出现的总次数；Nyk是类别yk下所有特征出现的总次数。对应到文本分类里，
如果单词word在一篇分类为label1的文档中出现了5次，那么Nlabel1,word的值会增加5。如果是去除了重复单词的，
那么Nlabel1,word的值会增加1。n是特征的数量，在文本分类中就是去重后的所有单词的数量。α的取值范围是[0,1]，
比较常见的是取值为1。
待预测样本中的特征xi在训练时可能没有出现，如果没有出现，则Nykxi值为0，如果直接拿来计算该样本属于某个分类的概率，
结果都将是0。在分子中加入α，在分母中加入αn可以解决这个问题。

可调整参数
alpha：浮点型，可选项，默认1.0，添加拉普拉修/Lidstone平滑参数
fit_prior：布尔型，可选项，默认True，表示是否学习先验概率，参数为False表示所有类标记具有相同的先验概率
class_prior：类似数组，数组大小为(n_classes,)，默认None，类先验概率
'''

import numpy as np
from sklearn.naive_bayes import MultinomialNB
#设X为范围[0,4]，的6*100矩阵
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
clf = MultinomialNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))