from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('Machine Learning/3.K-NearestNeighbor/source/data.csv')
X = data[['heat_value','heat_ratio','reroute_value']].values
Y = data[['target']].values.ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)# , random_state=22

from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

model1 = KNeighborsClassifier(n_neighbors=10)
model1.fit(X_train, Y_train)
score1 = model1.score(X_test, Y_test)

model2 = KNeighborsClassifier(n_neighbors=10, weights='distance')
model2.fit(X_train, Y_train)
score2 = model2.score(X_test, Y_test)

# model3 = RadiusNeighborsClassifier(n_neighbors=10, radius=200.0)
# model3.fit(X_train, Y_train)
# score3 = model3.score(X_test, Y_test)

print(score1, score2) #, score3

result1 = cross_val_score(model1, X, Y, cv=10)
result2 = cross_val_score(model2, X, Y, cv=10)
# result3 = cross_val_score(model3, X, Y, cv=10)

print(result1.mean(), result2.mean()) #, result3.mean()

predictions_binarized = model1.predict(X_test)

print('========计算召回率=======')
from sklearn.metrics import recall_score
print('Recall: %s' % recall_score(Y_test, predictions_binarized))

print('========计算F1分数=======')
from sklearn.metrics import f1_score
print('F1 score: %s' % f1_score(Y_test, predictions_binarized))

print('========生成综合报告=======')
from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions_binarized))












# #读取鸢尾花数据集
# # iris = load_iris()
# # x = iris.data
# # y = iris.target
# # k_range = range(1, 31)
# # k_error = []
# # #循环，取k=1到k=31，查看误差效果
# # for k in k_range:
# #     knn = KNeighborsClassifier(n_neighbors=k)
# #     #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
# #     scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
# #     print(scores)
# #     k_error.append(scores.mean())

# # #画图，x轴为k值，y值为误差值
# # plt.plot(k_range, k_error)
# # plt.xlabel('Value of K for KNN')
# # plt.ylabel('Error')
# # plt.show()


# # import matplotlib.pyplot as plt
# # import numpy as np
# # from matplotlib.colors import ListedColormap
# # from sklearn import neighbors, datasets

# # n_neighbors = 11

# # # 导入一些要玩的数据
# # # iris = datasets.load_iris()
# # # x = iris.data[:, :2]  # 我们只采用前两个feature,方便画图在二维平面显示
# # # y = iris.target
# # x = x[:, :2]  # 我们只采用前两个feature,方便画图在二维平面显示


# # h = .02  # 网格中的步长

# # # 创建彩色的图
# # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# # #weights是KNN模型中的一个参数，上述参数介绍中有介绍，这里绘制两种权重参数下KNN的效果图
# # for weights in ['uniform', 'distance']:
# #     # 创建了一个knn分类器的实例，并拟合数据。
# #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# #     clf.fit(x, y)

# #     # 绘制决策边界。为此，我们将为每个分配一个颜色
# #     # 来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
# #     x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
# #     y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
# #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# #                          np.arange(y_min, y_max, h))
# #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# #     # 将结果放入一个彩色图中
# #     Z = Z.reshape(xx.shape)
# #     plt.figure()
# #     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# #     # 绘制训练点
# #     plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
# #     plt.xlim(xx.min(), xx.max())
# #     plt.ylim(yy.min(), yy.max())
# #     plt.title("3-Class classification (k = %i, weights = '%s')"
# #               % (n_neighbors, weights))

# # plt.show()

# # ================================================
# # KNN模型分类，sklearn类库实现
# # （身高、体重）数据，预测
# # 2019-02-24
# # ================================================
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.neighbors import KNeighborsClassifier

# lb = LabelBinarizer()
# y_train_binarized = lb.fit_transform(y_train)
# print(y_train_binarized)  #1为男性


# K = 10
# clf = KNeighborsClassifier(n_neighbors=K)
# clf.fit(X_train, y_train_binarized.reshape(-1))
# prediction_binarized = clf.predict(np.array([155, 70]).reshape(1, -1))[0]
# predicted_label = lb.inverse_transform(prediction_binarized)
# print(predicted_label)
# # ================================================
# # KNN模型分类，sklearn类库实现
# # 测试集进行预测效果分析
# # 2019-02-24
# # ================================================

# X_test = np.array([
#     [168, 65],
#     [180, 96],
#     [160, 52],
#     [169, 67],
#     [178, 64],
#     [172, 59]
# ])
# y_test = ['male', 'male', 'female', 'female','male','female']
# y_test_binarized = lb.transform(y_test)
# print('Binarized labels: %s' % y_test_binarized.T[0])

# predictions_binarized = clf.predict(X_test)
# print('Binarized predictions: %s' % predictions_binarized)
# print('Predicted labels: %s' % lb.inverse_transform(predictions_binarized))
# print('=======计算正确率=====')
# from sklearn.metrics import accuracy_score
# print('Accuracy: %s' % accuracy_score(y_test_binarized, predictions_binarized))