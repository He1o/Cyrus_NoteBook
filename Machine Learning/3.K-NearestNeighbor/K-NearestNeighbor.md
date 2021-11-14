# k近邻算法

k近邻算法是一种基本分类与回归方法
## 1、历史背景
Evelyn Fix(1904-1965) 是一位数学家/统计学家，在伯克利攻读博士学位并继续在那里教授统计学。

Joseph Lawson Hodges Jr.(1922-2000)也是伯克利的一名统计学家，并从1944年开始参与与美国空军第二十空军（Twentieth Air Force of USAF）的统计合作。

这两位天才在1951年为美国空军制作的一份技术分析报告中相遇，在那里他们引入了一种非参数分类方法（判别分析）。他们从未正式发表过这篇论文，可能是因为所涉及的工作性质和保密性，特别是考虑到二战后不久的全球气氛。

接下来，Thomas Cover和Peter Hart在1967年证明了kNN分类的上限错误率

## 2、算法模型
kNN算法在某些条件下是一个通用的函数逼近器，但潜在的概念相对简单。kNN是一种监督学习算法，它在训练阶段简单地存储标记的训练样本。因此，kNN也被称为惰性学习算法，它对训练样本的处理推迟到做预测的时候才进行。  
假设训练数据集：
$$T=\{(\mathbf x_,y_1),(\mathbf x_2,y_2),...,(\mathbf x_N,y_N)\}$$
其中，${\displaystyle \mathbf x_i\in X\subseteq R^n}$ 表示实例的特征向量，$y\in Y$ 表示实例的类别。给定实例特征向量$\mathbf x$，输出所属的类 $y$ 。

具体过程：
1. 通过给定的距离度量，在训练集 $T$ 中找出与 $\mathbf x$ 最近邻的 $k$ 个点，涵盖这 $k$ 个点的 $\mathbf x$ 的领域 记作 $N_k(\mathbf x)$
2. 随后在 $N_k(\mathbf x)$ 中根据分类决策规则决定 $\mathbf x$ 的类别 $y$。

可以看出，kNN算法的主要三个要素分别为距离度量、$k$ 值和分类决策规则。
### 2.1 距离度量
距离度量有曼哈顿距离、欧式距离或更一般的闵式距离。

假设两个特征向量 $\mathbf{x}_i=(x_i^{(1)},x_i^{(2)},...,x_i^{(n)})^T$，$\mathbf{x}_j=(x_j^{(1)},x_j^{(2)},...,x_j^{(n)})^T$，$\mathbf{x}_i,\mathbf{x}_j$ 的 $L_p$距离定义为：
$$L_p(\mathbf{x}_i,\mathbf{x}_j)=\left(\sum_{l=1}^{n} |x_i^{(l)} - x_j^{(l)}|^p \right)^{\frac{1}{p}}$$

当 $p=1$ 时，称为曼哈顿距离，即
$$L_p(\mathbf{x}_i,\mathbf{x}_j)=\sum_{l=1}^{n} |x_i^{(l)} - x_j^{(l)}| $$

当 $p=2$ 时，称为欧式距离，即
$$L_p(\mathbf{x}_i,\mathbf{x}_j)=\left(\sum_{l=1}^{n} |x_i^{(l)} - x_j^{(l)}|^2 \right)^{\frac{1}{2}}$$

当 $p=\infty$，它是各个坐标距离的最大值，即
$$L_p(\mathbf{x}_i,\mathbf{x}_j)=\max_{l} |x_i^{(l)} - x_j^{(l)}| $$

>范数即为特征向量到原点的距离，表征自身的长度
### 2.2 $k$ 值的选择
如果选择较小的 $k$ 值，相当于用较小的领域中的训练数据进行预测，近似误差（approximation error）会减少，只有与输入实例较近的训练数据才会对预测结果产生影响。但缺点是估计误差（estimation error）会增大，预测结果对近邻的实例点非常敏感，容易发生过拟合。当 $k$ 为1时，称为最近邻算法，对于输入实例，将与其最近的数据点的类作为预测结果。

相反的，当使用较大的 $k$ 值时，意味着距离输入实例较远的训练数据也会对预测结果产生影响，使预测产生错误，容易发生欠拟合。当 $k$ 为N时，无论输入实例是什么，预测结果都将是训练数据中存在最多的类。

在应用中，$k$ 一般选取一个比较小的值，采用交叉验证法来选取最优的 $k$ 值。
 
### 2.3 决策规则
kNN算法在分类问题中决策规则往往是“多数表决”，即由输入实例的 $k$ 个近邻的训练实例中的多数类决定输入实例的类。事实上，多数表决（Majority vote）分为简单多数表决和特定多数表决，是要求满足一半数量以上或者特定数量，而不是占比最多的（Plurality vote），在二分类问题中两者没有区别，而在多分类问题中，不需要某一类投票数过半，超过N分之一就可以预测了。

在回归问题中，可使用“平均法”，即将这k个样本的实值输出标记的平均值作为预测结果，还可基于距离远近进行加权平均或加权投票，距离越近的样本权重越大。

## 3、模型优化

在考虑kNN算法时间复杂度之前，先看一下特征维度对算法的影响。

假设我们有100个训练实例均匀分布在 $(0,1]$ 区间， 它们的间隔为0.01单元。假设k值为3，就是找到查询点的三个最近邻，期望覆盖特征轴0.03的范围。当我们增加一个维度的时候，总体分布在区域为 $1$

kNN算法的时间复杂度是 $O(k*N*m)$ ，$N$ 是训练样本的数量， $m$ 是训练数据集的特征维度，由于 $N\ll m$ ，时间复杂度简化为 $O(k*N)$



http://www.atyun.com/37601.html
https://zhuanlan.zhihu.com/p/110066200
https://blog.csdn.net/sinat_30353259/article/details/80901746
http://www.scholarpedia.org/article/K-nearest_neighbor
https://zh.wikipedia.org/wiki/K-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95
https://zhuanlan.zhihu.com/p/45346117
https://sebastianraschka.com/pdf/lecture-notes/stat479fs18/02_knn_notes.pdf
https://zhuanlan.zhihu.com/p/23966698
https://zhuanlan.zhihu.com/p/45346117
https://zhuanlan.zhihu.com/p/53826008
https://www.cnblogs.com/eyeszjwang/articles/2429382.html