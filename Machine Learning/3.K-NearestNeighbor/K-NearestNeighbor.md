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


http://www.atyun.com/37601.html
https://zhuanlan.zhihu.com/p/110066200
https://blog.csdn.net/sinat_30353259/article/details/80901746
http://www.scholarpedia.org/article/K-nearest_neighbor
https://zh.wikipedia.org/wiki/K-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95