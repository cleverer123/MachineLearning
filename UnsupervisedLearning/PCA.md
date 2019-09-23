# PCA

> PCA（主成分分析），旨在找到数据中的主成分，并利用这些主成分表征原始数据从而达到降维的目的。在信号处理领域，我们认为信号具有较大方差，而噪声具有较小方差。因此我们不难引出PCA的目标即最大化投影方差，也就是让数据在主轴上投影的方差最大（在我们假设中方差最大的有用信号最大化减少了噪声的影响）。


## 最大方差理论：

对于给定的一组数据$\{ v_1, v_2, \dots, v_n\}$ 均为列向量。中心化后表示为$\{ x_1, x_2, \dots, x_n\} = \{ v_1 - \mu, v_2 - \mu, \dots, v_n - \mu \}$，其中$\mu = \frac{1}{n}\sum_{i=1}^{n}v_i $。寻找一个投影方向$\omega$，使得 $\{ x_1, x_2, \dots, x_n\}$ 在
$\omega$（单位向量）上投影方差最大（多分配给主成分）。向量$x_i$ 在 $\omega$ 上的投影坐标可以表示为$(x_i,\omega)=x^T_i \omega$，所以投影之后的方差可以表示为

$$
D(x) = \frac{1}{n}\sum_{i=1}^{n} (x_i^T \omega)^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i^T \omega)^T(x_i^T \omega) = \omega^T (\frac{1}{n}\sum_{i=1}^{n} x_i x_i^T ) \omega
$$​

$\frac{1}{n}\sum_{i=1}^{n} x_i x_i^T$ 是样本协方差矩阵，记为$\Sigma$。因此可将该问题表示成如下优化问题：

$$
\begin{aligned}
max_{\omega} \quad & \omega^T \Sigma \omega \\
s.t. \quad & \omega^T \omega=1 
\end{aligned}
$$

采用拉格朗日乘数法：

$$
L(w,\beta)=\omega^T \Sigma \omega - \beta (\omega^T \omega - 1)
$$

对$\omega$ 求偏导，$\frac{\partial L(w,\beta)}{\partial \omega} = 2\Sigma\omega - 2\beta\omega = 0$, 得$\Sigma\omega = \beta\omega $，代入 $D(x)$，得 $$D(x)=\omega^T \beta \omega = \beta.$$

**因此投影后的方差就是协方差矩阵的特征值。最大方差即为协方差矩阵最大的特征值，最佳投影方向就是最大特征值所对应的特征向量。** 我们将特征值从大到小排列，取特征值前d大对应的特征向量$\omega= [\omega_{1},...,\omega_{d}]^T $，通过以下映射的方式将高维样本映射到d维。

$$
x_i^* = w^T x_i
$$

## 最小平方误差



# PPCA
PPCA有如下优点：
- 可以推导出EM算法进行迭代求解。在主成份M定得小的情况下非常高效。
- 把PCA概率模型化结合EM算法可以处理数据缺失的情况。
- 多个PPCA模型可以混合，并用EM算法进行训练。
- 子空间的维度可以自动选择。
- 建立了似然函数，便于模型比较。
- PPCA建立了如何由隐变量产生观测变量的过程，可以进行采样产生样本数据。
- PPCA can be used to model class-conditional densities and hence be applied to classification problems.
- PPCA represents a constrained form of the Gaussian distribution in which the number of free parameters can be restricted while still allowing the model to capture thee dominant correlations in a data set.

PPCA是一个线性高斯框架的典型例子，所有的边缘分布和条件分布都是高斯的。


# 参考
- [机器学习面试必知：最大方差理论和最小平方误差理论下的PCA(主成分分析)的公式推导](https://blog.csdn.net/Neekity/article/details/87918977)
- [PRML读书笔记：Probabilistic PCA - victor的文章 - 知乎]
(https://zhuanlan.zhihu.com/p/28000014)