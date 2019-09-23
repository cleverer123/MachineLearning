# 变分推断

## 从EM算法到变分推断：

$$
\begin{aligned}
\log P(X) &= \log \int P(X,Z) dZ = \log \int Q(Z) \frac{P(X,Z)}{Q(Z)} dZ  \\
& \ge \int Q(Z) \log \frac{P(X,Z)}{Q(Z)} dZ
\end{aligned}
$$

Jesen不等式等号成立条件：$\frac{P(X,Z)}{Q(Z)} = const. $ 推出：$Q(Z) = P(Z|X)$

EM算法在E步计算联合分布的条件概率期望得到**似然的下界**$\mathcal{L}(Q)$，M步通过最大化该期望更新参数。

考察这个下界:
$$
\begin{aligned}
\mathcal{L}(Q) &= \int Q(Z) \log \frac{P(X,Z)}{Q(Z)}dZ\\
&= \int Q(Z) \log \frac{P(Z|X)P(X)}{Q(Z)}dZ \\
&= \int Q(Z) \log \frac{P(Z|X)}{Q(Z)}dZ + \log P(X)\\
&= -KL[Q(Z) \parallel P(Z|X)] + \log P(X) \\
\end{aligned}
$$

$$
\begin{aligned}
\log P(X) = \mathcal{L}(Q) + KL[Q(Z) \parallel P(Z|X)]
\end{aligned}
$$

可以看到，EM算法中利用Jesen不等式，近似忽略的部分就是KL散度。实际上变分推断是就通过KL散度在将$Q(Z)$分布与$P(Z|X)$分布做近似。变分推断的优化目标同样也是最大化$\mathcal{L}(Q)$。变分推断中，这个下届叫做**证据下界**(ELBO)$\mathcal{L}(Q)$。


## 从KL散度推导变分推断

如果从KL散度的角度出发，同样能推导出以上结果。

从KL散度出发推导变分，以及变分的求解，见我的另一篇[博客](http://chenliu.science/2018/10/05/Bayesian-Variational-Inference/)

## 与MCMC算法比较

MCMC方法是利用马尔科夫链取样来近似后验概率，变分法是利用优化结果来近似后验概率，那么我们什么时候用MCMC，什么时候用变分法呢？

首先，MCMC相较于变分法计算上消耗更大，但是它可以保证取得与目标分布相同的样本，而变分法没有这个保证：它只能寻找到近似于目标分布一个密度分布，但同时变分法计算上更快，由于我们将其转化为了优化问题，所以可以利用诸如随机优化(stochastic optimization)或分布优化(distributed optimization)等方法快速的得到结果。所以当数据量较小时，我们可以用MCMC方法消耗更多的计算力但得到更精确的样本。当数据量较大时，我们用变分法处理比较合适。

另一方面，后验概率的分布形式也影响着我们的选择。比如对于有多个峰值的混合模型，MCMC可能只注重其中的一个峰而不能很好的描述其他峰值，而变分法对于此类问题即使样本量较小也可能优于MCMC方法。[变分推断——深度学习第十九章 - 川陀学者的文章 - 知乎](https://zhuanlan.zhihu.com/p/49401976)