# 广义线性模型

设想一个分类或者回归问题，要预测一些随机变量 $y$ 的值，作为 $x$ 的一个函数。要导出适用于这个问题的广义线性模型，就要对我们的模型、给定 $x$ 下 $y$ 的条件分布来做出以下三个假设：

1. $y | x; \theta ∼ Exponential Family(\eta)$，即给定 $x$ 和 $\theta, y$ 的分布属于指数分布族，是一个参数为 $\eta$ 的指数分布。——假设1
2. 给定 $x$，目的是要预测对应这个给定 $x$ 的 $T(y)$ 的期望值。咱们的例子中绝大部分情况都是 $T(y) = y$，这也就意味着我们的学习假设 $h$ 输出的预测值 $h(x)$ 要满足 $h(x) = E[y|x]$。 （注意，这个假设通过对 $h_\theta(x)$ 的选择而满足，在逻辑回归和线性回归中都是如此。例如在逻辑回归中， $h_\theta (x) = [p (y = 1|x; \theta)] =[ 0 \cdot p (y = 0|x; \theta)+1\cdot p(y = 1|x;\theta)] = E[y|x;\theta]$。**译者注：这里的$E[y|x$]应该就是对给定$x$时的$y$值的期望的意思。**）——假设2
3. 自然参数 $\eta$ 和输入值 $x$ 是线性相关的，$\eta = \theta^T x$，或者如果 $\eta$ 是有值的向量，则有$\eta_i = \theta_i^T x$。——假设3

## 指数分布族

定义一下指数组分布（exponential family distributions）。如果一个分布能用下面的方式来写出来，我们就说这类分布属于指数族：
$$
p(y;\eta) =b(y)exp(\eta^TT(y)-a(\eta)) \qquad \text{(6)}
$$
上面的式子中，$\eta$ 叫做此分布的**自然参数** （natural parameter，也叫**典范参数 canonical parameter**） ； $T(y)$ 叫做**充分统计量（sufficient statistic）** ，我们目前用的这些分布中通常 $T (y) = y$；而 $a(\eta)$ 是一个**对数分割函数（log partition function）。** $e^{−a(\eta)}$ 这个量本质上扮演了归一化常数（normalization constant）的角色，也就是确保 $p(y; \eta)$ 的总和或者积分等于$1$。

写过一篇[关于指数分布族的博客](http://chenliu.science/2018/09/13/Exponential-Family/)

### 伯努利分布

伯努利分布的均值是$\phi$，也写作 $Bernoulli(\phi)$，确定的分布是 $y \in \{0, 1\}$，因此有 $p(y = 1; \phi) = \phi$; $p(y = 0;\phi) = 1−\phi$。

$$
\begin{aligned}
p(y;\phi) & = \phi ^y(1-\phi)^{1-y}\\
& = exp(y \log \phi + (1-y)\log(1-\phi))\\
& = exp( (log (\frac {\phi}{1-\phi}))y+\log (1-\phi) )\\
\end{aligned}
$$

因此，自然参数（natural parameter）就给出了，即 $\eta = log (\frac   \phi {1 − \phi})$。 我们翻转这个定义，用$\eta$ 来解 $\phi$ 就会得到 $\phi = 1/ (1 + e^{−\eta} )$，正好是Sigmoid函数。

从广义线性模型的角度理解逻辑回归的函数为$1/ (1 + e^{−z} )$：一旦我们假设以 $x$ 为条件的 $y$ 的分布是伯努利分布，那么根据广义线性模型和指数分布族的定义，假设函数的形式：$h_\theta(x) = 1/ (1 + e^{−\theta^T x})$

### 多项式分布

从广义线性模型的角度假设以 $x$ 为条件的 $y$ 的分布是多项式分布，那么假设函数的形式就是Softmax函数。参考cs229-1.9。
