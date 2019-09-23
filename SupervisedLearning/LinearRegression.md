# 线性回归

---

## 1、数据

$X \in R^{m \times p}$$，m个样本，样本具有p个特征。$$Y \in R^{m}$。

## 2、假设

假设某个样本 $\mathbf{x}$ 的特征与输出$y$是线性的：
$$
y = h_{\theta}\mathbf{x} = \theta_0 + \theta_1 x_1 + \dots + \theta_p x_p = \theta^T \mathbf{x}
$$

## 3、损失函数

$$
J(\theta) 
= \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
=\frac{1}{2} (X\theta-y)^T(X\theta - y)
$$

## 4、正规方程

> $$\Delta_{A^T} Tr(ABA^TC) = B^T A^T C^T + B A^T C$$
> $$\Delta_{\theta} X\theta = X^T$$ 
$$
\frac{\delta J(\theta)}{\delta \theta} = X^T(X\theta - y)
$$

$$
\theta = (X^TX)^{-1}X^Ty
$$

## 5、随机梯度下降

$$
\begin{aligned}
\theta_j =& \theta_j - \alpha \frac{\delta}{\delta \theta_j} J(\theta) \\ 
=& \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}  \\ 
=& \theta_j + \alpha \frac{1}{m}(y-X\theta)^Tx_j
\end{aligned}
$$

## 6、岭回归 - 正则化

前面通过正规方程得到的 w 的估计：
$$
\hat{w} = (X^TX)^{-1}X^Ty
$$
但是，当我们有 N 个样本，每个样本有 $x_i \in R^p$， 当 N < p 时， $X^TX$ 不可逆， 容易造成过拟合。

岭回归通过在矩阵 $X^TX$ 上加一个 $\lambda I$ 来使得矩阵可逆， 此时的 w 的估计：
$$
\hat{w} = (X^TX + \lambda I)^{-1}X^Ty
$$
而岭回归本质上是对 $L(w)$  进行 L2 正则化。

$$
\begin{aligned}
J(w) &= \sum_{i=1}^N ||w^Tx_i - y_i ||^2 + \lambda w^Tw \\
&= (w^TX^T - Y^T)(Xw + Y) + \lambda w^Tw \\
&= w^TX^TXw - 2w^TX^TY  - Y^TY + \lambda w^Tw \\
&= w^T(X^TX + \lambda I)w - 2w^TX^TY - Y^TY
\end{aligned}
$$
那么对 $w$ 的极大似然估计有：

$$
\hat{w} = argmax \, J(w) 
\frac{\delta J(w)}{\delta w} = 2(X^TX + \lambda I)w - 2 X^TY = 0
$$

那么我们就解得：
$$
\hat{w} = (X^TX + \lambda I)^{-1}X^Ty
$$
因此说， 岭回归本质上是 **线性回归 + L2 正则化**， 从而达到抑制过拟合的效果。

## 7、从概率角度看线性回归

假设特征与标签存在这样的关系：
$$
y^{(i)} = f_{\theta}(x^{(i)}) + \varepsilon^{(i)} = \theta^Tx^{(i)} + \varepsilon^{(i)}
$$

其中，$$\varepsilon^{(i)}$$ 是误差项。用于存放由于建模所忽略的变量导致的效果，或者随机的噪音信息。进一步假设 $\epsilon ^{(i)}$ 是独立同分布的 (IID ，independently and identically distributed) ，服从高斯分布（Gaussian distribution ，也叫正态分布 Normal distribution）。其平均值为 $0$，方差（variance）为 $\sigma ^2$。这样就可以把这个假设写成 $ \epsilon ^{(i)} ∼ N (0, \sigma ^2)$ 。然后 $ \epsilon ^{(i)} $ 的密度函数就是：
$$
p(\epsilon ^{(i)} )= \frac 1{\sqrt{2\pi}\sigma} exp (- \frac {(\epsilon ^{(i)} )^2}{2\sigma^2})
$$
这意味着：
$$
y^{(i)} |x^{(i)};\theta  \sim N(\theta^Tx^{(i)}, \sigma^2); $$
$$
P(y^{(i)}|x^{(i)};\theta) = \frac{1}{\sqrt{2\pi\sigma}} exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2 \sigma^2})
$$
​         现在，给定了$y^{(i)}$ 和 $x^{(i)}$之间关系的概率模型了，用什么方法来选择咱们对参数 $\theta$ 的最佳猜测呢？最大似然法（maximum likelihood）告诉我们要选择能让数据的似然函数尽可能大的 $\theta$。也就是说，咱们要找的 $\theta$ 能够让函数 $L(\theta)$ 取到最大值。

​         除了找到 $L(\theta)$ 最大值，我们还以对任何严格递增的 $L(\theta)$ 的函数求最大值。如果我们不直接使用 $L(\theta)$，而是使用对数函数，来找**对数似然函数 $l(\theta)$** 的最大值，那这样对于求导来说就简单了一些：
$$
\begin{aligned} l(\theta) &=\log L(\theta) \\
&=\log \prod ^m _{i=1} \frac 1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}) \\
&= \sum ^m *{i=1}log \frac 1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}) \\
&= m \log \frac 1{\sqrt{2\pi}\sigma}- \frac 1{\sigma^2}\cdot \frac 12 \sum^m_{i=1} (y^{(i)}-\theta^Tx^{(i)})^2 \end{aligned}
$$
因此，对 $l(\theta)$ 取得最大值也就意味着下面这个子式取到最小值：

$$ \frac 12 \sum^m _{i=1} (y^{(i)}-\theta^Tx^{(i)})^2 $$

到这里我们能发现这个子式实际上就是 $J(\theta)$，也就是最原始的最小二乘代价函数（least-squares cost function）。

总结一下也就是：在对数据进行概率假设的基础上，最小二乘回归得到的 $\theta$ 和最大似然法估计的 $\theta$ 是一致的。所以这是一系列的假设，其前提是认为最小二乘回归（least-squares regression）能够被判定为一种非常自然的方法，这种方法正好就进行了最大似然估计（maximum likelihood estimation）。（要注意，对于验证最小二乘法是否为一个良好并且合理的过程来说，这些概率假设并不是必须的，此外可能（也确实）有其他的自然假设能够用来评判最小二乘方法。）



# 参考

- [CS229 课程讲义中文翻译  #第一章](https://github.com/Kivy-CN/Stanford-CS-229-CN/blob/master/Markdown/cs229-notes1.md)

