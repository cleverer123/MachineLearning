## EM算法推导

对于$m$个样本观察数据$x=(x^{(1)},x^{(2)},...x^{(m)})$中，找出样本的模型参数θ, 极大化模型分布的对数似然函数如下：
$$
\begin{aligned}
l(\theta) &= \sum\limits_{i=1}^m logP(x^{(i)};\theta), \\
\theta &= arg \max \limits_{\theta} l(\theta)
\end{aligned}
$$

如果我们得到的观察数据有未观察到的隐含数据$z=(z^{(1)},z^{(2)},...z^{(m)})$，此时我们的极大化模型分布的对数似然函数如下：

$$
l(\theta) = \sum\limits_{i=1}^m logP(x^{(i)};\theta) = \sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}P(x^{(i)}， z^{(i)};\theta)
$$


上面这个式子可能没有办法直接求出$\theta$。对于每个$i$，引入一个$i^{(i)}$的新的未知的分布$Q_i(z^{(i)}), \Sigma_{z^(i)} Q_i(z^{(i)}) = 1, Q_i(z^{(i)}) \ge 0 $。

$$
\begin{aligned}\tag{1}
\sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}P(x^{(i)}， z^{(i)};\theta) 
& = \sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}Q_i(z^{(i)})\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \\ 
& \geq  \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} 
\end{aligned}
$$

上式采用了Jensen不等式，如果要满足Jensen不等式的等号，则有

$$
\frac{P(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})} = c
$$

其中常数 $c$ 不依赖 $z^{(i)}$。故要实现这一条件，需满足：
$$
Q_i(z^{(i)})\propto P(x^{(i)},z^{(i)};\theta)
$$

又由于 $\sum_z Q_i(z^{(i)}) = 1$ 是一个分布，因此，将其归一化得到：

$$
\begin{aligned}
Q_i(z^{(i)}) &= \frac{P(x^{(i)},z^{(i)};\theta)}{\sum_z P(x^{(i)},z;\theta)} \\
&= \frac{P(x^{(i)},z^{(i)};\theta)}{P(x^{(i)};\theta)} \\
&= P(z^{(i)}|x^{(i)};\theta)
\end{aligned}
$$

如果$Q_i(z^{(i)}) = P( z^{(i)}|x^{(i)};\theta))$, 那我们求到了包含隐藏数据的对数似然的一个下界，此为E步：
$$
\sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} 
$$


如果我们能极大化这个下界，则也在尝试极大化我们的对数似然。即我们需要最大化下式，此为M步：

$$\tag{2}
arg \max \limits_{\theta} \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

## EM算法流程
　　　　现在我们总结下EM算法的流程。
输入：观察数据$x=(x^{(1)},x^{(2)},...x^{(m)})$，联合分布$p(x,z ;\theta)$，条件分布$p(z|x; \theta)$，最大迭代次数$J$。

- (1) 随机初始化模型参数$\theta$;
- (2) 迭代J次：
    - (a)  E步：计算联合分布的条件概率期望：
$$
Q_i(z^{(i)}) = P( z^{(i)}|x^{(i)};\theta))
$$
$$
L(\theta) = \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} 
$$
    - (b) M步：最大化$L(\theta)$，更新$\theta$
$$
\theta = argmax_{\theta}  L(\theta) 
$$
    - (c) 如果$\theta$收敛，结束算法；否则继续迭代。

输出：模型参数$\theta$。


# Reference

- [EM算法原理总结](https://www.cnblogs.com/pinard/p/6912636.html)