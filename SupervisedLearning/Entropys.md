# 熵

## 信息熵
$$
H(p) = -\int p(x) log p(x) dx
$$

## 相对熵

$$
D(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)}dx
$$

相对熵又叫KL散度。

## 交叉熵

$$
CE(p, q) &= -\int p(x) \log q(x) dx &= - \int p(x) \log p(x) dx + \int p(x) \log \frac{p(x)}{q(x)} dx
&= H(p) + D(p \parallel q) 
$$

当p已知时，优化交叉熵等价于相对熵，即未知分布与已知分布的相似度。
