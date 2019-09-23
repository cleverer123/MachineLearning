# SVM
SVM 是一种**二分类模型**， 它的目的是寻找一个超平面来对样本进行分割，分割的依据是**间隔最大化**，最终转化为一个凸二次规划问题来求解。

## 理解SVM，起源于线性分类器。

    给定一些数据点，它们分别属于两个不同的类，现在要找到一个线性分类器把这些数据分成两类。如果用x表示数据点，用y表示类别（y可以取1或者-1，分别代表两个不同的类），一个线性分类器的学习目标便是要在n维的数据空间中找到一个超平面（hyper plane），这个超平面的方程可以表示为：
$$
w^Tx + b = 0
$$

点 $x$  到超平面$w^Tx + b = 0$ 的距离为：

$$
\begin{aligned}
& \frac{|w^Tx + b|}{||w||} \\
& ||w|| = \sqrt{w_1^2 + ... + w_n^2}
\end{aligned}
$$

## 函数间隔与几何间隔

在超平面$w^Tx+b=0$确定的情况下，$|w^Tx+b|$ 能够表示点$x$到距离超平面的远近，而通过观察$w^Tx+b$的符号与类标记y的符号是否一致可判断分类是否正确，所以，可以用$(y(w^Tx+b))$的正负性来判定或表示分类的正确性。于此，我们便引出了函数间隔（functional margin）的定义:

$$
\hat\gamma^{(i)}=y^{(i)}(w^Tx+b)
$$

给定一个训练集 $S = \{(x^{(i)},y^{(i)}); i = 1, ..., m\}$，我们将数据集$S$对于超平面$(w, b)$的函数间隔定义为所有训练样本中的函数间隔的最小值。记作 $\hat \gamma$，可以写成：
$$
\hat\gamma= \min_{i=1,...,m}\hat\gamma^{(i)}
$$

但这样定义的函数间隔，如果按比例缩放w和b，则函数间隔的值也会相应的缩放（虽然此时超平面没有改变）。因为我们希望通过结果的符号来判断类别，不受其大小的影响。

我们可以对法向量w加些约束条件，从而引出真正定义点到超平面的距离--几何间隔（geometrical margin）的概念。

假定对于一个点 $x^{(i)}$ ，$w$ 是垂直于超平面的一个向量，$\gamma^{(i)}$为样本$x^{(i)}$到超平面的距离。

那么利用空间几何知识：

$$
\gamma^{(i)}=\frac{w^Tx^{(i)}+b}{\parallel w\parallel }=(\frac{w}{\parallel w\parallel })^Tx^{(i)}+\frac{b}{\parallel w\parallel }
$$

这个解是针对图中 $A$ 处于训练样本中正向部分这种情况，即$y^{(i)}=1$，这时候位于“正向(positive)”一侧就是很理想的情况。如果更泛化一下，就可以定义对应训练样本 $(x^{(i)}, y^{(i)})$ 的几何间隔 $(w, b)$ 为：

$$
\gamma^{(i)}=y^{(i)}((\frac{w}{\parallel w\parallel })^Tx^{(i)}+\frac{b}{\parallel w\parallel })
$$

>几何间隔其实就是样本点到超平面的距离乘以标签。

通过由前面的分析可知：函数间隔不适合用来最大化间隔值，因为在超平面固定以后，可以等比例地缩放w的长度和b的值，这样可以使得的值任意大，亦即函数间隔可以在超平面保持不变的情况下被取得任意大。但几何间隔因为除以了$\parallel w\parallel$，使得在缩放w和b的时候几何间隔的值是不会改变的，它只随着超平面的变动而变动，因此，这是更加合适的一个间隔。换言之，这里要找的最大间隔分类超平面中的“间隔”指的是几何间隔。

最后，给定一个训练集 $S = \{(x^{(i)}, y^{(i)}); i = 1, ..., m\}$，我们也可以将数据集$S$对于超平面$(w, b)$的**几何间隔**定义为所有训练样本中的几何间隔的最小值：
$$
\gamma=\min_{i=1,...,m}\gamma^{(i)}
$$

## 最优间隔分类器

对一个数据点进行分类，当超平面离数据点的“间隔”越大，分类的确信度（confidence）也越大。所以，为了使得分类的确信度尽量高，需要让所选择的超平面能够最大化这个“间隔”值。

于是最大间隔分类器（maximum margin classifier）的目标函数可以定义为：

$$
\begin{aligned}
max_{\gamma,w,b} \quad & \gamma \\
s.t. \quad & y^{(i)}(w^Tx^{(i)}+b) \geq \gamma,\quad i=1,...,m\\
&\parallel w\parallel =1 \\
\end{aligned}
$$

![1](img\1.jpg)
![2](img\2.jpg)

从几何的角度去思考，我们的最大间隔就是$\frac{2}{\parallel w\parallel}$.

那么我们的最优化问题转化为：
$$
min \quad \frac{1}{2} ||w||^2 \quad \\ st. y_i(w^Tx_i + b) \geq 1
$$

## 5 拉格朗日对偶性(Lagrange duality) 

咱们先把 SVMs 以及最大化边界分类器都放到一边，先来谈一下约束优化问题的求解。

例如下面这样的一个问题：
$$
\begin{aligned}
min_w \quad & f(w)& \\
s.t. \quad &h_i(w) =0,\quad i=1,...,l\\
\end{aligned}
$$
可能有的同学还能想起来这个问题可以使用拉格朗日乘数法(method of Lagrange multipliers)来解决。（没见过也不要紧哈。）在这个方法中，我们定义了一个**拉格朗日函数(Lagrangian)** 为：
$$
L(w,\beta)=f(w)+\sum^l_{i=1}\beta_i h_i(w)
$$
上面这个等式中，这个 $\beta_i$ 就叫做**拉格朗日乘数(Lagrange multipliers)。** 然后接下来让 对 $L$ 取偏导数，使其为零：
$$
\frac{\partial L }{\partial w_i} =0; \quad \frac{\partial L }{\partial \beta_i} =0;
$$
然后就可以解出对应的 $w$ 和 $\beta$ 了。在本节，我们对此进行一下泛化，扩展到约束优化(constra_ined optimization)的问题上，其中同时包含不等约束和等式约束。由于篇幅限制，我们在本课程$^2$不能讲清楚全部的拉格朗日对偶性(do the theory of Lagrange duality justice)，但还是会给出主要的思路和一些结论的，这些内容会用到我们稍后的最优边界分类器的优化问题(optimal margin classifier’s optimization problem)。

> 2 对拉格朗日对偶性该兴趣的读者如果想要了解更多，可以参考阅读 R. T. Rockefeller (1970) 所作的《凸分析》(Convex Analysis)，普林斯顿大学出版社(Princeton University Press)。

下面这个，我们称之为**主** 最优化问题(primal optimization problem)：
$$
\begin{aligned}
min_w \quad & f(w)& \\
s.t. \quad & g_i(w) \le 0,\quad i=1,...,k\\
& h_i(w) =0,\quad i=1,...,l\\
\end{aligned}
$$
要解决上面这样的问题，首先要定义一下**广义拉格朗日函数(generalized Lagrangian)：**
$$
L(w,\alpha,\beta)=f(w)+\sum^k_{i=1}\alpha_ig_i(w)+\sum^l_{i=1}\beta_ih_i(w)
$$
上面的式子中， $\alpha_i$ 和 $\beta_i$ 都是**拉格朗日乘数(Lagrange multipliers)**。设有下面这样一个量(quantity)：
$$
\theta_{P}(w)=\max_{\alpha,\beta:\alpha_i \geq 0}L(w,\alpha,\beta)
$$
上式中的 $“P”$ 是对 “primal” 的简写。设已经给定了某些 $w$。如果 $w$ 不能满足某些主要约束，（例如对于某些 $i$ 存在 $g_i(w) > 0$ 或者 $h_i(w) \neq 0$），那么咱们就能够证明存在下面的等式关系：
$$
\begin{aligned}
\theta_P(w)&=\max_{\alpha,\beta:\alpha_i \geq 0} f(w)+\sum^k_{i=1}\alpha_ig_i(w)+\sum^l_{i=1}\beta_ih_i(w) &\text{(1)}\\
&= \infty &\text{(2)}\\
\end{aligned}
$$
与之相反，如果 $w$ 的某些特定值确实能满足约束条件，那么则有 $\theta_P(w) = f(w)$。因此总结一下就是：
$$
\theta_P(w)= \begin{cases} f(w) & \text {if w satisfies primal constraints} \\\infty & \text{otherwise} \end{cases}
$$
因此，如果 $w$ 的所有值都满足主要约束条件，那么$\theta_P$的值就等于此优化问题的目标量(objective in our problem)，而如果约束条件不能被满足，那 $\theta_P$的值就是正无穷了(positive infinity)。所以，进一步就可以引出下面这个最小化问题(minimization problem)：
$$
\min_w \theta_P(w)=\min_w \max_{\alpha,\beta:\alpha_i\geq0} L(w,\alpha,\beta)
$$
这个新提出的问题与之前主要约束问题有一样的解，所以还是同一个问题。为了后面的一些内容，我们要在这里定义一个目标量的最优值(optimal value of the objective)$p ^\ast = min_w \theta_P (w)$；我们把这个称为 主要优化问题的**值** (value of the primal problem)。

接下来咱们来看一个稍微不太一样的问题。我们定义下面这个 $\theta_D$：
$$
\theta_D(\alpha,\beta)=\min_w L(w,\alpha,\beta)
$$
上面的式子中，$“D”$ 是 “dual” 的缩写。这里要注意，在对$\theta_P$ 的定义中，之前是对 $\alpha$, $\beta$ 进行优化(找最大值)，这里则是找 $w$ 的最小值。

现在我们就能给出这个**对偶**优化问题了：
$$
\max_{\alpha,\beta:\alpha_i\geq 0} \theta_D(\alpha,\beta)  = \max_{\alpha,\beta:\alpha_i\geq 0} \min_w L(w,\alpha,\beta)
$$
这个形式基本就和我们之前看到过的主要约束问题(primal problem)是一样的了，唯一不同是这里的“max” 和 “min” 互换了位置。我们也可以对这种对偶问题对象的最优值进行定义，即 $d^\ast = \max_{\alpha,\beta:\alpha_i\geq 0} \theta_D(w)$。

主要约束问题和这里的对偶性问题是怎么联系起来的呢？通过下面的关系就很容易发现
$$
d^\ast = \max_{\alpha,\beta:\alpha_i\geq 0}\min_w L(w,\alpha,\beta) \leq \min_w \max_{\alpha,\beta:\alpha_i\geq 0}L(w,\alpha,\beta)  =p^\ast
$$
（你应该自己证明一下，这里以及以后的一个函数的最大的最小值“max min”总是小于等于最小的最大值“min max”。）不过在某些特定的情况下，就会有二者相等的情况：
$$
d^\ast =p^\ast
$$
这样就可以解对偶问题来代替原来的主要约束问题了。接下来咱们就来看看导致上面二者相等的特定条件是什么。

假设 $f$ 和 $g_i$ 都是凸的(convex$^3$)，$h_i$ 是仿射的(affine$^4$)。进一步设 $g$ 是严格可行的(strictly feasible)；这就意味着会存在某个 $w$，使得对所有的 $i$ 都有 $g_i(w) < 0$。

基于上面的假设，可知必然存在 $w^\ast$，$\alpha^\ast$， $\beta^\ast$ 满足$w^\ast$ 为主要约束问题(primal problem)的解，而$\alpha^\ast$，$\beta^\ast$ 为对偶问题的解，此外存在一个 $p^\ast = d^\ast = L(w^\ast,\alpha^\ast, \beta^\ast)$。另外，$w^\ast$，$\alpha^\ast$， $\beta^\ast$这三个还会满足**卡罗需-库恩-塔克条件(Karush-Kuhn-Tucker conditions, 缩写为 KKT)，** 如下所示：
$$
\begin{aligned}
\frac{\partial}{\partial w_i}L(w^\ast,\alpha^\ast,\beta^\ast) &= 0,\quad i=1,...,n & \text{(3)}\\
\frac{\partial}{\partial \beta_i}L(w^\ast,\alpha^\ast,\beta^\ast)&= 0 ,\quad i=1,...,l &  \text{(4)}\\
\alpha_i^\ast g_i(w^\ast)&= 0,\quad i=1,...,k & \text{(5)}\\
g_i(w^\ast)&\leq 0,\quad i=1,...,k & \text{(6)}\\
\alpha_i^\ast &\geq 0,\quad i=1,...,k &\text{(7)}\\
\end{aligned}
$$
反过来，如果某一组 $w^\ast,\alpha^\ast,\beta^\ast$ 满足 KKT 条件，那么这一组值就也是主要约束问题(primal problem)和对偶问题的解。

这里咱们要注意一下等式$(5)$，这个等式也叫做 **KKT 对偶互补** 条件(dual complementarity condition)。这个等式暗示，当$\alpha_i^\ast > 0$ 的时候，则有 $g_i(w^\ast) = 0$。（也就是说，$g_i(w) \leq 0$ 这个约束条件存在的话，则应该是相等关系，而不是不等关系。）后面的学习中，这个等式很重要，尤其对于表明 SVM 只有少数的“支持向量(Support Vectors)”；在学习 SMO 算法的时候，还可以用 KKT 对偶互补条件来进行收敛性检测(convergence test)。

SVM 优化

- SVM 的优化问题为：
  $$
  min \quad \frac{1}{2} ||w||^2 \quad \\ st. \quad  g_i(w) = 1- y_i(w^Tx_i + b) \leq 0
  $$



- 构造拉格朗日函数：
  $$
  min_{w,b}max_{\lambda} L(w, b, \lambda) = \frac{1}{2} ||w||^2 + \sum_{i=1}^n \lambda_i (1- y_i(w^Tx_i + b) ) \\
  s.t. \lambda_i \geq 0
  $$

- 利用强对偶性转化：
  $$
  max_{\lambda}min_{w,b} \, L(w, b, \lambda)
  $$
  对参数 $w, b$ 求偏导有：
  $$
  \frac{\delta L}{\delta w} = w - \sum_{i=1}^n \lambda_i x_i y_i = 0 \\
  \frac{\delta L}{\delta b} = \sum_{i=1}^n \lambda_i y_i = 0
  $$
  得到：
  $$
  w =  \sum_{i=1}^n \lambda_i x_i y_i \\
  \sum_{i=1}^n \lambda_i y_i = 0
  $$
  将两式带入到 $L(w, b, \lambda)$ 中有：
  $$
  \begin{aligned}
  min_{w,b} \, L(w, b, \lambda) &=\sum_{j=1}^n \lambda_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j y_i y_j x_i^Tx_j
  \end{aligned}
  $$

参数$\lambda$的求解采用SMO算法，这里就不多介绍了。

#  软间隔SVM

软间隔允许部分样本点不满足约束条件：
$$
1 - y_i (w^Tx_i + b) \leq 0
$$


软间隔优化目标：

$$
\begin{aligned}
min_{w,b} \quad & \frac{1}{2} ||w||^2 + C \sum_{i=1}^m \xi_i \\ 
st. \quad & y^{(i)}(w^Tx^{(i)} + b) \geq 1 \quad \xi_i \geq 0
\end{aligned}
$$

# 参考
- [支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)

