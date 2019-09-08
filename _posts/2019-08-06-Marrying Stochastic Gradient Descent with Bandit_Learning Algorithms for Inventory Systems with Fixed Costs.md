---
layout:     post                    # 使用的布局（不需要改）
title:      Summary of Online Learning             # 标题 
date:       2019-08-06            # 时间
author:     BY Zhipeng Liang                    # 作者
header-img: img/what-is-reinforcement-learning-the-complete-guide.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Python
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


## Learning Algorithms for Inventory Systems with Fixed Costs

[Paper Link](https://poseidon01.ssrn.com/delivery.php?ID=848111097086113100109080089083120096032042072006036091011094029091113095122120090065023097018011121012001114085100031029005083025009038044078119027071074107077094027038050071029093111098016004023107105079093024078109068097011066121122001031068113104112&EXT=pdf)

This blog is attached with [Slides](<https://liangzp.github.io/Online-Learning/Learning_Inventory.html>)

Contributions:

First nonparametric learning algorithm to solve stochastic inventory system with fixed cost under censored demand



Notation:

$Proj_[a,b](x)=max\{a,min\{x,b\}\}$



Model:

Demand in period t $D_t$ ands identically distributed continuous random variables 

Dynamics:

1. The firm observes the beginning on-hand inventory level $x_t$

2. The firm makes an ordering decision $q_t\geq0$. The ending on-hand inventory level becomes $y_t=x_t+q_t$ (No lead time)

3. The demand $D_t$ is realized to be $d_t$

4. The total cost for period t is given by
   $$
   C_t(x_t,q_t,d_t)=K\mathbb{1}_{q_t>0}+cq_t+h(x_t+q_t-d_t)^++p(x_t+q_t-d_t)^-
   $$







5.The inventory next period is $x_{t+1}=[x_t+q_t-d_t]^+$



Optimal policy:
$$
\pi^*=\arg \min_{\pi}\lim_{T\rightarrow\infty}\sup\frac{1}{T}\mathbb{E}[\sum_{t=1}^TC_t^\pi]
$$
Regret:
$$
R_T(\pi)=\mathbb{E}[\sum_{t=1}^TC_t^\pi]-\mathbb{E}[\sum_{t=1}^TC_t^{\pi^*}]
$$
Assumption:

1. Demand is iid
2. The probability density function $f(\cdot)$ of demand D is bounded
3. The warehouse storage capacity is $\beta$.



Renewal Reward Theorem:
$$
\lim_{T\rightarrow\infty}\frac{1}{T}\mathbb{E}[\sum_{t=1}^TC_t^{(\delta,S)}]=\frac{\mathbb{E}[H(\delta,S)]}{\mathbb{E}[L(\delta,S)]}
$$
where
$$
\begin{aligned}
&\tau_i=inf\{t\geq 1:y_t=S\}\\
L(\delta,S)=\tau_{i+1}-\tau_i \ \ &\ \ H(\delta,S)=c(x_{\tau_i}-x_{\tau_{i+1}})+\sum_{t=\tau_i}^{\tau_{i+1}-1}C_t\\
\end{aligned}
$$


Transforming the Objective:

Notice that $p(x_t+q_t-d_t)^-=pd_t-pmin(x_t+q_t,d_t)$, the first term is independent of policy and the second term is policy-dependent and observable(?). Therefore, we can ignore the first term.
$$
\tilde{C}_t(x_t,q_t,d_t)=C_t(x_t,q_t,d_t)-pd_t
$$

$$
G(\delta,S)=c(x_{\tau_i}-x_{\tau_{i+1}})+\sum_{t=\tau_i}^{\tau_{i+1}-1}\tilde{C}_t
$$

Denote $V(\delta,S)=\frac{\mathbb{G}(\delta,S)}{\mathbb{L}(\delta,S)}$.

**The rest can be found in the slides.**




