---
layout:     post                    # 使用的布局（不需要改）
title:      Hedging the Drift: Learning to Optimize under Non-Stationarity             # 标题 
date:       2019-08-01            # 时间
author:     BY Zhipeng Liang                    # 作者
header-img: img/what-is-reinforcement-learning-the-complete-guide.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 在线学习
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
### Hedging the Drift: Learning to Optimize under Non-Stationarity 

*by Wang Chi Cheung, David Simchi-Levi and Ruihao Zhu*

This paper mainly concerns about Multi-armed Bandits problem when environment evolves according to the time, so called non-stationary environment. Multi-armed Bandit (MAB) problems are mainly studied in two streams, the stochastic and adversarial environments. In th former, the model uncertainty is static and the partial feedback is corrupted by a mean zeron random noise. The decision maker aims at estimating the latent static environment using historical data and converging to a static optimal decision. In the latter, the model is dynamically changed by an adversary and the DM strives to hedge against the changes, the compete favorably in comparison to certain benchmark policies. This classification can be understand in the way whether the randomness is knowable or unknowable.

Starting from (Besbes et al. 2014, 2015), a stream of research works focuses on MAB problems in a drifting environment, which is a hybrid of a stochastic and an adversarial environment. Simplily applying conventional algorithms is not suitable in such a setting. Strategies for the stochastic environments can quickly deteriorate under non-stationarity as historical data might "expire" while the permission of an confronting adversary in the adversarial settings counld be too pessimistic. Thanks to the framework of (Besbes et al. 2014, 2015), considering driting environment by introducing variance budge $B_T(=\Theta(T^{\rho}$ for some $\rho\in(0,1)$$))$, which is the the total change in a T step problem, provides an benchmark for estimating the performance of strategies in drifting environment. 

Apart from drifting environment, numerous research works consider the switching environment where the time horizon is partitioned into at most S intervals, and it switches from one stochastic environment to another across different intervals.

This paper has mainly two contributions:
1. When the variation budget $B_T$ is known, this paper characterize the lower bound of dynamic regret and develop a tuned Sliding Window Upper-Confidence-Bound (SW-UCB) algorithm.
2. When the variation budget $B_T$ is unknow, they propose a novel Bandit-over-Bandit (BOB) framework, using SW-UCB as sub-routine and EXP3 algorithm to adaptively tuning its window length.

##### Model Setting

In each round $t\in 1,\dots, T$, a decision set $D_t\supseteq \mathbb{R}^d$ is presented to the DM and it has to choose an action $X_t\in D_t$. Afterwards, the reward:
$$
Y_t=<X_t,\theta_t>+\eta_t
$$
is revealed. Without loss of generosity, the reward $<X_t,\theta_t>$ is normalized so that $|<X,\theta_t>|\leq 1$ for all $X\in D_t$ and t.

The vector of parameter $\theta_t$ is an unknown d-dimensional vector and $\eta_t$ is a random noise drawn i.i.d. from an unknown sub-Gaussian distribution with variance proxy R, i.e. $\mathbb{E}[\eta_t]=0$ and $\forall \lambda\in \mathbb{R}$ we have
$$
\mathbb{E}[exp(\lambda \eta_t)]\leq exp(\frac{\lambda^2 R^2}{2})
$$



Consider the general drifting environment where the sum of l2 differences of consecutive $\theta_t$'s should be bounded by some variation budget $B_T$=$\Theta(T^\rho)$ for some $\rho\in(0,1)$, i.e.,
$$
\sum_{t=1}^{T-1}\|\theta_{t+1}-\theta_t\|\leq B_T
$$
Dynamic regret of a given policy $\pi$ is defined as:
$$
R_T(\pi)=sup_{\theta_{1:T}\in \Theta(B_T)} \mathbb{E}[\sum_{t=1}^T<x^*_t-X_t,\theta_t>]
$$
where $x_t^*=argmax_{x\in D_t}<x,\theta_t>$.



##### Lower Bound

THEOREM 1. For any $T\geq d$ and $B_T\geq d^{-1/2}$, the dynamic regret of any policy $\pi$ satisfies $R_T(\pi)=\Omega(d^{\frac{2}{3} }B_T^{\frac{1}{3}}T^{\frac{2}{3}})$

LEMMA 4 For any $T_0\geq \sqrt{d}/2$ and $D=\{x\in R^d: \|x\|\leq 1\}$, then there exists a $\theta\in\{\pm \sqrt{d/4T_0}\}^d$, such that the worst case regret of any algorithm for linear bandits with unknown parameter $\theta$ is $\Omega(d\sqrt{T_0})$。

Proof:  Assume that the nature can change the parameters every H rounds, and any pair of $\theta, \theta'\in \{\pm \sqrt{d/4H}\}^d$, the difference between $\theta$ and $\theta'$ is upper bounded as  $\sqrt{\sum_{i=1}^d \frac{4d}{4H}}=\frac{d}{\sqrt{H}}$, and there are at most $\lfloor T/H \rfloor$ changes across the whole time horizon, and the total variation is at most $B=\frac{T}{H}\cdot \frac{d}{\sqrt{H}}=dTH^{-\frac{3}{2}}$.  In order to satisfy the variation budget, $B\leq B_T$, this indicates that $H\geq (dT)^{\frac{2}{3}}B_T^{-\frac{2}{3}}$. Taking $H=\lceil (dT)^{\frac{2}{3}}B_T^{-\frac{2}{3}}\rceil$, the worst case regret is $\Omega(d^{\frac{2}{3}}B^{\frac{1}{3}}T^{\frac{2}{3}})$.



##### Sliding Window Regularized Least Squares Estimator

The difference $\hat{\theta}_t-\theta_t$ has the following expression:
$$
V_{t-1}^{-1}\sum_{s=1}X_sX_s^T(\theta_s-\theta_t)+V_{t-1}^{-1}(\sum_{s=1\lor (t-w)^{t-1}}\eta_sX_t-\lambda\theta_t)
$$


The first term is the estimation inaccuracy due to the no-stationarity; while the second term is the estimation error due to random noise.



**LEMMA 1.**
$$
\|V_{t-1}^{-1}\sum_{s=1}^{t-1} X_sX_s^T(\theta_s-\theta_t)\|\leq \sum_{s=1}^{t-1} \|\theta_s-\theta_t\|
$$
Proof Sketch:

1. Transform $\theta_s-\theta_t$ into consecutive $\theta_s-\theta_{s+1}$

.2. $\lambda_{max}(V_{t-1}^{-1}(\sum_{s=1\lor (t-w)}^{p}))\leq 1$

**LEMMA 2.** (Improved Algorithms for Linear Stochastic Bandits by Abbasi-Yadkori et al. 2011)
$$
\|\sum_{s=1\lor(t-w)}^{t-1}\eta_sX_s-\lambda\theta_t\|_{V_{t-1}^{-1}}\leq R\sqrt{dln(\frac{1+wL^2/\lambda}{\delta})}+\sqrt{\lambda}S
$$
holds with probability at least $1-\delta$.

This LEMMA can be directly derived from Theorem 2 in (Abbasi-Yadkori et al. 2011).



**Theorem 2** For any $t\in1\dots,T$ and any $\delta\in[0,1]$, we have with probability at least $1-\delta$
$$
|x^T(\hat{\theta}_t-\theta_t)|\leq L\sum_{s=1\lor(t-w)}^{t-1}\|\theta_s-\theta_{s+1}\|+\beta\|x\|_{V_{t-1}^{-1}}
$$


Dynamic regret:
$$
\begin{aligned}
<x_t^*-X_t,\theta_t>&=<x_t^*,\theta_t>-<X_t,\theta_t>\\
&=\underbrace{<x_t^*,\theta_t-\hat{\theta_t}>}_{Theorem 2}+\underbrace{<x_t^*,\hat{\theta_t}>}_{UCB}-\underbrace{<X_t,\theta_t>}_{Theorem 2}
\end{aligned}
$$
$$\Rightarrow$$
$$
<x^*_t-X_t,\theta_t>\leq 2L \sum_{s=1\lor (t-w)}^{t-1} \|\theta_s-\theta_{s+1}\|+2\beta \|X_t\|_{V_{t-1}^{-1}}
$$
**Theorem 3**
$$
\begin{aligned}
\mathbb{E}[Regret\_T(SW\_UCB \ algorithm)]&=\sum_{t=1}^{T-1}<x^*_t-X_t,\theta_t>\\
&\leq 2LwB_t+2\beta \sum_{t=1}^T(\|X_t\|_{V_{t-1}^{-1}}\land 1)
\end{aligned}
$$


The second term:
$$
\begin{aligned}
\sum_{t=1}^T(\|X_t\|_{V_{t-1}^{-1}}\land 1)&\leq \sqrt{T}\sqrt{\sum_{t=1}^T(\|X_t\|_{V_{t-1}^{-1}}^2\land 1)}\leq \sqrt{\sum_{i=0}^{\lceil T/w\rceil-1}\sum_{t=i\cdot w+1}^{(i+1)w}1\land \|X_t\|^2_{V_{t-1}^2}}\\
&\leq \sqrt{\sum_{i=0}^{\lceil T/w\rceil-1}\sum_{t=i\cdot w+1}^{(i+1)w}1\land \|X_t\|^2_{\overline{V}_{t-1}^2}}\\
&\leq  T\sqrt{\frac{2d}{w}ln(\frac{d\lambda+wL^2}{d\lambda})}(Lemma \ 11\  of\  (Abbasi-Yadkori et al. 2011))
\end{aligned}
$$

$$
\mathbb{E}[Regret_T(SW_UCB \ algorithm)]\leq 2LwB_T+2\beta T(\sqrt{\frac{2d}{w}ln(\frac{d\lambda+wL^2}{d\lambda})}+2T\delta\\
\leq \tilde{O}(wB_T+\frac{dT}{\sqrt{w}})
$$

By choosing 

$w=O((dT)^{2/3}B_t^{-2/3})$ and $\delta=1/T$,  we have $\mathbb{E}[Regret_T(SW\_UCB \ algorithm)]=\tilde{O}(d^{2/3}B_T^{1/3}T^{2/3})$

if $B_T$ is not unknown taking $w=O((dT)^{2/3})$ and $\delta=1/T$, we have $\mathbb{E}[Regret_T(SW\_UCB \ algorithm)]=\tilde{O}(d^{2/3}(B_T+1)T^{2/3})$





##### Bandit-over-Bandits

**Theorem 4**
$$
\begin{aligned}
\mathbb{E}[Regret_T(BOB algorithm)]&=\mathbb{E}[\sum_{t=1}^T<x_t^*,\theta_t>-\sum_{t=1}^T<X_t,,\theta_t>\\
&=\underbrace{\mathbb{E}[\sum_{t=1}^T<x_t^*,\theta_t>-\sum_{i=1}^{\lceil T/H\rceil}\sum_{t=(i-1)H+1}^{i\cdot H\land T}<X_t(w^\dagger),\theta_t>]}_{first\ term}\\&+\underbrace{\mathbb{E}[\sum_{i=1}^{\lceil T/H\rceil}\sum_{t=(i-1)H+1}^{i\cdot H\land T}<X_t(w^\dagger),\theta_t>-\sum_{i=1}^{\lceil T/H\rceil}\sum_{t=(i-1)H+1}^{i\cdot H\land T}<X_t(w_i),\theta_t>]}_{second \ term}
\end{aligned}
$$
The First term=$\tilde{O}(w^{\dagger}B_T+\frac{dT}{\sqrt{w^{\dagger}}})$

The second term$\leq \mathbb{E}[\tilde{O}(Q\sqrt{\frac{|J|T}{H}})]$, where Q is the maximum absolute sum of rewards of any block as random variable. (Auer et al. 2002a)

LEMMA 3. With probability at least 1-2/T, Q does not exceed $H+2R\sqrt{Hln(T/\sqrt{H})}$, i.e.
$$
Pr(Q\leq H+2R\sqrt{Hln(T/\sqrt{H})})\geq 1-\frac{2}{T}
$$
FACT (Rigollet and Hutter 2018):
$$
Pr(|\sum_{t=(i-1)H+1}^{i\cdot H\land T}\eta_t|\geq 2R\sqrt{H ln \frac{T}{\sqrt{H}}})\leq \frac{2H}{T^2}
$$

$$
Q=|\sum_{t=(i-1)H+1}^{i\cdot H\land T}<X_t, \theta_t>\eta_t|\leq H+|\sum_{t=(i-1)H+1}^{i\cdot H\land T}\eta_t|
$$

Therefore, the second term


$$
\begin{aligned}
\leq \mathbb{E}[\tilde{O}(Q\sqrt{\frac{|J|T}{H}})]\leq &(H+2HR\sqrt{lnT})*Pr(Q\leq H+2R\sqrt{Hln\frac{T}{\sqrt{H}}})+Pr(Q\leq H+2R\sqrt{Hln\frac{T}{\sqrt{H}}})\\
&=\tilde{O}(\sqrt{H|J|T})+T\cdot \frac{2}{T}\\
&=\tilde{O}(\sqrt{H|J|T})
\end{aligned}
$$





EXP3.S:y=1.0589198165496876x+-2.031075494645885
SW_UCB:y=0.9344154836606042x+-2.6797493935069974



BOB:y=0.9618974510609959x+-1.999059722100245
SW_UCB:y=1.56266615722359x+-3.21379878505043







EXP3.S:y=0.9529570403659773x+-1.406305036534438
SW_UCB:y=0.7955559909479708x+-1.034963364818518





BOB:y=1.061702735961192x+-1.8792975489796089
SW_UCB:y=1.9093382314556537x+-6.44241440616039