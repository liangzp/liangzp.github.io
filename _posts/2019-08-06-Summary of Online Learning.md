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


## Summary of Online Learning 

### Analysis of Thompson Sampling for the Multi-armed Bandit Problem
[link](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf)

Shipra Agrawal, Navin Goyal

*Abstract:* This paper shows that Thompson Sampling algorithm achieves logarithmic expected regret for the stochastic multi-armed bandit problem.

Results:
**The expect regret for stochastic two-armed bandit problem:**
$$
O(\frac{ln T}{\Delta}+\frac{1}{\Delta^3})
$$

Proof Sketch:

Assume the first arm is the best arm and second is sub-optimal. To bound regret, we need to bound the times of playing the second arm $\mathbb{E}[k_2(T)]$.
$$
\begin{aligned}
\mathbb{E}[k_2(T)]&\leq L+\mathbb{E}[\sum_{j=j_0}^{T-1}Y_j]\\
&\leq L+\sum_{j=j_0}^{T-1}(\mathbb{E}[\mathbb{E}[\min\{X(j,s(j),\mu_2+\frac{\Delta}{2}),T\}|s(j)]]+T\cdot\mathbb{E}[\sum_{t=t_j+1}^{t_{j+1}-1}I(\theta_2(t)>\mu_2+\frac{\Delta}{2})])\\
&\leq L+\sum_{j=j_0}^{T-1} \mathbb{E}[\mathbb{E}[\min\{X(j,s(j),\mu_2+\frac{\Delta}{2}),T\}|s(j)]]+T\cdot\sum_{t=1}^T Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L)
\end{aligned}
$$
**Importantly, we don't need to pay attention to the real time t. All we need to focus is the pseudo-time, whose unit step is defined by the step when sub-optimal arm is pulled.** Therefore, L is the time steps after the second arm has been played $L=24(lnT)/\Delta^2​$ times. $Y_j​$ is the time step between jth time of pulling the first arm and (j+1)th time of pulling the first arm. This event can be transformed into $\theta_1(t)>\theta_2(t)​$ where $\theta_1\sim Beta(s_1,t_1-s_1)​$ and $\theta_2\sim Beta(s_2,t_2-s_2)​$. However, it is not easy to directly derive the probability of $\theta_1(t)>\theta_2(t)​$, therefore, the author use an upper bound $\theta_1(t)>\mu_2+\frac{\Delta}{2}​$ and  $\theta_2(t)<\mu_2+\frac{\Delta}{2}​$ . And then we can see the event arm 2 is pull as an bernoulli experiment.



First, we bound the third term. The intuition is below:
$$
\overbrace{\mu_2\Rightarrow \frac{s_2(t)}{k_2(t)}}^{Chernoff \ Bound\ to \ solve \ term\ 2}\underbrace{\Rightarrow \theta_2(t)\sim Beta(s_t(t),k_2(t)-s_2(t))}_{solve\ term\ 1}
$$
Details:
$$
\begin{aligned}
Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L)=&Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L,\frac{s_2(t)}{k_2(t)}\leq \mu_2+\frac{\Delta}{4})+\\&
Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L,\frac{s_2(t)}{k_2(t)}>\mu_2+\frac{\Delta}{4})\\
&\leq Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L,\frac{s_2(t)}{j_2(t)}\leq \mu_2+\frac{\Delta}{4})\\
&+Pr(k_2(t)\geq L,\frac{s_2(t)}{k_2(t)}>\mu_2+\frac{\Delta}{4})\\
&=Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L,\frac{s_2(t)}{k_2(t)}\leq \mu_2+\frac{\Delta}{4})\\
&+\sum_{i=L}^{T}Pr(k_2(t)=i,\frac{s_2(t)}{k_2(t)}>\mu_2+\frac{\Delta}{4})\\
&\leq Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L,\frac{s_2(t)}{k_2(t)}\leq \mu_2+\frac{\Delta}{4})\\
& +\sum_{i=L}^{T} e^{-2l\Delta^2/16}(Chernoff \ bounds)\\
&\leq Pr(\theta_2(t)>\mu_2+\frac{\Delta}{2},k_2(t)\geq L,\frac{s_2(t)}{k_2(t)}\leq \mu_2+\frac{\Delta}{4})+\frac{1}{T^2}\\
&\leq \sum_{i=L}^{T}Pr(\theta_2(t)>\frac{s_2(t)}{k_2(t)}+\frac{\Delta}{4},k_2(t)=i)+\frac{1}{T^2}\\
&\leq \sum_{i=L}^{T} e^{-2l(\frac{\Delta}{4})^2}+\frac{1}{T^2}\\
&\leq \frac{2}{T^2}
\end{aligned}
$$
Then we bound the second term $\mathbb{E}[\mathbb{E}[\min\{X(j,s(j),\mu+\frac{\Delta}{2}),T\}|s(j)]]$:

Fact:  $\mathbb{E}[X(j,s(j),\mu_2+\frac{\Delta}{2})]=\frac{1}{F_{j+1,y}^B(s(j))}-1=\frac{1}{1-F_{s+1,j-s+1(y)}^{beta}(y)}-1$

Notice that $s(j)\sim B(j,\mu_1)$, therefore when j is large enough (when $j\geq 4(ln T)/\Delta'^2$) where $\Delta'=\Delta/2$, we can know that s(j)/j approximate $\mu_1$, which is larger than $\mu_2+\frac{\Delta}{2}$ with large probability. Therefore, with large probability, $\frac{1}{F_{j+1,y}^B(s(j))}-1$ is very close to 0 since $F_{j+1,y}^B(s(j))$ becomes sufficiently close to 1 when F concentrate to $j(\mu_2+\frac{\Delta}{2})$ and $s(j)>j(\mu_2+\frac{\Delta}{2})$. Therefore, second arm can only produce significant regret with low probability.

The authors formulate the intuition above by classify j into large case and small case.

For Small j (when $j< 4(ln T)/\Delta'^2$)

Given s(j), we have:
$$
\begin{aligned}
\mathbb{E}[\mathbb{E}[X(j,s(j),\mu+\frac{\Delta}{2})|s(j)]]&=\mathbb{E}[\frac{1}{F_{j+1,y}^B(s(j))}]-1=\sum_{s=0}^{j}\frac{f_{k,\mu_1}^B (s)}{F_{j+1,y}^B (s)}-1\\
&=\sum_{s=0}^{\lceil y(j+1)\rceil}\frac{f_{k,\mu_1}^B (s)}{F_{j+1,y}^B (s)}+\sum_{s=\lceil y(j+1)\rceil}^{j}\frac{f_{k,\mu_1}^B (s)}{F_{j+1,y}^B (s)}-1\\
&\leq \sum_{s=0}^{\lfloor yj \rfloor}\frac{f_{k,\mu_1}^B (s)}{F_{j+1,y}^B (s)}+\sum_{s=\lfloor yj \rfloor}^{\lceil y(j+1)\rceil}\frac{f_{k,\mu_1}^B (s)}{F_{j+1,y}^B (s)}+2-1\\
&\leq  \sum_{s=0}^{\lfloor yj \rfloor}\frac{f_{k,\mu_1}^B (s)}{(1-y) f_{j,y}^B(s)}+\sum_{s=\lfloor yj \rfloor}^{\lceil y(j+1)\rceil}\frac{2}{1-y}（or \frac{R^y}{1-y}e^{-Dj}） \\&\text{this term exists if}\lfloor yj \rfloor<\lceil yj\rceil < \lceil y(j+1)\rceil \text{and dependes on j}\\
&\leq \frac{\mu_1}{\mu_1-y}e^{-Dj} +\sum_{s=\lfloor yj \rfloor}^{\lceil y(j+1)\rceil}\frac{2}{1-y}（or \frac{R^y}{1-y}e^{-Dj}） \ \text{(I omit the details here)}
\end{aligned}
$$
The second inequation follows the fact that $F_{j+1,y}^B (s)=(1-y)F_{j,y}^B (s)+yF_{j,y}^B (s-1)\geq (1-y)F_{j,y}^B (s) $ and $F_{j,y}^B(s)\geq f_{j,y}^B(s)$.

Finally, use some inequalities, we can come to the conclusion. I upload the pdf of this paper with my handwriting notes involving more details.

**The expect regret for stochastic N-armed bandit problem:**
$$
O((\sum_{i=2}^N\frac{1}{\Delta^2_i})^2lnT)
$$

### Further Optimal Regret Bounds for Thompson Sampling

[link](<https://arxiv.org/pdf/1209.3353.pdf>)

Shipra Agrawal, Navin Goyal

Results:

Theorem 1.(Problem-dependent bound) For the N-armed stochastic bandit problem, Thompson Sampling algorithm has expected regret:
$$
\mathbb{E}[R(T)]\leq(1+\epsilon)\sum_{i=2}^N\frac{lnT}{d(\mu_i,\mu_1)}\Delta_i+O(\frac{N}{\epsilon^2})
$$
Theorem 2.(Problem-independent bound)For the N-armed stochastic bandit problem, Thompson Sampling algorithm has expected regret:
$$
\mathbb{E}[R(T)]\leq O(\sqrt{NTlnT})
$$


Previous results:

Lower bound for any consistent algorithms:
$$
\mathbb{E}[R(T)]\geq [\sum_{i=2}^N\frac{\Delta_i}{d(\mu_i,\mu_1)}+o(1)]lnT
$$
UCB1 instance-dependent bound:
$$
\mathbb{E}[R(T)]\leq [8\sum_{i=2}^T\frac{1}{\Delta_i}]lnT+(1+\pi^2/3)(\sum_{i=2}^N\Delta_i)
$$
UCB1 Instance-independent bound:
$$
\mathbb{E}[R(T)]=O(\sqrt{NTlnT})
$$

