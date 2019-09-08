---
layout:     post                    # 使用的布局（不需要改）
title:      Garbage In, Reward Out: Bootstrapping Exploration in Multi-Armed Bandits             # 标题 
subtitle:   HKUST Summer Research Task 4 #副标题
date:       2019-08-06            # 时间
author:     BY Zhipeng Liang                    # 作者
header-img: img/what-is-reinforcement-learning-the-complete-guide.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    -Online Learning
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


## Garbage In, Reward Out: Bootstrapping Exploration in Multi-Armed Bandits

*This blog is attached with [My Slides](https://liangzp.github.io/Online-Learning/GIRO.html)*

Arguably, the most used and stuied exploration straties in multi-armed and contextual bandits are the $\epsilon$-greedy policy, the optimism in the fact of uncertainty (e.g. UCB) and Thompson sampling. $\epsilon$-greedy policy is easy to implement but also statistically suboptimal. UCB is computationally efficient in multi-armed and linear bandits but may become suboptimal when reward function is non-linear. Thompson Sampling demonstrates excellent performance and optimal regret bound, but sampling from posterior in general setting is always costly. 

To address these problems, bootstrapping exploration has been proposed. To be specific, bootstrapping exploration consists of two components: bootstrapping which is used to introduce sampling variance and pseudo reward, which is used to introduce optimism. Compares with the straties mentioned above, bootstrapping is easy to implement because we don't need to construce problem-specific confidence sets or posterios like UCB and Thompson Sampling. And it is competitive in empirical performance.

In the following analysis, we assume that the unique optimal arm is 1.

Based on algorithm 1, the paper provide theorem 1 to help us formulate the intuition that balancing exploration and exploitation is important to keep a low regret. Theorem 1 tells us that the upper bound of the regret is consists of two terms, $a_i$ and $b_i$. Intuitively, if we want to bound $a_i$, we want 1/Q-1 as small as possible. Since probability is always less than 1. therefore, we hope the probability of $\mu$ sampled from history of arm 1 can become very close to 1. If the sampling method can ensure $\hat{\mu}$ concentration to the true one, we know if we set tau is the average between $\mu_1$ and $\\mu_i$, then when s is large enough, we can assure this is large in a high probability. That how exploitation contributes to the low regret. But how can we ensure $Q_{1,s}$ is still large enough even when s is small? That's why we need optimism and exploration. Since s is small and we cannot identity the optimal one, we just let each arm enjoys a large $\hat{\mu}$, $\hat{\mu}$ is larger than $\tau$ with a high probability. Then $Q_{1,s}(\tau_i)$ is high enough. Of course this is also enlarge $b_i$, but by carefully design the algorithm, we can still keep a good balance. 



In order to ensure enough exploration, one simple idea is to use bootstrapping to add some sampling variance in the algorithm. But such a naive method cannot improve it in fact and it may incur linear regret, this is what Lemma 1 says.  Consider a Bernoulli bandit with 2 arms. At the very beginning, the algorithm will decide which arm to pull if there estimators equal. Then algorithm will pull arm 1 and arm 2 once. There is always a positive probability for arm 1 producing 0 reward and arm 2 producing 1 reward, while the algorithm will prefer arm 2. Then a linear regret will incur.



The problem of this algorithm is that once arm 1 produce 0 reward at the first pull, then the estimator of it will remain zero. To solve this problem, Thompson Sampling uses posterior to give it positive probability to pull it in the future. While in this paper, they add pseudo reward in the history. Even when arm 1 fails at the first time, there is still some 1 in its history and it is hopefully we can sample them into the bootstrap sample and finally reactive arm 1.



Here the paper provides us informal justification of their algorithm. They justify two things: first the mean of the bootstrap sample approach to a scaled and shifted true parameters. So we can see that the differentiation of the estimators between optimal and suboptimal arm becomes small when a grows. We can also understand the optimism in the following way. The paper defines a algorithm is optimistic when any unfavorable history is less likely than being optimistic under that history, i.e.
$$
\mathbb{P}(\hat{\mu}\geq (\mu_i+a)/\alpha | E)\geq \mathbb{P}[E]
$$

where:

$$
E=\{V_{i,s}/(\alpha s)=(\mu_i+a)/\alpha-\epsilon\}
$$
Then they prove if $\hat{\mu}|V_{i,s}$ and $V_{i,s}/(\alpha s)\sim$ Normal distribution, then $a\geq \frac{1}{3}$ is sufficient to derive such a conclusion. Of course normal distribution is unrealistic assumption, so they need novel analysis in the part c in the appendix.

As we can see in Theorem 2, bootstrapping exploration can also achieve regret bound match with UCB1 and in numerical experiments, it demonstrates promising potential in general setting.





