---
layout:     post                    # 使用的布局（不需要改）
title:      Summary of Online Learning             # 标题 
date:       2019-08-16            # 时间
author:     BY Zhipeng Liang                    # 作者
header-img: img/what-is-reinforcement-learning-the-complete-guide.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - Online Learning
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


## Metric Learning

#### Introduction

A possible informal formulation of the metric learning problem could be given as follows: given an input distance function d(x,y) between objects x and y (for example, the Euclidean distance), along with supervised information regarding an ideal distance, construct a new distance function d˜(x,y) which is “better” than the original distance function. This survey will focus, for the most part, on learning distance functions d˜(x,y) of the form d(f(x),f(y)) for some function f — that is, we learn some mapping f and utilize the original distance function over the mapped data. We will denote this approach as global metric learning methods, since they learn a single mapping f to be applied to all the data. The fact that the supervised information is
a function of the ideal distance (or similarity) is key to distinguishing the methods we study in this survey from other existing techniques such as dimensionality reduction methods or classification techniques. Thus, in this survey we will mainly focus on metric learning as a supervised learning problem.


We will break down global metric learning into two subclasses —linear and nonlinear. For both cases, we will mainly focus on the case where the input distance function is the Euclidean distance, i.e., d(x,y) = \|x − y\|_2. In the linear case, we aim to learn a linear mapping based on supervision, which we can encode as a matrix G such that the learned distance is \|Gx − Gy\|_2. To achieve convexity, many methods assume that G is square and full-rank, leading to convex optimization problems with positive semi-definiteness constraints.

Nonlinear methods for global metric learning is that the distance function is the mroe general $d(x,y)=\|f(x)-f(y)\|_2$. One of the most well-understood and effective techniques for learning such nonlinear mappings is to extend linear methods
via kernelization. The basic idea is to learn a linear mapping in the feature space of some potentially nonlinear function φ; that is, the distance function may be written d(x,y) = \|Gφ(x) − Gφ(y)\|_2, where φ may be a nonlinear function.
 
#### Distance Learning via Linear Transformations (Mahalanobis metric learning)
the Euclidean distance between two whitened variables is simply the Mahalanobis distance. Note that, in the case of the wine example discussed earlier, the use of the Mahalanobis distance would avoid the scenario in which one feature dominates in the computation of the Euclidean distance, as the data has implicitly been whitened. This generalized notion of a Mahalanobis distance exactly captures the idea of learning a global linear transformation.

##### Model

$$
\min_{A\in dom(A)} \ \ \ \ r(A)\\
s.t.\ c_i(X^TAX)\leq 0,\ 1\leq i\leq m
$$

where c(X^TAX) can take various definiton, e.g.:
$$
c(X^TAY)=max(0,d_A(x_i,x_j)-u),\ \(i,j)\in S\\
c(X^TAY)=max(0,l-d_A(x_i,x_j)),\ \(i,j)\in D
$$

Thetwo most popular forms of supervision for metric learning are given by (a) similarity/dissimilarity constraints, and (b) relative distance constraints. In many applications, the form of the side information is governed by the application. For instance, suppose we are applying metric learning for face identity, and we want to gather supervision from human
subjects. It is typically easier for a subject to provide relative distance constraints than similarity and dissimilarity constraints (i.e., it is possible to say that image a is more similar to b than to c, but it may be difficult to determine whether an arbitrary pair of images should be considered similar or dissimilar). On the other hand, if we have a fully supervised training data set consisting of class labels for all training data, it is straightforward and standard to create similarity constraints
for all pairs of objects of the same class and dissimilarity constraints for pairs of objects of different classes.

