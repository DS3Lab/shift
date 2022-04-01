# Successive Halving

## Budget

Minimal Budget: the budget needed to bootstrap successive halving. At least, we need

$$
B_{min}=M \times \lceil \log_\eta M \rceil
$$

Once we have a budget larger than this, we are guaranteed to have exactly $1$ model remaining.

The remained single model is reduced from $2$ models, and the remained model will not be processed only, i.e., with the minimal budget, we at least give samples to two models.

## Pulls and Chunks

At $k$th step, the number of remaining model is defined by

$$
S_k = \lceil \frac{M}{\eta^{k+1}} \rceil
$$

At $k$th step, we need $r_k$ additional pulls, where

$$
r_k = \lfloor \frac{B}{S_{k-1}\lceil\log_\eta M \rceil} \rfloor
$$

In this step, each model will be given $r_k \times C$ more samples. The two models will be given, in total, 

$$
\sum_{k=0}^{K}r_k\times C
$$

samples, where $K=\lceil \log_\eta M \rceil-1$.

Hence, if we want the two models to see the entire dataset, we have

$$
\sum_{k=0}^{\lceil \log_\eta M \rceil-1}r_k\times C \geq N
$$

which leads to 

$$
C\geq \frac{N}{\sum_{k=0}^{\lceil \log_\eta M \rceil-1}r_k}
$$
