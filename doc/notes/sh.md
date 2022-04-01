# Successive Halving Sanity Checking

Input:

- Budget $B=32$
- $n=8$ arms.
- $l_{i,k}$ for the $k$th loss from the $i$th arm.

Initialisation:

- $S_0 = [n]$ (8 arms)

Steps:
For $k = 0, 1, \cdots [log_2(n)]-1=2$

---

- $k = 0\implies r_k=[\frac{B}{|S_k| * log_2(n)}]=[\frac{32}{8 * 3}]=1$
- Pull each arm $r_k=1$ times.
- $R_k=\sum_{j=0}^k r_j=1$
- Sort $S_0$ in ascending order with respect to $l$.
- $S_1=\{i\in S_0: \ell_{\sigma_{k(i)}, R_k}\leq \ell_{\sigma_{k(|S_0|/2)}, R_k}\}$

(In our case, 8 arms go through 0-4095 points, 4 arms left)

---

- $k=1\implies r_k=[\frac{B}{|S_1| * log_2(n)}]=[\frac{32}{4*3}]=2$.
- Pull each arm $r_k=2$ times.
- $R_k=\sum_{j=0}^k r_j=r_0+r_1=3$.
- Sort $S_1$ in ascending order with respect to $l$.
- $S_2=\{i\in S_1: \ell_{\sigma_{k(i)}, R_k}\leq \ell_{\sigma_{k(|S_0|/2)}, R_k}\}$

(In our example, 4 arms are pulled twice, they go through 4095-8191-12287, 2 arms left)

---

- $k=2\implies r_k=[\frac{B}{|S_2| * log_2(n)}]=[\frac{32}{2*3}]=5$.
- Pull each arm $r_k=5$ times.
- $R_k=\sum_{j=0}^k r_j=r_0+r_1+r_2=8$.
- Sort $S_2$ in ascending order with respect to $l$.
- $S_3=\{i\in S_2: \ell_{\sigma_{k(i)}, R_k}\leq \ell_{\sigma_{k(|S_0|/2)}, R_k}\}$

(In our example, 2 arms are pulled $5$ times, they go through 4095-8191-12287-16383-20479-24575-25000, only 4 pulls are possible)
