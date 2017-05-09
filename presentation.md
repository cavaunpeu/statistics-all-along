# Discriminative vs. Generative Models
---

# ML is:

---
| dogs | cats | turtles |
|---------|----------|----------|
| 3       | 4        | 3        |
| 1       | 9        | 6        |
| 2       | 2        | 7        |
| 5       | 1        | 2        |
| 3       | 3        | 1        |
| 6       | 5        | 4        |
| 4       | 8        | 9        |
| 7       | 1        | 4        |

---

```python
X = df[['dogs', 'cats']]
y = df['turtles']

ŷ = model.predict(X)
```
---

So, how do we do this?

---

```python
import sklearn
```

---

This presentation starts and ends here. Your head is currently above water; we're going to dive into the pool, touch the bottom, then come back to the surface.

---

# Linear regression

---

```python
y = weights^T * X
```

In fancy math:

$$
\hat{y} = \theta^Tx
$$

---
$\hat{y}$ is computed as a function of both $x$ and $\theta$. We'd like this value to be close to the to the true $y$.

We quantify this "closeness" with the "mean squared error" in the canonical case. This is defined exactly as it reads:

$$\mathcal{L} = \sum\limits_{i=1}^m (\theta^Tx^{(i)} - y^{(i)})^2$$

For clarity, now in semi-obnoxious Python:

```python
error = guess - true
loss = (error**2).sum()
```

---
> "I do machine learning; I should probably learn statistics, though, too?"

-- me at some point; hopefully none of you

---

`df['dogs']` is a "random variable." It doesn't depend on anything.

`df['turtles']` is a "random variable." It depends on `df['dogs']` and `df['cats']`.

Goal: change `weights` so as to minimize our error function.

---

This is a probability distribution. It is a lookup table for the likelihood of observing each unique outcome of a random variable.

```python
cats = {2: .17, 5: .24, 3: .11, 9: .48}
```

Typically, probability distributions take parameters which control their shape. Let's define a function to illustrate this:

```python
def prob_distribution_cats(λ):
    return distribution

In [1]: prob_distribution_cats(λ=5)
Out[1]: {2: .37, 5: .14, 3: .13, 9: .36}

In [2]: prob_distribution_cats(λ=84)
Out[2]: {2: .09, 5: .32, 3: .17, 9: .42}
```

The possible values of a random variable are dictated by a probability distribution.

---

What probability distributions dictate the values of our random variables?

Let's start by writing our data as draws from a distribution.

$$
P(\theta), P(X), P(y\vert X, \theta)
$$

---

Which probability distributions describe our data? Let's start with $y$, and assume it is distributed *normally* for now, i.e.

$$
y \sim \mathcal{N}(\mu, \sigma^2)
$$

where $\mathcal{N}$ gives the normal distribution, i.e.

![](https://mathbitsnotebook.com/Algebra2/Statistics/normalturqa.jpg)

Naturally, will assume that $\mu = \theta^Tx$, and $\sigma$ describes some irreducible error in our estimate of $y$. Irreducible error means: $y$ really depends on some other input - e.g. `df['zebras']` - that we haven't included in our model.

---

The values of $y$ are distributed as $y \sim \mathcal{N}(\theta^Tx, \sigma^2)$.

The probability of drawing a specific value of $y^{(i)}$ given $x^{(i)}$ and $\theta$ is given by the normal likelihood function:

$$
P(y^{(i)}\vert x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma}\exp{\bigg(-\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}\bigg)}
$$

-- Carl Friedrich Gauss, 1809

---

Goal: change `weights` so as to minimize our error function.

Which should we choose?

---

A Moroccan walks into a bar. He's wearing a football jersey that's missing a sleeve. He has a black eye, and blood on his jeans. How did he most likely spend his day?

1. At home, reading a book.
2. Training for a bicycle race.
3. At the WAC vs. Raja game drinking Casablanca beers with his friends - all of whom are MMA fighters and hate the other team.

---

Which `weights` maximize the likelihood of having observed the $y$ that we did?

This is called the **maximum likelihood estimate**. To compute it, we simply pick the `weights` that maximize $P(y^{(i)}\vert x^{(i)}; \theta)$ from above. However, we're not just concerned about one outcome $y^{(i)}$; instead, we care about them all.

---

Assuming that $y^{(i)}$ values are independent of one another, we can write their *joint* likelihood as follows:

$$
P(y\vert x; \theta) = \prod\limits_{i=1}^{m}P(y^{(i)}\vert x^{(i)}; \theta)
$$

_**There is nothing scary about this product; whereas the lone term gives the likelihood of one data point, the product gives the likelihood of having observed all data points. This is akin to:**_

```python
die = choose_from([1, 2, 3, 4, 5, 6])
P(die > 2, die < 4, die == 1) = 4/6 * 3/6 * 1/6
```

---

Since probabilities are numbers in $[0, 1]$, the product of a bunch of probabilities gets very small, very quick. For this reason, we often take the natural logarithm: the log of a product becomes the sum of logs.

$$
\begin{align*}
\log{P(y\vert x; \theta)}
&= \log{\prod\limits_{i=1}^{m}P(y^{(i)}\vert x^{(i)}; \theta)}\\
&= \sum\limits_{i=1}^{m}\log{P(y^{(i)}\vert x^{(i)}; \theta)}\\
&= \sum\limits_{i=1}^{m}\log{\frac{1}{\sqrt{2\pi}\sigma}\exp{\bigg(-\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}\bigg)}}\\
&= \sum\limits_{i=1}^{m}\log{\frac{1}{\sqrt{2\pi}\sigma}} + \sum\limits_{i=1}^{m}\log{\exp{\bigg(-\frac{(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}\bigg)}}\\
&= m\log{\frac{1}{\sqrt{2\pi}\sigma}} - \frac{1}{2\sigma^2}\sum\limits_{i=1}^{m}(y^{(i)} - \theta^Tx^{(i)})^2\\
&= C_1 - C_2\sum\limits_{i=1}^{m}(y^{(i)} - \theta^Tx^{(i)})^2\\
\end{align*}
$$

_**Maximizing the log-likelihood of our data with respect to $\theta$, i.e. `weights`, is equivalent to maximizing the negative mean squared error between $y$ and $\hat{y}$.**_

---

Most optimization routines, i.e. strategies for actually solving for $\theta$, minimize instead of maximize. As such, in machine learning, we both do and say:

_**Minimizing the log-likelihood of our data with respect to $\theta$, i.e. `weights`, is equivalent to minimizing the mean squared error between $y$ and $\hat{y}$.**_

---

Logistic regression:

$$
p = \frac{1}{1 + e^{-\theta^Tx}}
$$

Canonical loss function:

$$
\mathcal{L} = \sum\limits_{i = 1}^m y^{(i)}\log{p^{(i)}} + (1 - y^{(i)})\log{(1 - p^{(i)})}
$$

---

In probabilistic terms:

In logistic regression, $y$ is a binary random variable: it is a thing that takes values in $\{0, 1\}$. Given $x$ and $\theta$, which likelihood function tells us the number of $1$'s we can expect to see over $n$ trials?

![](http://www.boost.org/doc/libs/1_52_0/libs/math/doc/sf_and_dist/graphs/binomial_pdf_1.png)

---

Given a binomial likelihood:

$$P(y\vert x; \theta) = \prod\limits_{i = 1}^m(p^{(i)})^{y^{(i)}}(1 - p^{(i)})^{1 - y^{(i)}}$$

If this looks confusing:
- Disregard the left side
- Ask yourself: what is the probability of observing the following specific sequence of coin flips, where $P(\text{heads}) = .7$:

$$
\begin{align*}
P(\text{heads}, \text{tails}, \text{heads}, \text{heads}, \text{heads})
&= (.7^1 * .3^0)(.7^0 * .3^1)(.7^1 * .3^0)(.7^1 * .3^0)\\
&= .7 * .3 * .7 * .7\\
&= .102899
\end{align*}
$$

---

The negative log-likelihood gives:

$$
\begin{align*}
-\log{P(y\vert x; \theta)}
&= -\log{\prod\limits_{i = 1}^m(p^{(i)})^{y^{(i)}}(1 - p^{(i)})^{1 - y^{(i)}}}\\
&= -\sum\limits_{i = 1}^m\log{\bigg((p^{(i)})^{y^{(i)}}(1 - p^{(i)})^{1 - y^{(i)}}\bigg)}\\
&= -\sum\limits_{i = 1}^m\log{(p^{(i)})^{y^{(i)}} + \log{(1 - p^{(i)})^{1 - y^{(i)}}}}\\
&= -\sum\limits_{i = 1}^my^{(i)}\log{(p^{(i)})} + (1 - y^{(i)})\log{(1 - p^{(i)})}\\
\end{align*}
$$

This will look familiar as well.

# Resources
- [Andrew Ng](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
