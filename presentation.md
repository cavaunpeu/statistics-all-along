# you've been doing statistics all along
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

---

![](./figures/bayes_theorem_betanalpha.png)

---

Bayes Theoreom:

$$
P(y \vert X) = \frac{P(X\vert y)P(y)}{P(X)} = \frac{P(X, y)}{P(X)}
$$

---

# Multi-class classification

---

| dogs | cats | headache |
|---------|----------|----------|
| 3       | 4        | high        |
| 1       | 9        | high        |
| 2       | 2        | low        |
| 5       | 1        | medium        |
| 3       | 3        | low        |
| 6       | 5        | high        |
| 4       | 8        | low        |
| 7       | 1        | medium        |

---

Discriminative models jump directly to estimating $P(y \vert X)$ without modeling its component parts $P(X\vert y), P(y)$ and $P(X)$.

In effect, they can distinguish Darija from French, but can't speak either.

---

Generative models model $P(X\vert y), P(y)$ and $P(X)$. To generate new data, we draw from $P(y)$, then $P(X\vert y)$. To make predictions, we solicit $P(y), P(X\vert y)$, and $P(X)$, then employ Bayes rule outright.

Generative models speak fluent Darija and French. They can distinguish the two because duh, they speak each one and know their differences.

---

# Generative models

---

# The joint distribution

Generative models start by modeling the *joint distribution* of our data, i.e. $P(X, y) = P(X\vert y)P(y)$. Wikipedia (via [this](http://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm) StackOverflow thread, I think) gives a great example of the difference between joint and conditional distributions:

![](./figures/joint_vs_conditional_distributions.png)

---

# Naive Bayes

$$
P_{\theta}(x, y) = P_{\theta}(x\vert y)P_{\theta}(y) = P_{\theta}(y)\prod\limits_{i=1}^m P_{\theta}(x_i\vert y)
$$

- Estimate $\hat{\theta}$ as before
- To make a prediction:

```python
np.argmax([P(x, y) for y in ['low', 'medium', 'high']])
```

The denominator is not important: it is the same throughout.

---

Finally, to *generate likely data given a class*,

1. Draw a class from $P(y)$
2. Draw data from $P(x, y)$

---

> “What I cannot create, I do not understand.”

-- Richard Feynman

---

# Discriminative models

Define 3 separate models. Do the same linear combination. Biggest number wins.

$$
\hat{y}_{\text{low}} = \theta_{\text{low}}^Tx\\
\hat{y}_{\text{medium}} = \theta_{\text{medium}}^Tx\\
\hat{y}_{\text{high}} = \theta_{\text{high}}^Tx\\
$$

These are *proportional* to the joint distribution of the respective class and the data observed.

---

$$
\begin{equation}
P(y\vert x)
= \frac{P(y, x)}{P(x)}
= \frac{e^{\tilde{y}}}{\sum\limits_{y} e^{\tilde{y}}}
= \frac{e^{\big(\sum\limits_{i}w_ix_i\big)_{\tilde y}}}{\sum\limits_{y} e^{\big(\sum\limits_{i}w_ix_i\big)_{\tilde y}}}
\end{equation}
$$

$e$, because the numerator needs to be bigger than the denominator.

---

We won't compute the true $P(y, x)$; our model will not learn the true distribution of data within each class.

Instead:

$$
P(y\vert x) = \frac{P(y, x)}{P(x)} = \frac{\tilde{P}(y, x)}{\text{normalizer}}
$$

This, of course, you'll recognize as the softmax function.

---

```python
linear_regression(loss=mean_squared_error).fit()
```

Maximize the likelihood of the normally-distributed response variable under some set of weights.

---

```python
logistic_regression(loss=log_loss).fit()
```

Maximize the likelihood of the binomially-distributed response variable under some set of weights.

---

```python
naive_bayes(loss=negative_log_joint_likelihood).predict()
```

Compute the joint probability $P(X, y)$ of the data and response variables, then take the argmax.

---

```python
neural_network(loss=categorical_cross_entropy).predict()
```

Compute the conditional distribution $P(y\vert x)$ of the data and response variables. Therein, an *unnormalized* joint probability is computed — not the real thing.

---

Machine learning libraries like scikit-learn are a terrific resource (and it is only in rare circumstance that you should hand-roll a model yourself).

This said, please do note: when you're calling `.fit()` and `.predict()`, you've been doing statistics all along.

---

# Resources
- [Stanford University CS229](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
- [StackOverflow discussion on discriminative vs. generative models](http://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm)
- [CrossValidated discussion on discriminative vs. generative models](https://stats.stackexchange.com/questions/12421/generative-vs-discriminative)
- [On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive Bayes](http://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes.pdf)

---
