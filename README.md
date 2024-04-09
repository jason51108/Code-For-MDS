# Statistical Inference for Multidimensional Scaling

## 1. Introduction
The Latent Space Model (LSM) is a type of probabilistic model employed in Link Prediction in Complex Networks. The primary objective of probabilistic models is to uncover latent structures within observed networks and subsequently utilize the learned model to predict missing links. Given a target network, $G=(V,E)$, the probabilistic model optimizes a predefined objective function to establish a model, represented by a set of parameters $\Theta$, which best fits the observed data of the target network. Subsequently, the conditional probability $P(a_{ij}=1|\Theta)$ is utilized to estimate the probability of the non-existence of a link $(i,j)$ given the parameters $\Theta$.


## 2. Generalized Latent Space Model
Let each node $i$ be assigned with a latent position $\bf \theta_i \in \R^{k}$ in the $k$-dimensional latent space and a degree parameter $\alpha_i \in \R$ that accounts for node heterogeneity. Then for each node pair $1 \leq i < j \leq n$, we assume that entries $a_{ij}$ are conditionally independent of the parameter $\gamma_{ij}$
$$
    a_{ij} \sim p(\cdot \mid \gamma_{ij}) \quad \text{with} \quad \gamma_{ij}=\sigma(m_{ij})=\sigma(\alpha_i+\alpha_j-\Vert\bf \theta_i-\bf \theta_j\Vert^2)
$$
where each edge $a_{ij}$ is a random variable following the distribution $p(\cdot \mid \gamma_{ij})$ with the parameter $\gamma_{ij}$ determined by the latent position $(\bf \theta_i, \bf \theta_j)$ and degree parameters $(\alpha_i,\alpha_j)$ through a link function $\sigma(\cdot)$. Typically, $\sigma(\cdot)$ is a smooth and increasing function, ensuring that the model exhibits the desirable property that higher distance similarity of latent positions and greater node heterogeneity values lead to higher expected values of $a_{ij}$. 

We mainly discuss the case where the probability density function $p(\cdot \mid \gamma_{ij})$ belongs to the exponential family distribution. The probability density or mass function of $a_{ij}$ takes an exponential family form
$f_{ij}\left(a_{ij}|m_{ij},\varphi_{ij}\right)=\exp\left\{\varphi_{ij}^{-1}\{a_{ij}m_{ij}-b_{ij}\left(m_{ij}\right)\}+c_{ij}\left(a_{ij},\varphi_{ij}\right)\right\}$, where $b_{ij}(\cdot)$ and $c_{ij}(\cdot)$ are pre-specied functions, $\varphi_{ij}$ is a dispersion parameter. The probability density function $f_{ij}(\cdot)$ depends on variables $i$ and $j$, therefore the function can be of different types. We give some examples below.

**Binomial model.** In this case, the values of the $a_{ij} \in \{0,1\}$ are binary. The probability mass function of Binomial is $p_{ij}(a_{ij}\mid\gamma_{ij})=\gamma_{ij}^{a_{ij}}\left(1-\gamma_{ij}\right)^{1-a_{ij}}$, let $m_{ij}=\log (\gamma_{ij}/{1-\gamma_{ij}})$, $p_{ij}\left(a_{ij}\mid m_{ij}\right)=\exp\{a_{ij}m_{ij}-\log\left(1+e^{m_{ij}}\right)\}$, where $\varphi_{ij}=1$, $b_{ij}(m_{ij}) = \log\left(1+e^{m_{ij}}\right)\}$ and $c_{ij}(a_{ij},\varphi_{ij})=1$

**Poisson model.** Under the circumstances, link function is $\sigma(x)=e^x$. The probability mass function is $p_{ij}(a_{ij}\mid \gamma_{ij})=\exp\{a_{ij}\log \gamma_{ij} - \gamma_{ij} - \log (a_{ij}!)\} = \exp\{a_{ij}m_{ij} - e^{m_{ij}} -\log (a_{ij}!)\}$
$\varphi_{ij}=1$, $b_{ij}(m_{ij})=e^{m_{ij}}$ and $c_{ij}(a_{ij},\varphi_{ij})=-\log(a_{ij}!)$.

**Normal model.** For a continuous variable $a_{ij}$, we assume $f_{ij}$ to be a normal density function, where $\varphi_{ij}$ is the variance, $b_{ij}(m_{ij})=m_{ij}^2/2$ and $c_{ij}(a_{ij},\varphi_{ij})=-a_{ij}^2/(2\varphi_{ij})-(\log (2\pi \varphi_{ij}))/2$.

## 3. Estimation

### 3.1 Likelihood-based Estimation

### 3.2 Alternating minimization algorithm