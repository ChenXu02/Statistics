# Detailed Derivation of the Variational Gaussian Process ELBO

```latex
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\usepackage{hyperref}
\usepackage{bbm}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}

\section*{Detailed Derivation of the Variational Gaussian Process ELBO}

\subsection*{1. Gaussian Process Regression: Background}

We consider a regression problem with training data $\{(x_i, y_i)\}_{i=1}^N$, where $x_i \in \mathbb{R}^D$ and $y_i \in \mathbb{R}$.  A Gaussian Process (GP) prior is placed on a latent function $f: \mathbb{R}^D \to \mathbb{R}$.  Specifically,
\[
  f(\cdot)\;\sim\;\mathcal{GP}\bigl(0,\;k(\cdot,\cdot)\bigr),
\]
which implies that for any finite collection of inputs $\{x_i\}_{i=1}^N$, the corresponding latent vector 
\[
  \mathbf{f} \;=\; [\,f(x_1),\,f(x_2),\,\dots,\,f(x_N)\,]^\top \;\in\;\mathbb{R}^N
\]
follows a multivariate normal distribution:
\[
  p(\mathbf{f}) \;=\; \mathcal{N}\bigl(\mathbf{f}\,\bigm|\,\mathbf{0},\,K_{NN}\bigr),
\]
where $\bigl[K_{NN}\bigr]_{ij} = k(x_i, x_j)$ is the $N\times N$ kernel (covariance) matrix evaluated at the training inputs.

We assume independent Gaussian observation noise on the outputs:
\[
  y_i \;=\; f(x_i) \;+\; \varepsilon_i, 
  \quad
  \varepsilon_i \sim \mathcal{N}(0,\;\sigma_n^2), 
  \quad i=1,\dots,N.
\]
Thus the likelihood factorizes as
\[
  p(\mathbf{y}\,\bigm|\mathbf{f}) 
  \;=\; \prod_{i=1}^N p(y_i \mid f_i)
  \;=\; \prod_{i=1}^N \mathcal{N}\bigl(y_i \,\bigm|\, f_i,\;\sigma_n^2\bigr).
\]

The marginal likelihood (evidence) is
\[
  p(\mathbf{y})
  \;=\; \int_{\mathbb{R}^N} p(\mathbf{y}\mid \mathbf{f})\,p(\mathbf{f})\,\mathrm{d}\mathbf{f}
  \;=\; \mathcal{N}\bigl(\mathbf{y}\,\bigm|\mathbf{0},\,K_{NN} + \sigma_n^2 I_N\bigr).
\]
Computing and optimizing $\log p(\mathbf{y})$ requires $\mathcal{O}(N^3)$ time and $\mathcal{O}(N^2)$ storage, due to inversion and determinant of $N\times N$ matrices.

\subsection*{2. Sparse Gaussian Process via Inducing Variables}

To scale Gaussian Processes to large $N$, we introduce $M$ \emph{inducing points} 
\[
  Z = \{\,z_j\in\mathbb{R}^D\mid j=1,\dots,M\}, 
  \quad M \ll N,
\]
and define the corresponding inducing variables
\[
  \mathbf{u} \;=\; [\,f(z_1),\,f(z_2),\,\dots,\,f(z_M)\,]^\top \;\in\;\mathbb{R}^M.
\]
Under the joint GP prior, the pair $(\mathbf{f}, \mathbf{u})$ is jointly Gaussian:
\[
  \begin{pmatrix} \mathbf{f} \\[4pt] \mathbf{u} \end{pmatrix}
  \;\sim\;
  \mathcal{N}\!\Biggl(\,
    \begin{pmatrix} \mathbf{0} \\[2pt] \mathbf{0} \end{pmatrix},
    \begin{pmatrix}
      K_{NN} & K_{NM} \\[4pt]
      K_{MN} & K_{MM}
    \end{pmatrix}
  \Biggr),
\]
where
\begin{itemize}
  \item $K_{NN} \in \mathbb{R}^{N\times N}$ has entries $[K_{NN}]_{ij} = k(x_i, x_j)$,
  \item $K_{MM} \in \mathbb{R}^{M\times M}$ has entries $[K_{MM}]_{jk} = k(z_j, z_k)$,
  \item $K_{NM} \in \mathbb{R}^{N\times M}$ has entries $[K_{NM}]_{i j} = k(x_i, z_j)$, and $K_{MN} = K_{NM}^\top$.
\end{itemize}

From this joint Gaussian we can write the conditional distribution of $\mathbf{f}$ given $\mathbf{u}$:
\[
  p(\mathbf{f}\mid \mathbf{u})
  \;=\;
  \mathcal{N}\Bigl(\mathbf{f}\,\Bigm|\,
    K_{NM}\,K_{MM}^{-1}\,\mathbf{u},\;
    \underbrace{K_{NN} - K_{NM}\,K_{MM}^{-1}\,K_{MN}}_{=\;\widetilde{K}}
  \Bigr).
\]
The prior on the inducing variables is
\[
  p(\mathbf{u}) \;=\; \mathcal{N}\bigl(\mathbf{u}\mid \mathbf{0},\,K_{MM}\bigr).
\]
Hence the full generative story is:
\[
  p(\mathbf{u}) 
  \;=\; \mathcal{N}(\mathbf{u}\mid \mathbf{0},\,K_{MM}), 
  \quad
  p(\mathbf{f}\mid \mathbf{u}) 
  \;=\; \mathcal{N}\bigl(\mathbf{f}\mid K_{NM}K_{MM}^{-1}\mathbf{u},\,\widetilde{K}\bigr),
  \quad
  p(\mathbf{y}\mid \mathbf{f}) 
  \;=\; \prod_{i=1}^N \mathcal{N}(y_i\mid f_i,\sigma_n^2).
\]
The joint density is therefore
\[
  p(\mathbf{y}, \mathbf{f}, \mathbf{u})
  \;=\;
  p(\mathbf{y}\mid\mathbf{f})
  \;p(\mathbf{f}\mid\mathbf{u})
  \;p(\mathbf{u}).
\]
Marginalizing out both $\mathbf{f}$ and $\mathbf{u}$ recovers the true marginal likelihood
\[
  p(\mathbf{y})
  \;=\;
  \iint 
    p(\mathbf{y}\mid \mathbf{f})\,
    p(\mathbf{f}\mid \mathbf{u})\,
    p(\mathbf{u})
  \;\mathrm{d}\mathbf{f}\,\mathrm{d}\mathbf{u},
\]
which is still $\mathcal{O}(N^3)$ to compute exactly.

\subsection*{3. Variational Inference: Introducing \texorpdfstring{$q(\mathbf{f},\mathbf{u})$}{q(f,u)}} 

We aim to approximate the true intractable posterior 
\[
  p(\mathbf{f}, \mathbf{u} \mid \mathbf{y})
  \;=\;
  \frac{p(\mathbf{y}, \mathbf{f}, \mathbf{u})}{p(\mathbf{y})}
\]
by a tractable variational distribution 
\[
  q(\mathbf{f}, \mathbf{u})
  \;=\;
  p(\mathbf{f}\mid \mathbf{u})\;q(\mathbf{u}),
\]
where $q(\mathbf{u})$ is chosen as a free-form Gaussian:
\[
  q(\mathbf{u}) \;=\; \mathcal{N}\bigl(\mathbf{u} \mid \bm{m},\,S\bigr),
  \qquad
  S \in \mathbb{R}^{M\times M}, 
  \;S \succ 0.
\]
Thus the variational parameters are the mean $\bm{m}\in\mathbb{R}^M$ and the covariance $S$ for $q(\mathbf{u})$, plus any kernel hyperparameters and inducing locations $Z$.

Define the variational free energy (Evidence Lower Bound, ELBO) as
\[
  \mathcal{L}
  \;=\; \mathbb{E}_{q(\mathbf{f},\mathbf{u})}\bigl[\log p(\mathbf{y}, \mathbf{f}, \mathbf{u})\bigr]
  \;-\; \mathbb{E}_{q(\mathbf{f},\mathbf{u})}\bigl[\log q(\mathbf{f}, \mathbf{u})\bigr].
\]
Since $\log p(\mathbf{y}) = \log p(\mathbf{y},\mathbf{f},\mathbf{u}) - \log p(\mathbf{f},\mathbf{u}\mid \mathbf{y})$, one can show by adding and subtracting that
\[
  \log p(\mathbf{y})
  \;=\;
  \mathcal{L}
  \;+\;
  \mathrm{KL}\bigl[q(\mathbf{f},\mathbf{u})\,\|\,p(\mathbf{f},\mathbf{u}\mid \mathbf{y})\bigr],
\]
and since $\mathrm{KL}(\cdot)\ge 0$, we have $\mathcal{L} \le \log p(\mathbf{y})$.  Hence maximizing $\mathcal{L}$ tightens the lower bound.

\subsection*{4. Expanding and Simplifying the ELBO}

First, write out each term:
\[
\begin{aligned}
  \mathcal{L} 
  &= \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}\Bigl[\log p(\mathbf{y}\mid \mathbf{f}) 
      + \log p(\mathbf{f}\mid \mathbf{u}) 
      + \log p(\mathbf{u}) \Bigr] 
    \;-\;
    \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}\Bigl[\log q(\mathbf{f}, \mathbf{u})\Bigr].
\end{aligned}
\]
Since $q(\mathbf{f}, \mathbf{u}) = p(\mathbf{f}\mid\mathbf{u})\,q(\mathbf{u})$, we have
\[
  \log q(\mathbf{f}, \mathbf{u})
  = \log p(\mathbf{f}\mid \mathbf{u}) + \log q(\mathbf{u}).
\]
Thus
\[
  \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}\bigl[\log q(\mathbf{f},\mathbf{u})\bigr]
  = \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}\bigl[\log p(\mathbf{f}\mid \mathbf{u})\bigr]
  \;+\; \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}\bigl[\log q(\mathbf{u})\bigr].
\]
Similarly,
\[
  \mathbb{E}_{q(\mathbf{f}, \mathbf{u})}\bigl[\log p(\mathbf{f}\mid \mathbf{u})\bigr]
  \quad\text{appears in both positive and negative terms, hence will cancel out.}
\]
So we group terms:
\[
\begin{aligned}
  \mathcal{L}
  &= \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y}\mid \mathbf{f})]}_{(A)}
    + \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{f}\mid \mathbf{u})]}_{(B)}
    + \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{u})]}_{(C)} \\[-1ex]
  &\quad
    -\,\underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{f}\mid \mathbf{u})]}_{(B)}
    \;-\; \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log q(\mathbf{u})]}_{(D)}.
\end{aligned}
\]
The terms $(B)$ cancel out, leaving
\[
  \mathcal{L}
  = \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y}\mid \mathbf{f})]}_{(A)}
    \;+\; \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{u})] - \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log q(\mathbf{u})]}_{(C)-(D)}.
\]
Since $\log p(\mathbf{u})$ and $\log q(\mathbf{u})$ depend only on $\mathbf{u}$, we have
\[
  \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{u})]
  = \mathbb{E}_{q(\mathbf{u})}[\log p(\mathbf{u})], 
  \quad
  \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log q(\mathbf{u})]
  = \mathbb{E}_{q(\mathbf{u})}[\log q(\mathbf{u})].
\]
Thus
\[
  \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{u})] 
  \;-\; \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log q(\mathbf{u})]
  = -\,\mathrm{KL}\bigl[q(\mathbf{u}) \,\bigl\|\, p(\mathbf{u})\bigr].
\]
Hence
\[
  \mathcal{L}
  = \underbrace{\mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y}\mid \mathbf{f})]}_{\text{Term (A)}}
    \;-\; \mathrm{KL}\bigl[q(\mathbf{u})\,\|\,p(\mathbf{u})\bigr].
\]

\subsection*{5. Evaluating Term (A): $\displaystyle \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y}\mid \mathbf{f})]$}

We have
\[
  p(\mathbf{y}\mid \mathbf{f})
  = \prod_{i=1}^N p(y_i\mid f_i),
  \quad
  p(y_i \mid f_i)
  = \mathcal{N}(y_i \mid f_i, \sigma_n^2).
\]
Therefore
\[
  \log p(\mathbf{y}\mid \mathbf{f})
  = \sum_{i=1}^N \log p(y_i \mid f_i)
  = \sum_{i=1}^N \left[
    -\tfrac12 \log(2\pi\sigma_n^2)
    \;-\; \tfrac{1}{2\sigma_n^2}\,(y_i - f_i)^2
  \right].
\]
Taking expectation under $q(\mathbf{f},\mathbf{u}) = p(\mathbf{f}\mid \mathbf{u})\,q(\mathbf{u})$ yields
\[
  \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y}\mid \mathbf{f})]
  = \mathbb{E}_{q(\mathbf{u})}\Bigl[
    \mathbb{E}_{p(\mathbf{f}\mid \mathbf{u})}\bigl[\log p(\mathbf{y}\mid \mathbf{f})\bigr]
  \Bigr]
  = \sum_{i=1}^N 
    \mathbb{E}_{q(\mathbf{u})}
    \Bigl[\,\mathbb{E}_{p(f_i \mid \mathbf{u})}\bigl[\log p(y_i \mid f_i)\bigr]\Bigr].
\]
Define the marginal distribution of each latent $f_i$ under the variational scheme:
\[
  q(f_i)
  \;=\;
  \int_{\mathbb{R}^M} p(f_i \mid \mathbf{u})\,q(\mathbf{u})\,\mathrm{d}\mathbf{u}.
\]
Because $p(f_i \mid \mathbf{u})$ is Gaussian (its mean and variance can be read off from the conditional $p(\mathbf{f}\mid \mathbf{u})$), the marginal $q(f_i)$ is also Gaussian.  Thus
\[
  \mathbb{E}_{q(\mathbf{u})}\bigl[\mathbb{E}_{p(f_i \mid \mathbf{u})} [\log p(y_i \mid f_i)]\bigr]
  = \mathbb{E}_{q(f_i)}\bigl[\log p(y_i \mid f_i)\bigr].
\]
Hence Term (A) can be written as
\[
  \mathbb{E}_{q(\mathbf{f},\mathbf{u})}[\log p(\mathbf{y}\mid \mathbf{f})]
  = \sum_{i=1}^N \mathbb{E}_{q(f_i)}\bigl[\log p(y_i \mid f_i)\bigr].
\]

Putting everything together, the ELBO is
\[
  \boxed{
    \mathcal{L}
    \;=\;
    \sum_{i=1}^N \mathbb{E}_{q(f_i)}\bigl[\log p(y_i \mid f_i)\bigr]
    \;-\; \mathrm{KL}\bigl[q(\mathbf{u}) \,\|\, p(\mathbf{u})\bigr].
  }
\]

\subsection*{6. Final Expression and Interpretation}

Thus we have derived the Variational Gaussian Process ELBO.  To restate:

\begin{align*}
  \mathcal{L}
  &= \sum_{i=1}^N \mathbb{E}_{q(f_i)}\bigl[ \log p(y_i \mid f_i) \bigr]
     \;-\;
     \mathrm{KL}\bigl[q(\mathbf{u}) \,\bigm\|\, p(\mathbf{u})\bigr],
  \\[6pt]
  \text{where} 
  \quad
  &q(\mathbf{u}) = \mathcal{N}(\mathbf{u} \mid \bm{m},\,S),
  \qquad
  q(f_i) = \int p(f_i \mid \mathbf{u})\,q(\mathbf{u})\,\mathrm{d}\mathbf{u}.
\end{align*}

Key points:
\begin{itemize}
  \item The term $\displaystyle \sum_{i=1}^N \mathbb{E}_{q(f_i)}[\log p(y_i \mid f_i)]$ encourages the predictive distribution of each latent $f_i$ to explain the observed label $y_i$ well.
  \item The KL term $\mathrm{KL}[q(\mathbf{u}) \| p(\mathbf{u})]$ regularizes the variational posterior $q(\mathbf{u})$ to stay close to the prior $p(\mathbf{u})$ on the inducing variables.
  \item All computations involving $\mathbf{f}$ are done \emph{conditionally} on $\mathbf{u}$ and integrated out analytically, so no $\mathcal{O}(N^3)$ inversion of $K_{NN}$ is required.
  \item The resulting computational complexity for evaluating and optimizing $\mathcal{L}$ is $\mathcal{O}(N\,M^2 + M^3)$, assuming $M \ll N$.
\end{itemize}

This completes the detailed step-by-step derivation of the ELBO for the Variational Sparse Gaussian Process.

\end{document}
