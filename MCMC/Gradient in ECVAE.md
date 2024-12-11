# Gradient Calculation for MCMC Sampling in Energy-Calibrated Variational Learning


In the MCMC sampling process for Energy-Calibrated Variational Learning, the gradient of the loss function is essential for updating the latent variable $z$. Since the conditional density $-\log p_{\beta, \theta}(z̃ | x)$ is generally intractable, we approximate and compute it indirectly.

## 1. Using the Unnormalized Joint Distribution
Instead of directly calculating $-\log p_{\beta, \theta}(z̃ | x)$, we use the joint distribution:
$p_{\beta, \theta}(z̃ | x) \propto p_{\beta, \theta}(z̃, x),$
where the joint distribution can be decomposed as:
$p_{\beta, \theta}(z̃, x) = p_{\beta, \theta}(x | z̃) \cdot p_{\theta}(z̃).$

The gradient for $-\log p_{\beta, \theta}(z̃ | x)$ simplifies to:
$\nabla_{z̃} \log p_{\beta, \theta}(z̃ | x) = \nabla_{z̃} \log p_{\beta, \theta}(z̃, x).$

## 2. Decomposition of the Joint Distribution
### a. Likelihood Term
The likelihood $p_{\beta, \theta}(x | z̃)$ is modeled by the decoder $g_\beta(z̃)$, typically as a Gaussian distribution:
$p_{\beta, \theta}(x | z̃) \propto \exp\left(-\frac{\|x - g_\beta(z̃)\|_ 2^2}{2\sigma_x^2}\right),$
where $g_\beta(z̃)$ is the output of the decoder.

### b. Prior Term
The prior $p_{\theta}(z̃)$ is often assumed to be a standard Gaussian:
$p_{\theta}(z̃) \propto \exp\left(-\frac{\|z̃\|_2^2}{2}\right).$

### c. Combined Log-Joint
The joint distribution's log form becomes:
$\log p_{\beta, \theta}(z̃, x) = -\frac{\|x - g_\beta(z̃)\|_2^2}{2\sigma_x^2} - \frac{\|z̃\|_2^2}{2} + C,$
where $C$ is a constant that can be ignored in gradient calculations.

## 3. Gradient Computation
The gradient of $\log p_{\beta, \theta}(z̃, x)$ with respect to $z̃$ has two components:

### a. Gradient from the Likelihood Term
$\nabla_{z̃} \left(-\frac{\|x - g_\beta(z̃)\|_ 2^2}{2\sigma_x^2}\right) = \frac{1}{\sigma_x^2} \nabla_{z̃} g_\beta(z̃)^\top \left(x - g_\beta(z̃)\right),$
where $\nabla_{z̃} g_\beta(z̃)$ is computed via backpropagation through the decoder.

### b. Gradient from the Prior Term
$\nabla_{z̃} \left(-\frac{\|z̃\|_2^2}{2}\right) = -z̃.$

### c. Combined Gradient
Adding these terms:
$\nabla_{z̃} \log p_{\beta, \theta}(z̃, x) = \frac{1}{\sigma_x^2} \nabla_{z̃} g_\beta(z̃)^\top \left(x - g_\beta(z̃)\right) - z̃.$

## 4. Incorporating the Distance Constraint
To ensure that $z̃$ does not deviate too far from the initial $z$, a distance constraint term is added:
$\mathcal{L}(z̃) = -\log p_{\beta, \theta}(z̃, x) + \frac{\|z̃ - z\|_2^2}{2\sigma_z^2}.$

The gradient of the full loss is:
$\nabla_{z̃} \mathcal{L}(z̃) = -\nabla_{z̃} \log p_{\beta, \theta}(z̃, x) + \frac{z̃ - z}{\sigma_z^2}.$

Expanding the terms, the final gradient is:
$\nabla_{z̃} \mathcal{L}(z̃) = -\frac{1}{\sigma_x^2} \nabla_{z̃} g_\beta(z̃)^\top \left(x - g_\beta(z̃)\right) + z̃ - \frac{z̃ - z}{\sigma_z^2}.$

## 5. Summary
During MCMC sampling, the loss is not directly computed but its gradient is derived using the unnormalized joint distribution $p_{\beta, \theta}(z̃, x)$. This gradient combines contributions from the decoder (likelihood), prior, and distance constraint, ensuring effective and stable updates to $z̃$.
