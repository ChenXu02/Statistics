# Origin and Development of Energy Score

**Energy Score** is a statistical measure derived from the study of differences between random variables and distributions. It originated from the concept of "energy" in statistical physics and later evolved into a tool for comparing probability distributions, testing independence, and evaluating multidimensional models. Its theoretical foundation is closely related to the concept of Energy Distance.

---

## **1. Origin of Energy Distance**

### **(1) Conceptual Origin**
The term "energy" is inspired by the concept of **potential energy** in statistical physics. In physics, the interaction between particles in a system can be described using energy based on distances. Statisticians adopted this analogy to study differences between probability distributions.

### **(2) Early Research**
- **Gábor J. Székely et al. (1980s-2000s):**
  Gábor J. Székely and collaborators introduced and developed the concept of **Energy Statistics** during the 1980s to early 2000s.
  - Their goal was to create a sample-based, nonparametric statistical method to measure differences between high-dimensional distributions or dependencies between variables.
  - **Energy Distance** became the core tool of this framework.

### **(3) Definition of Energy Distance**
Energy Distance $D(P, Q)$ measures the difference between two distributions $P$ and $Q$ as:
$D(P, Q) = 2\mathbb{E}[d(X, Y)] - \mathbb{E}[d(X, X')] - \mathbb{E}[d(Y, Y')]$
where:
- $X, X' \sim P$, and $Y, Y' \sim Q$ are independent samples from $P$ and $Q$, respectively.
- $d(\cdot, \cdot)$ is a distance function (commonly Euclidean distance).

This formula is inspired by potential energy differences in physical systems:
- $\mathbb{E}[d(X, Y)]$: Expected distance between samples from $P$ and $Q$.
- $\mathbb{E}[d(X, X')]$: Expected intra-sample distance within $P$.
- $\mathbb{E}[d(Y, Y')]$: Expected intra-sample distance within $Q$.

---

## **2. Extension to Energy Score**

### **(1) From Energy Distance to Energy Score**
Energy Score generalizes the concept of Energy Distance, particularly for **multidimensional distributions** and **multi-distribution comparisons**. It aims to assess whether a set of observations comes from a target distribution or to compare similarities among multiple distributions.

### **(2) Definition of Energy Score**
For observed multidimensional random variables $\{x_1, x_2, \dots, x_N\}$ with a target distribution $Q$, the Energy Score is defined as:
$\text{Energy Score} = \mathbb{E}_ {Y \sim Q} [d(x, Y)] - \frac{1}{2} \mathbb{E}_{Y, Y' \sim Q} [d(Y, Y')]$
where:
- $x$ is the observed sample.
- $Y, Y'$ are samples drawn from the target distribution $Q$.
- $d(\cdot, \cdot)$ is a distance function.

### **(3) Interpretation**
- Energy Score evaluates the "alignment" between the observations and the target distribution.
- By comparing distances between the observations and target samples, it assesses how well the generative model or assumed distribution fits the data.

---

## **3. Theoretical Background**

### **(1) Theoretical Motivation**
- **Key Characteristics of Energy Statistics:**
  1. **Nonparametric:** Energy statistics do not rely on specific distributional assumptions, making them suitable for complex and non-standard distributions.
  2. **High-dimensional Compatibility:** Both Energy Distance and Energy Score handle high-dimensional data effectively by focusing on distances rather than dimensionality or specific distribution shapes.
  3. **Robustness:** They are robust to noise and outliers.

### **(2) Relation to Other Metrics**
Energy Distance and Energy Score are related to other metrics for distribution comparison but have unique properties:
- **Relation to Wasserstein Distance:**
  Energy Distance is equivalent to the square root of a specific form of the Wasserstein Distance.
- **Relation to Maximum Mean Discrepancy (MMD):**
  While similar in concept, Energy Distance uses Euclidean distances directly, whereas MMD relies on kernel-based methods.

### **(3) Connection to Statistical Tests**
Energy Statistics are also used to construct nonparametric hypothesis tests, such as:
- **Two-sample test:** Comparing whether two samples come from the same distribution.
- **Independence test:** Testing whether two random variables are dependent.

---

## **4. Applications**

### **(1) Evaluating Generative Models**
Energy Score is widely used to evaluate the performance of generative models like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). It is particularly advantageous because:
- It does not rely on distributional assumptions.
- It effectively handles high-dimensional and complex data.

### **(2) Multivariate Dependency Analysis**
Energy Score can be applied to detect nonlinear dependencies or relationships between variables, especially in high-dimensional datasets.

### **(3) Statistical Tests**
Energy Statistics are used in various nonparametric tests, such as:
- $k$-sample tests: Comparing whether multiple samples share the same distribution.
- Paired tests: Evaluating whether observed data matches a hypothesized distribution.

### **(4) Image Processing and Physical Modeling**
Energy Score is used to compare simulation results with real-world data in applications like image generation and physical simulations.

---

## **5. Contributions and Key References**
The development of Energy Score and related methods is largely attributed to:
- **Gábor J. Székely** and **Maria L. Rizzo**, whose work laid the theoretical foundation for Energy Statistics.

Key references include:
1. Székely, G. J., & Rizzo, M. L. (2004). *Testing for Equal Distributions in High Dimension.*
2. Székely, G. J., & Rizzo, M. L. (2005). *A New Test for Multivariate Normality.*

---

## **6. Summary**
Energy Score originates from the concept of energy in physics and has been developed into a powerful tool for distribution comparison. Its core ideas rely on differences in distances or potential energy, with the following features:
- **Flexible:** Applicable to diverse distribution shapes and high-dimensional data.
- **Nonparametric:** Free of specific distributional assumptions.
- **Practical:** Widely used for generative model evaluation, independence testing, and distribution comparison.

Its development has provided a robust framework for high-dimensional data analysis and complex distribution modeling.
