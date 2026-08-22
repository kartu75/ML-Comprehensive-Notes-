# Stanford CS229 Machine Learning — Complete Notes (Part 3)
---

# Chapter 15: EM Algorithm & Factor Analysis

## 15.1 Factor Analysis

### Motivation

Suppose x ∈ ℝⁿ but the data really lies in a lower-dimensional subspace (or near it). Factor analysis finds hidden continuous factors that explain the observed variables.

**Model:**
```
x = Λz + μ + ε
```

Where:
- x ∈ ℝⁿ = observed data
- z ∈ ℝᵏ = latent factors (k << n)
- Λ ∈ ℝⁿˣᵏ = factor loadings matrix
- μ ∈ ℝⁿ = mean of x
- ε ∈ ℝⁿ = noise

### Distributional Assumptions

```
z ~ N(0, I)     (latent factors: standard Gaussian)
ε ~ N(0, Ψ)     (noise: Gaussian with diagonal covariance)
```

Therefore: x | z ~ N(Λz + μ, Ψ)

And the marginal: x ~ N(μ, ΛΛᵀ + Ψ)

### What Factor Loadings Tell Us

Λ ∈ ℝⁿˣᵏ: each column is one factor, each row shows how much that variable is influenced:

```
Example: 6 exam scores (Math, Physics, Chem, English, History, Geography), k=2 factors

           Factor1    Factor2
           (Science)  (Humanities)
Math:      [0.9,       0.1   ]
Physics:   [0.8,       0.1   ]
Chemistry: [0.7,       0.2   ]
English:   [0.1,       0.8   ]
History:   [0.1,       0.9   ]
Geography: [0.2,       0.7   ]
```

### Learning with EM

**E-Step:** Compute posterior of z given x:
```
E[z | x] = Λᵀ(ΛΛᵀ + Ψ)⁻¹(x - μ)
Cov(z | x) = I - Λᵀ(ΛΛᵀ + Ψ)⁻¹Λ
```

**M-Step:** Update Λ, μ, Ψ to maximise expected log-likelihood.

### Rotation Ambiguity

Factor Analysis has a fundamental non-identifiability: if Λ is a solution, so is ΛR for any orthogonal matrix R. The factors are only determined up to rotation — this is why naming factors requires domain knowledge.

### Factor Analysis vs PCA

| Factor Analysis | PCA |
|---|---|
| Probabilistic model with noise | No probabilistic model |
| Noise term Ψ separates signal from noise | No explicit noise separation |
| Factors need not be orthogonal | Components are orthogonal |
| More interpretable | Faster, deterministic |
| Fit by EM | Fit by eigendecomposition |

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=2, random_state=42)
fa.fit(X)

# Factor loadings
print(fa.components_)   # shape (k, n): k factors, n variables

# Transform data to factor space
Z = fa.transform(X)     # shape (m, k)
```

---

# Chapter 16: PCA and ICA

## 16.1 Principal Component Analysis (PCA)

### Motivation

Find the directions in feature space along which the data has the greatest variance. Project onto these directions to reduce dimensionality while retaining maximum information.

### Algorithm

**Step 1:** Centre the data:
```
x⁽ⁱ⁾ := x⁽ⁱ⁾ - μ    where μ = (1/m) Σᵢ x⁽ⁱ⁾
```

**Step 2:** Compute the covariance matrix:
```
Σ = (1/m) Xᵀ X    (X is the centred data matrix, shape m × n)
```

**Step 3:** Compute eigenvectors and eigenvalues of Σ:
```
Σ u = λ u
```

Sort eigenvectors by decreasing eigenvalue.

**Step 4:** Project data onto top k eigenvectors:
```
z⁽ⁱ⁾ = Uₖᵀ x⁽ⁱ⁾
```

Where Uₖ = [u₁, u₂, ..., uₖ] (the top k eigenvectors, shape n × k).

### Geometric Interpretation

**Principal Component 1 (PC1):** Direction of maximum variance in the data.

**PC2:** Direction of maximum variance, orthogonal to PC1.

**PCk:** Direction of maximum variance, orthogonal to all previous PCs.

```
Data scattered diagonally:

x₂
7|          *
6|        *
5|      *     → PC1 is the diagonal direction (most spread)
4|    *       → PC2 is perpendicular (no spread)
3|  *
 |____________ x₁

After PCA (keep only PC1):
Points described by single number (position along diagonal)
2D → 1D with zero information loss
```

### Choosing k (Number of Components)

**Explained Variance Ratio:**
```
EVR(k) = Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ⁿ λᵢ
```

Choose k such that EVR(k) ≥ 0.95 (retain 95% of variance).

**Scree Plot:** Plot eigenvalues vs index, pick the "elbow."

### Reconstruction

```
x̂⁽ⁱ⁾ = Uₖ z⁽ⁱ⁾ + μ    (approximate reconstruction)
```

**Reconstruction error:** ||x⁽ⁱ⁾ - x̂⁽ⁱ⁾||² = sum of dropped eigenvalues.

### Applications

1. **Dimensionality reduction:** Compress features before feeding to a model
2. **Visualisation:** Reduce to 2D or 3D for plotting
3. **Noise reduction:** Drop low-variance components (noise)
4. **Anomaly detection:** Points with high reconstruction error = anomalies
5. **Finance:** PCA on stock returns reveals market/sector/idiosyncratic factors

### Critical Implementation Notes

1. **Always scale before PCA:** PCA is sensitive to feature scales. Use StandardScaler first.
2. **Fit on train, transform both:** `scaler.fit_transform(X_train)`, `scaler.transform(X_test)`. Same for PCA.
3. **PCA is unsupervised:** Never uses y labels.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Scale first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)  # keep 95% of variance automatically
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original features: {X_train.shape[1]}")
print(f"PCA features: {X_train_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

# Visualise in 2D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.show()
```

## 16.2 Independent Component Analysis (ICA)

### The Cocktail Party Problem

Multiple independent sources (speakers) mix linearly, producing observed signals:
```
x = As    (x: observed signals, A: mixing matrix, s: independent sources)
```

Goal: recover s from x, without knowing A.

### The ICA Model

```
x = As  →  s = Wx    where W = A⁻¹
```

**Assumptions:**
- Sources s are **statistically independent** (knowing one tells you nothing about another)
- Sources are **non-Gaussian** (crucial — see below)
- Mixing is linear

### Why Non-Gaussian?

**Central Limit Theorem:** The sum of independent random variables becomes more Gaussian.

So: mixtures x are MORE Gaussian than the original sources s.

**ICA principle:** Find unmixing W such that components Wx are LEAST Gaussian → most independent.

Gaussianity is measured by **kurtosis** (4th moment) or **negentropy**.

### FastICA Algorithm

1. Preprocess: centre and whiten x (decorrelate)
2. For each component: initialise random w
3. Iterate:
   ```
   w := E[x g(wᵀx)] - E[g'(wᵀx)] w
   w := w / ||w||
   ```
   Where g is a non-linear function (e.g., tanh or x³)
4. Deflation: ensure independence from previously found components
5. Repeat for k components

### ICA vs PCA

| ICA | PCA |
|---|---|
| Maximises statistical independence | Maximises variance |
| Non-Gaussian sources assumed | No distributional assumption |
| Unique solution (up to scaling/ordering) | Rotation ambiguous |
| Finds causal structure | Finds variance structure |
| Signal separation | Dimensionality reduction |

### When to Use ICA

ICA is appropriate when data is a **linear mixture of independent non-Gaussian sources**:
- EEG brain signals (different brain regions)
- Financial returns (market/sector/idiosyncratic factors)
- Speech/audio separation

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=2, random_state=42)
S_recovered = ica.fit_transform(X_mixed)  # recovered sources

# Mixing matrix estimate
A_estimate = ica.mixing_
```

---

# Chapter 17: Independent Component Analysis & Reinforcement Learning Introduction

## 17.1 ICA — Additional Details

### Identifiability Issues

ICA cannot determine:
1. **Scale:** Can multiply each source by a constant and divide the mixing matrix column by same constant
2. **Order:** Cannot determine which recovered signal corresponds to which original source

These ambiguities don't matter for practical applications (signal separation).

### ICA for Feature Learning

Beyond signal separation, ICA can learn meaningful features:
- Applied to image patches → learns edge detectors similar to V1 visual cortex
- Applied to audio → learns frequency-selective filters similar to cochlear hair cells

## 17.2 Introduction to Reinforcement Learning

### The RL Framework

An **agent** interacts with an **environment** to maximise cumulative reward.

```
Agent → Action a → Environment
Agent ← State s, Reward r ← Environment
```

**Components:**
- State space S: all possible states
- Action space A: all possible actions
- Transition function P(s' | s, a): probability of reaching state s' after taking action a in state s
- Reward function R(s, a): immediate reward for taking action a in state s
- Discount factor γ ∈ [0,1]: weights future rewards

**Goal:** Find a policy π: S → A that maximises expected cumulative discounted reward:
```
E[Σₜ₌₀^∞ γᵗ R(sₜ, aₜ)]
```

---

# Chapter 18: MDPs & Value/Policy Iteration

## 18.1 Markov Decision Processes (MDPs)

An MDP is a formal framework for sequential decision making with the **Markov property**: the future depends only on the current state, not the history.

**Formal definition:** MDP = (S, A, P, R, γ)

## 18.2 Value Functions

### State Value Function V^π(s)

Expected cumulative reward starting from state s, following policy π:
```
V^π(s) = E_π[Σₜ₌₀^∞ γᵗ R(sₜ, aₜ) | s₀ = s]
```

### Bellman Equation for V^π

```
V^π(s) = R(s, π(s)) + γ Σ_{s'} P(s' | s, π(s)) V^π(s')
```

Current value = immediate reward + discounted expected future value.

### Optimal Value Function V*

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
```

### Action-Value Function Q^π(s, a)

Expected cumulative reward starting from state s, taking action a, then following π:
```
Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')
```

### Optimal Policy

```
π*(s) = argmax_a Q*(s, a) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
```

## 18.3 Value Iteration

Iteratively compute V* using the Bellman optimality equation:

```
Algorithm:
1. Initialise V(s) = 0 for all s
2. Repeat until convergence:
   V(s) := max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]   for all s
3. Extract policy: π(s) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
```

**Convergence:** Value iteration converges to V* (guaranteed for any γ < 1).

**Complexity per iteration:** O(|S|² |A|) — expensive for large state spaces.

## 18.4 Policy Iteration

Alternative to value iteration — alternate between policy evaluation and improvement:

```
Algorithm:
1. Initialise π randomly
2. Repeat until π doesn't change:
   a. Policy Evaluation:  compute V^π (solve Bellman equations)
   b. Policy Improvement: π(s) := argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

**Policy evaluation** (step 2a) requires solving a system of linear equations:
```
V^π = R^π + γ P^π V^π
→ V^π = (I - γP^π)⁻¹ R^π
```

**Convergence:** Policy iteration typically converges in fewer iterations than value iteration (but each iteration is more expensive).

## 18.5 Value Iteration vs Policy Iteration

| Value Iteration | Policy Iteration |
|---|---|
| Simple update | More complex (linear solve) |
| Many cheap iterations | Fewer expensive iterations |
| Approximate policy during learning | Always has explicit policy |
| Better for large state spaces | Better when few states |

---

# Chapter 19: Continuous State MDPs & Model Simulation

## 19.1 The Curse of Dimensionality

Discrete MDPs with small state spaces (like grid worlds) can be solved exactly with value/policy iteration. But real-world problems have continuous, high-dimensional state spaces.

**Example:** A car's state includes position (x,y), velocity (vₓ,vᵧ), orientation θ, angular velocity ω → 6D continuous state space. Cannot tabulate V(s) for all s.

## 19.2 Discretisation

**Approach:** Discretise continuous state space into a grid.

**Problem:** Curse of dimensionality — number of states grows exponentially with dimensions.

For a 10D state space with 100 values per dimension:
```
States = 100^10 = 10^20  ← completely intractable
```

## 19.3 Value Function Approximation

Instead of tabulating V(s) for every state, approximate it with a parameterised function:
```
V(s) ≈ θᵀφ(s)    (linear approximation)
V(s) ≈ f_θ(s)    (neural network approximation → Deep RL)
```

**Fitted Value Iteration:**
1. Sample states s₁, ..., sm
2. For each state, compute target: yᵢ = max_a [R(sᵢ,a) + γ E[V(s')]]
3. Fit a regression model: V ≈ argmin_V Σᵢ (V(sᵢ) - yᵢ)²
4. Use fitted V as new value estimate
5. Repeat

## 19.4 Simulator / Model

For continuous MDPs, we often have a **simulator** that models transitions:
```
s_{t+1} = f(sₜ, aₜ) + noise
```

**Physics-based simulators:** Exact dynamics from equations of motion.

**Learned simulators (model-based RL):** Learn f from data, then plan using the learned model.

## 19.5 Linear Quadratic Regulator (LQR)

Special case of continuous MDP with:
- Linear dynamics: sₜ₊₁ = Asₜ + Baₜ + noise
- Quadratic cost: cost = sᵀQs + aᵀRa

**Optimal policy is linear:** π(s) = -Ks (linear state feedback).

Can be solved exactly using the **Riccati equation** — foundational in control theory.

---

# Chapter 20: Reward Model & Linear Dynamical Systems

## 20.1 Reward Shaping

In many real RL problems, rewards are **sparse** — the agent only receives reward rarely (e.g., winning a game).

**Reward shaping:** Add additional reward signals to guide learning:
```
R'(s, a, s') = R(s, a, s') + F(s, a, s')
```

Where F is a shaping function. Under certain conditions (F = γΦ(s') - Φ(s)), the optimal policy is preserved.

## 20.2 Inverse Reinforcement Learning (IRL)

Instead of specifying the reward function, learn it from expert demonstrations:

```
Normal RL:  Given R → find π
IRL:        Given π* (expert) → find R
            Then: Given R → find π
```

**Applications:**
- Learning to drive from human demonstrations
- Robot learning from human demonstrations
- Understanding human/animal behaviour

## 20.3 Linear Dynamical Systems (LDS)

### State Space Model

```
sₜ₊₁ = Asₜ + Baₜ + wₜ    (dynamics equation, wₜ ~ N(0,Q))
xₜ = Csₜ + vₜ              (observation equation, vₜ ~ N(0,R))
```

Where:
- sₜ = hidden state (latent)
- xₜ = observed output
- aₜ = control input
- wₜ, vₜ = process and observation noise

### Kalman Filter

The optimal algorithm for state estimation in a LDS:

**Prediction step:**
```
s_{t|t-1} = A s_{t-1|t-1} + B aₜ
Σ_{t|t-1} = A Σ_{t-1|t-1} Aᵀ + Q
```

**Update step:**
```
Kₜ = Σ_{t|t-1} Cᵀ (C Σ_{t|t-1} Cᵀ + R)⁻¹   (Kalman gain)
s_{t|t} = s_{t|t-1} + Kₜ(xₜ - C s_{t|t-1})
Σ_{t|t} = (I - KₜC) Σ_{t|t-1}
```

**Applications:** GPS navigation, financial time series, robot localisation, aerospace.

### EM for LDS (Kalman Smoother)

When A, B, C, Q, R are unknown, use EM:
- **E-Step:** Run Kalman smoother to estimate hidden states
- **M-Step:** Update model parameters given estimated states

---

# Chapter 21: RL Debugging & Diagnostics

## 21.1 Common RL Problems and Fixes

### Problem 1: Reward Signal Too Sparse

**Symptom:** Agent takes many steps but rarely gets any reward signal → slow or no learning.

**Fixes:**
- Reward shaping
- Curriculum learning (start with easy tasks)
- Demonstrations (imitation learning first, then RL)

### Problem 2: Exploration vs Exploitation

**Problem:** Agent exploits what it knows → never discovers better strategies.

**Fixes:**
- ε-greedy: with probability ε, take random action
- Softmax exploration: action probabilities proportional to Q values
- Upper Confidence Bound (UCB): explore states with high uncertainty
- Entropy regularisation: encourage diverse action distributions

### Problem 3: Non-Stationary Environments

When the environment changes over time, old experience becomes irrelevant.

**Fix:** Experience replay with prioritisation, periodic resets, meta-learning.

### Problem 4: Credit Assignment

With delayed rewards, which past actions caused the current reward?

**Fix:** Eligibility traces, advantage estimation, n-step returns.

## 21.2 Diagnosing RL Algorithms

### Is the problem the reward function?

Run the algorithm with a hand-crafted, dense reward. If it learns → reward function was the problem.

### Is the problem exploration?

Visualise what states the agent visits. If it always stays in a small region → exploration problem.

### Is the problem the model?

For model-based RL: test the model's prediction accuracy. If poor → model problem.

## 21.3 Policy Gradient Methods

Beyond value-based methods, directly optimise the policy:

```
J(π_θ) = E_π[Σₜ γᵗ r(sₜ, aₜ)]

∇_θ J(π_θ) = E_π[Σₜ ∇_θ log π_θ(aₜ|sₜ) · Gₜ]
```

Where Gₜ is the return from time step t.

**REINFORCE algorithm:**
```
For each episode:
    Generate trajectory τ = (s₀,a₀,r₀, s₁,a₁,r₁, ...)
    For each time step t:
        Gₜ = Σₖ≥t γ^{k-t} rₖ
        θ := θ + α · ∇_θ log π_θ(aₜ|sₜ) · Gₜ
```

**Problem:** High variance in gradient estimates.

**Fix:** Subtract a baseline b(s) (e.g., value function):
```
∇_θ J ≈ ∇_θ log π_θ(a|s) · (Q(s,a) - V(s))
       = ∇_θ log π_θ(a|s) · A(s,a)    (advantage function)
```

## 21.4 Actor-Critic Methods

Combine policy gradient (actor) with value function (critic):

```
Actor:  π_θ(a|s) — the policy (decides what action to take)
Critic: V_w(s)   — estimates value (tells actor how good the state is)
```

**Update rules:**
```
Critic: minimize (rₜ + γV_w(sₜ₊₁) - V_w(sₜ))²
Actor: θ := θ + α · ∇_θ log π_θ(aₜ|sₜ) · (rₜ + γV_w(sₜ₊₁) - V_w(sₜ))
```

**Modern variants:** PPO (Proximal Policy Optimisation), A3C, SAC — used in state-of-the-art RL systems.

---

# Appendix A: Mathematics Review

## A.1 Linear Algebra

**Matrix multiplication:** (m×n)(n×p) = (m×p)

**Transpose:** (AB)ᵀ = BᵀAᵀ

**Matrix inverse:** AA⁻¹ = I (only for square, non-singular matrices)

**Trace:** tr(A) = Σᵢ Aᵢᵢ (sum of diagonal elements)

**Determinant:** |A| or det(A)

**Eigendecomposition:** Av = λv → A = QΛQᵀ (for symmetric A)

**Gradient rules:**
```
∇_x (xᵀa) = a
∇_x (xᵀAx) = (A + Aᵀ)x = 2Ax (if A symmetric)
∇_A tr(AB) = Bᵀ
∇_A log|A| = A⁻ᵀ = (A⁻¹)ᵀ
```

## A.2 Probability

**Bayes' Rule:**
```
P(A|B) = P(B|A)P(A) / P(B)
```

**Gaussian distribution:**
```
N(x; μ, σ²) = (1/√2πσ²) exp(-(x-μ)²/2σ²)
```

**Multivariate Gaussian:**
```
N(x; μ, Σ) = (1/(2π)^{n/2} |Σ|^{1/2}) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

**Expected value:** E[X] = Σₓ x·P(X=x) or ∫ x·f(x)dx

**Variance:** Var(X) = E[(X-E[X])²] = E[X²] - (E[X])²

**Covariance:** Cov(X,Y) = E[(X-E[X])(Y-E[Y])]

## A.3 Calculus

**Chain rule:** d/dx f(g(x)) = f'(g(x)) · g'(x)

**Partial derivative:** ∂/∂xᵢ f(x₁,...,xₙ)

**Gradient:** ∇_x f = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ

**Hessian:** H_{ij} = ∂²f/∂xᵢ∂xⱼ

---

# Appendix B: Key Formulas Quick Reference

## B.1 Linear Regression

```
Hypothesis:     hθ(x) = θᵀx
Cost:           J(θ) = (1/2m)||Xθ - y||²
Gradient:       ∇J = (1/m)Xᵀ(Xθ - y)
Update:         θ := θ - α∇J
Normal equation: θ = (XᵀX)⁻¹Xᵀy
```

## B.2 Logistic Regression

```
Sigmoid:    g(z) = 1/(1+e^{-z})
Hypothesis: hθ(x) = g(θᵀx)
Cost:       J = -(1/m)[yᵀlog(h) + (1-y)ᵀlog(1-h)]
Gradient:   ∇J = (1/m)Xᵀ(h-y)    [same form as linear!]
```

## B.3 Softmax

```
P(y=k|x) = exp(θₖᵀx) / Σⱼ exp(θⱼᵀx)
Loss:       L = -log P(y=true class)
```

## B.4 SVM

```
Primal:  min ½||θ||²  s.t. y⁽ⁱ⁾(θᵀx⁽ⁱ⁾+b) ≥ 1
Dual:    max Σᵢ αᵢ - ½ Σᵢⱼ αᵢαⱼy⁽ⁱ⁾y⁽ʲ⁾⟨x⁽ⁱ⁾,x⁽ʲ⁾⟩
Kernel:  K(x,z) = φ(x)ᵀφ(z)
RBF:     K(x,z) = exp(-||x-z||²/2σ²)
```

## B.5 Neural Networks

```
Forward:  z^[l] = W^[l]a^[l-1] + b^[l],  a^[l] = g(z^[l])
Backward: δ^[L] = a^[L] - y
          δ^[l] = (W^[l+1])ᵀδ^[l+1] ⊙ g'(z^[l])
          ∂L/∂W^[l] = δ^[l](a^[l-1])ᵀ
Adam:     m := β₁m + (1-β₁)g,  v := β₂v + (1-β₂)g²
          θ := θ - α·m̂/√v̂
```

## B.6 PCA

```
Covariance: Σ = (1/m)XᵀX
Eigen:      Σu = λu
Project:    z = Uₖᵀx
Reconstruct: x̂ = Uₖz + μ
EVR:        Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ⁿ λᵢ
```

## B.7 GMM/EM

```
E-step: rᵢₖ = πₖN(xᵢ;μₖ,Σₖ) / Σⱼ πⱼN(xᵢ;μⱼ,Σⱼ)
M-step: μₖ = Σᵢrᵢₖxᵢ/Nₖ,  Nₖ = Σᵢrᵢₖ
        πₖ = Nₖ/m
```

## B.8 MDPs

```
Bellman: V*(s) = max_a[R(s,a) + γΣ_{s'}P(s'|s,a)V*(s')]
Policy:  π*(s) = argmax_a[R(s,a) + γΣ_{s'}P(s'|s,a)V*(s')]
```

---

# Appendix C: Practical ML Checklist

## C.1 Before Training

```
□ Understand the problem (regression/classification/clustering?)
□ Explore the data (shape, types, distributions, missing values)
□ Check class balance
□ Scale features (StandardScaler for most models)
□ Split data properly (stratified for classification, chronological for time series)
□ Establish baseline (predict mean/majority class)
```

## C.2 During Training

```
□ Monitor train and validation loss
□ Check for vanishing/exploding gradients (neural nets)
□ Use appropriate loss function for the task
□ Start with simple model, add complexity as needed
□ Use cross-validation for reliable performance estimates
```

## C.3 After Training

```
□ Evaluate on held-out test set ONCE
□ Analyse errors (confusion matrix, residuals)
□ Check feature importances
□ Test on edge cases
□ Consider calibration (are predicted probabilities reliable?)
```

## C.4 Model Selection Guide

```
Problem type → Recommended models:

Regression (small data, interpretable):      Linear Regression, Ridge, Lasso
Regression (medium data):                    Random Forest, XGBoost
Regression (large data, complex):            Neural Network

Binary classification (interpretable):       Logistic Regression, SVM
Binary classification (best performance):    XGBoost, Random Forest
Binary classification (images/text deep):    Neural Network

Multiclass classification:                   Same as binary, output k neurons + softmax

Text classification:                         Naive Bayes (baseline), Logistic Regression

Clustering:                                  K-Means (spherical clusters), GMM (general)

Dimensionality reduction:                    PCA (variance), Factor Analysis (interpretable)

Time series:                                 ARIMA (statistical), LSTM (deep learning)
```

## C.5 Hyperparameter Tuning Priorities

```
1. Learning rate (most impactful)
2. Model complexity (depth, width, n_estimators)
3. Regularisation strength (C, lambda, dropout)
4. Batch size
5. Other architecture choices
```

---

# Appendix D: Common Mistakes and How to Avoid Them

## D.1 Data Leakage

**Mistake:** Fitting scaler/PCA on all data before splitting.

**Fix:** Always fit preprocessing on training data only.
```python
# WRONG
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# RIGHT
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## D.2 Wrong Loss Function

| Task | Wrong | Right |
|---|---|---|
| Binary classification | MSE | BCEWithLogitsLoss |
| Multiclass | BCELoss | CrossEntropyLoss |
| Regression | CrossEntropyLoss | MSELoss |

## D.3 Not Shuffling Training Data

Always shuffle (except time series):
```python
train_test_split(X, y, shuffle=True)  # default
DataLoader(dataset, shuffle=True)     # PyTorch
```

## D.4 Double Applying Activation

```python
# WRONG: sigmoid in model + BCEWithLogitsLoss (applies sigmoid again)
model = nn.Sequential(..., nn.Sigmoid())
loss = nn.BCEWithLogitsLoss()(pred, y)

# RIGHT: no sigmoid in model
model = nn.Sequential(..., nn.Linear(64, 1))
loss = nn.BCEWithLogitsLoss()(pred, y)

# OR: sigmoid in model + BCELoss (not BCEWithLogitsLoss)
model = nn.Sequential(..., nn.Sigmoid())
loss = nn.BCELoss()(pred, y)
```

## D.5 Not Calling model.eval()

```python
# WRONG: dropout still active during inference
predictions = model(X_test)

# RIGHT: disable dropout/batchnorm for inference
model.eval()
with torch.inference_mode():
    predictions = model(X_test)
```

## D.6 Forgetting optimizer.zero_grad()

```python
# WRONG: gradients accumulate across batches
for batch in dataloader:
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

# RIGHT: zero gradients before each backward pass
for batch in dataloader:
    optimizer.zero_grad()  # clear previous gradients
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
```

---

*These notes cover Stanford CS229 Machine Learning (Autumn 2018) by Andrew Ng. All 21 lectures are covered, from supervised learning through unsupervised learning and reinforcement learning.*
