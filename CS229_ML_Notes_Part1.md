# Stanford CS229 Machine Learning — Complete Notes
### Based on Andrew Ng's Autumn 2018 Lecture Series
---

# Chapter 1: Introduction to Machine Learning

## 1.1 What is Machine Learning?

Machine learning is the science of getting computers to learn from data without being explicitly programmed for every task.

**Arthur Samuel (1959):** "Field of study that gives computers the ability to learn without being explicitly programmed."

**Tom Mitchell (1998):** "A computer program is said to learn from experience E with respect to some task T and performance measure P, if its performance at T, as measured by P, improves with experience E."

**Example:** Spam detection
- Task T: classify email as spam or not
- Experience E: labelled emails (spam/not spam)
- Performance P: accuracy of classification

## 1.2 Types of Machine Learning

### Supervised Learning
You have labelled data — input-output pairs (x, y). The model learns a mapping from x to y.

**Regression:** y is continuous (predict house price, stock return)
**Classification:** y is discrete (spam/not spam, cat/dog/iguana)

### Unsupervised Learning
You only have inputs x, no labels. The model finds structure in data on its own.

Examples: clustering (K-means), dimensionality reduction (PCA), density estimation (GMM)

### Reinforcement Learning
An agent learns by interacting with an environment, receiving rewards or penalties.

Examples: game playing (AlphaGo), robotics, trading agents

## 1.3 Key Terminology

| Term | Meaning |
|---|---|
| Training set | Data used to fit the model |
| Test set | Held-out data for final evaluation |
| Features (x) | Input variables |
| Label/Target (y) | Output variable |
| Parameters (θ) | Weights learned during training |
| Hypothesis (h) | The function the model learns |
| Loss/Cost | How wrong the model is |

## 1.4 The Machine Learning Workflow

```
1. Collect data
2. Choose model architecture
3. Define loss function
4. Optimise (gradient descent)
5. Evaluate on test set
6. Deploy
```

---

# Chapter 2: Linear Regression and Gradient Descent

## 2.1 Problem Setup

Given a dataset of m training examples:
```
{(x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), ..., (x⁽ᵐ⁾, y⁽ᵐ⁾)}
```

Where x⁽ⁱ⁾ ∈ ℝⁿ (n features) and y⁽ⁱ⁾ ∈ ℝ (continuous output).

**Goal:** Learn a function hθ(x) that maps x to y.

## 2.2 The Hypothesis

For linear regression:
```
hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

Where x₀ = 1 (bias term) and θ = [θ₀, θ₁, ..., θₙ]ᵀ.

In matrix notation: **hθ(X) = Xθ**

## 2.3 Cost Function (MSE)

We measure how wrong our model is using Mean Squared Error:

```
J(θ) = (1/2m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

The ½ is for mathematical convenience (cancels when differentiating).

**Goal:** Find θ that minimises J(θ).

## 2.4 Gradient Descent

Gradient descent iteratively updates parameters in the direction that reduces the loss.

**Update rule:**
```
θⱼ := θⱼ - α · (∂J/∂θⱼ)
```

Where α is the learning rate (step size).

**Computing the gradient:**
```
∂J/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾
```

**In vector form:**
```
θ := θ - (α/m) · Xᵀ(Xθ - y)
```

**Intuition:** If prediction > truth (positive error), decrease θ. If prediction < truth (negative error), increase θ.

### Variants of Gradient Descent

| Variant | Per update uses | Updates per epoch |
|---|---|---|
| Batch GD | All m examples | 1 |
| Stochastic GD (SGD) | 1 example | m |
| Mini-batch GD | b examples (batch) | m/b |

**Practical choice:** Mini-batch (b=32 or 64) balances speed and stability.

## 2.5 The Normal Equation (Closed-Form Solution)

For linear regression, there is an exact analytical solution:

```
θ = (XᵀX)⁻¹Xᵀy
```

**Derivation:** Set ∇θJ(θ) = 0 and solve for θ.

| Normal Equation | Gradient Descent |
|---|---|
| No learning rate needed | Needs learning rate tuning |
| O(n³) — slow for large n | Scales well to large n |
| Exact solution | Iterative approximation |
| Fails if XᵀX is singular | Always works |

**Use Normal Equation when:** n < 10,000 features.
**Use Gradient Descent when:** n is very large.

## 2.6 Feature Scaling

Features with very different scales cause gradient descent to converge slowly.

**Standardisation (Z-score normalisation):**
```
x' = (x - μ) / σ
```

Result: mean = 0, std = 1. Always scale before gradient descent.

## 2.7 Probabilistic Interpretation

Why MSE? Assume y⁽ⁱ⁾ = θᵀx⁽ⁱ⁾ + ε⁽ⁱ⁾ where ε ~ N(0, σ²).

Then:
```
p(y⁽ⁱ⁾ | x⁽ⁱ⁾; θ) = (1/√2πσ) · exp(-(y⁽ⁱ⁾ - θᵀx⁽ⁱ⁾)² / 2σ²)
```

Maximising log-likelihood gives exactly the same solution as minimising MSE. MSE is the maximum likelihood estimator under Gaussian noise.

## 2.8 Key Code

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print(model.coef_, model.intercept_)
print(f"R²: {model.score(X_test_scaled, y_test):.4f}")
```

---

# Chapter 3: Locally Weighted & Logistic Regression

## 3.1 Locally Weighted Regression (LWR)

### Motivation
Linear regression fits ONE global line to all data. What if the relationship is locally linear but globally non-linear?

### The Algorithm
For each query point x, fit a local weighted linear regression:

```
Minimise: Σᵢ w⁽ⁱ⁾(y⁽ⁱ⁾ - θᵀx⁽ⁱ⁾)²
```

Where the weight function is:
```
w⁽ⁱ⁾ = exp(-(x⁽ⁱ⁾ - x)² / 2τ²)
```

- Points close to query x → weight ≈ 1 (high influence)
- Points far from query x → weight ≈ 0 (low influence)
- τ (bandwidth) controls how wide the neighbourhood is

### Key Properties
- **Non-parametric:** no fixed θ — must refit for every new prediction
- **Computationally expensive:** O(n) per prediction
- **τ too small:** overfits (only uses nearby points)
- **τ too large:** underfits (uses all points equally, degenerates to linear regression)

**Practical use:** Rarely used in practice due to computational cost. Modern alternatives (Random Forest, XGBoost) are better.

## 3.2 Logistic Regression

### Problem Setup
Binary classification: y ∈ {0, 1}.

We need a model that outputs probabilities in [0, 1] — linear regression doesn't guarantee this.

### The Sigmoid Function

```
g(z) = 1 / (1 + e⁻ᶻ)
```

Properties:
- Output always in (0, 1)
- g(0) = 0.5
- As z → +∞, g(z) → 1
- As z → -∞, g(z) → 0
- Derivative: g'(z) = g(z)(1 - g(z))

### The Hypothesis

```
hθ(x) = g(θᵀx) = 1 / (1 + e^(-θᵀx))
```

Interpretation: **P(y=1 | x; θ)** — probability that y=1 given x.

Decision boundary: predict y=1 if hθ(x) ≥ 0.5, i.e., when θᵀx ≥ 0.

### Cost Function: Binary Cross-Entropy

MSE doesn't work for classification (non-convex with sigmoid). Use:

```
J(θ) = -(1/m) Σᵢ [y⁽ⁱ⁾ log(hθ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1-hθ(x⁽ⁱ⁾))]
```

**Intuition:**
- If y=1 and hθ(x) → 1: loss → 0 (correct, confident)
- If y=1 and hθ(x) → 0: loss → ∞ (wrong, confident — penalised heavily)
- If y=0 and hθ(x) → 0: loss → 0
- If y=0 and hθ(x) → 1: loss → ∞

### Gradient of Cross-Entropy

Surprisingly, the gradient has the same form as linear regression:

```
∂J/∂θⱼ = (1/m) Σᵢ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾
```

This falls out of the GLM framework (Chapter 4).

### Probabilistic Interpretation

Cross-entropy is the maximum likelihood estimator assuming y | x ~ Bernoulli(hθ(x)).

### Multi-class: Softmax Regression

For k classes:
```
P(y=k | x) = exp(θₖᵀx) / Σⱼ exp(θⱼᵀx)
```

- k separate weight vectors, one per class
- Probabilities sum to 1 (unlike k separate logistic regressions)
- Loss: cross-entropy = -log(P(true class))

### Key Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

# Chapter 4: Perceptron & Generalized Linear Models

## 4.1 The Perceptron

### The Model
The oldest classification algorithm — the building block of neural networks.

```
hθ(x) = 1 if θᵀx ≥ 0
         0 if θᵀx < 0
```

Uses a step function (hard threshold) instead of sigmoid.

### Perceptron Update Rule

```
θⱼ := θⱼ + α(y⁽ⁱ⁾ - hθ(x⁽ⁱ⁾)) · xⱼ⁽ⁱ⁾
```

Or equivalently using error = y - ŷ:
- Correct prediction → error = 0 → no update
- y=1, predicted 0 → error = +1 → increase weights
- y=0, predicted 1 → error = -1 → decrease weights

### Key Limitation
Perceptron only converges if data is **linearly separable**. If not, it never converges. This limitation (famously the XOR problem) stalled neural network research for decades.

**Historical note:** One perceptron = one neuron. Stacking many perceptrons = neural network (Chapter 10).

## 4.2 Newton's Method

Alternative to gradient descent using second-order information.

### Scalar Version
```
θ := θ - f'(θ) / f''(θ)
```

Uses the curvature (second derivative) to take smarter steps toward the minimum.

### Vector Version (Newton-Raphson)
```
θ := θ - H⁻¹ · ∇θJ(θ)
```

Where:
- ∇θJ(θ) = gradient vector (first derivatives)
- H = Hessian matrix (second derivatives), where Hᵢⱼ = ∂²J/∂θᵢ∂θⱼ

### Newton vs Gradient Descent

| Newton's Method | Gradient Descent |
|---|---|
| Uses 2nd derivatives (curvature) | Uses only 1st derivatives |
| Faster convergence (quadratic) | Slower (linear) |
| O(n²) memory for Hessian | O(n) memory |
| Infeasible for large n (millions of params) | Scales well |

**Practical use:** Newton's method is used for small models. Deep learning always uses gradient descent variants (Adam, SGD).

## 4.3 Exponential Family Distributions

Many probability distributions share the same mathematical form:

```
p(y; η) = b(y) · exp(η·T(y) - a(η))
```

Where:
- η = natural parameter
- T(y) = sufficient statistic (usually just y)
- b(y) = base measure
- a(η) = log partition function (normalising constant)

### Distributions in the Exponential Family

| Distribution | Used for |
|---|---|
| Gaussian | Linear regression (continuous y) |
| Bernoulli | Logistic regression (binary y) |
| Poisson | Count data |
| Multinomial | Multiclass classification |

## 4.4 Generalized Linear Models (GLMs)

GLM is the **unified framework** that connects all regression/classification algorithms.

### Three Components of Every GLM

**1. Random Component:** What distribution does y follow?
```
Gaussian → linear regression
Bernoulli → logistic regression
Poisson → Poisson regression
```

**2. Systematic Component:** Linear combination of features:
```
η = θᵀx
```

**3. Link Function:** Connects η to the prediction:
```
Linear regression: ŷ = η (identity link)
Logistic regression: ŷ = sigmoid(η) (logit link)
Poisson regression: ŷ = exp(η) (log link)
```

### Constructing a GLM
1. Choose a distribution from the exponential family
2. Set η = θᵀx
3. Predict E[y|x] — this automatically gives the right model

**Key insight:** Logistic regression isn't an arbitrary choice — it's the natural result of assuming y follows a Bernoulli distribution in the GLM framework. The sigmoid falls out of the math automatically.

### Softmax Regression as GLM
Multinomial distribution → softmax link → multiclass classification. Same unified framework.

---

# Chapter 5: GDA & Naive Bayes (Generative Learning Algorithms)

## 5.1 Discriminative vs Generative Models

### Discriminative (everything so far)
Learn P(y | x) directly — given x, predict y.

Examples: Logistic Regression, SVM, Neural Networks

### Generative
Learn P(x | y) — what does x look like given each class?

Then use Bayes rule to get P(y | x):
```
P(y | x) = P(x | y) · P(y) / P(x)
```

- P(x | y) = likelihood
- P(y) = prior
- P(x) = evidence (normalising constant, often ignored in classification)

**Key difference:** Generative models learn each class separately. Discriminative models learn the boundary directly.

## 5.2 Gaussian Discriminant Analysis (GDA)

### Assumptions

**y ~ Bernoulli(φ):**
```
P(y=1) = φ
P(y=0) = 1 - φ
```

**x | y=0 ~ N(μ₀, Σ) and x | y=1 ~ N(μ₁, Σ):**
```
P(x | y=k) = (1/√(2π|Σ|)) · exp(-½(x-μₖ)ᵀΣ⁻¹(x-μₖ))
```

Both classes share the same covariance Σ but have different means μ₀ and μ₁.

### MLE Parameter Estimates

```
φ = (1/m) Σᵢ 1{y⁽ⁱ⁾=1}

μ₀ = Σᵢ 1{y⁽ⁱ⁾=0} · x⁽ⁱ⁾ / Σᵢ 1{y⁽ⁱ⁾=0}

μ₁ = Σᵢ 1{y⁽ⁱ⁾=1} · x⁽ⁱ⁾ / Σᵢ 1{y⁽ⁱ⁾=1}

Σ = (1/m) Σᵢ (x⁽ⁱ⁾ - μ_{y⁽ⁱ⁾})(x⁽ⁱ⁾ - μ_{y⁽ⁱ⁾})ᵀ
```

**Intuition:** μ₀ is just the mean of all x vectors where y=0. μ₁ is the mean of all x where y=1.

### Prediction

For a new point x:
```
Compare: P(x|y=1)·P(y=1) vs P(x|y=0)·P(y=0)
Predict: class with higher score
```

### GDA vs Logistic Regression

| GDA | Logistic Regression |
|---|---|
| Makes Gaussian assumption | No distributional assumption |
| More efficient when assumption holds | More robust when assumption violated |
| Less data needed if Gaussian | Needs more data |
| Can fail badly if not Gaussian | Works on most real data |

**Rule:** If data is approximately Gaussian → GDA. Otherwise → Logistic Regression.

## 5.3 Naive Bayes

### Motivation
GDA models P(x | y) as a multivariate Gaussian — requires estimating a full covariance matrix. For high-dimensional data (e.g., text with 10,000+ words), this is infeasible.

### The Naive Bayes Assumption

Features are **conditionally independent** given the class:

```
P(x₁, x₂, ..., xₙ | y) = Π_{j=1}^n P(xⱼ | y)
```

"Naive" because this independence assumption is almost never literally true — but works surprisingly well in practice.

### The Classification Rule

```
P(y | x) ∝ P(y) · Π_j P(xⱼ | y)

Score(class k) = P(y=k) · P(x₁|y=k) · P(x₂|y=k) · ... · P(xₙ|y=k)
```

Predict the class with the highest score.

### Learning Parameters

**Prior:**
```
P(y=k) = (number of examples with class k) / m
```

**For text (Multinomial NB):**
```
P(word_j | y=k) = (count of word_j in class k docs) / (total words in class k docs)
```

### Laplace Smoothing

Problem: if a word never appears in training for class k, P(word|class k) = 0 and the whole product = 0.

Solution — Laplace smoothing:
```
P(xⱼ=v | y=k) = (count(xⱼ=v, y=k) + 1) / (count(y=k) + |V|)
```

Add 1 to numerator, add vocabulary size to denominator.

### Variants

| Variant | Features | Use case |
|---|---|---|
| MultinomialNB | Word counts (0,1,2,...) | Text classification |
| BernoulliNB | Binary (word present/absent) | Short texts |
| GaussianNB | Continuous features | Structured data |

### Key Code

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

# Text preprocessing
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# ComplementNB handles class imbalance better
model = ComplementNB()
model.fit(X_train_vec, y_train)
print(f"Accuracy: {model.score(X_test_vec, y_test):.4f}")
```

---

# Chapter 6: Support Vector Machines (SVMs)

## 6.1 Motivation

Many decision boundaries can separate two classes. Which one is best?

**Answer:** The one with the **maximum margin** — the widest possible gap between the boundary and the closest points of each class.

## 6.2 Functional and Geometric Margins

### Functional Margin
```
γ̂⁽ⁱ⁾ = y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b)
```

- Positive → correctly classified
- Negative → misclassified
- Large positive → confident correct prediction

**Problem:** Not scale invariant. Multiply θ by 2 → functional margin doubles, but boundary unchanged.

### Geometric Margin
```
γ⁽ⁱ⁾ = y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) / ||θ||
```

The actual geometric distance from the point to the boundary. Scale invariant.

```
geometric margin = functional margin / ||θ||
```

## 6.3 The Optimal Margin Classifier

### Optimisation Problem

```
maximise  γ
subject to  y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) / ||θ|| ≥ γ  for all i
```

By convention, set functional margin = 1 for support vectors:

```
minimise  ½||θ||²
subject to  y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) ≥ 1  for all i
```

This is a **convex quadratic programming** problem — guaranteed global optimum.

**Geometric margin = 1/||θ||**, so minimising ||θ||² maximises the margin.

## 6.4 Support Vectors

Support vectors are the training points closest to the decision boundary — the points where y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) = 1 exactly.

**Key property:** Only support vectors determine the boundary. All other points can be removed without changing the boundary.

## 6.5 Lagrangian Duality

### Primal Problem
```
minimise  ½||θ||²
subject to  y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) ≥ 1
```

### Lagrangian
```
L(θ, b, α) = ½||θ||² - Σᵢ αᵢ[y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) - 1]
```

### Dual Problem (after taking derivatives and substituting)
```
maximise  Σᵢ αᵢ - ½ Σᵢ Σⱼ αᵢαⱼy⁽ⁱ⁾y⁽ʲ⁾<x⁽ⁱ⁾, x⁽ʲ⁾>
subject to  αᵢ ≥ 0,  Σᵢ αᵢy⁽ⁱ⁾ = 0
```

### Representer Theorem
The optimal θ can always be written as:
```
θ = Σᵢ αᵢy⁽ⁱ⁾x⁽ⁱ⁾
```

Crucially, αᵢ = 0 for all non-support vectors. Only support vectors contribute to θ.

**Everything depends on dot products** ⟨x⁽ⁱ⁾, x⁽ʲ⁾⟩ — this enables the kernel trick.

## 6.6 Soft Margin SVM

Real data isn't perfectly linearly separable. Allow some violations with slack variables ξᵢ:

```
minimise  ½||θ||² + C Σᵢ ξᵢ
subject to  y⁽ⁱ⁾(θᵀx⁽ⁱ⁾ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```

**C parameter:**
- Large C → narrow margin, fewer violations → can overfit
- Small C → wide margin, more violations → more regularisation

## 6.7 Hinge Loss

SVM minimises hinge loss:
```
Loss = max(0, 1 - y·(θᵀx))
```

- Correctly classified AND outside margin → loss = 0 (ignored)
- Inside margin or misclassified → positive loss

**Difference from logistic regression:**
- Logistic regression: ALL points affect the boundary (log loss)
- SVM: only support vectors (points near boundary) affect it

## 6.8 SVM vs Logistic Regression

| Property | Logistic Regression | SVM |
|---|---|---|
| Output | Probabilities | Hard classifications |
| Loss | Log loss (all points) | Hinge loss (boundary points only) |
| Robustness | Sensitive to outliers | Robust (ignores far-away points) |
| Speed on large data | Fast | Slow (kernel methods) |
| Probabilistic | Yes | No (unless calibrated) |

## 6.9 Key Code

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Always scale for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Linear SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)

# RBF kernel for non-linear
model_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
model_rbf.fit(X_train_scaled, y_train)
```

---

# Chapter 7: Kernels

## 7.1 The Problem with Linear Boundaries

Some data cannot be separated by a straight line:

```
Example: inner/outer circle data
* * * * *
*  · · · *    ← no straight line separates * from ·
* * * * *
```

**Solution:** Map data to a higher-dimensional space where it becomes linearly separable.

## 7.2 Feature Maps

Define a function φ: ℝⁿ → ℝᵖ (where p >> n) that maps original features to a higher-dimensional space.

**Example:** For x = [x₁, x₂]:
```
φ(x) = [x₁, x₂, x₁², x₂², x₁x₂]
```

In this 5D space, a linear boundary corresponds to a curved (quadratic) boundary in the original 2D space.

**Problem:** Computing φ(x) explicitly is expensive, especially for very high (or infinite) dimensional φ.

## 7.3 The Kernel Trick

The SVM dual formulation only needs **dot products** between training examples:
```
⟨x⁽ⁱ⁾, x⁽ʲ⁾⟩
```

A kernel function computes the dot product in the transformed space **without explicitly computing φ**:
```
K(x, z) = φ(x)ᵀφ(z)
```

This is the kernel trick: get the benefits of high-dimensional feature maps at the cost of computing a simple function K(x, z).

## 7.4 Common Kernels

### Linear Kernel
```
K(x, z) = xᵀz
```
Equivalent to no transformation. Use when data is already linearly separable or high-dimensional (text).

### Polynomial Kernel
```
K(x, z) = (xᵀz + c)^d
```
Corresponds to φ mapping to all polynomial features of degree ≤ d.

### RBF / Gaussian Kernel
```
K(x, z) = exp(-||x - z||² / 2σ²)
```

Properties:
- Corresponds to **infinite-dimensional** feature map
- K(x,z) → 1 when x ≈ z (very similar)
- K(x,z) → 0 when x and z are far apart
- σ² controls the width (bandwidth)

In sklearn: `gamma = 1/(2σ²)`
- High gamma → narrow bumps → complex boundary → can overfit
- Low gamma → wide bumps → smooth boundary → more regularisation

### Mercer's Condition
A function K is a valid kernel if and only if the kernel matrix K (where Kᵢⱼ = K(x⁽ⁱ⁾, x⁽ʲ⁾)) is symmetric positive semi-definite for any set of inputs.

## 7.5 Making Predictions with Kernels

After training:
```
θ = Σᵢ αᵢy⁽ⁱ⁾x⁽ⁱ⁾
```

Prediction for new point x:
```
θᵀx = Σᵢ αᵢy⁽ⁱ⁾ K(x⁽ⁱ⁾, x)
```

Sum over support vectors only (αᵢ = 0 for non-support vectors).

## 7.6 When to Use Which Kernel

| Kernel | Best for |
|---|---|
| Linear | High-dimensional data (text, genes), already separable |
| Polynomial | Known polynomial relationship |
| RBF (Gaussian) | Low-dimensional data with complex boundary (default choice) |

**Rule of thumb:**
- Text classification (10,000+ features) → linear kernel
- Image/audio/tabular with complex patterns → RBF
- When unsure → try RBF first

## 7.7 SVM Beyond Classification

Kernels make SVMs powerful for:
- **Regression (SVR):** Support Vector Regression
- **Anomaly detection:** One-class SVM
- **Structured prediction:** String kernels, graph kernels

## 7.8 Key Code

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Tune C and gamma together
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"Test score: {grid.score(X_test_scaled, y_test):.4f}")
```
