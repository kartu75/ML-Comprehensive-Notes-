# Stanford CS229 Machine Learning — Complete Notes (Part 2)
---

# Chapter 8: Data Splits, Models & Cross-Validation

## 8.1 The Bias-Variance Tradeoff

Every model makes errors. These errors come from two sources:

### Bias
Error from wrong assumptions in the model (underfitting).
- Model too simple to capture the true pattern
- High bias → high training error AND high test error
- Example: fitting a straight line to curved data

### Variance
Error from sensitivity to small fluctuations in the training data (overfitting).
- Model memorises training data instead of learning the pattern
- High variance → low training error but high test error
- Example: a degree-100 polynomial that wiggles through every training point

### The Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise
```

```
Model Complexity →
Bias²:     high ──────────────────→ low
Variance:  low  ─────────────────→ high
Total:     \___/  ← sweet spot
```

**Underfitting:** high bias, low variance
**Overfitting:** low bias, high variance
**Goal:** Find the sweet spot that minimises total error.

## 8.2 Train / Dev / Test Splits

### Why Three Sets?

If you tune hyperparameters based on test set performance, you're leaking information — your model implicitly fits the test set.

**Solution:** Three separate sets:
- **Training set (60-80%):** Model learns from this
- **Validation/Dev set (10-20%):** Tune hyperparameters here
- **Test set (10-20%):** Final evaluation, touched ONCE at the very end

### Split Sizes by Dataset Size

| Dataset Size | Train | Dev | Test |
|---|---|---|---|
| Small (<1000) | 70% | 15% | 15% |
| Medium (1k-100k) | 80% | 10% | 10% |
| Large (>1M) | 98% | 1% | 1% |

Larger datasets → smaller dev/test percentages still give reliable estimates.

### Code

```python
from sklearn.model_selection import train_test_split

# Step 1: split off test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Step 2: split remaining into train/dev
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
# 0.176 ≈ 15/85 → gives ~15% of original as dev
```

## 8.3 Cross-Validation

### Motivation
A single train/dev split might get lucky or unlucky. Cross-validation gives a more reliable performance estimate.

### K-Fold Cross-Validation

Split training data into k equal folds. Train k times, each time using a different fold as validation:

```
Fold 1: [VAL | TRAIN | TRAIN | TRAIN | TRAIN]
Fold 2: [TRAIN | VAL | TRAIN | TRAIN | TRAIN]
Fold 3: [TRAIN | TRAIN | VAL | TRAIN | TRAIN]
Fold 4: [TRAIN | TRAIN | TRAIN | VAL | TRAIN]
Fold 5: [TRAIN | TRAIN | TRAIN | TRAIN | VAL]
```

Average the k validation scores for the final estimate.

**Benefits:**
- Every point gets used for both training and validation
- More reliable than single split
- Especially useful for small datasets

**Typical choice:** k=5 or k=10.

### GridSearchCV (Automatic Hyperparameter Tuning)

Combines cross-validation with exhaustive hyperparameter search:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# Manual cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV mean: {scores.mean():.4f}, std: {scores.std():.4f}")

# Automatic hyperparameter search with CV
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best C: {grid.best_params_}")
print(f"Test score: {grid.score(X_test, y_test):.4f}")
```

### Time Series Cross-Validation

For time series data, you cannot shuffle — future data must never be in training:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

Each fold only trains on past data and validates on future data.

## 8.4 Diagnosing Bias vs Variance

| Train Error | Dev Error | Diagnosis | Fix |
|---|---|---|---|
| High | High | High bias (underfitting) | Bigger model, more features, more epochs |
| Low | High | High variance (overfitting) | More data, regularisation, simpler model |
| Low | Low | Good fit | — |
| High | Low | Bug in code | Check implementation |

## 8.5 Model Selection Criteria

Beyond accuracy, consider:

**AIC (Akaike Information Criterion):**
```
AIC = 2k - 2ln(L)
```
Where k = number of parameters, L = likelihood. Lower is better. Penalises complexity.

**BIC (Bayesian Information Criterion):**
```
BIC = k·ln(m) - 2ln(L)
```
Penalises complexity more than AIC. Prefer BIC for large datasets.

## 8.6 Evaluation Metrics

### Regression Metrics

**MSE (Mean Squared Error):**
```
MSE = (1/m) Σ(ŷᵢ - yᵢ)²
```

**RMSE:** √MSE — in original units, interpretable.

**R² (Coefficient of Determination):**
```
R² = 1 - SS_res/SS_tot = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²
```
- R²=1: perfect fit
- R²=0: model no better than predicting the mean
- R²<0: model worse than predicting the mean

### Classification Metrics

**Confusion Matrix:**
```
              Predicted 0    Predicted 1
Actual 0     TN (True Neg)  FP (False Pos)
Actual 1     FN (False Neg)  TP (True Pos)
```

**Accuracy:** (TP+TN) / (TP+TN+FP+FN) — misleading when classes imbalanced.

**Precision:** TP / (TP+FP) — of all predicted positives, how many were actually positive?

**Recall (Sensitivity):** TP / (TP+FN) — of all actual positives, how many were captured?

**F1 Score:** 2·(Precision·Recall) / (Precision+Recall) — harmonic mean, balances both.

**Precision-Recall Tradeoff:**
- Raise threshold → higher precision, lower recall
- Lower threshold → higher recall, lower precision

**When to use which:**
- Spam filter → high precision (don't want to lose real emails)
- Medical diagnosis → high recall (don't want to miss disease)
- Balanced → F1 score

---

# Chapter 9: Learning Theory

## 9.1 PAC Learning

**PAC = Probably Approximately Correct**

Answers: how many training examples do we need to guarantee our model is approximately correct with high probability?

```
m ≥ (1/ε) · (ln|H| + ln(1/δ))
```

Where:
- ε = error tolerance (how accurate we want to be)
- δ = failure probability (1-δ = confidence level)
- |H| = size of the hypothesis class (number of possible models)

**Key insight:** More complex models (larger |H|) require more training data to generalise well.

## 9.2 VC Dimension

Measures the complexity of a model class by how many points it can "shatter" (correctly classify with any labelling).

**Shattering:** A hypothesis class H shatters a set of points S if for every possible labelling of S, there exists some h ∈ H that achieves zero error.

**VC dimension (VC(H)):** The size of the largest set that H can shatter.

**Examples:**
- Linear classifiers in ℝ²: VC(H) = 3 (can shatter any 3 points, but not always 4)
- Linear classifiers in ℝⁿ: VC(H) = n+1

### Sauer's Lemma
```
|H(S)| ≤ Σₖ₌₀^{VC(H)} C(m, k) ≈ O(m^{VC(H)})
```

## 9.3 Generalisation Bound

With probability ≥ 1-δ, for any h ∈ H:
```
ε(h) ≤ ε̂(h) + √((VC(H)·ln(em/VC(H)) + ln(1/δ)) / m)
```

Where ε(h) = true error, ε̂(h) = training error.

**Interpretation:** True error ≤ Training error + Complexity penalty.

**Key conclusions:**
- More data → smaller gap between training and test error
- More complex model → needs more data to generalise

## 9.4 Empirical Risk Minimisation (ERM)

ERM: choose the hypothesis that minimises training error.

```
ĥ = argmin_{h∈H} ε̂(h)
```

Under PAC learning assumptions, ERM gives a good approximation of the best possible hypothesis when training set is large enough.

## 9.5 Bias-Variance Revisited Formally

```
E[(h(x) - y)²] = Bias²[h(x)] + Var[h(x)] + σ²
```

- Bias = E[h(x)] - f(x) where f is the true function
- Variance = E[(h(x) - E[h(x)])²]
- σ² = irreducible noise

## 9.6 No Free Lunch Theorem

**Statement:** Averaged over all possible problems, every algorithm performs equally.

**Implication:** There is no universally best algorithm. Any algorithm that performs better on some problems must perform worse on others.

**Practical consequence:** Always try multiple algorithms on your specific problem and evaluate empirically. The best model depends on the data, not just the algorithm.

## 9.7 Regularisation as a Complexity Penalty

Regularisation explicitly controls the bias-variance tradeoff by penalising model complexity.

**L2 (Ridge):**
```
J(θ) = Loss(θ) + λ||θ||²
```
Shrinks all weights towards zero. Never exactly zero.

**L1 (Lasso):**
```
J(θ) = Loss(θ) + λ||θ||₁
```
Drives some weights to exactly zero → sparse solution → feature selection.

**Elastic Net:**
```
J(θ) = Loss(θ) + λ₁||θ||₁ + λ₂||θ||²
```
Combination of L1 and L2.

**In sklearn:**
- `LogisticRegression(penalty='l2', C=1.0)` ← C = 1/λ, default
- `LogisticRegression(penalty='l1', solver='liblinear')`
- `Ridge(alpha=λ)` for linear regression
- `Lasso(alpha=λ)` for linear regression

---

# Chapter 10: Decision Trees & Ensemble Methods

## 10.1 Decision Trees

### What is a Decision Tree?

A hierarchical model that makes predictions by asking a sequence of binary questions about features.

```
                [Size > 1500?]
               /              \
       [Bedrooms > 2?]    [Income > 80k?]
       /          \           /        \
  [Leaf: $200k] [Leaf: $250k] [Leaf: $400k] [Leaf: $320k]
```

Each internal node = a feature and threshold
Each leaf node = a prediction (class or value)

### How Trees Split

At each node, find the split that maximally reduces impurity.

**For Classification — Gini Impurity:**
```
Gini(S) = 1 - Σₖ pₖ²  = Σₖ pₖ(1-pₖ)
```
- Gini = 0: perfectly pure (all same class)
- Gini = 0.5: maximum impurity (50/50 split, binary case)

**Gini Gain for a split:**
```
Gain = Gini(parent) - [|Sleft|/|S| · Gini(Sleft) + |Sright|/|S| · Gini(Sright)]
```

Pick the split with maximum Gini Gain.

**Alternative: Information Gain (Entropy):**
```
Entropy(S) = -Σₖ pₖ log₂(pₖ)
```
Information Gain = Entropy(parent) - weighted average Entropy(children)

**For Regression — Variance Reduction:**
```
MSE(S) = (1/|S|) Σᵢ (yᵢ - ȳ)²
```
Leaf prediction = mean of y values in that leaf.

### Key Hyperparameters

| Parameter | Effect |
|---|---|
| max_depth | Limits tree depth → controls overfitting |
| min_samples_split | Min samples to split a node |
| min_samples_leaf | Min samples in a leaf |
| max_features | Features considered at each split |

**Without max_depth:** Tree grows until leaves are pure → perfect training accuracy → severe overfitting.

### Advantages and Disadvantages

**Advantages:**
- Interpretable (can visualise and explain)
- No feature scaling needed
- Handles both numerical and categorical features
- Fast inference

**Disadvantages:**
- High variance (small data changes → very different tree)
- Prone to overfitting
- Not great for regression (piecewise constant predictions)

## 10.2 Ensemble Methods

Ensemble = combine multiple models to get better predictions than any single model.

**Why it works:** If individual models make different errors, averaging cancels out errors.

**Three main approaches:**
1. Bagging (Bootstrap Aggregating)
2. Boosting
3. Stacking

## 10.3 Bagging and Random Forest

### Bagging

Train multiple models on different **bootstrap samples** (random samples with replacement) of the training data. Average predictions.

```
Bootstrap 1: sample m points with replacement → train model 1
Bootstrap 2: sample m points with replacement → train model 2
...
Bootstrap B: sample m points with replacement → train model B

Prediction = average(model1, model2, ..., modelB)  # regression
           = majority vote(model1, ..., modelB)      # classification
```

**Effect:** Reduces variance without increasing bias. Particularly effective for high-variance models (deep trees).

### Random Forest

Bagging + extra randomness at each split:

At each node, only consider a **random subset of features** (typically √n for classification, n/3 for regression).

This de-correlates the trees — if one feature is very strong, it won't dominate every tree.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf = RandomForestClassifier(
    n_estimators=100,    # number of trees
    max_depth=None,      # let trees grow fully (bagging handles overfitting)
    max_features='sqrt', # features at each split
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importance (very useful!)
print(rf.feature_importances_)
```

### Out-of-Bag (OOB) Error

Each tree is trained on ~63% of the data (bootstrap). The remaining ~37% (OOB samples) can be used as a validation set for free — no need for a separate dev split.

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB score: {rf.oob_score_:.4f}")
```

## 10.4 Boosting

Train models **sequentially** — each model focuses on fixing the errors of the previous one.

### AdaBoost (Adaptive Boosting)

1. Start with equal weights for all training examples
2. Train weak classifier (decision stump — depth=1 tree)
3. Increase weights of misclassified examples
4. Train next classifier on reweighted data
5. Repeat T times
6. Final prediction = weighted vote of all classifiers

**Model weight:** More accurate classifiers get higher vote weight:
```
αₜ = (1/2) ln((1-εₜ)/εₜ)
```
Where εₜ = weighted error rate of classifier t.

**Example weight update:**
```
wᵢ → wᵢ · exp(-αₜ · yᵢ · hₜ(xᵢ))
```
Misclassified examples get higher weights, correctly classified get lower.

### Gradient Boosting

Instead of reweighting examples, each new tree is fit to the **residuals** (errors) of the current model:

```
Round 1: fit f₁ to y
         residuals = y - f₁(x)

Round 2: fit f₂ to residuals₁
         residuals = y - f₁(x) - f₂(x)

Round 3: fit f₃ to residuals₂
         ...

Final: F(x) = f₁(x) + f₂(x) + f₃(x) + ...
```

**Connection to gradient descent:** Residuals are the negative gradient of MSE loss. Fitting each tree to residuals = doing one step of gradient descent in function space.

## 10.5 XGBoost

XGBoost (eXtreme Gradient Boosting) extends gradient boosting with:

1. **Regularisation:** L1 and L2 penalties on tree weights → prevents overfitting
2. **Tree pruning:** Grows trees and prunes back (max_depth is a hard limit)
3. **Parallel processing:** Column block structure for fast tree construction
4. **Handling missing values:** Built-in sparsity-aware split finding
5. **Caching:** Cache-aware access patterns → faster

### XGBoost Objective

```
Obj(θ) = L(θ) + Ω(f)
```

Where:
- L(θ) = loss function (MSE for regression, log-loss for classification)
- Ω(f) = regularisation term on tree structure

### Key Hyperparameters

| Parameter | Effect |
|---|---|
| n_estimators | Number of trees (more → better but slower) |
| learning_rate | Shrinkage — how much each tree contributes |
| max_depth | Max depth of each tree |
| subsample | Fraction of training data per tree |
| colsample_bytree | Fraction of features per tree |
| reg_alpha | L1 regularisation |
| reg_lambda | L2 regularisation |

**Rule:** Lower learning_rate + more trees = better but slower. Common: lr=0.01-0.1, trees=100-1000.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,  # stop if no improvement for 10 rounds
        verbose=False)
```

## 10.6 Bagging vs Boosting

| Property | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| Tree training | Parallel (independent) | Sequential (each fixes previous) |
| Error reduced | Variance | Bias AND variance |
| Overfitting risk | Low (hard to overfit) | Can overfit if not regularised |
| Hyperparameter sensitivity | Low | Higher |
| Performance | Very good baseline | Usually best on structured data |
| Speed | Fast | Slower but more accurate |

## 10.7 Stacking

Train multiple diverse models, then train a **meta-model** on their predictions:

```
Level 0 models:
  Logistic Regression → prediction₁
  Random Forest       → prediction₂
  XGBoost            → prediction₃

Level 1 meta-model:
  Input: [prediction₁, prediction₂, prediction₃]
  Output: final prediction
```

Used mainly in Kaggle competitions for last 1-2% improvement.

---

# Chapter 11: Introduction to Neural Networks

## 11.1 Motivation

Linear models draw straight-line boundaries. Real data is often non-linear. Neural networks learn non-linear representations by stacking layers of linear transformations + non-linear activations.

**Key insight:** A neural network is just logistic regression stacked in layers.

## 11.2 The Neuron

Each neuron computes:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b  (linear combination)
a = g(z)                                          (activation function)
```

Where g is a non-linear activation function.

## 11.3 Neural Network Architecture

### Layers
- **Input layer:** receives raw features x
- **Hidden layers:** intermediate representations (nobody directly observes these)
- **Output layer:** final prediction

### Notation for Layer l

```
z^[l] = W^[l] a^[l-1] + b^[l]   (linear step)
a^[l] = g^[l](z^[l])             (activation step)
```

Where:
- W^[l] ∈ ℝ^{n^[l] × n^[l-1]} (weight matrix)
- b^[l] ∈ ℝ^{n^[l]} (bias vector)
- n^[l] = number of neurons in layer l

## 11.4 Why Hidden Layers?

Without activation functions, stacking layers = single linear layer:
```
a^[2] = W^[2](W^[1]x + b^[1]) + b^[2] = (W^[2]W^[1])x + (W^[2]b^[1] + b^[2])
```
= single linear transformation. Useless for non-linear patterns.

With activations: each layer transforms the representation into something more useful. Deep networks learn hierarchical features:
```
Images: pixels → edges → shapes → parts → objects
Text:   characters → words → phrases → sentences → meaning
```

## 11.5 Activation Functions

### ReLU (Rectified Linear Unit)
```
g(z) = max(0, z)
g'(z) = 1 if z > 0, else 0
```
**Default choice for hidden layers.** Fast, prevents vanishing gradient, sparse activations.

**Dying ReLU problem:** If z always negative, neuron always outputs 0 and stops learning.

### Leaky ReLU
```
g(z) = max(0.01z, z)
g'(z) = 1 if z > 0, else 0.01
```
Fixes dying ReLU — small gradient even for negative inputs.

### Sigmoid
```
g(z) = 1 / (1 + e^{-z})
g'(z) = g(z)(1 - g(z))  ≤ 0.25
```
**Use for binary output layer.** Avoid in hidden layers (vanishing gradient).

### Tanh
```
g(z) = (e^z - e^{-z}) / (e^z + e^{-z})
g'(z) = 1 - tanh²(z) ≤ 1
```
Output between -1 and 1. Better than sigmoid for hidden layers (zero-centred) but still suffers vanishing gradient in deep networks.

### Softmax
```
g(z)_k = e^{z_k} / Σⱼ e^{z_j}
```
**Use for multiclass output layer.** Outputs probability distribution over k classes.

### GELU (Gaussian Error Linear Unit)
```
g(z) ≈ z · Φ(z)
```
Used in transformers (BERT, GPT). Smooth approximation to ReLU.

## 11.6 Counting Parameters

For a network with architecture [n₀, n₁, n₂, ..., nL]:

```
Parameters in layer l:
Weights: n^[l] × n^[l-1]
Biases: n^[l]
Total layer l: n^[l] × (n^[l-1] + 1)
```

**Example:** Architecture [784, 128, 64, 10]:
```
Layer 1: 784 × 128 + 128 = 100,480
Layer 2: 128 × 64 + 64 = 8,256
Layer 3: 64 × 10 + 10 = 650
Total: 109,386 parameters
```

## 11.7 Forward Propagation

```
Input: x = a^[0]

For l = 1, 2, ..., L:
    z^[l] = W^[l] a^[l-1] + b^[l]
    a^[l] = g^[l](z^[l])

Output: ŷ = a^[L]
```

In matrix form for m training examples:
```
Z^[l] = W^[l] A^[l-1] + b^[l]    (shape: n^[l] × m)
A^[l] = g^[l](Z^[l])
```

## 11.8 Loss Functions for Neural Networks

**Binary classification:**
```
L(ŷ, y) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Multiclass classification (cross-entropy):**
```
L(ŷ, y) = -Σₖ yₖ log(ŷₖ)
```

**Regression (MSE):**
```
L(ŷ, y) = (ŷ - y)²
```

## 11.9 Universal Approximation Theorem

A neural network with a single hidden layer and enough neurons can approximate any continuous function to any desired accuracy.

**Implication:** Theoretically, one hidden layer is enough. In practice, deeper networks learn more efficiently with fewer total parameters.

## 11.10 PyTorch Implementation

```python
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)   # binary: 1 output, no sigmoid (BCEWithLogitsLoss handles it)
)

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.inference_mode():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    accuracy = (preds == y_test).float().mean()
```

---

# Chapter 12: Backpropagation & Improving Neural Networks

## 12.1 Backpropagation

Backpropagation is the algorithm for computing gradients of the loss with respect to all parameters in a neural network. It applies the **chain rule** of calculus from output to input.

### Chain Rule Review

For composed functions f(g(x)):
```
df/dx = (df/dg) · (dg/dx)
```

### Backprop Algorithm

**Forward pass:** compute and cache all z^[l] and a^[l].

**Output layer (layer L) error:**
```
δ^[L] = ∂L/∂z^[L] = (a^[L] - y) ⊙ g'^[L](z^[L])
```

For cross-entropy + sigmoid: δ^[L] = a^[L] - y (simplified).

**Propagate backwards through layer l:**
```
δ^[l] = (W^[l+1])ᵀ δ^[l+1] ⊙ g'^[l](z^[l])
```

**Gradients for parameters:**
```
∂L/∂W^[l] = δ^[l] (a^[l-1])ᵀ
∂L/∂b^[l] = δ^[l]
```

**Intuition:** δ^[l] is the "error signal" at layer l — how responsible each neuron is for the final loss. Errors propagate backwards from output to input.

### Vanishing Gradient Problem

In deep networks with sigmoid/tanh activations, gradients shrink as they propagate backwards:

```
g'(z) = g(z)(1-g(z)) ≤ 0.25

After 10 layers: gradient ≈ 0.25^10 ≈ 10^{-6}
```

Early layers barely learn — their weights don't update.

**Solution:** Use ReLU in hidden layers. g'(z) = 1 for z > 0 → no shrinking.

## 12.2 Weight Initialisation

### Why Not Zero?

If W^[l] = 0 for all l, then all neurons in a layer compute identical outputs → identical gradients → always identical → no learning (symmetry breaking failure).

### Xavier / Glorot Initialisation (for Tanh)
```
W ~ N(0, √(2/(n^[l-1] + n^[l])))
```
or uniform in [-√(6/(n^[l-1]+n^[l])), +√(6/(n^[l-1]+n^[l]))]

### He / Kaiming Initialisation (for ReLU)
```
W ~ N(0, √(2/n^[l-1]))
```

**Rule:** Use He initialisation with ReLU, Xavier with Tanh/Sigmoid. PyTorch uses Kaiming uniform by default for `nn.Linear`.

## 12.3 Batch Normalisation

### Problem
As training progresses, the distribution of each layer's inputs shifts ("internal covariate shift"), slowing training.

### Solution: Batch Norm
Normalise each layer's inputs to have mean=0, std=1, then scale and shift:

```
μ_B = (1/m) Σᵢ zᵢ
σ²_B = (1/m) Σᵢ (zᵢ - μ_B)²
z̃ᵢ = (zᵢ - μ_B) / √(σ²_B + ε)
aᵢ = γ z̃ᵢ + β
```

Where γ and β are learnable parameters (scale and shift).

**Benefits:**
- Faster training (higher learning rates possible)
- Reduces sensitivity to weight initialisation
- Acts as regularisation (reduces need for dropout)
- Placed between z^[l] and g^[l](z^[l]) (before activation)

```python
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.BatchNorm1d(64),  # before activation
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

## 12.4 Dropout

### Idea
During each training forward pass, randomly set each neuron's output to 0 with probability p (typically p=0.2-0.5).

```python
nn.Dropout(p=0.5)  # 50% of neurons randomly zeroed during training
```

### Why it Works
- Forces the network to learn redundant representations (can't rely on any single neuron)
- Acts as training an ensemble of exponentially many networks
- Different "thinned" networks at each step

### Important Notes
- Only active during training, not inference
- `model.train()` enables dropout; `model.eval()` disables it
- Scale outputs during training (or divide by (1-p) at test time) to maintain expected values

## 12.5 Optimisation Algorithms

### SGD with Momentum
```
v := βv + (1-β)∇θJ
θ := θ - α·v
```
Momentum (β≈0.9) accumulates gradients → faster in consistent directions, dampens oscillations.

### RMSProp
```
s := βs + (1-β)(∇θJ)²
θ := θ - α · ∇θJ / (√s + ε)
```
Adapts learning rate per parameter — divides by RMS of recent gradients.

### Adam (Adaptive Moment Estimation)
Combines momentum + RMSProp:

```
m := β₁m + (1-β₁)∇θJ        (first moment — momentum)
v := β₂v + (1-β₂)(∇θJ)²     (second moment — RMSProp)

m̂ = m / (1-β₁ᵗ)             (bias correction)
v̂ = v / (1-β₂ᵗ)

θ := θ - α · m̂ / (√v̂ + ε)
```

**Default hyperparameters:** β₁=0.9, β₂=0.999, ε=10⁻⁸, α=0.001.

**Adam is the default choice** for most deep learning tasks.

### Learning Rate Scheduling

```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# In training loop:
for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()
```

## 12.6 Regularisation for Neural Networks

### L2 Weight Decay
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Early Stopping
Stop training when validation loss stops improving:

```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(1000):
    train_loss = train()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

# Chapter 13: Debugging ML Models & Error Analysis

## 13.1 The Debugging Framework

When your model isn't performing well, diagnose BEFORE trying fixes.

### Step 1: Check Train vs Test Error

```
High train error, High test error → High Bias (underfitting)
Low train error, High test error  → High Variance (overfitting)
Low train error, Low test error   → Model is good
High train error, Low test error  → Bug in code (check implementation)
```

### Step 2: Human-Level Performance Comparison

```
Human error:    1%
Train error:    8%   ← avoidable bias = 8%-1% = 7%
Dev error:      10%  ← variance = 10%-8% = 2%
```

**Avoidable bias >> variance → fix bias first** (bigger model, more features, train longer)
**Variance >> avoidable bias → fix variance first** (more data, regularisation)

### Step 3: Ceiling Analysis

Determine the maximum improvement from fixing each component:

```
Current system accuracy: 72%

Fix face detection:      72% → 77%  (+5%)
Fix eye recognition:     77% → 80%  (+3%)
Fix nose recognition:    80% → 82%  (+2%)
Fix mouth recognition:   82% → 85%  (+3%)
Fix facial expression:   85% → 90%  (+5%)
Fix postprocessing:      90% → 91%  (+1%)
```

Focus effort on components with highest ceiling gains.

## 13.2 Fixes for High Bias

| Symptom | Fix |
|---|---|
| Training error high | Add more layers/neurons |
| Training error high | Train for more epochs |
| Training error high | Add more features |
| Training error high | Reduce regularisation |
| Training error high | Try different architecture |

## 13.3 Fixes for High Variance

| Symptom | Fix |
|---|---|
| Train << Test error | Collect more training data |
| Train << Test error | Add regularisation (L1, L2, dropout) |
| Train << Test error | Reduce model complexity |
| Train << Test error | Early stopping |
| Train << Test error | Feature selection |
| Train << Test error | Data augmentation |

## 13.4 Error Analysis

Manually examine misclassified examples in the dev set.

**Process:**
1. Sample ~100 misclassified dev examples
2. Categorise errors by type
3. Prioritise fixes for most common error types

**Example for spam classifier:**
```
Misclassified emails (100 samples):
Drug emails:         12%
Fake domain emails:  35%
Unusual format:      43%
Other:               10%
```

→ Fix unusual format first (highest impact)

## 13.5 Data Mismatch

Sometimes train and test distributions differ (e.g., train on studio photos, test on phone photos).

**Detection:**
```
Training set error:  1%
Training-dev error:  9%  ← large gap → high variance (not data mismatch)
Dev error:           10%

OR

Training set error:  1%
Training-dev error:  2%  ← small gap → not variance
Dev error:           10% ← large gap vs training-dev → DATA MISMATCH
```

**Fix data mismatch:**
- Collect more data similar to test distribution
- Data synthesis/augmentation to match test distribution

## 13.6 Learning Curves

Plot training and validation error vs number of training examples:

```
Error
  |
  |  ─── Train error
  |       ─────────────────────────
  |  ─── Dev error
  |  \
  |   \__________________________
  |_________________________________ m (training examples)
```

**High bias:** Both curves plateau at high error, close together.
**High variance:** Training error is low, large gap to dev error.
**Fix:** More data helps variance but not bias.

---

# Chapter 14: Expectation-Maximization (EM) Algorithm

## 14.1 The Problem: Latent Variable Models

Some models have **latent (hidden) variables** z that we cannot observe directly.

**Chicken-and-egg problem:**
- To estimate parameters θ → need to know z
- To infer z → need to know parameters θ

EM breaks this deadlock by alternating.

## 14.2 The EM Algorithm

**Goal:** Maximise the log-likelihood of observed data:
```
ℓ(θ) = Σᵢ log p(x⁽ⁱ⁾; θ) = Σᵢ log Σ_z p(x⁽ⁱ⁾, z; θ)
```

The sum inside the log makes this hard to optimise directly.

**EM Framework:**

**E-Step (Expectation):** Compute the posterior distribution over z given current parameters:
```
Qᵢ(z⁽ⁱ⁾) = p(z⁽ⁱ⁾ | x⁽ⁱ⁾; θ)
```

**M-Step (Maximisation):** Update parameters to maximise the expected log-likelihood:
```
θ := argmax_θ Σᵢ Σ_{z⁽ⁱ⁾} Qᵢ(z⁽ⁱ⁾) log [p(x⁽ⁱ⁾, z⁽ⁱ⁾; θ) / Qᵢ(z⁽ⁱ⁾)]
```

## 14.3 Jensen's Inequality

EM relies on Jensen's inequality to construct a lower bound on log-likelihood.

**Statement:** For a concave function f:
```
f(E[X]) ≥ E[f(X)]
```

**For log (concave):**
```
log(E[X]) ≥ E[log(X)]
```

**Application to EM:**
```
log p(x; θ) = log Σ_z p(x, z; θ)
             = log Σ_z Q(z) · [p(x, z; θ) / Q(z)]
             ≥ Σ_z Q(z) log [p(x, z; θ) / Q(z)]    ← Jensen's inequality
             = ELBO (Evidence Lower BOund)
```

EM maximises this lower bound instead of the intractable log-likelihood directly.

## 14.4 EM Guarantees

1. **Monotonic improvement:** Log-likelihood increases (or stays the same) at every iteration: ℓ(θ^{t+1}) ≥ ℓ(θ^t)
2. **Convergence:** Always converges (but may converge to local maximum)
3. **Initialisation sensitivity:** Different starting points may give different solutions → run multiple times

## 14.5 EM for Gaussian Mixture Models (GMM)

### Model
Data is generated by:
1. Choose component k with probability πₖ (mixing weights)
2. Sample x from Gaussian N(μₖ, Σₖ)

```
p(x) = Σₖ πₖ · N(x; μₖ, Σₖ)
```

Hidden variable z: which component generated each point.

### E-Step: Compute Responsibilities

```
rᵢₖ = p(z⁽ⁱ⁾=k | x⁽ⁱ⁾) = πₖ · N(x⁽ⁱ⁾; μₖ, Σₖ) / Σⱼ πⱼ · N(x⁽ⁱ⁾; μⱼ, Σⱼ)
```

rᵢₖ = how much is point i "responsible" to component k.

### M-Step: Update Parameters

```
Nₖ = Σᵢ rᵢₖ                                         (effective # points in component k)

μₖ = (1/Nₖ) Σᵢ rᵢₖ x⁽ⁱ⁾                            (weighted mean)

Σₖ = (1/Nₖ) Σᵢ rᵢₖ (x⁽ⁱ⁾ - μₖ)(x⁽ⁱ⁾ - μₖ)ᵀ        (weighted covariance)

πₖ = Nₖ / m                                          (mixing weight)
```

### GMM vs K-Means

| K-Means | GMM/EM |
|---|---|
| Hard assignments (0 or 1) | Soft assignments (probabilities) |
| Finds cluster centers only | Finds full distribution (mean + covariance) |
| Fast | Slower |
| Special case of EM (σ→0) | General probabilistic model |

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Soft assignments
responsibilities = gmm.predict_proba(X)

# Hard assignments
labels = gmm.predict(X)

# Parameters
print(gmm.means_)       # μₖ for each component
print(gmm.covariances_) # Σₖ for each component
print(gmm.weights_)     # πₖ for each component

# Log-likelihood
print(gmm.score(X))
```
