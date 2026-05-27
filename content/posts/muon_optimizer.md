---
title: "The Muon Optimizer: From Matrix Basics to State-of-the-Art Training"
date: 2025-01-25
draft: true
math: true
summary: "A deep dive into the Muon optimizer - understanding orthogonalization, Newton-Schulz iterations, and why throwing away gradient magnitudes makes training faster."
tags: ["deep-learning", "optimization", "math", "muon"]
showToc: true
tocOpen: true
---

In late 2024, a new optimizer called **Muon** emerged from the competitive world of neural network speedrunning. It trained models significantly faster than AdamW while using *less* memory. The secret? Instead of using gradients directly, Muon *orthogonalizes* them first.

But what does "orthogonalize a gradient" even mean? And why would throwing away information (the gradient magnitudes) make optimization *better*?

This post builds up the intuition from first principles. We'll start with what matrices really do, understand what makes orthogonal matrices special, and then see why orthogonalizing gradients is such a powerful idea. Along the way, we'll prove the key mathematical results and visualize everything with Python.

---

## Part 1: Understanding Matrices as Transformations

Before we can understand orthogonalization, we need to deeply understand what matrices *do* to vectors and space itself.

### 1.1 Matrices Are Functions on Vectors

A matrix $W$ isn't just a grid of numbers — it's a **function** that transforms vectors. When you multiply a vector $\mathbf{x}$ by a matrix $W$, you get a new vector $\mathbf{y} = W\mathbf{x}$.

For a 2×2 matrix and 2D vectors:

$$W = \begin{pmatrix} a & b \\\ c & d \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} x_1 \\\ x_2 \end{pmatrix}$$

$$W\mathbf{x} = \begin{pmatrix} a \cdot x_1 + b \cdot x_2 \\\ c \cdot x_1 + d \cdot x_2 \end{pmatrix}$$

Each component of the output is a **linear combination** of the input components. This is why matrix multiplication is called a **linear transformation**.

### 1.2 What Linear Transformations Can Do

Linear transformations can:
- **Stretch** or **compress** space along certain directions
- **Rotate** space
- **Reflect** space (flip it)
- **Shear** space (skew it)
- **Project** onto lower dimensions (collapse some directions)

What they *cannot* do:
- Curve straight lines
- Move the origin

**Key insight:** Every linear transformation preserves the origin and maps straight lines to straight lines (or points).

### 1.3 Visualizing Transformations

Let's see what different matrices do to a grid of points:

<iframe
  src="/muon/matrix-transforms.html"
  width="100%"
  height="850"
  style="border:0; border-radius: 80px;"
  loading="lazy">
</iframe>

*Explore how different matrices transform space. Watch how the unit circle becomes an ellipse, and how the grid lines bend and stretch.*

The key observation: **different matrices stretch space by different amounts in different directions**. Some matrices stretch uniformly (like scaling), while others stretch a lot in one direction and compress in another. This uneven stretching is exactly what makes optimization hard — and what Muon fixes.

---

## Part 2: The Singular Value Decomposition (SVD)

The SVD is one of the most important decompositions in linear algebra. It reveals the hidden structure of any matrix.

### 2.1 Quick Background: Orthogonal and Orthonormal

Before diving in, let's clarify some terminology:

**Orthogonal vectors:** Two vectors are orthogonal if they're perpendicular — their dot product is zero:
$$\mathbf{u} \cdot \mathbf{v} = 0$$

**Orthonormal vectors:** Vectors that are both orthogonal to each other AND have unit length (length = 1):
$$\mathbf{u} \cdot \mathbf{v} = 0 \quad \text{and} \quad \|\mathbf{u}\| = \|\mathbf{v}\| = 1$$

**Orthogonal matrix:** A square matrix where:
- Every column is perpendicular to every other column
- Every column has unit length (length = 1)
- (Same is true for the rows)

This is summarized by: $Q^T Q = I$

Why? The $(i,j)$ entry of $Q^T Q$ is the dot product of column $i$ and column $j$. For this to equal $I$ (ones on diagonal, zeros elsewhere), columns must be mutually perpendicular (off-diagonal = 0) and unit length (diagonal = 1).

**Example:** A 45° rotation matrix:

$$Q = \begin{pmatrix} 0.707 & -0.707 \\\ 0.707 & 0.707 \end{pmatrix}$$

Column 1: $(0.707, 0.707)$ — length = $\sqrt{0.5 + 0.5} = 1$ ✓

Column 2: $(-0.707, 0.707)$ — length = $1$ ✓

Dot product: $(0.707)(-0.707) + (0.707)(0.707) = -0.5 + 0.5 = 0$ ✓ (perpendicular)

Let's verify $Q^T Q = I$:

$$Q^T Q = \begin{pmatrix} 0.707 & 0.707 \\\ -0.707 & 0.707 \end{pmatrix} \begin{pmatrix} 0.707 & -0.707 \\\ 0.707 & 0.707 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\\ 0 & 1 \end{pmatrix} = I \quad \checkmark$$

This will be crucial for understanding why orthogonal matrices are "perfect" transformations.

### 2.2 The SVD Theorem

**Theorem (Singular Value Decomposition):** Every matrix $A \in \mathbb{R}^{m \times n}$ can be written as:

$$A = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal (columns are orthonormal)
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$ is orthogonal (columns are orthonormal)

The diagonal entries $\sigma_i$ are called **singular values**.

### 2.3 Geometric Interpretation: Rotate → Stretch → Rotate

The SVD tells us that *any* linear transformation can be decomposed into three simple steps:

1. **$V^T$: Rotate** the input space to align with special directions
2. **$\Sigma$: Stretch** along each axis by the singular values
3. **$U$: Rotate** the result to the final orientation

Let's prove this geometrically. Consider what $A$ does to the unit sphere:

$$\{\mathbf{x} : \|\mathbf{x}\| = 1\} \xrightarrow{A} \{A\mathbf{x} : \|\mathbf{x}\| = 1\}$$

The image is an **ellipsoid**! The semi-axes of this ellipsoid have lengths $\sigma_1, \sigma_2, \ldots$ and point in the directions given by the columns of $U$.

<iframe
  src="/muon/svd-visualization.html"
  width="100%"
  height="950"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*See the SVD in action. The unit circle is first rotated by $V^T$, then stretched by $\Sigma$ into an ellipse, then rotated by $U$ to the final orientation. Drag the sliders to modify the matrix and watch the decomposition update.*

### 2.4 Background: Eigenvalues and Symmetric Matrices

Before we dive into why SVD always exists, we need a few key concepts.

**Eigenvalues and Eigenvectors**

For a square matrix $A$, an **eigenvector** is a special direction that the matrix just stretches (or shrinks) without rotating:

$$A\mathbf{v} = \lambda \mathbf{v}$$

The scalar $\lambda$ is the **eigenvalue** — how much the matrix stretches in that direction:
- $\lambda = 2$: vectors in that direction double in length
- $\lambda = -1$: vectors flip direction (and keep their length)
- $\lambda = 0$: vectors get collapsed to zero

**Example:** For the matrix $A = \begin{pmatrix} 2 & 0 \\\ 0 & 3 \end{pmatrix}$:

- $\mathbf{v}_1 = \begin{pmatrix} 1 \\\ 0 \end{pmatrix}$ is an eigenvector with $\lambda_1 = 2$
- $\mathbf{v}_2 = \begin{pmatrix} 0 \\\ 1 \end{pmatrix}$ is an eigenvector with $\lambda_2 = 3$

You can verify: $A\mathbf{v}_1 = \begin{pmatrix} 2 \\\ 0 \end{pmatrix} = 2\mathbf{v}_1$ ✓

<iframe
  src="/muon/eigenvector-demo.html"
  width="100%"
  height="780"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Drag the angle slider to rotate the input vector (blue). Watch the output vector (purple). When you align with an eigenvector direction, the output points the same way — only stretched, not rotated!*

**Symmetric Matrices**

A matrix $A$ is **symmetric** if $A = A^T$ (equals its own transpose). This happens when $A_{ij} = A_{ji}$ for all entries.

$$\text{Symmetric: } \begin{pmatrix} 1 & 2 \\\ 2 & 3 \end{pmatrix} \qquad \text{Not symmetric: } \begin{pmatrix} 1 & 2 \\\ 5 & 3 \end{pmatrix}$$

Symmetric matrices have two magical properties:
1. **All eigenvalues are real numbers** (not complex)
2. **Eigenvectors for different eigenvalues are perpendicular**

**Positive Semi-Definite (PSD) Matrices**

This concept looks scary but is actually simple once you see what it means.

**Step 1: Understanding the quadratic form $\mathbf{x}^T A \mathbf{x}$**

This expression takes a vector $\mathbf{x}$ and a matrix $A$, and outputs a single number. Let's compute it for a concrete example:

$$A = \begin{pmatrix} 2 & 1 \\\ 1 & 3 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 1 \\\ 2 \end{pmatrix}$$

First, compute $A\mathbf{x}$:
$$A\mathbf{x} = \begin{pmatrix} 2 & 1 \\\ 1 & 3 \end{pmatrix} \begin{pmatrix} 1 \\\ 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 1 + 1 \cdot 2 \\\ 1 \cdot 1 + 3 \cdot 2 \end{pmatrix} = \begin{pmatrix} 4 \\\ 7 \end{pmatrix}$$

Then, compute $\mathbf{x}^T (A\mathbf{x})$:
$$\mathbf{x}^T A\mathbf{x} = \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 4 \\\ 7 \end{pmatrix} = 1 \cdot 4 + 2 \cdot 7 = 18$$

So for this $A$ and $\mathbf{x}$, the quadratic form gives us 18.

**Step 2: What "positive semi-definite" means**

A symmetric matrix $A$ is **positive semi-definite** if this quadratic form is never negative:

$$\mathbf{x}^T A \mathbf{x} \geq 0 \quad \text{for ALL possible vectors } \mathbf{x}$$

No matter what vector you plug in, you always get zero or a positive number out.

**Step 3: Why is this useful?**

Think of $\mathbf{x}^T A \mathbf{x}$ as measuring "energy" or "cost." A PSD matrix never gives you negative energy — it's like a bowl that always curves upward:

- **PSD matrix**: The quadratic form defines a bowl shape (minimum at origin)
- **Non-PSD matrix**: The quadratic form has a saddle point (goes negative in some directions)

**Step 4: Connection to eigenvalues**

Here's the key insight. If $A$ has eigenvalue $\lambda$ with eigenvector $\mathbf{v}$, then:

$$\mathbf{v}^T A \mathbf{v} = \mathbf{v}^T (\lambda \mathbf{v}) = \lambda (\mathbf{v}^T \mathbf{v}) = \lambda \|\mathbf{v}\|^2$$

Since $\|\mathbf{v}\|^2 > 0$ (eigenvectors aren't zero), the sign of $\mathbf{v}^T A \mathbf{v}$ equals the sign of $\lambda$.

So if ANY eigenvalue is negative, plugging in its eigenvector gives a negative result — violating PSD!

**Conclusion:** A symmetric matrix is PSD if and only if all eigenvalues are $\geq 0$.

**Step 5: Example of a NON-PSD matrix**

$$B = \begin{pmatrix} 1 & 0 \\\ 0 & -2 \end{pmatrix}$$

This has eigenvalues $\lambda_1 = 1$ and $\lambda_2 = -2$. For $\mathbf{x} = \begin{pmatrix} 0 \\\ 1 \end{pmatrix}$:

$$\mathbf{x}^T B \mathbf{x} = \begin{pmatrix} 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\\ 0 & -2 \end{pmatrix} \begin{pmatrix} 0 \\\ 1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \end{pmatrix} \begin{pmatrix} 0 \\\ -2 \end{pmatrix} = -2$$

Negative! So $B$ is not PSD.

<iframe
  src="/muon/psd-visualization.html"
  width="100%"
  height="580"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Move the sliders to test different points. The PSD matrix (left) is always green (≥ 0). The non-PSD matrix (right) goes red when you increase x₂ — that's the negative eigenvalue direction!*

**Why $A^T A$ is Always Symmetric PSD**

Here's the key fact for the SVD proof. For *any* matrix $A$ (not necessarily square), $A^T A$ is always symmetric and PSD. Let's prove both.

**Proof that $A^T A$ is symmetric:**

We need to show $(A^T A)^T = A^T A$.

Using the rule $(XY)^T = Y^T X^T$:
$$(A^T A)^T = A^T (A^T)^T = A^T A \quad \checkmark$$

**Proof that $A^T A$ is positive semi-definite:**

We need to show $\mathbf{x}^T (A^T A) \mathbf{x} \geq 0$ for all $\mathbf{x}$.

The trick is to regroup the expression. Using the rule $(XY)^T = Y^T X^T$:

$$\mathbf{x}^T (A^T A) \mathbf{x} = \mathbf{x}^T A^T A \mathbf{x} = (A\mathbf{x})^T (A\mathbf{x})$$

But $(A\mathbf{x})^T (A\mathbf{x})$ is just the dot product of $A\mathbf{x}$ with itself — which is the squared length!

$$= \|A\mathbf{x}\|^2 \geq 0 \quad \checkmark$$

Squared lengths are always non-negative, so $A^T A$ is PSD.

**Concrete example:** Let $A = \begin{pmatrix} 1 & 2 \\\ 3 & 4 \end{pmatrix}$ and $\mathbf{x} = \begin{pmatrix} 1 \\\ 1 \end{pmatrix}$.

First, $A\mathbf{x} = \begin{pmatrix} 3 \\\ 7 \end{pmatrix}$.

Then $\|A\mathbf{x}\|^2 = 3^2 + 7^2 = 58$.

And $A^T A = \begin{pmatrix} 10 & 14 \\\ 14 & 20 \end{pmatrix}$, so $\mathbf{x}^T A^T A \mathbf{x} = \begin{pmatrix} 1 & 1 \end{pmatrix} \begin{pmatrix} 24 \\\ 34 \end{pmatrix} = 58$ ✓

Same answer, as expected.

**The Spectral Theorem**

This is one of the most beautiful results in linear algebra:

> **Spectral Theorem:** Every symmetric matrix can be written as $A = V \Lambda V^T$, where $V$ is orthogonal and $\Lambda$ is diagonal.

**What this means:**

- $V$ is orthogonal — its columns are the eigenvectors, and they're perpendicular to each other with unit length
- $\Lambda$ is diagonal — the eigenvalues sit on the diagonal: $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots)$

**Why it's useful:** This decomposes any symmetric matrix into "rotate → scale each axis → rotate back":

$$A\mathbf{x} = V \Lambda V^T \mathbf{x}$$

1. $V^T \mathbf{x}$: Rotate to the eigenvector coordinate system
2. $\Lambda (\cdot)$: Scale each coordinate by its eigenvalue
3. $V (\cdot)$: Rotate back to original coordinates

**For PSD matrices:** All eigenvalues are $\geq 0$, so $\Lambda$ has non-negative diagonal entries.

**Why symmetric matrices are special:** Non-symmetric matrices might have complex eigenvalues, or eigenvectors that aren't perpendicular. Symmetric matrices are "nice" — everything stays real and orthogonal.

This is why $A^TA$ in the next section has non-negative eigenvalues and orthonormal eigenvectors — it's symmetric PSD, so the Spectral Theorem applies!

### 2.5 Proof Sketch of SVD Existence

**Why does SVD always exist?** Here's the key insight:

Consider $A^T A$. This is a symmetric positive semi-definite matrix, so it has an orthonormal basis of eigenvectors with non-negative eigenvalues:

$$A^T A = V \Lambda V^T$$

where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots)$ with $\lambda_i \geq 0$.

Define $\sigma_i = \sqrt{\lambda_i}$ and $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots)$.

For each $\sigma_i > 0$, define:
$$\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$$

**Claim:** The $\mathbf{u}_i$ are orthonormal.

**Proof:** We need to show $\mathbf{u}_i^T \mathbf{u}_j = 1$ if $i=j$, and $0$ otherwise.

$$\mathbf{u}_i^T \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} (A\mathbf{v}_i)^T (A\mathbf{v}_j) = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T A^T A \mathbf{v}_j$$

Since $A^T A \mathbf{v}_j = \lambda_j \mathbf{v}_j$ (eigenvector equation):

$$= \frac{\lambda_j}{\sigma_i \sigma_j} \mathbf{v}_i^T \mathbf{v}_j$$

The $\mathbf{v}_i$ are orthonormal, so $\mathbf{v}_i^T \mathbf{v}_j = 0$ when $i \neq j$, and $= 1$ when $i = j$.

When $i = j$: using $\lambda_j = \sigma_j^2$, we get $\frac{\sigma_j^2}{\sigma_j^2} \cdot 1 = 1$. ✓

Now we can verify $A = U\Sigma V^T$ by checking $A\mathbf{v}_i = \sigma_i \mathbf{u}_i$. ∎

### 2.6 The Singular Values Reveal Matrix "Size"

The singular values tell us how much a matrix stretches space:

- **Largest singular value** $\sigma_1 = \|A\|_2$: The **spectral norm** — maximum stretching in any direction
- **Smallest singular value** $\sigma_n$: Minimum stretching (if zero, the matrix collapses some direction)
- **Sum of singular values** $\sum \sigma_i = \|A\|_*$: The **nuclear norm**
- **Sum of squared singular values** $\sum \sigma_i^2 = \|A\|_F^2$: The **Frobenius norm** squared

### 2.7 Computing Singular Values: An Example

Let's compute the SVD of a simple matrix by hand:

$$A = \begin{pmatrix} 3 & 0 \\\ 0 & 1 \end{pmatrix}$$

**Step 1:** Compute $A^T A$:
$$A^T A = \begin{pmatrix} 3 & 0 \\\ 0 & 1 \end{pmatrix}\begin{pmatrix} 3 & 0 \\\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 9 & 0 \\\ 0 & 1 \end{pmatrix}$$

**Step 2:** Find eigenvalues of $A^T A$:
$$\det(A^T A - \lambda I) = (9-\lambda)(1-\lambda) = 0$$
$$\lambda_1 = 9, \quad \lambda_2 = 1$$

**Step 3:** Singular values:
$$\sigma_1 = \sqrt{9} = 3, \quad \sigma_2 = \sqrt{1} = 1$$

**Step 4:** Eigenvectors of $A^T A$ give us $V$:
$$\mathbf{v}_1 = \begin{pmatrix} 1 \\\ 0 \end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix} 0 \\\ 1 \end{pmatrix}$$

**Step 5:** Compute $U$ from $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$:
$$\mathbf{u}_1 = \frac{1}{3}\begin{pmatrix} 3 \\\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\\ 0 \end{pmatrix}, \quad \mathbf{u}_2 = \frac{1}{1}\begin{pmatrix} 0 \\\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\\ 1 \end{pmatrix}$$

So for this diagonal matrix, $U = V = I$ and $\Sigma = A$. This makes sense: a diagonal matrix is already in its "stretched" form!

---

## Part 3: The Condition Number and Why It Matters

### 3.1 Definition of Condition Number

The **condition number** of a matrix is:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_n}$$

If $\sigma_{\min} = 0$ (matrix is not full rank), we say $\kappa(A) = \infty$.

### 3.2 Geometric Meaning

The condition number measures **how unevenly a matrix stretches space**.

- $\kappa = 1$: All singular values are equal. The matrix stretches uniformly in all directions (like scaling by a constant times a rotation).
- $\kappa = 10$: The matrix stretches 10× more in one direction than another.
- $\kappa = 1000$: Huge disparity. The matrix nearly collapses one direction while greatly stretching another.

<iframe
  src="/muon/condition-number.html"
  width="100%"
  height="600"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Visualize condition number. Adjust the singular values and watch how the transformed ellipse becomes more elongated as condition number increases.*

### 3.3 Why Condition Number Kills Optimization

Consider gradient descent on a quadratic loss:

$$\mathcal{L}(\theta) = \frac{1}{2} \theta^T H \theta$$

where $H$ is the Hessian (matrix of second derivatives).

The gradient is $\nabla \mathcal{L} = H\theta$, and gradient descent updates:

$$\theta_{t+1} = \theta_t - \eta H \theta_t = (I - \eta H) \theta_t$$

**Convergence Rate Analysis:**

Let $H = U \Lambda U^T$ be the eigendecomposition. In the eigenbasis:

$$\tilde{\theta}_{t+1} = (I - \eta \Lambda) \tilde{\theta}_t$$

Each component evolves independently:

$$\tilde{\theta}_{t+1}^{(i)} = (1 - \eta \lambda_i) \tilde{\theta}_t^{(i)}$$

For convergence, we need $|1 - \eta \lambda_i| < 1$ for all $i$, which means:

$$0 < \eta < \frac{2}{\lambda_{\max}}$$

The **convergence rate** for component $i$ is $|1 - \eta \lambda_i|$. The overall convergence is limited by the slowest component.

**Optimal learning rate:** $\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}$

At this learning rate, the convergence rate is:

$$\rho = \frac{\kappa - 1}{\kappa + 1}$$

where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number.

**The problem:** As $\kappa \to \infty$, $\rho \to 1$, meaning convergence becomes arbitrarily slow.

| Condition Number | Convergence Rate | Iterations to reduce error by $10\times$ |
|-----------------|------------------|------------------------------------------|
| $\kappa = 1$ | 0 | 1 |
| $\kappa = 10$ | 0.82 | 12 |
| $\kappa = 100$ | 0.98 | 115 |
| $\kappa = 1000$ | 0.998 | 1150 |

### 3.4 The Zig-Zag Problem Visualized

When the condition number is high, gradient descent exhibits the characteristic "zig-zag" pattern:

<iframe
  src="/muon/zigzag-demo.html"
  width="100%"
  height="600"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Watch gradient descent struggle on an ill-conditioned landscape. The optimizer oscillates in the steep direction while making slow progress in the flat direction.*

The gradient points mostly toward the steep direction (where the loss changes quickly), but that's not the direction toward the minimum! The optimizer wastes effort fighting oscillations instead of making progress.

---

## Part 4: Orthogonal Matrices — The Perfect Transformations

### 4.1 Definition and Equivalent Conditions

A square matrix $Q$ is **orthogonal** if any of these equivalent conditions hold:

1. $Q^T Q = I$ (columns are orthonormal)
2. $Q Q^T = I$ (rows are orthonormal)
3. $Q^{-1} = Q^T$ (transpose is inverse)
4. $\|Q\mathbf{x}\| = \|\mathbf{x}\|$ for all $\mathbf{x}$ (preserves lengths)
5. All singular values equal 1

Let's prove these equivalences.

**Proof (1 ⟹ 4):**

$$\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^T(Q\mathbf{x}) = \mathbf{x}^T Q^T Q \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|^2$$

Taking square roots: $\|Q\mathbf{x}\| = \|\mathbf{x}\|$. ∎

**Proof (4 ⟹ 5):**

The singular values satisfy $\sigma_i = \max_{\|\mathbf{x}\|=1, \mathbf{x} \perp \mathbf{v}_1,\ldots,\mathbf{v}_{i-1}} \|A\mathbf{x}\|$.

If $\|Q\mathbf{x}\| = \|\mathbf{x}\|$ for all $\mathbf{x}$, then in particular for unit vectors:
$$\sigma_i = \max_{\|\mathbf{x}\|=1, \ldots} \|Q\mathbf{x}\| = \max_{\|\mathbf{x}\|=1, \ldots} 1 = 1$$

So all singular values are 1. ∎

**Proof (5 ⟹ 1):**

If all singular values are 1, then $\Sigma = I$, so:
$$Q = U I V^T = U V^T$$
$$Q^T Q = (UV^T)^T(UV^T) = VU^T UV^T = V V^T = I$$ ∎

### 4.2 Orthogonal Matrices Preserve Everything

Orthogonal matrices preserve:

1. **Lengths:** $\|Q\mathbf{x}\| = \|\mathbf{x}\|$
2. **Angles:** $\cos\theta = \frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|} = \frac{(Q\mathbf{x})^T(Q\mathbf{y})}{\|Q\mathbf{x}\|\|Q\mathbf{y}\|}$
3. **Distances:** $\|Q\mathbf{x} - Q\mathbf{y}\| = \|Q(\mathbf{x}-\mathbf{y})\| = \|\mathbf{x}-\mathbf{y}\|$
4. **Volumes:** $|\det(Q)| = 1$

**Proof of angle preservation:**

$$(Q\mathbf{x})^T(Q\mathbf{y}) = \mathbf{x}^T Q^T Q \mathbf{y} = \mathbf{x}^T \mathbf{y}$$

And we already know $\|Q\mathbf{x}\| = \|\mathbf{x}\|$, $\|Q\mathbf{y}\| = \|\mathbf{y}\|$. ∎

### 4.3 Orthogonal = Rotation (and Reflection)

In 2D, every orthogonal matrix is either:

**Rotation by angle $\theta$:**
$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\\ \sin\theta & \cos\theta \end{pmatrix}, \quad \det(R_\theta) = 1$$

**Reflection (rotation + flip):**
$$\begin{pmatrix} \cos\theta & \sin\theta \\\ \sin\theta & -\cos\theta \end{pmatrix}, \quad \det = -1$$

In higher dimensions, orthogonal matrices are compositions of rotations in 2D planes and possibly reflections.

### 4.4 The Condition Number of Orthogonal Matrices

Since all singular values of an orthogonal matrix are 1:

$$\kappa(Q) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{1}{1} = 1$$

**Orthogonal matrices have perfect conditioning.** They transform space without any distortion — every direction is treated equally.

<iframe
  src="/muon/orthogonal-demo.html"
  width="100%"
  height="550"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Compare orthogonal vs non-orthogonal transformations. The orthogonal matrix rotates the grid while preserving the shape of the unit circle. The non-orthogonal matrix distorts it into an ellipse.*

---

## Part 5: Orthogonalizing Matrices — Finding the Nearest Orthogonal Matrix

### 5.1 The Orthogonal Procrustes Problem

Given a matrix $A$, what is the **nearest orthogonal matrix** to it?

This is called the **Orthogonal Procrustes Problem**:

$$Q^* = \arg\min_{Q: Q^TQ = I} \|A - Q\|_F$$

where $\|\cdot\|_F$ is the Frobenius norm.

### 5.2 Solution via SVD

**Theorem:** If $A = U\Sigma V^T$ is the SVD of $A$, then the nearest orthogonal matrix is:

$$Q^* = UV^T$$

This is obtained by replacing all singular values with 1.

**Proof:**

For any orthogonal $Q$:

$$\|A - Q\|_F^2 = \|A\|_F^2 - 2\text{tr}(A^T Q) + \|Q\|_F^2$$

Since $\|Q\|_F^2 = n$ (sum of squared entries of orthogonal matrix) and $\|A\|_F^2$ is fixed, we need to maximize:

$$\text{tr}(A^T Q) = \text{tr}(V\Sigma U^T Q)$$

Let $W = U^T Q V$. Since $U$ and $V$ are orthogonal, $W$ is orthogonal iff $Q$ is orthogonal.

$$\text{tr}(V\Sigma U^T Q) = \text{tr}(\Sigma U^T Q V) = \text{tr}(\Sigma W)$$

For diagonal $\Sigma$ with entries $\sigma_i \geq 0$:

$$\text{tr}(\Sigma W) = \sum_i \sigma_i W_{ii}$$

Since $W$ is orthogonal, $|W_{ii}| \leq 1$. This is maximized when $W_{ii} = 1$ for all $i$, which means $W = I$.

$$W = I \implies U^T Q V = I \implies Q = UV^T$$ ∎

### 5.3 Geometric Interpretation

The SVD gives us $A = U\Sigma V^T$:
- $V^T$: First rotation
- $\Sigma$: Stretching (the "bad" part that distorts)
- $U$: Second rotation

Orthogonalization removes the stretching:
- $Q^* = UV^T$: Just the rotations, no stretching

We keep the "rotational content" of the matrix and throw away the "stretching content."

### 5.4 What Orthogonalization Does to Singular Values

| Before ($A$) | After ($UV^T$) |
|--------------|----------------|
| $\sigma_1 = 5$ | $\sigma_1 = 1$ |
| $\sigma_2 = 3$ | $\sigma_2 = 1$ |
| $\sigma_3 = 0.1$ | $\sigma_3 = 1$ |

The large singular values get shrunk. The small singular values get amplified. Everything becomes equal.

<iframe
  src="/muon/orthogonalization-demo.html"
  width="100%"
  height="700"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Watch orthogonalization in action. Input a matrix (or use random), see its SVD, and observe how the singular values all become 1 in the orthogonalized version.*

---

## Part 6: Why Orthogonalize Gradients?

Now we can understand Muon's core insight.

### 6.1 The Problem: Gradient Matrices Are Ill-Conditioned

In neural networks, the gradient of the loss with respect to a weight matrix $W$ often has **high condition number**.

Why? Consider a linear layer $\mathbf{y} = W\mathbf{x}$. The gradient is:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T$$

This is an **outer product** of the upstream gradient and the input. Outer products have rank 1!

Over a batch, we sum many rank-1 matrices:

$$G = \sum_{i=1}^{B} \mathbf{g}_i \mathbf{x}_i^T$$

The result is typically **nearly low-rank**: a few directions dominate while others have tiny values.

**Empirical observation:** Gradient matrices in typical neural network training have condition numbers of 100-10,000 or more.

### 6.2 What Happens When We Use Raw Gradients

When we update with the raw gradient $G$:

$$W \leftarrow W - \eta G$$

- **Large singular values of $G$**: These directions get huge updates, potentially causing oscillation
- **Small singular values of $G$**: These directions get tiny updates, causing slow learning

But here's the key insight: **those "small" directions often contain important learning signal!**

Just because a direction has small gradient magnitude doesn't mean it's unimportant. It might be:
- A rare but critical feature
- A direction that requires many small, consistent updates
- Information that's being drowned out by noise in dominant directions

### 6.3 Orthogonalization: Equal Updates in All Directions

When we orthogonalize the gradient:

$$\tilde{G} = \text{orthogonalize}(G) = UV^T$$

Every direction gets the **same update magnitude**:

- Large singular value directions: scaled down to 1
- Small singular value directions: scaled up to 1

The directions themselves (encoded in $U$ and $V$) are preserved — we still move in the right directions. But the magnitudes are equalized.

### 6.4 The Spectral Steepest Descent Interpretation

There's a deeper theoretical justification. Consider this optimization problem:

> **Constrained Update Problem:** Find $\Delta W$ that maximizes loss reduction, subject to a bound on how much the layer's *output* changes.

Formally:

$$\Delta W^* = \arg\max_{\Delta W} \left[ -\text{tr}(G^T \Delta W) \right] \quad \text{s.t.} \quad \|\Delta W\|_2 \leq \epsilon$$

where $\|\cdot\|_2$ is the spectral norm (largest singular value).

**Theorem:** The solution is $\Delta W^* = \epsilon \cdot UV^T$ where $G = U\Sigma V^T$.

**Proof:**

By Hölder's inequality for Schatten norms:

$$|\text{tr}(G^T \Delta W)| \leq \|G\|_* \cdot \|\Delta W\|_2$$

where $\|G\|_* = \sum_i \sigma_i$ is the nuclear norm.

Equality holds when $\Delta W$ and $G$ are "aligned" — specifically, when $\Delta W = c \cdot UV^T$ for some scalar $c$.

Under the constraint $\|\Delta W\|_2 \leq \epsilon$ and noting $\|UV^T\|_2 = 1$, the maximum is achieved at $\Delta W^* = \epsilon \cdot UV^T$. ∎

**Interpretation:** Orthogonalization implements **steepest descent in the spectral norm**. Instead of asking "what direction reduces loss most per unit of weight change (Frobenius norm)," we ask "what direction reduces loss most per unit of *output perturbation* (spectral norm)."

This is a more natural metric for neural networks, where we care about how updates affect the layer's behavior, not just the raw size of weight changes.

### 6.5 Learning Rate Transfer

Another benefit: orthogonalized updates have consistent magnitude regardless of matrix size.

For a regular gradient $G \in \mathbb{R}^{m \times n}$:
$$\|G\|_F \approx \sqrt{mn} \cdot \text{(typical entry size)}$$

For an orthogonalized gradient $UV^T$:
$$\|UV^T\|_F = \sqrt{\min(m,n)}$$

This means the same learning rate works across layers of different sizes — no need for careful per-layer tuning.

---

## Part 7: The Newton-Schulz Iteration

Computing SVD is expensive: $O(mn \cdot \min(m,n))$ for an $m \times n$ matrix. For large matrices in neural networks, this is prohibitive.

The Newton-Schulz iteration provides a way to approximate orthogonalization using only matrix multiplications — which GPUs are optimized for.

### 7.1 The Matrix Sign Function

First, some background. The **matrix sign function** for a matrix $A$ with no eigenvalues on the imaginary axis is:

$$\text{sign}(A) = A(A^2)^{-1/2}$$

For a symmetric positive semi-definite matrix, this equals $UV^T$ where $A = U\Sigma V^T$.

More intuitively: if $A = U\Sigma V^T$, then $\text{sign}(A) = U \cdot \text{sign}(\Sigma) \cdot V^T$, where sign is applied element-wise to the diagonal of $\Sigma$.

Since singular values are non-negative, $\text{sign}(\sigma) = 1$ for $\sigma > 0$ and $\text{sign}(0) = 0$.

**Key insight:** The matrix sign function is exactly orthogonalization (for full-rank matrices)!

### 7.2 Newton's Method for Matrix Sign

The scalar equation $x = \text{sign}(y)$ can be written as finding the fixed point of:

$$x^2 = 1 \quad \text{(for } y \neq 0 \text{)}$$

Newton's method for solving $f(x) = x^2 - 1 = 0$:

$$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} = x_k - \frac{x_k^2 - 1}{2x_k} = \frac{1}{2}\left(x_k + \frac{1}{x_k}\right)$$

### 7.3 The Matrix Version: Newton Iteration

Lifting to matrices:

$$X_{k+1} = \frac{1}{2}(X_k + X_k^{-1})$$

**Problem:** This requires matrix inversion, which is expensive and numerically unstable.

### 7.4 Newton-Schulz: Inversion-Free Iteration

The Newton-Schulz iteration cleverly avoids inversion:

$$X_{k+1} = \frac{1}{2}X_k(3I - X_k^T X_k)$$

**Why does this work?**

For orthogonal matrices, $X^TX = I$, so:
$$X_{k+1} = \frac{1}{2}X_k(3I - I) = X_k$$

Orthogonal matrices are fixed points!

For nearly-orthogonal matrices, the iteration pushes them toward orthogonality.

### 7.5 Convergence Analysis

Let $X = U\Sigma V^T$ be the SVD. The Newton-Schulz iteration acts on singular values as:

$$\sigma \mapsto \frac{1}{2}\sigma(3 - \sigma^2) = \frac{3\sigma - \sigma^3}{2}$$

Let's call this function $\phi(\sigma)$.

**Claim:** For $\sigma \in (0, \sqrt{3})$, the iteration $\sigma_{k+1} = \phi(\sigma_k)$ converges to 1.

**Proof:**

First, note that $\phi(1) = \frac{3 - 1}{2} = 1$, so 1 is a fixed point.

$\phi'(\sigma) = \frac{3 - 3\sigma^2}{2}$, so $\phi'(1) = 0$.

This means convergence near 1 is **quadratic** (Newton's method property):

$$|\sigma_{k+1} - 1| = |\phi(\sigma_k) - \phi(1)| \approx \frac{1}{2}|\phi''(1)||\sigma_k - 1|^2$$

For stability, we need $|\phi(\sigma)| < \sqrt{3}$ when $|\sigma| < \sqrt{3}$. One can verify this holds. ∎

<iframe
  src="/muon/newton-schulz-convergence.html"
  width="100%"
  height="650"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Visualize Newton-Schulz convergence. See how different starting singular values converge to 1 over iterations. Notice the quadratic convergence rate near 1.*

### 7.6 The Quintic Variant: Faster Convergence

Muon uses a **quintic polynomial** instead of cubic:

$$\phi(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$$

with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$.

**Why quintic?**

- More free parameters allow optimizing for faster convergence
- The coefficients were tuned to maximize convergence speed over the expected range of singular values
- Still has the fixed point $\phi(1) = 1$ and $\phi'(1) = 0$ (quadratic convergence)

The matrix iteration becomes:

$$X_{k+1} = aX_k + X_k(bX_k^TX_k + c(X_k^TX_k)^2)$$

### 7.7 Numerical Stability and bfloat16

A crucial advantage of Newton-Schulz over direct SVD: **it's stable in low precision**.

The coupled Newton iteration (matrix inversion method) requires at least float32 to avoid numerical issues. But Newton-Schulz can run stably in **bfloat16**, which is 2× faster on modern GPUs.

The trick is to normalize the input first:
$$X_0 = G / \|G\|_F$$

This ensures all singular values start in a reasonable range.

### 7.8 The Complete Algorithm

```python
def newton_schulz(G, steps=5, eps=1e-7):
    """
    Approximately orthogonalize matrix G using Newton-Schulz iteration.

    Args:
        G: Input matrix of shape (m, n)
        steps: Number of iterations (5 is usually sufficient)
        eps: Small constant for numerical stability

    Returns:
        Approximately orthogonal matrix of same shape as G
    """
    # Optimized quintic coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Convert to bfloat16 for speed
    X = G.bfloat16()

    # Normalize for numerical stability
    X = X / (X.norm() + eps)

    # Handle non-square matrices: work with the smaller dimension
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T  # Now X is (n, m) with n <= m

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T           # (n, n) matrix
        B = b * A + c * A @ A  # Quintic terms
        X = a * X + B @ X      # Update

    # Restore original shape
    if transpose:
        X = X.T

    return X
```

### 7.9 Computational Cost

For an $m \times n$ matrix with $n \leq m$:

**Per iteration:**
- $X @ X^T$: $O(n^2 m)$ FLOPs
- $A @ A$: $O(n^3)$ FLOPs
- $B @ X$: $O(n^2 m)$ FLOPs

Total per iteration: $O(n^2 m + n^3) \approx O(n^2 m)$ for $m > n$.

**For 5 iterations:** $O(5 n^2 m) = O(n^2 m)$

**Comparison to training FLOPs:**
- Forward + backward for a linear layer: $O(nmB)$ where $B$ is batch size
- Newton-Schulz overhead: $O(n^2 m)$
- Ratio: $n/B$

For typical values ($n = 4096$, $B = 4096$): overhead is about **0.5%** of training compute.

---

## Part 8: The Complete Muon Algorithm

### 8.1 Putting It All Together

Muon combines three ideas:

1. **Momentum**: Accumulate gradient information over time
2. **Nesterov acceleration**: Look-ahead for faster convergence
3. **Orthogonalization**: Equalize update magnitudes across directions

### 8.2 The Algorithm

For each 2D parameter matrix $W$:

**Initialize:** $m_0 = 0$

**At each step $t$:**

1. **Compute gradient:**
   $$g_t = \nabla_W \mathcal{L}(\theta_t)$$

2. **Update momentum (Nesterov style):**
   $$m_t = \beta \cdot m_{t-1} + g_t$$

   The Nesterov variant actually computes gradient at a "look-ahead" position:
   $$g_t = \nabla_W \mathcal{L}(\theta_t + \beta \cdot m_{t-1})$$

3. **Orthogonalize momentum:**
   $$\tilde{m}_t = \text{NewtonSchulz}(m_t, \text{steps}=5)$$

4. **Apply update:**
   $$W_{t+1} = W_t - \eta \cdot \tilde{m}_t$$

### 8.3 Why Orthogonalize Momentum Instead of Gradient?

We orthogonalize the momentum $m_t$ rather than the raw gradient $g_t$ because:

1. **Momentum is smoother**: Averaging over time reduces noise
2. **Better direction estimate**: The accumulated momentum better reflects the true optimization direction
3. **Computational efficiency**: We orthogonalize once per step, not once per gradient

### 8.4 Handling Different Parameter Types

Orthogonalization only makes sense for 2D matrices. Muon uses different strategies for different parameters:

| Parameter Type | Strategy |
|---------------|----------|
| Linear layer weights (2D) | Muon (orthogonalize) |
| Conv filters (4D) | Reshape to 2D, Muon, reshape back |
| Biases (1D) | AdamW |
| LayerNorm parameters (1D) | AdamW |
| Embeddings | AdamW |
| Output layer | AdamW |

**Why AdamW for embeddings and output layer?**

These layers have different optimization dynamics:
- Embeddings are lookup tables, not transformations
- The output layer connects to the loss directly and benefits from Adam's adaptivity

### 8.5 Full Implementation

```python
import torch
from torch.optim import Optimizer

def newton_schulz(G, steps=5, eps=1e-7):
    """Approximately orthogonalize matrix G."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X.type_as(G)


class Muon(Optimizer):
    """
    Muon optimizer: Momentum with Newton-Schulz orthogonalization.

    Args:
        params: Parameters to optimize (should be 2D matrices)
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    """

    def __init__(self, params, lr=0.02, momentum=0.95,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum,
                       nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get or initialize momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']

                # Update momentum
                buf.mul_(momentum).add_(grad)

                # Nesterov: use gradient at look-ahead position
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Orthogonalize if 2D
                if p.ndim == 2:
                    update = newton_schulz(update, steps=ns_steps)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss
```

### 8.6 Practical Usage

```python
# Separate parameters for Muon vs AdamW
muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    if param.ndim == 2 and 'embed' not in name and 'head' not in name:
        muon_params.append(param)
    else:
        adamw_params.append(param)

# Create optimizers
muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
adamw_opt = torch.optim.AdamW(adamw_params, lr=3e-4)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()

    muon_opt.step()
    adamw_opt.step()

    muon_opt.zero_grad()
    adamw_opt.zero_grad()
```

---

## Part 9: Why Muon Works — The Complete Picture

### 9.1 Memory Efficiency

| Optimizer | Buffers per Parameter |
|-----------|----------------------|
| SGD | 0 |
| SGD + Momentum | 1 (momentum) |
| Adam/AdamW | 2 (first moment + second moment) |
| **Muon** | **1 (momentum)** |

Muon uses **50% less optimizer memory** than AdamW.

For a 7B parameter model:
- AdamW: 7B × 2 × 4 bytes = 56 GB optimizer states
- Muon: 7B × 1 × 4 bytes = 28 GB optimizer states

This is a significant practical advantage for training large models.

### 9.2 The Optimization Landscape View

<iframe
  src="/muon/muon-vs-adam.html"
  width="100%"
  height="750"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Race Muon against AdamW on an ill-conditioned landscape. Watch how Muon's orthogonalized updates navigate directly toward the minimum while Adam oscillates.*

### 9.3 Why Orthogonalization Beats Adaptivity

Adam's approach: Adapt the learning rate for each parameter based on its gradient history.

$$\theta \leftarrow \theta - \eta \frac{m}{\sqrt{v} + \epsilon}$$

This helps, but it's treating the symptom (varying gradient magnitudes) rather than the cause (ill-conditioned updates).

Muon's approach: Fix the conditioning directly.

$$\theta \leftarrow \theta - \eta \cdot \text{orthogonalize}(m)$$

By equalizing all directions, we eliminate the need for adaptive scaling.

### 9.4 Empirical Results

| Benchmark | AdamW | Muon | Speedup |
|-----------|-------|------|---------|
| CIFAR-10 speedrun | 3.3 A100-sec | 2.6 A100-sec | 1.27× |
| NanoGPT | 1.0× | 0.74× | 1.35× |
| 1.5B param LLM | 13.3 H100-hours | 10 H100-hours | 1.33× |

Muon consistently achieves **25-35% faster training** while using less memory.

---

## Part 10: Recent Developments and Variants

### 10.1 Scaling to Large Models

Initial experiments showed Muon working well up to 1.5B parameters. Recent work (2025) has demonstrated:

- **Muon scales to LLM training**: Validated on models up to 7B parameters
- **Consistent speedups**: 20-35% faster convergence maintained at scale
- **Compatible with distributed training**: Works with FSDP, tensor parallelism

### 10.2 NorMuon: Reducing NS Steps

The Newton-Schulz iteration is the main computational overhead. **NorMuon** introduces:

- **Spectral preconditioning**: Better initialization reduces required steps
- **5 → 4 steps**: 20% reduction in orthogonalization cost
- **2.8× speedup** in orthogonalization overhead

### 10.3 Dion: Communication-Efficient Distributed Muon

For distributed training, **Dion** proposes:

- **Low-rank orthogonalization**: Approximate orthogonalization with rank-$k$ SVD
- **Amortized power iteration**: Cheaper than full Newton-Schulz
- **Reduced communication**: Better suited for multi-node training

### 10.4 What's Next?

Open research directions:

1. **Adaptive NS steps**: Use fewer steps when momentum is already nearly orthogonal
2. **Hybrid approaches**: Combine Muon's orthogonalization with Adam's adaptivity
3. **Layer-specific tuning**: Different settings for attention vs MLP layers
4. **Convergence theory**: Rigorous analysis of Muon's convergence properties

---

## Summary

### The Key Ideas

1. **Matrices transform space** — SVD reveals they rotate, stretch, then rotate again

2. **Condition number** — Ratio of max to min stretching; high values make optimization hard

3. **Orthogonal matrices** — All singular values = 1; perfect conditioning; pure rotation

4. **Orthogonalization** — Replace singular values with 1; find nearest orthogonal matrix

5. **Newton-Schulz** — Fast iterative orthogonalization using only matrix multiplications

6. **Muon** — Orthogonalize momentum before applying updates; equal treatment of all directions

### The Algorithm at a Glance

```
For each 2D weight matrix W:
    1. Compute gradient g
    2. Update momentum: m ← β·m + g
    3. Orthogonalize: m̃ ← NewtonSchulz(m)
    4. Update weights: W ← W - η·m̃
```

### When to Use Muon

| Use Muon | Use AdamW |
|----------|-----------|
| Linear layers | Embeddings |
| Attention projections | Output layer |
| MLP weights | Biases |
| Conv layers (reshaped) | LayerNorm params |

### The Evolution of Optimizers

```
SGD        →  Momentum   →  Adam       →  AdamW      →  Muon
(basic)       (velocity)    (adaptive)    (proper WD)   (orthogonal)

Problem:      Solution:     Problem:      Solution:     Solution:
oscillation   smooth        ad-hoc        decouple      fix conditioning
              gradient      adaptivity    regularizer   at source
```

---

## References

- [Keller Jordan's original Muon blog post](https://kellerjordan.github.io/posts/muon/)
- [Jeremy Bernstein's theoretical derivation](https://jeremybernste.in/writing/deriving-muon)
- [Muon is Scalable for LLM Training (arXiv:2502.16982)](https://arxiv.org/abs/2502.16982)
- [NorMuon: Making Muon more efficient (arXiv:2510.05491)](https://arxiv.org/abs/2510.05491)
- [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
- [NVIDIA Emerging Optimizers documentation](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html)

*Found this useful? Check out my other deep dives on [optimizers](/posts/optimizers-deep-dive) and [entropy](/posts/entropy-explorer).*
