---
title: "A Visual Journey Through Deep Learning Optimizers"
date: 2025-01-21
draft: true
math: true
summary: "From vanilla gradient descent to AdamW - understanding optimizers through interactive visualizations and rigorous math."
tags: ["deep-learning", "optimization", "math", "interactive"]
cover:
    image: "optimizers/cover.png"
    alt: "Optimizer trajectories on a loss landscape"
    caption: "Different optimizers taking different paths to the minimum"
showToc: true
tocOpen: true
---

Training neural networks is fundamentally an optimization problem. We have a loss function $\mathcal{L}(\theta)$ that measures how poorly our model performs, and we want to find parameters $\theta^*$ that minimize it:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

But here's the catch: neural networks can have millions or billions of parameters. We can't just solve this analytically. Instead, we rely on **iterative optimization** — repeatedly nudging our parameters in directions that reduce the loss.

This post takes you on a journey through the evolution of optimization algorithms, from the simplest gradient descent to the sophisticated AdamW optimizer used to train modern LLMs. Along the way, we'll see *why* each improvement was necessary through interactive visualizations.

---

## 1. Vanilla Gradient Descent

The simplest approach: follow the gradient downhill.

### The Algorithm

The gradient $\nabla_\theta \mathcal{L}$ points in the direction of steepest *increase* of the loss. To minimize, we go the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta$ is the **learning rate** — how big a step we take.

### The Problem

The same learning rate is applied to all parameters. Consider a loss landscape that's:
- **Steep** in one direction (small changes → big loss changes)
- **Flat** in another direction (small changes → tiny loss changes)

If you set $\eta$ based on the steep direction, you'll crawl in the flat direction.
If you set $\eta$ based on the flat direction, you'll overshoot and oscillate in the steep direction.

<iframe
  src="/optimizers/vanilla-gd.html"
  width="100%"
  height="850"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Try adjusting the learning rate above. Notice how it's impossible to find a single learning rate that works well in both directions of this elongated loss landscape.*

### Mathematical Intuition

For a quadratic loss $\mathcal{L}(\theta) = \frac{1}{2}\theta^T H \theta$ where $H$ is the Hessian, the optimal learning rate is related to the eigenvalues of $H$. If eigenvalues vary widely (high **condition number**), no single learning rate is optimal.

---

## 2. Momentum

**Intuition:** A ball rolling downhill accumulates velocity.

### The Problem with Vanilla GD

In ravines (long, narrow valleys), vanilla GD oscillates back and forth across the valley while making slow progress along it:

```
Without momentum:     With momentum:
    ↓                     ↓
    ↑                     ↓
    ↓                     ↓
    ↑                     ↓
  (oscillates)          (rolls through)
```

### The Solution

Instead of using the gradient directly, we maintain a **velocity** that accumulates gradients over time:

$$m_t = \beta \cdot m_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot m_t$$

where:
- $m_t$ is the momentum (velocity)
- $\beta \approx 0.9$ is the momentum coefficient
- The momentum *remembers* past gradients

### Why It Works

1. **Oscillating gradients cancel out:** In the steep direction, gradients alternate signs, so momentum averages them toward zero
2. **Consistent gradients accumulate:** In the flat direction, gradients point the same way, so momentum builds up speed

<iframe
  src="/optimizers/momentum.html"
  width="100%"
  height="900"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Compare vanilla GD and momentum on the same landscape. Watch how momentum smooths out oscillations and accelerates in consistent directions.*

### The Math: Exponential Moving Average

Unrolling the recurrence:

$$m_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \beta^3 g_{t-3} + \cdots$$

This is an **exponentially weighted sum** of past gradients. Recent gradients have more influence, but history matters.

---

## 3. RMSprop

**Intuition:** Adapt the learning rate for each parameter based on its gradient history.

### The Idea

Different parameters have different gradient magnitudes:
- Parameters with **large gradients** → they're in a steep region → take smaller steps
- Parameters with **small gradients** → they're in a flat region → take larger steps

RMSprop tracks the *magnitude* of recent gradients using an exponential moving average of squared gradients:

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t$$

where:
- $v_t$ tracks the average squared gradient (per-parameter)
- $\sqrt{v_t}$ estimates the RMS (root mean square) of recent gradients
- $\epsilon \approx 10^{-8}$ prevents division by zero

### Per-Parameter Adaptive Learning Rate

The effective learning rate for each parameter is:

$$\eta_{\text{effective}} = \frac{\eta}{\sqrt{v_t} + \epsilon}$$

- Large historical gradients → large $v_t$ → small effective learning rate
- Small historical gradients → small $v_t$ → large effective learning rate

**Automatic per-parameter learning rate adaptation!**

<iframe
  src="/optimizers/rmsprop.html"
  width="100%"
  height="1250"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*See how RMSprop automatically adjusts step sizes. Parameters in steep directions get scaled down, while parameters in flat directions get scaled up.*

---

## 4. Adam = Momentum + RMSprop

**The key insight:** Why choose between momentum and adaptive learning rates when you can have both?

### The Algorithm

Adam maintains *two* moving averages:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{(first moment / momentum)}$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad \text{(second moment / RMSprop)}$$

The update combines both:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

- **$m_t$** provides momentum (direction smoothing)
- **$v_t$** provides adaptive scaling (step size)

### Typical Values

- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (RMSprop decay — slower, more stable)
- $\epsilon = 10^{-8}$
- $\eta = 0.001$ (common default)

---

## 5. Bias Correction

### The Problem

Both $m$ and $v$ are initialized to zero. In the first few steps:

$$m_1 = \beta_1 \cdot 0 + (1 - \beta_1) \cdot g_1 = (1-\beta_1) \cdot g_1 = 0.1 \cdot g_1$$

We're averaging with zeros! The estimate is **biased toward zero**.

### The Fix

Divide by a correction factor that accounts for the missing history:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**At $t=1$:**
$$\hat{m}_1 = \frac{0.1 \cdot g_1}{1 - 0.9^1} = \frac{0.1 \cdot g_1}{0.1} = g_1 \quad \checkmark$$

**As $t \to \infty$:**
$$1 - \beta^t \to 1 \implies \hat{m}_t \to m_t$$

The correction vanishes as we accumulate enough history.

<iframe
  src="/optimizers/bias-correction.html"
  width="100%"
  height="1220"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Visualize how bias correction compensates for zero-initialization. Watch the correction factor decrease over time as more history accumulates.*

---

## 6. Full Adam Algorithm

Putting it all together:

$$t \leftarrow t + 1$$
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

<iframe
  src="/optimizers/adam.html"
  width="100%"
  height="1200"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Watch Adam navigate the loss landscape. You can see all the components working together: momentum smoothing the path, adaptive scaling adjusting step sizes, and bias correction ensuring good behavior early on.*

### The Grand Comparison

Now let's see how all these optimizers compare on the same landscape:

<iframe
  src="/optimizers/optimizer-comparison.html"
  width="100%"
  height="1250"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Race all four optimizers on the same elongated valley. Watch how SGD struggles, Momentum overshoots, RMSprop adapts, and Full Adam combines the best of both worlds to converge fastest.*

---

## 7. Weight Decay vs L2 Regularization

Now we need to discuss regularization — and a subtle but important distinction.

### L2 Regularization

Add a penalty term to the loss:

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|^2$$

The gradient becomes:

$$\nabla \mathcal{L}_{\text{reg}} = \nabla \mathcal{L} + \lambda \theta$$

The regularization term pushes weights toward zero.

### Weight Decay

Shrink weights directly during the update:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla \mathcal{L} - \eta \cdot \lambda \cdot \theta_t$$

Or equivalently:

$$\theta_{t+1} = (1 - \eta \lambda) \cdot \theta_t - \eta \cdot \nabla \mathcal{L}$$

### For Vanilla SGD: They're Equivalent!

With SGD update $\theta_{t+1} = \theta_t - \eta \cdot g_t$:

**L2:** $g_t = \nabla \mathcal{L} + \lambda \theta_t$
$$\theta_{t+1} = \theta_t - \eta(\nabla \mathcal{L} + \lambda \theta_t) = \theta_t - \eta \nabla \mathcal{L} - \eta \lambda \theta_t$$

**Weight decay:**
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L} - \eta \lambda \theta_t$$

**Identical!** This is why people often use the terms interchangeably.

---

## 8. Why Adam + L2 Regularization is Broken

Here's where things get interesting. With Adam, L2 regularization and weight decay are **not equivalent**.

### The Problem

With L2, the gradient becomes $g_t = \nabla \mathcal{L} + \lambda \theta_t$.

Adam's adaptive scaling applies to this *combined* gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda \theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda \theta_t)^2$$

The weight decay term $\lambda \theta_t$ gets scaled by $\frac{1}{\sqrt{v_t}}$.

### Why This is Wrong

Consider a weight with:
- **Small gradient history** (small $v_t$) → large effective learning rate
- The weight decay also gets amplified!

Or:
- **Large gradient history** (large $v_t$) → small effective learning rate
- The weight decay gets suppressed!

**We don't want weight decay to depend on gradient history.** Large weights should decay consistently, period.

<iframe
  src="/optimizers/adam-l2-problem.html"
  width="100%"
  height="850"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*This visualization shows how the same weight decay coefficient produces very different actual decay rates depending on the gradient history of each parameter.*

---

## 9. AdamW: The Fix

**Solution:** Apply weight decay *outside* of Adam's adaptive mechanism.

### The Algorithm

```python
# Standard Adam update (without L2 in gradient)
m = β1 * m + (1 - β1) * grad
v = β2 * v + (1 - β2) * grad²
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
adam_update = m_hat / (sqrt(v_hat) + eps)

# Apply both updates separately
param = param - lr * adam_update        # Adam step
param = param - lr * λ * param          # Weight decay (decoupled!)
```

Or combined:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

### The Key Difference

| Approach | Weight Decay Effect |
|----------|-------------------|
| Adam + L2 | $\theta \leftarrow \theta - \eta \cdot \frac{\lambda\theta}{\sqrt{v} + \epsilon}$ (scaled by gradient history) |
| AdamW | $\theta \leftarrow \theta - \eta \lambda \theta$ (consistent regardless of gradients) |

<iframe
  src="/optimizers/adamw.html"
  width="100%"
  height="900"
  style="border:0; border-radius: 8px;"
  loading="lazy">
</iframe>

*Compare Adam+L2 vs AdamW. Notice how AdamW provides more consistent regularization across parameters with different gradient histories.*

---

## Summary

| Method | What It Does | Key Innovation |
|--------|-------------|----------------|
| **SGD** | Basic gradient descent | Simplest baseline |
| **Momentum** | Accumulates velocity | Smooths gradients, accelerates in consistent directions |
| **RMSprop** | Adapts per-parameter learning rate | Scales updates by gradient history |
| **Adam** | Momentum + RMSprop | Best of both worlds |
| **AdamW** | Adam + proper weight decay | Decouples regularization from adaptation |

### The Evolution

```
SGD: Too slow, oscillates
 ↓
Momentum: Smoother, faster
 ↓
RMSprop: Adaptive step sizes
 ↓
Adam: Momentum + Adaptation
 ↓
AdamW: Proper regularization
```

### When to Use What

- **SGD + Momentum:** Often generalizes better, but requires more tuning. Used in some vision tasks.
- **Adam/AdamW:** Faster convergence, less sensitive to hyperparameters. Default for most deep learning, especially NLP/LLMs.
- **AdamW over Adam:** Always prefer AdamW when using weight decay (which you almost always should).

The journey from vanilla gradient descent to AdamW represents decades of research, each step addressing a specific shortcoming of the previous approach. Understanding these algorithms deeply helps you debug training issues, choose appropriate hyperparameters, and know when to deviate from defaults.

---

*Found this useful? Check out my other deep dives on [KL Divergence](/posts/entropy-explorer) and [LLM-as-Judge](/posts/llm-judge).*
