---
title: "The Curious Case of KL Divergence"
date: 2025-09-01
draft: true
math: true
summary: "In depth understanding of KLD starting from Entropy to RL."
tags: ["math", "information-theory"]
cover:
    image: "entropy-explorer.gif"
    alt: "Shannon Entropy Visualization - showing entropy changes as probability distribution shifts"
---

Over years, KL divergence (KLD) is something that can be found in different areas of the Machine learning world. Be it Knowledge Distillation, or Semi-Supervised learning and now even to train LLMs with Reinforcement Learning. In all of these, the goal is always to bring two distributions closer. But there is a subtle difference on how KLD is defined and used. We have Forward KL and Backward KL. In supervised learning setup, forward KL is used and in RL setup, reverse KL is used. In this article, we'll see what Forward and Reverse KL are and what their properties are. We'll investigate in which scenarios it makes sense to use Reverse and Forward KL and also study the reasons behind them. Importantly, we'll explore why forward KL is not suitable for RL setups (where online policy sampling is required) and why reverse KL is not suitable for knowledge distillation (due to mode-seeking behavior and blind spots that lose teacher knowledge).

Don't worry if this sounds abstract—we'll start from the fundamentals. We'll build up from [information theory](#information-in-the-shannon-sense) and [entropy](#entropy), derive the [mathematics of KL divergence](#2-kl-divergence-definition-and-derivation), explore [forward vs reverse KL behaviors](#3-forward-vs-reverse-kl-behaviors-mode-covering-vs-mode-seeking), and by the end you'll have a complete understanding of when and why to use each variant.

## Information (in the Shannon Sense)

Let’s get a working intuition for *information* in the context of information theory. While the term is familiar in everyday language, Claude Shannon gave it a precise mathematical meaning that captures a simple but powerful insight:

> **Information measures surprise** — how much uncertainty is resolved when an event occurs.

If an event is very likely, learning that it happened doesn’t teach you much. But when something *unlikely* occurs, you gain insight. That’s what Shannon formalized with the definition:

$$
I(x) = -\log_2 p(x)
$$

Here, \\( p(x) \\) is the probability of an event \\( x \\), and \\( I(x) \\) is the amount of information (in **bits**) gained by observing \\( x \\). The base-2 logarithm reflects the fact that we measure information in binary, in terms of how many yes/no decisions (bits) are needed to identify the outcome.

---

## Why use \\( \log p \\)? An Axiomatic Approach

This formula isn’t arbitrary, but it emerges naturally from a few reasonable assumptions about how we expect information to behave. Suppose we define a function \\( I(p) \\) to represent the information content of an event with probability \\( p \\). The requirements are:

### Axiom A1 — Rarity Implies More Information

The rarer an event, the more information it should convey. So \\( I(p) \\) should **decrease** as \\( p \\) increases.

An event that is guaranteed (\\( p = 1 \\)) conveys **no** information:

$$
I(1) = 0
$$


### Axiom A2 — Additivity for Independent Events

If two *independent* events occur, say one with probability \\( p \\) and the other with \\( q \\), then the information gained from both should be the **sum** of the individual informations:

$$
I(p \cdot q) = I(p) + I(q)
$$

This is essential if we want to talk about how information accumulates across independent events (e.g. flipping a coin twice = 2 bits).


### Axiom A3 — Continuity

The function \\( I(p) \\) should behave smoothly. Small changes in probability shouldn’t cause abrupt jumps in information. In other words, \\( I(p) \\) should be **continuous**.

---

### 🔑 The Only Solution: \\( I(p) = -\log p \\)

From Axiom A2, we get a functional equation:

$$
I(pq) = I(p) + I(q)
$$

The only real-valued functions on \\( (0, 1] \\) that satisfy this and are continuous and monotonic are of the form:

$$
I(p) = -k \log_b p
$$

The negative sign ensures that less probable events have **more** information. The constant \\( k \\) and base \\( b \\) simply determine the **unit** of measurement.

If we choose base 2 (binary), and set \\( k = 1 \\), we get the familiar form:

$$
I(p) = -\log_2 p
$$

This measures information in **bits** — the amount of binary decisions needed to resolve uncertainty.

## Bits, Information and Probability

As we saw in the previous section on Information, it's definition has a few intuitive properties:

* If an event is **certain** (\\( p = 1 \\)), then:

  $$
  I(x) = -\log_2(1) = 0 \text{ bits}
  $$

  You gain nothing — it was expected.

* If an event is **unlikely** (e.g., \\( p = 0.001 \\)), then:

  $$
  I(x) = -\log_2(0.001) \approx 9.97 \text{ bits}
  $$

  Its occurrence is surprising and highly informative.


## A Guiding Example: Guessing a Number

Suppose I think of a number between 1 and 8, each equally likely. Then each outcome has \\( p = 1/8 \\), so the information from guessing correctly is:

$$
I(x) = -\log_2(1/8) = 3 \text{ bits}
$$

That makes sense: 3 yes/no questions are sufficient to identify the correct number (since \\( 2^3 = 8 \\)).

Now, let’s say I think of a number between 1 and 1024, and you **guess** it correctly on the first try. That outcome had probability \\( 1/1024 \\), so its information content is:

$$
I(x) = -\log_2(1/1024) = 10 \text{ bits}
$$

The surprise is greater — and the informational value reflects it.
<iframe
  src= "/entropy-explorer/information.html"
  width="100%"
  height="1000"
  style="border:0"
  loading="lazy">
</iframe>

## Why Not Just Use Probability?

It's tempting to ask: if probability already tells us how rare an event is, why invent a new quantity?

Because **probability only measures uncertainty *before* an event happens**. Once the event occurs, we want a measure of how much *uncertainty was resolved*. We also want this measure to:

* **Add up** across independent events,
* Reflect how many binary decisions are needed, and
* Behave smoothly with respect to changes in probability.

Probability can't do this. But information can.

## Bottom Line

* **Monotonicity**, **additivity**, and **continuity** lead us directly to \\( I(x) = -\log p(x) \\).
* The base determines the **unit** — base 2 gives us **bits**.
* Information isn't about "amount of data"; it's about **how unexpected** an event was, and how much it told us.

Once you see this, information theory becomes less about abstract formulas, and more about understanding surprise, learning, and the structure of uncertainty itself.

---

## Entropy

Entropy, the root of all solutions :)
Information in previous section we focused on specific events. Entropy is the **expected** information over all the events.

So, we try to find the expected value of a distribution.
For a discrete *distribution* \\(P\\) over outcomes \\(x\\):

$$
H(P)=\-\sum_x P(x)\log P(x).
$$

Suppose we define a random variable \\(X\\) representing the outcome of a fair coin toss. Naturally, we want to know: on average, how many [bits](#bits-information-and-probability) are needed to encode the outcomes produced by this distribution?

This is exactly what entropy measures — the expected amount of information per event drawn from a distribution.

Using Shannon's entropy formula:
$$
H(X)\=\-\sum_x p(x)\log_2 p(x)\=\\mathbb{E}_{X\sim p}\big[-\log_2 p(X)\big].
$$

### Understanding the Notation: $\mathbb{E}_{X\sim p}$

Let's break down this notation piece by piece, since it's central to everything that follows:

**What does $\mathbb{E}_{X\sim p}\big[-\log_2 p(X)\big]$ mean?**

* $\mathbb{E}[\cdot]$ = **expectation** = weighted average
* $X \sim p$ = "**X follows distribution p**" = we sample outcomes according to the probabilities specified by p
* $-\log_2 p(X)$ = the [information content](#information-in-the-shannon-sense) when outcome X occurs
* **Together**: "average information when we sample outcomes according to distribution p"

**Clarifying X vs P:**

* **$P$ (or $p$)** = the **probability distribution** (the blueprint/rules that assign probabilities)
  * Example: For a coin, $p(\text{Heads}) = 0.5$ and $p(\text{Tails}) = 0.5$
* **$X$** = the **random variable** (the actual outcome we observe)
  * Example: $X$ could be Heads or Tails
* **$X \sim P$** = "$X$ is drawn from $P$" means: when we observe $X$, each outcome appears with probability given by $P$

**How the expectation becomes a sum:**

When we write $\mathbb{E}_{X\sim p}\big[-\log_2 p(X)\big]$, we're computing:
* For each possible outcome $x$ that $X$ could be
* Multiply its information $-\log_2 p(x)$ by the probability $p(x)$ of observing it
* Sum across all outcomes: $\sum_x p(x) \cdot [-\log_2 p(x)]$

**Why "sampling from P" matters:**

Notice that in entropy $H(P)$, we use $P$ for **both**:
1. **Sampling**: which outcomes we expect to see (via $X \sim p$)
2. **Evaluation**: which probabilities we use to compute information (via $p(X)$)

This dual role of $P$ is crucial. Later, we'll see what happens when we sample from one distribution but evaluate probabilities from a *different* distribution—that's where things get interesting.

---

**Example: Fair Coin**

For a fair coin, both outcomes — heads (H) and tails (T) — have equal probability:

$$
H(X) = -[0.5 \cdot \log_2(0.5) + 0.5 \cdot \log_2(0.5)] = 0.5 \cdot 1 + 0.5 \cdot 1 = 1 \text{ bit}
$$

**Breaking down the expectation calculation:**

Using $\mathbb{E}_{X\sim p}\big[-\log_2 p(X)\big]$:
* When $X = \text{Heads}$ (happens with probability 0.5): we get $-\log_2(0.5) = 1$ bit
* When $X = \text{Tails}$ (happens with probability 0.5): we get $-\log_2(0.5) = 1$ bit
* **Average**: $0.5 \times 1 + 0.5 \times 1 = 1$ bit

So the entropy of a fair coin toss is 1 bit — which aligns with our intuition: a single binary question (e.g. "Is it heads?") is enough to fully describe the outcome.

But if the coin was biased, 

Let’s now consider an unfair coin, where the probability of heads is much higher than tails:

* \\(P(H)\\) = 0.9
* \\(P(T)\\) = 0.1

We compute:

$$
H(X) = -[0.9 \cdot \log_2(0.9) + 0.1 \cdot \log_2(0.1)]
$$

$$
\approx -[0.9 \cdot (-0.152) + 0.1 \cdot (-3.322)]
$$

$$
\approx 0.1368 + 0.3322 = 0.469 \text{ bits}
$$


So, the entropy of this biased coin is approximately 0.469 bits — noticeably less than 1 bit.

This makes intuitive sense: if the coin lands heads 90% of the time, the outcome is more predictable. There's less uncertainty, and thus less information gained from each toss.

**Think of it this way**: entropy measures the average "surprise" per outcome. With a fair coin, every flip is equally surprising (1 bit of information). With a biased coin, most flips give you little new information ("heads again, as expected"), so the average surprise is much lower (0.469 bits). The more predictable the distribution, the lower the entropy.

Here is another example showing how as we become more confident of next word being `dog`, the entropy drops.

<iframe
  src= "/entropy-explorer/entropy_explorer.html"
  width="100%"
  height="520"
  style="border:0"
  loading="lazy">
</iframe>

## Cross-entropy

### The Intuition: When Your Model Gets It Wrong

Imagine you're training a language model to predict the next word. After analyzing thousands of movie reviews, you notice that when people write "The movie was", they complete it with:

* **"amazing"**: 50% of the time
* **"terrible"**: 40% of the time
* **"okay"**: 10% of the time

This is the **true distribution** $P$ — how people actually complete this phrase in real data.

**First, let's think about entropy** $H(P)$:

Even if you had a *perfect* model that knew these exact probabilities, there's still inherent uncertainty. You can't predict exactly which word comes next; you can only know that it's "amazing" half the time, "terrible" 40% of the time, etc.

The entropy $H(P) = \mathbb{E}_{X\sim P}[-\log P(X)]$ measures this baseline unpredictability. It's the average surprise you'd experience if you had the correct probabilities.

For our example: $H(P) \approx 1.36$ bits (you can verify this using the formula we learned).

**Now, cross-entropy: What if your model is wrong?**

You train a language model, but it learns incorrectly. Your model $Q$ thinks people are overly optimistic, predicting:

* **"amazing"**: 90%
* **"terrible"**: 5%
* **"okay"**: 5%

Now when you encounter real reviews (drawn from the true distribution $P$):

* 50% of the time, you see "amazing" — your model expected this 90% of the time, so you're *less* surprised than reality warrants ($-\log(0.9) \approx 0.15$ bits vs the true $-\log(0.5) \approx 1$ bit)
* **40% of the time, you see "terrible"** — but your model only expected this 5% of the time! You're *way more* surprised ($-\log(0.05) \approx 4.32$ bits vs the true $-\log(0.4) \approx 1.32$ bits)
* 10% of the time, you see "okay" — again, model expected only 5%, so you're more surprised than you should be

The **cross-entropy** $H(P,Q)$ measures your average surprise when:
* Reality follows the true distribution $P$ (which words actually appear)
* But you're using your model's wrong probabilities $Q$ to measure surprise

For this example: $H(P,Q) \approx 2.18$ bits — much higher than the entropy of 1.36 bits!

The excess surprise (2.18 - 1.36 = 0.82 bits) is the cost of having the wrong model. **This is exactly what "cross-entropy loss" measures when training LLMs** — it penalizes the model for assigning low probabilities to words that actually appear in the training data. By minimizing this cross-entropy, we force the model's predictions $Q$ to better match the true distribution $P$ of the data.

### See It In Action

Try the interactive visualization below. Watch color-coded particles representing each word floating in a force field:
- **Left panel (Reality - Distribution P)**: Shows the baseline with calm, steady particle motion — this is what perfect predictions look like
- **Right panel (Your Model - Distribution Q)**: When your predictions are wrong, particles become CHAOTIC! They explode with speed, glow intensely, and create extreme turbulence proportional to the surprise level

Notice the green "P:" values next to each slider showing the true distribution. Adjust the sliders to change your model's predictions. When Q predicts "terrible" at only 5% but it appears 40% of the time, watch the red particles in the right panel go absolutely WILD! Can you calm the chaos by making Q match P?

<iframe
  src="/entropy-explorer/cross_entropy.html"
  width="100%"
  height="550"
  style="border:4; margin: 50px 0; display: block;"
  loading="lazy">
</iframe>

---

### The Mathematics

**The Formula:**

$$
H(P, Q) = \mathbb{E}_{X\sim P}[-\log Q(X)] = -\sum_x P(x)\,\log Q(x).
$$

**The key difference from [entropy](#entropy):** In entropy $H(P)$, we used $P$ for both sampling outcomes and measuring surprise. Cross-entropy breaks this:
* **$X \sim P$**: Sample outcomes from the true distribution $P$ (what actually happens)
* **$-\log Q(X)$**: Measure surprise using model $Q$'s probabilities (what we predicted)

This becomes the sum: $\sum_x P(x) \cdot [-\log Q(x)]$ — weight each outcome's Q-based surprise by how often it actually occurs under $P$.

**Key Properties:**

* $H(P,Q) = H(P)$ when $Q = P$ (perfect model)
* $H(P,Q) > H(P)$ when $Q \neq P$ (the worse the approximation, the higher the cross-entropy)
* Minimizing cross-entropy = making $Q$ closer to $P$

> **Support matching:** $Q(x)$ must be $> 0$ wherever $P(x) > 0$. If your model says an outcome is impossible ($Q(x) = 0$) but it actually happens ($P(x) > 0$), cross-entropy becomes infinite.

---


## 2) KL Divergence: Definition and Derivation

### Definition

$$
D_{\mathrm{KL}}(P\|Q)\;=\;\sum_x P(x)\,\log\frac{P(x)}{Q(x)}.
$$

### Derivation from [cross-entropy](#cross-entropy)

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&= \sum_x P(x)\,\log P(x)\;-\;\sum_x P(x)\,\log Q(x) \\
&= \bigl[-\sum_x P(x)\,\log Q(x)\bigr] \;-\; \bigl[-\sum_x P(x)\,\log P(x)\bigr] \\
&= H(P, Q) \;-\; H(P).
\end{aligned}
$$

* $H(P)$ is a **constant** if $P$ is fixed (e.g., teacher distribution). See the [Entropy](#entropy) section for more details.
* Minimizing $D_{\mathrm{KL}}(P\|Q)$ ↔ minimizing [cross-entropy](#cross-entropy) $H(P,Q)$.

> KL is **asymmetric**: $D_{\mathrm{KL}}(P\|Q)\neq D_{\mathrm{KL}}(Q\|P)$. The direction matters.

---

## 3) Forward vs. Reverse KL: Behaviors (Mode-Covering vs. Mode-Seeking)

### Forward KL (mode-covering)

$$
D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}=\mathbb{E}_{x\sim P}[\log P(x)-\log Q(x)].
$$

* Weighted by **$P(x)$**. If $P(x)>0$ and $Q(x)\ll P(x)$, the penalty is huge.
* Strongly discourages **missing any mass** where $P$ is nonzero → **mode-covering**.

### Reverse KL (mode-seeking)

$$
D_{\mathrm{KL}}(Q\|P)=\sum_x Q(x)\log\frac{Q(x)}{P(x)}=\mathbb{E}_{x\sim Q}[\log Q(x)-\log P(x)].
$$

* Weighted by **$Q(x)$**. If $Q(x)=0$ somewhere $P(x)>0$, that point contributes **zero**.
* Penalizes placing mass where $P\approx 0$, but not for **missing** modes → **mode-seeking**.

### Toy picture (bimodal teacher, unimodal student)

* Teacher $P$: two peaks at $-3$ and $+3$.
* Student $Q$: one Gaussian.

  * **Forward KL**: inflate variance and center between peaks to **cover both**.
  * **Reverse KL**: pick **one** peak and ignore the other (mode-seeking).

> For the mathematical details on how these behaviors emerge from the gradients, see [Section 4: Gradients You Actually Optimize](#4-gradients-you-actually-optimize).

---

## 4) Gradients You Actually Optimize

### 4.1 Forward KL gradient (low variance)

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q_\theta)&=H(P,Q_\theta)-H(P) \\
\nabla_\theta D_{\mathrm{KL}}(P\|Q_\theta)
&= \nabla_\theta H(P,Q_\theta) \\
&= -\sum_x P(x)\,\nabla_\theta \log Q_\theta(x) \\
&= -\mathbb{E}_{x\sim P}\bigl[\nabla_\theta \log Q_\theta(x)\bigr].
\end{aligned}
$$

* Expectation under **fixed** $P$ → **low-variance**, standard cross-entropy gradient.
* If $Q_\theta(x)\to 0$ while $P(x)>0$, $\nabla_\theta\log Q_\theta(x)=\frac{\nabla_\theta Q_\theta(x)}{Q_\theta(x)}$ blows up → **strong correction**. This is why forward KL **covers** all teacher modes.

### 4.2 Reverse KL gradient via the **log-derivative trick**

Start with

$$
D_{\mathrm{KL}}(Q_\theta\|P)=\sum_x Q_\theta(x)\,\bigl[\log Q_\theta(x)-\log P(x)\bigr].
$$

Differentiate:

$$
\begin{aligned}
\nabla_\theta D
&= \sum_x \nabla_\theta Q_\theta(x)\,[\log Q_\theta(x)-\log P(x)]
\;+\; \sum_x Q_\theta(x)\,\nabla_\theta\log Q_\theta(x).
\end{aligned}
$$

Use $\sum_x \nabla_\theta Q_\theta(x)=0$ (probabilities sum to 1), so the second sum vanishes:

$$
\nabla_\theta D
= \sum_x \nabla_\theta Q_\theta(x)\,[\log Q_\theta(x)-\log P(x)].
$$

Apply the **log-trick** $\nabla_\theta Q_\theta(x)=Q_\theta(x)\,\nabla_\theta\log Q_\theta(x)$:

$$
\boxed{
\nabla_\theta D_{\mathrm{KL}}(Q_\theta\|P)
= \mathbb{E}_{x\sim Q_\theta}
\Big[(\log Q_\theta(x)-\log P(x))\,\nabla_\theta\log Q_\theta(x)\Big].
}
$$

* This is a **REINFORCE-style** estimator: you must **sample from $Q_\theta$**.
* If $Q_\theta(x)\approx 0$, that $x$ is **never sampled** → **zero-gradient blind spot**.

---

## 5) Blind Spots and Variance (Why Reverse KL Is Tricky)

### Zero-gradient blind spots

* If $P(x)>0$ but $Q_\theta(x)=0$, the reverse-KL gradient at $x$ is zero because you never sample that $x$ from $Q_\theta$.
* The student may **never learn** that token even though the teacher says it matters.

### Variance explosion

* Reverse-KL gradient uses random samples from an **evolving** student distribution and a “reward” $\log Q-\log P$ that can swing widely → **high variance**, needs baselines/huge batches/clipping.
* Forward-KL gradient samples from **fixed** $P$ and is just cross-entropy → **low variance** and **sample-efficient**.

---

## 6) Knowledge Distillation: Why Forward KL Wins

**Setup:** Teacher $P(\cdot\mid\text{context})$ gives soft targets; Student $Q_\theta$ tries to match them.

* **Objective:** minimize $D_{\mathrm{KL}}(P\|Q_\theta)=H(P,Q_\theta)-H(P)$ ([forward KL](#3-forward-vs-reverse-kl-behaviors-mode-covering-vs-mode-seeking)).
* **Gradient:** $-\mathbb{E}_{x\sim P}\nabla_\theta\log Q_\theta(x)$ (see [gradient derivation](#41-forward-kl-gradient-low-variance)).
* **Properties:** stable, low-variance, **mode-covering** (no blind spots), trivial to implement (it's the usual [cross-entropy](#cross-entropy)).

> **What about the [entropy](#entropy) term $H(P)$?** It's a constant w\.r.t. $\theta$ and doesn't affect gradients—you don't try to "increase" or "decrease" it during student training.

---

## 7) “We only want task-relevant modes.” Do we need reverse KL?

Often you **don’t** want the student to copy *every* quirk of the teacher—you want **task adaptation**. You still don’t need pure reverse KL. Better options:

* **Filter / reweight the teacher** on your task data: compute a **task-conditioned teacher** $P_{\text{task}}$ (e.g., mask or down-weight unwanted tokens, sharpen on desired ones) and minimize **forward KL** $D_{\mathrm{KL}}(P_{\text{task}}\|Q)$. You retain low variance while focusing learning where it matters.
* **Temperature & top-p/k** on teacher outputs to emphasize salient modes, then forward-KL.
* **Focal/importance weighting** of the cross-entropy.
* **α-divergences / Rényi**: interpolate between mode-covering and mode-seeking without jumping to pure reverse KL.

---

## 8) PPO on a KL-only reward = Reverse-KL Minimization

Suppose you ignore task rewards and set the PPO reward to

$$
r(x)=\log P(x)-\log Q_\theta(x).
$$

PPO's policy-gradient (ignoring clipping for clarity, see [reverse KL gradient](#42-reverse-kl-gradient-via-the-log-derivative-trick)) is

$$
\nabla_\theta J=\mathbb{E}_{x\sim Q_\theta}\big[A(x)\,\nabla_\theta\log Q_\theta(x)\big],
$$

with $A(x)=r(x)-b$. Since $\mathbb{E}_{x\sim Q}[\nabla_\theta\log Q]=0$, the baseline term drops and

$$
\nabla_\theta J
=\mathbb{E}_{x\sim Q_\theta}\big[(\log P(x)-\log Q_\theta(x))\,\nabla_\theta\log Q_\theta(x)\big]
= -\,\nabla_\theta D_{\mathrm{KL}}(Q_\theta\|P).
$$

So PPO on this reward **maximizes** $J$ $\Longleftrightarrow$ **minimizes** reverse KL. It inherits reverse-KL’s drawbacks: **high variance**, **blind spots**, **mode-seeking**, and **lower sample-efficiency** than forward-KL distillation when you already have teacher samples.

---

## 9) Interpreting the Magnitude of KL

* $D_{\mathrm{KL}}(\cdot\|\cdot)\ge 0$; it’s $0$ iff $P=Q$.
* Not symmetric; no triangle inequality (not a metric).
* **Forward KL large** → the student **underestimates** regions the teacher cares about.
* **Reverse KL large** → the student **over-allocates** mass where the teacher says probability is tiny/zero.
* Can be **infinite** if support mismatch occurs (e.g., $P(x)>0$ but $Q(x)=0$ for forward KL).

---

## 10) Applications You’ll See in Practice

* **Supervised learning**: cross-entropy ⇔ minimizing $D_{\mathrm{KL}}(P\|Q)$.
* **Knowledge distillation**: forward KL to match teacher soft labels.
* **Variational inference**: often minimize $D_{\mathrm{KL}}(Q\|P)$ for tractability (mode-seeking is sometimes acceptable).
* **RL**: policy updates regularized by KL to prevent large steps (e.g., TRPO/PPO).
* **VAEs**: KL regularizes latent posteriors against a prior.

---

## 11) Frequently Asked Clarifications

* **“Why is the reverse-KL gradient zero at $Q(x)=0$?”**
  Because the gradient is an **expectation under $Q$**; events with $Q(x)=0$ are never sampled, contributing no gradient. (And the analytic term contains a factor $Q(x)$.)
* **“Does forward-KL have smaller gradient variance?”**
  Yes—samples come from fixed $P$, and the gradient is a plain cross-entropy expectation.
* **“What if I want ‘mode-seeking’ behavior?”**
  Prefer **task-conditioned forward KL** (filter/weight teacher) or **α-divergences**; avoid pure reverse KL in supervised distillation unless you accept its variance and blind-spot issues.

---

## 12) Mini Worked Example (discrete)

Teacher $P$:
$\;P(\text{A})=0.49,\;P(\text{B})=0.49,\;P(\text{C})=0.02.$

Student init $Q$:
$\;Q(\text{A})=0.98,\;Q(\text{B})=0.02,\;Q(\text{C})=0.$

* **Forward KL**: term for C is $0.02\log(0.02/0)\to\infty$ → huge gradient to increase $Q(\text{C})$. Student **covers** C.
* **Reverse KL**: term for C is $Q(\text{C})\log(Q/P)=0\cdot(\cdot)=0$ → **no gradient** for C. Student can happily ignore C (mode-seeking).

---

## 13) Summary Cheat-Sheet

* **[Entropy](#entropy)**: $H(P)=-\sum P\log P$.
* **[Forward KL](#forward-kl-mode-covering)**: $D(P\|Q)=H(P,Q)-H(P)$. [Low-variance gradients](#41-forward-kl-gradient-low-variance), **mode-covering**, needs **support matching** (finite only if $Q>0$ where $P>0$).
* **[Reverse KL](#reverse-kl-mode-seeking)**: $D(Q\|P)$ with [gradient](#42-reverse-kl-gradient-via-the-log-derivative-trick)
  $\mathbb{E}_{x\sim Q}[(\log Q-\log P)\nabla\log Q]$. **Mode-seeking**, **[blind spots](#5-blind-spots-and-variance-why-reverse-kl-is-tricky)**, **high variance**.
* **[Distillation](#6-knowledge-distillation-why-forward-kl-wins)**: Use **forward KL** ([cross-entropy](#cross-entropy)). To focus on task-relevant modes, **filter/weight the teacher** and still use forward KL.
* **[PPO on KL reward](#8-ppo-on-a-kl-only-reward--reverse-kl-minimization)**: exactly minimizes **reverse KL**; inherits its downsides for supervised settings.
* **Magnitude of KL**: "how mismatched," but direction matters; can be infinite with support mismatch.

If you want, I can add a small code snippet that numerically compares gradient variance for forward vs. reverse KL on a toy problem.
