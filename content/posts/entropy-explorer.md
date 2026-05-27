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

Don't worry if this sounds abstract. We'll start from the fundamentals. We'll build up from [information theory](#information-in-the-shannon-sense) and [entropy](#entropy), derive the [mathematics of KL divergence](#2-kl-divergence-definition-and-derivation), explore [forward vs reverse KL behaviors](#3-forward-vs-reverse-kl-behaviors-mode-covering-vs-mode-seeking), and by the end you'll have a complete understanding of when and why to use each variant.

## Information (in the Shannon Sense)

Let’s get a working intuition for *information* in the context of information theory. While the term is familiar in everyday language, Claude Shannon gave it a precise mathematical meaning that captures a simple but powerful insight:

> **Information measures surprise.** It quantifies how much uncertainty is resolved when an event occurs.

If an event is very likely, learning that it happened doesn’t teach you much. But when something *unlikely* occurs, you gain insight. That’s what Shannon formalized with the definition:

$$
I(x) = -\log_2 p(x)
$$

Here, \\( p(x) \\) is the probability of an event \\( x \\), and \\( I(x) \\) is the amount of information (in **bits**) gained by observing \\( x \\). The base-2 logarithm reflects the fact that we measure information in binary, in terms of how many yes/no decisions (bits) are needed to identify the outcome.

---

### Why use \\( \log p \\)? An Axiomatic Approach

This formula isn’t arbitrary, but it emerges naturally from a few reasonable assumptions about how we expect information to behave. Suppose we define a function \\( I(p) \\) to represent the information content of an event with probability \\( p \\). The requirements are:

#### Axiom A1: Rarity Implies More Information

The rarer an event, the more information it should convey. So \\( I(p) \\) should **decrease** as \\( p \\) increases.

An event that is guaranteed (\\( p = 1 \\)) conveys **no** information:

$$
I(1) = 0
$$


#### Axiom A2: Additivity for Independent Events

If two *independent* events occur, say one with probability \\( p \\) and the other with \\( q \\), then the information gained from both should be the **sum** of the individual informations:

$$
I(p \cdot q) = I(p) + I(q)
$$

This is essential if we want to talk about how information accumulates across independent events (e.g. flipping a coin twice = 2 bits).


#### Axiom A3: Continuity

The function \\( I(p) \\) should behave smoothly. Small changes in probability shouldn’t cause abrupt jumps in information. In other words, \\( I(p) \\) should be **continuous**.

---

#### 🔑 The Only Solution: \\( I(p) = -\log p \\)

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

This measures information in **bits**, the amount of binary decisions needed to resolve uncertainty.

## Bits, Information and Probability

As we saw in the previous section on Information, it's definition has a few intuitive properties:

* If an event is **certain** (\\( p = 1 \\)), then:

  $$
  I(x) = -\log_2(1) = 0 \text{ bits}
  $$

  You gain nothing. It was expected.

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

The surprise is greater, and the informational value reflects it.
<iframe
  src= "/entropy-explorer/information.html"
  width="90%"
  height="800"
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
* The base determines the **unit**. Base 2 gives us **bits**.
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

This is exactly what entropy measures: the expected amount of information per event drawn from a distribution.

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

This dual role of $P$ is crucial. Later, we'll see what happens when we sample from one distribution but evaluate probabilities from a *different* distribution. That's where things get interesting.

---

**Example: Fair Coin**

For a fair coin, both outcomes (heads H and tails T) have equal probability:

$$
H(X) = -[0.5 \cdot \log_2(0.5) + 0.5 \cdot \log_2(0.5)] = 0.5 \cdot 1 + 0.5 \cdot 1 = 1 \text{ bit}
$$

**Breaking down the expectation calculation:**

Using $\mathbb{E}_{X\sim p}\big[-\log_2 p(X)\big]$:
* When $X = \text{Heads}$ (happens with probability 0.5): we get $-\log_2(0.5) = 1$ bit
* When $X = \text{Tails}$ (happens with probability 0.5): we get $-\log_2(0.5) = 1$ bit
* **Average**: $0.5 \times 1 + 0.5 \times 1 = 1$ bit

So the entropy of a fair coin toss is 1 bit. This aligns with our intuition: a single binary question (e.g. "Is it heads?") is enough to fully describe the outcome.

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


So, the entropy of this biased coin is approximately 0.469 bits, noticeably less than 1 bit.

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

This is the **true distribution** $P$: how people actually complete this phrase in real data.

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

* 50% of the time, you see "amazing". Your model expected this 90% of the time, so you're *less* surprised than reality warrants ($-\log(0.9) \approx 0.15$ bits vs the true $-\log(0.5) \approx 1$ bit)
* **40% of the time, you see "terrible"**, but your model only expected this 5% of the time! You're *way more* surprised ($-\log(0.05) \approx 4.32$ bits vs the true $-\log(0.4) \approx 1.32$ bits)
* 10% of the time, you see "okay". Again, the model expected only 5%, so you're more surprised than you should be

The **cross-entropy** $H(P,Q)$ measures your average surprise when:
* Reality follows the true distribution $P$ (which words actually appear)
* But you're using your model's wrong probabilities $Q$ to measure surprise

$$
H(P,Q) = \mathbb{E}_{x \sim P}\big[-\log_2 Q(x)\big] = -\sum_x P(x)\,\log_2 Q(x)
$$

Read this as: **reality picks the word** (so we average over $P$), and **you bring the surprise meter** (so the term inside is $-\log_2 Q(x)$).

**Why $P$ weights but $Q$ sits inside the log.** The weighting is the *frequency of what actually happens*. Reality emits words according to $P$, so we average over $P$. The thing being weighted is *your* surprise, which depends only on what *your* model predicted. When "terrible" appears, your shock is $-\log_2 Q(\text{terrible})$ regardless of the true $P$. You don't get to retroactively un-shock yourself. Cross-entropy is exactly this: reality stays $P$, but the surprise meter is mis-calibrated to $Q$.

**Plugging in the numbers** (term = $P(x) \cdot -\log_2 Q(x)$):

| word | $P$ | $Q$ | $-\log_2 Q$ | contribution |
|---|---|---|---|---|
| amazing | 0.50 | 0.90 | 0.152 | $0.50 \times 0.152 = 0.076$ |
| terrible | 0.40 | 0.05 | 4.322 | $0.40 \times 4.322 = 1.729$ |
| okay | 0.10 | 0.05 | 4.322 | $0.10 \times 4.322 = 0.432$ |

$$
H(P,Q) \approx 0.076 + 1.729 + 0.432 \approx 2.24 \text{ bits}
$$

Notice the **"terrible"** row contributes ~1.73 of the 2.24 bits, almost 80% of the total. The model's worst miscalibration (predicting 5% when the truth is 40%) lands on an event that happens *often*, so it dominates the average. A bad prediction on a rare event would barely move the needle.

The excess surprise ($2.24 - 1.36 = 0.88$ bits) is the cost of having the wrong model. **This is exactly what "cross-entropy loss" measures when training LLMs.** It penalizes the model for assigning low probabilities to words that actually appear in the training data. By minimizing this cross-entropy, we force the model's predictions $Q$ to better match the true distribution $P$ of the data.

## KL Divergence: Definition and Derivation

### From Cross-Entropy to KL Divergence

We've seen how [cross-entropy](#cross-entropy) $H(P,Q)$ measures the total surprise when using model $Q$ to predict outcomes from the true distribution $P$. But how much of that surprise is **unavoidable** versus how much is **due to our model being wrong**?

Recall from our [movie review example](#cross-entropy):
* **Entropy** $H(P) \approx 1.36$ bits: the baseline uncertainty in the true distribution. This our ceiling. We can't do better than this.
* **Cross-entropy** $H(P,Q) \approx 2.24$ bits: total surprise using our wrong/approximation model

The difference between these, **0.88 bits**, is the **extra cost** of using the wrong model. This is exactly what KL divergence measures.

#### The Key Relationship

$$
\boxed{D_{\mathrm{KL}}(P\|Q) = H(P, Q) - H(P)}
$$

**What this tells us:**

* **$H(P)$** is constant (the true distribution doesn't change)
* **$H(P,Q)$** varies with your model $Q$
* **KL divergence isolates the "extra surprise"**: the cost of using an imperfect model
* **Minimizing cross-entropy** $H(P,Q)$ ⟺ **Minimizing KL divergence** $D_{\mathrm{KL}}(P\|Q)$

This is why in machine learning, we can use cross-entropy loss instead of computing KL divergence directly. They have identical gradients!

---

### The Log-Ratio Form

We can also derive KL divergence directly from the relationship above. Start with:

$$
D_{\mathrm{KL}}(P\|Q) = H(P, Q) - H(P)
$$

Substitute the definitions of cross-entropy and entropy:

$$
H(P, Q) = -\sum_x P(x)\log Q(x)
$$

$$
H(P) = -\sum_x P(x)\log P(x)
$$

Plug both in (the two minus signs combine to a plus on the second term):

$$
D_{\mathrm{KL}}(P\|Q) = -\sum_x P(x)\log Q(x) + \sum_x P(x)\log P(x)
$$

Both terms share the factor $P(x)$, so combine them into one sum:

$$
D_{\mathrm{KL}}(P\|Q) = \sum_x P(x)\big[\log P(x) - \log Q(x)\big]
$$

Apply the log subtraction rule $\log a - \log b = \log\tfrac{a}{b}$:

$$
\boxed{D_{\mathrm{KL}}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}}
$$

This is the standard **log-ratio** form. Equivalently, written as an expectation under $P$:

$$
D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right]
$$

**Understanding the log-ratio:**

For each event $x$:
* If $P(x) > Q(x)$: model **underestimates** → positive contribution (penalty)
* If $P(x) < Q(x)$: model **overestimates** → negative contribution
* If $P(x) = Q(x)$: perfect match → zero contribution

**Why the $P(x)$ weighting matters.** Each term in the sum is the *log-ratio* (how wrong the model is at $x$) multiplied by $P(x)$ (how often $x$ actually occurs). The model only gets penalized for mistakes on events that reality bothers to produce.

Think of two failure modes:

* **Underestimating a frequent event.** Reality says "this happens 50% of the time" but your model says 1%. The log-ratio $\log(0.5/0.01) \approx 5.6$ is enormous, *and* it gets weighted by $P(x) = 0.5$. Big number times big weight = huge penalty. The loss screams at you to fix this.
* **Overestimating a rare event.** Reality says "this happens 0.1% of the time" but your model thinks 10%. The log-ratio $\log(0.001/0.1) \approx -4.6$ is large in magnitude (negative, meaning $Q$ is too high), but it's weighted by $P(x) = 0.001$. Tiny weight muffles the penalty. The loss barely notices.

So the loss really only cares about one thing: **don't assign low probability to events that actually happen.** It's much more forgiving about wasting probability mass on events that *don't* happen.

**This is "mode-covering" behavior.** A "mode" is a peak in $P$, a region where probability mass concentrates (a frequent event/word/feature). Forward KL forces $Q$ to *cover every mode of $P$*, because missing one (assigning low $Q$ where $P$ is high) is exactly the failure case it punishes hardest. The cost of doing so is that $Q$ may also spread mass into regions where $P \approx 0$, but as we just saw, that's almost free in forward KL. The result: $Q$ ends up wider than $P$, blurring across all the modes, never confidently picking just one.

**Reverse KL flips the weighting.** Now look at $D_{\mathrm{KL}}(Q\|P) = \sum_x Q(x)\log\frac{Q(x)}{P(x)}$. The log-ratio is inverted ($\log\tfrac{Q}{P}$ instead of $\log\tfrac{P}{Q}$), and each term is weighted by $Q(x)$ instead of $P(x)$. That single change inverts which mistakes get punished.

Take the same two scenarios as before and watch the verdicts swap:

* **Underestimating a frequent event** ($P=0.5$, $Q=0.01$). Forward KL hated this. Reverse KL barely notices. The log-ratio $\log(0.01/0.5) \approx -5.6$ is large in magnitude, but it's weighted by $Q(x) = 0.01$. Tiny weight muffles the penalty. Reverse KL is fine with $Q$ abandoning a real mode of $P$, as long as $Q$ goes to (near) zero there.
* **Overestimating a rare event** ($P=0.001$, $Q=0.1$). Forward KL barely noticed. Reverse KL hates it. The log-ratio $\log(0.1/0.001) \approx 6.6$ is large, *and* it's weighted by $Q(x) = 0.1$. Big number times meaningful weight = huge penalty. Reverse KL screams at you for putting mass where $P$ says nothing happens.

So reverse KL really only cares about one thing: **don't assign mass to events that don't actually happen.** It's much more forgiving about ignoring real modes of $P$, as long as $Q$ stays out of them entirely.

**This is "mode-seeking" behavior.** If $P$ has two well-separated peaks, $Q$ can earn a low reverse-KL score by parking itself entirely on *one* peak and ignoring the other. Spreading mass into the valley between the peaks (where $P \approx 0$) is exactly the failure case reverse KL punishes hardest, so $Q$ would rather sit narrowly inside a single mode than stretch out to cover both. The result is the mirror image of forward KL: $Q$ ends up narrower than $P$, confident on one mode and blind to the rest.

#### See It: Mode-Covering vs. Mode-Seeking

The demo below makes both directions concrete. The true distribution $P$ (red) is bimodal with peaks at $x = -2$ and $x = 2$. Your model $Q$ (blue) is a single Gaussian, and you control its mean and width with the sliders. The two lower panels plot the per-$x$ contributions of each direction side by side: the orange panel shows $P(x)\log\frac{P(x)}{Q(x)}$ (forward KL, weighted by $P$) and the purple panel shows $Q(x)\log\frac{Q(x)}{P(x)}$ (reverse KL, weighted by $Q$). You can see exactly where each penalty lives.

**Three things to try:**

1. Click **"Mode-cover preset"**. $Q$ becomes wide and centered between the two modes. Forward KL is moderate, but **reverse KL is large**: the purple panel lights up across the central valley, because $Q$ is putting mass at $x \approx 0$ where $P \approx 0$, and reverse KL hates that. Meanwhile the orange (forward) panel stays calm: $Q$ has at least *some* mass on each mode, so $\log\frac{P}{Q}$ never blows up.
2. Click **"Mode-seek preset"**. $Q$ is narrow and sits on the right mode of $P$. Reverse KL drops to near zero (the purple panel flattens, because $Q$ has stopped putting mass where $P$ is silent), but **forward KL explodes**. The orange panel shows a huge spike at the abandoned left mode of $P$, where $P(x)$ is high but $Q(x) \approx 0$, making $\log\frac{P}{Q}$ blow up.
3. Try sliding $\sigma_Q$ slowly from 0.6 up to 2.4 with $\mu_Q = 0$. Watch forward KL drop monotonically as $Q$ widens to "cover" both modes, while reverse KL climbs as $Q$ spills more mass into the central valley. The two directions are pulling $Q$ in opposite directions: forward wants width, reverse wants concentration.

<iframe
  src="/entropy-explorer/mode_covering.html"
  width="100%"
  height="820"
  style="border:0"
  loading="lazy">
</iframe>

The takeaway: **forward KL would rather have $Q$ be too wide than miss a mode of $P$, while reverse KL would rather have $Q$ be too narrow than place mass where $P$ is silent.** Both preferences fall directly out of which distribution does the weighting in the sum: $P(x)$ punishes missing $P$'s mass, $Q(x)$ punishes wasting $Q$'s own mass on regions $P$ ignores.

---

## Knowledge Distillation: Why Forward KL is the Default

Knowledge distillation (KD) trains a small "student" model $Q_\theta$ to imitate a larger "teacher" model $P$. The teacher was trained at great cost (compute, data, scale); the student is the cheap deployable version. The question for us is: which direction of KL should the student minimize?

Both choices are well-defined. Let's write them out and follow the consequences.

### Forward KL distillation

Per input $x$, the teacher gives a soft distribution $P(\cdot|x)$ over outputs. Forward KL distillation minimizes the expected $D_{\text{KL}}(P \| Q_\theta)$ across the input distribution:

$$
\mathcal{L}_{\text{FKD}}(\theta) = \mathbb{E}_{x}\left[D_{\text{KL}}(P(\cdot|x) \| Q_\theta(\cdot|x))\right] = \mathbb{E}_{x}\mathbb{E}_{y \sim P(\cdot|x)}\left[\log \frac{P(y|x)}{Q_\theta(y|x)}\right].
$$

**Reading the nested expectation.** Two averages are stacked here. The outer one, $\mathbb{E}_x[\cdot]$, runs over **inputs**: you have a dataset of inputs (prompts for an LLM, images for a vision model, sentences for a translation model) and you draw an input $x$ from it. The inner one, $\mathbb{E}_{y \sim P(\cdot|x)}[\cdot]$, runs over **outputs**: for that fixed input, the teacher hands you a distribution $P(\cdot|x)$ over possible outputs $y$, and you average over outputs drawn according to that distribution. Read the whole expression as: *"pick an input from your dataset, look at the per-input KL between teacher and student over the output space, then average across inputs."* For an LLM, $x$ is the prefix of a sentence, $P(\cdot|x)$ is the teacher's predicted next-token distribution over the 50K+ vocabulary, and the inner expectation averages across what the teacher considers likely next tokens.

Drop the $\log P$ term (constant w.r.t. $\theta$, the student's parameters) and you are left with:

$$
\nabla_\theta \mathcal{L}_{\text{FKD}} = -\mathbb{E}_{x}\mathbb{E}_{y \sim P(\cdot|x)}\left[\nabla_\theta \log Q_\theta(y|x)\right].
$$

**This is cross-entropy with soft labels.** For small output spaces (image classification with 10 classes), you sum the soft labels exactly. For huge output spaces (LLM vocab of 50K+, or sequence-level distillation), you sample from the teacher.

Three things make this objective pleasant to optimize:

1. **You sample from $P$**, which is fixed. Once teacher logits are cached, or the teacher runs once per batch, there is no resampling cost as the student updates.
2. **Low variance.** The gradient is a plain expectation under a fixed distribution. No log-derivative trick, no REINFORCE estimator, no baselines needed.
3. **Mode-covering safety.** From the [analysis above](#the-log-ratio-form), forward KL punishes the student hardest when $Q_\theta(y|x) \approx 0$ at a $y$ the teacher considers likely. The student is forced to assign at least some mass to every teacher mode, so it cannot accidentally throw away a chunk of teacher knowledge.

This is Hinton, Vinyals, and Dean's original 2015 KD recipe in modern notation. It is also why distilled BERTs, distilled GPTs, and distilled vision models all use cross-entropy on soft targets: forward KL is the only choice that is both stable and gradient-cheap.

### Reverse KL distillation

The other direction, $D_{\text{KL}}(Q_\theta \| P)$, gives:

$$
\mathcal{L}_{\text{RKD}}(\theta) = \mathbb{E}_{x}\mathbb{E}_{y \sim Q_\theta(\cdot|x)}\left[\log \frac{Q_\theta(y|x)}{P(y|x)}\right].
$$

Already this looks different from forward KL. The sample $y$ comes from the *student* $Q_\theta$, not the teacher $P$. That single change creates a real obstacle: **the distribution we are averaging over depends on the parameters we are trying to optimize.** If we wiggle $\theta$, two things happen at once: the term inside the brackets changes (since it contains $Q_\theta$), AND the probability of drawing each particular $y$ changes (since we are sampling from $Q_\theta$ itself). A naive gradient that differentiates only the inside misses the second effect entirely.

#### A detour: the log-derivative trick

This is a problem ML hits over and over again. The classical fix is the **log-derivative trick**, also called the **score-function estimator**, also called the **REINFORCE estimator** in reinforcement learning. It is the workhorse identity behind every policy-gradient algorithm (PPO, A2C, TRPO, GRPO), behind variational inference with discrete latents, and now behind reverse-KL distillation.

The setup: you want $\nabla_\theta \mathbb{E}_{y \sim Q_\theta}[f(y)]$ for some function $f$. Expand the expectation as a sum, then differentiate:

$$
\nabla_\theta \sum_y Q_\theta(y) f(y) = \sum_y \big(\nabla_\theta Q_\theta(y)\big) f(y).
$$

Now use the identity $\nabla_\theta Q_\theta(y) = Q_\theta(y) \nabla_\theta \log Q_\theta(y)$ (which is just $\nabla \log u = \nabla u / u$ rearranged) to put $Q_\theta(y)$ back as a sampling weight:

$$
= \sum_y Q_\theta(y) \, f(y) \, \nabla_\theta \log Q_\theta(y) = \mathbb{E}_{y \sim Q_\theta}\big[f(y) \, \nabla_\theta \log Q_\theta(y)\big].
$$

**The intuition.** The gradient *raises the log-probability of samples where $f(y)$ is large and lowers it where $f(y)$ is small.* Think of $f$ as a "reward signal" weighting how strongly each sample should be reinforced or suppressed. This is exactly the policy-gradient intuition: take actions that paid off, take fewer actions that did not. Same math, different label.

The catch is variance. Because $y$ is random and $f(y)$ can swing widely from sample to sample, the estimator $f(y) \nabla_\theta \log Q_\theta(y)$ can be very noisy. You usually need either a baseline $b$ (subtracting $b$ from $f(y)$ does not change the expectation but can shrink variance dramatically) or large batches to average the noise out. Hold on to this. It comes back as the central practical issue.

#### Applying it to reverse KL

Inside the brackets of $\mathcal{L}_{\text{RKD}}$ we have $f(y) = \log Q_\theta(y|x) - \log P(y|x)$. This is *almost* the setup we just handled, except $f$ itself depends on $\theta$ through the $\log Q_\theta$ term. Differentiate the full expectation $\sum_y Q_\theta(y) f(y)$ with the product rule and you get two terms:

$$
\nabla_\theta \sum_y Q_\theta(y) f(y) = \underbrace{\sum_y \big(\nabla_\theta Q_\theta(y)\big) f(y)}_{\text{log-derivative trick gives } \mathbb{E}[f \, \nabla\log Q]} + \underbrace{\sum_y Q_\theta(y) \, \nabla_\theta f(y)}_{\text{extra term from } f \text{ depending on } \theta}.
$$

The first term is exactly the log-derivative-trick result we just derived. The second term simplifies beautifully: $\nabla_\theta f(y) = \nabla_\theta \log Q_\theta(y)$ (the $\log P$ piece does not depend on $\theta$), so the second term is $\sum_y Q_\theta(y) \nabla_\theta \log Q_\theta(y) = \sum_y \nabla_\theta Q_\theta(y) = \nabla_\theta \sum_y Q_\theta(y) = \nabla_\theta 1 = 0$. It vanishes because probabilities sum to one.

We are left with:

$$
\nabla_\theta \mathcal{L}_{\text{RKD}} = \mathbb{E}_{x}\mathbb{E}_{y \sim Q_\theta(\cdot|x)}\left[\big(\log Q_\theta(y|x) - \log P(y|x)\big) \nabla_\theta \log Q_\theta(y|x)\right].
$$

This is exactly a REINFORCE-style policy gradient with effective "reward" $-(\log Q_\theta - \log P)$, which is large and positive whenever the student is putting much more mass on $y$ than the teacher does. The training loop is now: sample some outputs from the student, compute that reward for each, weight the per-sample score function $\nabla_\theta \log Q_\theta$ by the reward, and take a step. Same shape as PPO.

#### What goes wrong in practice

With the derivation in hand, three issues immediately surface:

1. **You sample from $Q_\theta$**, which keeps moving. Every gradient step changes the sampling distribution, so samples from the previous step are stale. For sequence models this means actually *generating* from the student each batch, which is expensive.
2. **High variance.** The "reward" $\log Q_\theta - \log P$ can swing widely across samples, especially early in training when student and teacher disagree on most of the space. Practical implementations need baselines, large batches, or clipping to make the gradient usable.
3. **Zero-gradient blind spots.** If $Q_\theta(y|x) \approx 0$ for some $y$, that $y$ is never sampled, so it contributes nothing to the gradient. The per-$x$ contribution carries a factor of $Q(y)$ in front, and when $Q(y) = 0$, no signal reaches the student. The student can permanently ignore an entire teacher mode and the loss will not complain.

The third issue is the deal-breaker for classical KD. The whole point of distillation is to transfer the teacher's knowledge into the student. Reverse KL keeps the teacher knowledge the student already partially has, and *silently discards* anything it does not. There is no mechanism by which "the teacher cares about this region but I do not" gets corrected.

### A toy example to make the blind spot concrete

Teacher distribution over three outputs:
$$P(\text{A}) = 0.49,\quad P(\text{B}) = 0.49,\quad P(\text{C}) = 0.02.$$

Student initialization (a common early-training pathology where the student picks one mode):
$$Q(\text{A}) = 0.98,\quad Q(\text{B}) = 0.02,\quad Q(\text{C}) = 0.$$

**Forward KL contribution for C:** $0.02 \cdot \log(0.02 / 0) \to \infty$. The loss is infinite, or numerically enormous with epsilons, and the gradient pushes $Q(\text{C})$ up immediately.

**Reverse KL contribution for C:** $Q(\text{C}) \cdot \log(Q(\text{C})/P(\text{C})) = 0 \cdot \log(0/0.02) = 0$. The loss does not notice. The student happily stays on A forever, ignoring both B (which the teacher considers equally important) and C entirely.

Forward KL is the safer choice when your goal is to faithfully copy the teacher's distribution. Reverse KL converges to *a* solution that is locally consistent with $P$, but it does not have to be the *right* solution.

### The classical verdict

For traditional KD, like image classification with a few hundred classes, or BERT distillation with masked-token targets, anywhere the teacher's full soft distribution is available and the goal is faithful transfer, forward KL wins decisively. Lower variance, no blind spots, samples from a fixed distribution, gradient is just cross-entropy. There is no contest.

This is why basically every distillation paper before roughly 2023 uses forward KL (cross-entropy with soft labels), often with a temperature softening of $P$ to make the soft targets more informative. See Hinton, Vinyals, Dean (2015) for the original, and Sanh et al. (2019, DistilBERT) or Jiao et al. (2020, TinyBERT) for canonical LLM-era examples. All forward KL.

### The LLM twist: when reverse KL becomes interesting

The classical analysis quietly assumes you *want* the student to match every detail of the teacher. For autoregressive LLM distillation, that assumption starts to break.

The issue is the **long tail** of LLM token distributions. At each generation step, the teacher's softmax over 50K+ vocabulary tokens has a small number of high-probability tokens (the modes you want to learn) and a long tail of low-probability tokens that are mostly noise: alternate phrasings, slightly off-topic continuations, plausible-but-suboptimal next words.

Forward KL does not distinguish modes from noise. It says: match the entire distribution, including the long tail. For a small student with limited capacity, this means spreading probability mass thin across the tail. At inference time, sampling from a too-flat distribution produces hallucinated content. The tokens the student "covered" because the teacher gave them $10^{-4}$ probability turn into actual generation choices once the student is deployed, and many of them are not good choices.

**MiniLLM** (Gu et al., 2024) makes this case explicitly. They argue that reverse KL is preferable for LLM distillation precisely *because* its mode-seeking behavior lets the student ignore the teacher's long tail. The student concentrates on high-probability behaviors, generates more focused and less hallucinated outputs, and tolerates being meaningfully smaller than the teacher.

To deal with the gradient-variance and blind-spot problems, MiniLLM borrows from RL: policy-gradient style updates, single-step regularization to keep the student close to the teacher (preventing total mode collapse), and length-normalization. The implementation is much more complex than forward KL distillation, but the generation quality at small scales is meaningfully better.

### Beyond the binary

Forward KL and reverse KL are just two points on a spectrum of divergences. Several recent papers explore the middle ground:

- **GKD** (Agarwal et al., 2024), "On-Policy Distillation of Language Models", reframes the problem. The real issue is not the KL direction, it is the **distribution mismatch** between training-time samples (teacher) and inference-time samples (student). GKD uses on-policy student samples but evaluates with a generalized JSD that interpolates between forward and reverse. With the right interpolation parameter, you get low variance, no blind spots, and student-sample exposure that fixes the inference-time mismatch.
- **f-DISTILL** (Wen et al., 2023) generalizes to arbitrary f-divergences, of which forward KL, reverse KL, JSD, and total variation are all special cases. The empirical sweet spot is often somewhere between forward and reverse rather than at either extreme.
- **DistiLLM** (Ko et al., 2024) uses *skew-KL* variants, interpolating $P$ and $Q$ inside the log to bound variance and prevent the blind-spot pathology without giving up the mode-seeking benefit.

The common thread: pure reverse KL has real benefits for LLM distillation, but its drawbacks are real too, and modern recipes pay extra mathematical cost to extract the benefits without inheriting the drawbacks.

### Practical recipe

If you are distilling a model and want a working answer:

- **Classification, regression, structured prediction with full soft labels available:** Use forward KL (cross-entropy with soft targets). Add temperature if the teacher is overconfident. This is the right answer most of the time.
- **Sequence-level LLM distillation, want maximum quality, willing to invest engineering effort:** Look at GKD or MiniLLM. Expect to spend much more compute and tuning than vanilla forward KL.
- **You want a quick baseline for any LLM distillation task:** Forward KL on teacher samples. It is not optimal but it is robust and easy to debug. Iterate from there.

The deeper takeaway is that **the "correct" KL direction depends on what you are using the student for.** If you want faithful posterior matching, forward KL. If you want a deployable model that generates well in practice and you will tolerate losing some teacher diversity, reverse KL or its descendants. The choice is a product decision as much as a mathematical one.
