---
title: "Word Translation Without Parallel Data: A Beautiful Application of GANs and Linear Algebra"
date: 2026-05-26
draft: false
math: true
layout: scrolly
summary: "Three beauties meet in this paper. Words in one language end up internally aligned by meaning. For languages with shared structure, that arrangement is shaped almost the same way across both. A GAN and an SVD recover the rotation between them, strongly for close pairs, weakly for distant ones, with no parallel text."
tags: ["nlp", "linguistics", "linear-algebra", "embeddings", "gan", "procrustes", "svd"]
showToc: false
---

<section data-vis="hero">

Train word2vec on a giant English corpus. Train it again, completely separately, on a giant Italian corpus. You now have two 300-dimensional point clouds, one for each language's vocabulary, trained on different writers, different topics, different cultures.

Their shapes are nearly identical.

Not metaphorically. *Close enough that a single rigid rotation maps most of one onto the other.* The point for *cat* lands near *gatto*. *King* lands near *re*. *Three* lands near *tre*. A 300×300 orthogonal matrix, found without any dictionary or parallel text, recovers a working bilingual lexicon.

This is the central claim of [Word Translation <span class="standout">Without Parallel Data</span>](https://arxiv.org/pdf/1710.04087) by Conneau, Lample, Ranzato, Denoyer and Jégou (ICLR 2018), and it is true to a degree. The degree itself is the interesting part. The result is strong for languages with shared structure (the paper reaches 65 to 82 percent top-1 accuracy on English paired with Spanish, French, German, Italian), and substantially weaker for typologically distant pairs (English to Chinese clears 32 percent, less than half the Romance score). For pairs further apart still, with thinner shared history or differently-sourced corpora, the method degrades further. The gradient across those cases is the most informative empirical signal the paper produces. We will get to it.

What survives the qualification is that the result works at all. The fact that it does, even partially, draws on three separate beauties that are each worth pausing on before we look at the mechanics.

The first is the beauty of *embeddings*: that words in a *single* language, trained only to predict context, end up *internally aligned*. Semantically similar words sit close together. Semantic relationships become consistent geometric directions. None of this is designed in; it emerges from the statistics of natural text.

The second is the beauty of *language*: that this internal arrangement, for typologically and culturally close languages (similar grammar, overlapping vocabulary, comparable ways of carving up the world), ends up shaped almost the same way across them. Not universally. Not for every pair. But often enough, and strongly enough, to be useful.

The third is the beauty of *the right tools*: that a GAN and an SVD, neither designed for this problem, compose so cleanly that the answer falls out in closed form when the linguistic prior holds.

</section>

<section data-vis="directions">

## 1. The beauty of embeddings

Before 2013, words were strings. The word *cat* was no more related to *dog* than it was to *Tuesday*, as far as a computer was concerned. They were symbols, and symbols don't have neighborhoods.

Word2vec, published by Mikolov and colleagues in 2013, broke this. Take a tiny neural network. Train it to predict context words from a center word over a giant pile of text. Read the resulting 300-dimensional vectors off the network's weights. What you get encodes meaning *geometrically*.

*Cat* and *dog* end up close. *Monday*, *Tuesday*, *Wednesday* fall on a line. The vector from *man* to *woman* is, give or take, the same as the vector from *king* to *queen*. Gender becomes a direction. So does royalty. So does capital-to-country (*Paris→France*, *Berlin→Germany*). So does past tense (*walk→walked*, *think→thought*). So does comparative (*big→bigger*, *fast→faster*). Meaning grows axes.

This is genuinely strange. Nobody designed gender to be an axis. The network had no concept of gender, no concept of meaning, no language model in any deep sense. It was minimizing a prediction loss. The geometry was a side effect of the statistical structure of natural text. Distributional semantics, the old linguistic hypothesis that *you shall know a word by the company it keeps* (Firth, 1957), turned out to be not just a hypothesis but a constructive recipe.

That side effect is the foundation of a decade of NLP and the precondition for everything that follows in this paper. The visual on the right shows a few of the directions a real embedding learns, drawn as parallelograms in a 2D mock-up. The actual space lives in 300 dimensions, but the principle is the same: meaning has axes, and the axes turn out to be the things linguists have always wanted to name.

</section>

<section data-vis="alignment">

## 2. The beauty of language

The second beauty is what happens when you train *two* embedding spaces, on two different languages, completely independently.

They come out shaped the same way.

This is a fact about *language*, not about word2vec. The geometric structure that emerges inside a single language is not a structure of that language. It is a structure of the *world* the language describes, refracted through the patterns by which speakers talk about it.

Animals behave like animals regardless of who names them. Numbers count up the same way in every counting system. Kings stand in a stable relation to queens. Days form a seven-cycle. Categories cluster, hierarchies branch, opposites stretch along axes. These are properties of the referents, not of the words. Two training runs over disjoint corpora in different languages, given enough overlapping world, converge on geometrically similar embedding spaces because they're discovering, distributionally, the same underlying reality.

That is the linguistic prior this paper exploits. To use it, we need a single transformation $W$ that maps one language's embedding cloud onto the other's, point by point. If the two clouds genuinely share a shape, $W$ has to be *rigid*: it can rotate the cloud, it can reflect it, but it cannot stretch or shear, because any of those would change the shape itself.

In matrix terms, $W$ is an **orthogonal matrix**, characterized by the property $W^T W = I$. Orthogonal matrices are exactly the linear maps that preserve lengths and angles. Applied to a source-language embedding vector $\mathbf{x}$ (a single English word's 300-dimensional point), the product $W\mathbf{x}$ is a new 300-dimensional vector that lives in the target-language space, ideally sitting where the Italian translation of that word lives. The whole paper is a procedure for finding the right $W$.

The visual on the right is a toy 2D version of the situation. Two clouds with identical internal geometry sit at different orientations. Drag the warm cloud around the green one. When the right rotation is found, the page glows. In 300 dimensions, on real embedding data, a 300×300 orthogonal $W$ does the same job.

</section>

<section data-vis="mismatch">

## 3. Where the shape breaks (and why that's also beautiful)

The third beauty, which the paper handles less explicitly but which any linguist would insist on, is that language is also wonderfully varied. The shape-isomorphism is approximate, and the approximations are exactly where linguistics gets interesting.

Different languages turn the same concepts into different words. A meaning that one language captures with a single word may, in another, be split into several distinct words (along distinctions the first language doesn't bother to mark), or fused together with other concepts the first language keeps separate. Linguists call this **linguistic anisomorphism**. The visual on the right makes one example concrete: a single English word, *uncle*, fans out to five distinct Hindi words encoding whether the uncle is paternal or maternal, older or younger, by blood or by marriage. One point in English embedding space corresponds to five distinct points in Hindi space. A linear rotation cannot do that.

Other structural mismatches a single orthogonal transformation cannot resolve:

- **Basic color terms differ.** English *also* has plenty of words for shades of blue (cyan, navy, sky blue), but all of them are subordinate to the basic word *blue*. Russian, by contrast, has *two* basic color words at the same conceptual level: *голубой* (light blue) and *синий* (dark blue), no more related to each other than English *pink* and *red* are. The difference is not just vocabulary; psycholinguistic experiments (<a class="cite" href="https://www.pnas.org/doi/10.1073/pnas.0701644104">Winawer et al. 2007</a>) show Russian speakers categorically perceive the boundary between the two, with measurably faster color discrimination across the *синий* / *голубой* boundary, while English speakers see them as shades of one color. Hanunó'o, a Philippine language, has only four basic color terms total. Where one language has a single basic-level slot in concept space, another may have several.

- **Grammatical gender is a direction in some languages and not others.** Italian marks gender on every noun (*la tavola* is feminine, *il libro* is masculine, with no semantic reason for either). In an Italian embedding space, gender becomes a learnable axis. In English it largely is not. A rotation cannot create dimensions that the source representation never had.

- **Morphological richness changes what "a word" is.** Turkish builds entire phrases into single words (*evlerimizdekiler* means "the ones in our houses"), so a whitespace tokenizer turns every inflection (*evler*, *evlerim*, *evlerimizde*, *evlerimizdekiler*) into its own unrelated point in the embedding space. German keeps compound nouns whole, so *Donaudampfschiffahrtsgesellschaftskapitän* ("Danube steamship company captain") gets one embedding that shares nothing with *Schiff* or *Kapitän*. Mandarin, Japanese, and Thai have no whitespace at all, so word2vec pipelines relied on external segmenters (*jieba*, *MeCab*) whose choices were themselves a modeling decision. No rotation can undo any of this.

- **Polysemy patterns do not line up.** English *run* covers operating a business, executing software, flowing water, racing on foot, and seeking political office. Italian splits these senses across distinct words (*gestire*, *eseguire*, *scorrere*, *correre*, *candidarsi*). A single rotation cannot route one polysemous English point to five separate Italian ones.

- **Aspect and evidentiality get baked into verbs.** A Russian verb almost always tells you whether the action was completed or ongoing, and the contrast is so central that most verbs come in pairs: *читать* ("be reading") and *прочитать* ("read to completion") are two different verbs in Russian, two different points in the embedding space, but both translate to English *read*. Turkish goes further and marks the verb to indicate whether the speaker witnessed an event firsthand or heard about it secondhand; English handles that information optionally, with adverbs like "apparently" or "supposedly," not as a required grammatical feature. When source and target languages bake different distinctions into the verb itself, alignment becomes one-to-many in both directions.

- **Culture-specific concepts.** *Schadenfreude*, *komorebi*, *hygge*, *saudade*. Words that have no clean cross-linguistic counterpart sit in regions of their language's embedding space with no rotational image.

What an orthogonal transformation can do is align the parts of two embedding spaces that genuinely share structure. Where the structures diverge, the transformation either fails locally or smears unrelated words together. The paper's success is evidence that the shared part is large enough, for the language pairs it tests, to make a useful translation system. Not evidence that meaning is universal. The variation between languages is itself a kind of beauty: different windows onto the same world.

</section>

<section data-vis="tree">

## 4. How much of the shape actually transfers

The right way to read the paper's empirical results is as a *measurement* of how isomorphic two languages are.

The visual on the right places each language at its typological distance from English, color-coded by the paper's top-1 word translation accuracy. The picture is unmistakable. Languages where English shares more, ancestry, vocabulary inheritance, morphology, centuries of cultural contact, give 65 to 82 percent top-1 accuracy. Languages that share less drop into the 30s. English to Chinese, separated by family, typology, writing system, and most cultural reference points, performs less than half as well as English to Italian.

This is not a failure of the method. It is the method reporting back to us how isomorphic two languages actually are. The same orthogonal-rotation prior that lets the GAN find a good solution for English-Italian limits the ceiling for English-Chinese.

<a class="cite" href="https://aclanthology.org/P18-1072/">Søgaard, Ruder, and Vulić (2018)</a> made this critique explicit a year after the paper appeared. They showed that unsupervised alignment breaks down further when (i) language pairs are typologically distant, and (ii) the embeddings being aligned were trained on corpora from different domains, even *within* a single language pair. The geometric isomorphism is an empirical regularity that holds well for similar languages on similar corpora, and degrades as either dimension is pushed.

The breakdown is most severe at the far end of the resource axis. **Low-resource languages** carry a double penalty in this method. Their embeddings are weak to begin with, because word2vec needs hundreds of millions of tokens to find clean geometric structure, and many of the world's languages simply do not have that. The point cloud for a low-resource language is sparser, noisier, and less internally aligned than its high-resource counterpart. There is less shape to match in the first place. And low-resource languages tend, on average, to sit further away typologically and culturally from the high-resource languages anyone would want to align them to. The source representation is fragile, and the alignment prior is weak. Together the two failure modes compound, and the method often collapses outright.

This double penalty is one of the cleanest reasons modern multilingual models have absorbed the problem the 2017 paper was trying to solve. mBERT, XLM-R, and the multilingual LLMs that followed train a single network on dozens or hundreds of languages simultaneously, with a shared subword vocabulary and shared parameters across all of them. A low-resource language no longer has to discover its own internal geometry from a thin corpus; it inherits, by positive transfer, the structure already present in the high-resource neighbors that share grammatical or lexical features with it. The output is one embedding space where each language occupies a sub-region of a single shared geometry. There is no rotation to find. The languages are already in the same room. The 2017 paper's detour through GANs and Procrustes is, in this light, the move you make when you cannot afford to train one big model on everything at once. It is also why studying the method still pays off: it forces you to look at the linguistic prior in the open, where modern models tuck it inside the weights.

So the right framing is this. The paper exploits a *linguistic* prior of approximate cross-lingual structural similarity. That prior is strong enough for a large family of practically important language pairs to make unsupervised translation work. It is not strong enough to be called universal, and the field's understanding has matured to treat it as a graded resource rather than an axiom.

With that caveat properly placed, the math gets to do its work.

</section>

<section data-vis="procrustes-setup">

## 5. The beauty of SVD

Suppose for a moment that someone hands you a small dictionary: $n$ source words each paired with their known target translation. Stack the source vectors as rows of $X \in \mathbb{R}^{n \times d}$ and the target vectors as rows of $Y \in \mathbb{R}^{n \times d}$. You want the orthogonal $W$ that, when multiplied with $X$, gets you as close as possible to $Y$:

$$W^* = \arg\min_{W^T W = I} \, \| W X^T - Y^T \|_F^2$$

The visual on the right shows what we're after: a single rotation that pulls the warm cloud onto the green cloud and shrinks the per-pair residual lines to zero. Watch the rotation sweep through different angles; the readout at the bottom tracks the residual as it falls.

This is the **orthogonal Procrustes problem**, named with grim humor after the Greek innkeeper who fit guests to his bed by stretching short ones and chopping long ones. Schönemann gave the answer in closed form in 1966, and the answer is one application of the **Singular Value Decomposition**.

Before we use it, the SVD itself deserves a moment.

</section>

<section data-vis="svd">

### A sixty-second tour of SVD

The Singular Value Decomposition is one of the most beautiful results in linear algebra. It says that *any* matrix $M$, no matter what it does to space, can be written as

$$M = U \, \Sigma \, V^T$$

Three operations. Rotate by $V^T$. Stretch along orthogonal axes by $\Sigma$. Rotate by $U$. That is what every linear map is, at heart.

The visual on the right makes this visceral. A small smiley face goes through the three steps. First $V^T$ rotates it. Then $\Sigma$ stretches it into an ellipse, with the two singular values $\sigma_1, \sigma_2$ telling you exactly how much stretching happens in each direction. Then $U$ rotates the stretched face into its final orientation. The composed transform is the action of $M$.

This factorization is *everywhere*. It is the math underneath PCA. It is the math underneath low-rank image compression. It is the math underneath pseudoinverses, recommendation systems, latent semantic analysis. And it is the math that closes the form for Procrustes.

Here is the connection. If $M$ is *already* a pure rotation (no stretching), then $\Sigma = I$ and $M = U V^T$. If $M$ is *almost* a pure rotation but contaminated with some stretching, the closest pure rotation to it is exactly $U V^T$, obtained by computing the SVD and throwing the stretching part away. The SVD is a tool that takes any matrix and extracts the rotation hidden inside.

</section>

<section data-vis="procrustes-svd">

### Solving Procrustes

Here is the recipe. Form the matrix:

$$M = Y^T X$$

This $d \times d$ matrix summarizes how the source and target clouds are correlated direction-by-direction. The $(i, j)$ entry is the sum, over all $n$ word pairs, of the $i$-th coordinate of $y$ times the $j$-th coordinate of $x$. Wherever the target has a strong signal in direction $i$ when the source has a strong signal in direction $j$, $M_{ij}$ picks it up.

If the source and target were related by an exact rotation $R$ (so $y_k = R x_k$ for every pair), then $M = R \, X^T X$. The factor $X^T X$ is symmetric and full of self-correlation. In the SVD of $M$, that self-correlation collapses into the singular values, and the rotation $R$ falls out as the $U V^T$ part.

In practice the embeddings are noisy and the relationship is approximate, so $M$ is almost a rotation but contaminated. The SVD separates the two:

$$M = U \, \Sigma \, V^T$$

The rotations $U$ and $V$ encode the alignment we care about. The singular values in $\Sigma$ encode the noise and the scaling we want to discard. The best orthogonal $W$ is what you get by stripping $\Sigma$ out:

$$W^* = U V^T$$

One factorization, the global optimum. No learning rate, no luck.

### Why this is the answer (an optional walk-through)

If you trust the result, skip ahead. If you want the proof:

The cost we want to minimize is the squared distance between $WX$ and $Y$, summed over all points. With the Frobenius norm (the square root of the sum of squared element-wise entries):

$$\| W X^T - Y^T \|_F^2 = \sum_{i,j} ( W X^T - Y^T )_{ij}^2$$

Expand the square the way you'd expand $(a - b)^2 = a^2 - 2ab + b^2$:

$$\| W X^T - Y^T \|_F^2 = \| W X^T \|_F^2 - 2 \, \text{tr}( W X^T Y ) + \| Y^T \|_F^2$$

The trace appears because the Frobenius dot product of two matrices is $\sum_{i,j} A_{ij} B_{ij} = \text{tr}(A^T B)$.

Two of the three terms do not depend on $W$. The third equals $\| X^T \|_F^2$ because $W$ is orthogonal and orthogonal maps preserve length. So minimizing the cost is the same as *maximizing* the middle term:

$$\max_{W} \; \text{tr}( W^T M ) \quad \text{where } M = Y^T X$$

Substitute the SVD $M = U \Sigma V^T$ and define $Z = V^T W^T U$. Because $W$, $U$, $V$ are orthogonal, so is $Z$. The objective becomes

$$\text{tr}(Z \Sigma) = \sum_i Z_{ii} \, \sigma_i$$

with every $\sigma_i \geq 0$. To make this as large as possible, push every $Z_{ii}$ to 1. Orthogonal matrices have $|Z_{ii}| \leq 1$, and the only orthogonal matrix with all diagonal entries equal to 1 is the identity:

$$Z = I \quad \Longleftrightarrow \quad V^T W^T U = I \quad \Longleftrightarrow \quad W = U V^T$$

There. Provably optimal. No gradient descent could improve it.

</section>

<section data-vis="adversarial">

## 6. The beauty of GANs

Now strip away the dictionary. We have $X$ and $Y$ as two unordered point clouds and no idea which row corresponds to which. We cannot form $M = Y^T X$ because we don't know the pairings. SVD has nothing to chew on.

What we need is a tool that matches *distributions* without point-by-point correspondence. The right tool, invented by <a class="cite" href="https://arxiv.org/abs/1406.2661">Goodfellow et al. in 2014</a>, is the **generative adversarial network**.

A GAN is a beautiful thing. You have two players. A **generator** produces samples from some distribution. A **discriminator** is a small classifier that tries to tell those samples apart from real ones. They train against each other. The discriminator gets better at catching fakes; the generator gets better at fooling it; the discriminator updates again. Equilibrium is reached when the discriminator is at chance, fifty-fifty, no better than guessing.

What makes this beautiful is what the generator never sees. It does not see the real samples directly. It does not see which output it should produce in any given case. All it sees is the gradient of the discriminator's confidence: "outputs that look like this get caught." From that single thin signal, the generator learns to match the entire target distribution.

For our problem, the generator is a single matrix $W$. Given a source embedding $x$, it outputs $Wx$. The discriminator looks at vectors and decides if they came from $\{Wx_i\}$ (transformed source) or from $\{y_j\}$ (real target). They train against each other. Critically, the discriminator never knows which $x$ pairs with which $y$. It only sees distributions in bulk.

The paper adds one stabilizer to keep $W$ close to orthogonal, a soft penalty applied after each gradient step:

$$W \leftarrow (1 + \beta) W - \beta (W W^T) W$$

This pulls $W$ toward the orthogonal manifold without imposing hard constraints that would make gradient flow ugly.

When the discriminator drops to chance, the distributions have collapsed onto each other. To the extent the two clouds genuinely share a shape, the only $W$ that can make them collapse is, up to ambiguities like global reflection, the right rotation.

</section>

<section data-vis="adversarial">

## 7. Why GANs stop short, and why that's the point

In practice, even on language pairs where the shape-isomorphism is strong, the GAN does not finish the job. This is not a flaw of the method. It is the reason the rest of the paper exists.

GAN training is stochastic. The gradients are noisy. The discriminator never quite reaches equilibrium. The soft orthogonality penalty is *just* a penalty, so $W$ drifts off the orthogonal manifold and gets pulled back, never sitting exactly on it. And the discriminator only ever sees bulk distributional similarity; it has no way to tell the generator *which* source point should map to *which* target point.

So adversarial training gets you to a $W$ that places each source cloud roughly where its matching target cloud lives. Close enough that the most confident nearest-neighbor pairs under the current $W$ are usually correct. Not close enough to be a translation matrix.

The animation on the right plays this out. The orange cloud is $WX$. The green cloud is the target $Y$. Thin lines connect each $Wx_i$ to its true $y_i$, showing the residual error. Watch the orange cloud rotate into the right neighborhood, jitter around there, and refuse to settle. Then a single Procrustes step erases the residual instantly. The lines collapse. The clouds coincide.

That contrast, *jitter then snap*, is the architecture of the method.

</section>

<section data-vis="adversarial">

## 8. The synthesis: bootstrap then close the form

Here is the move. After adversarial training, $W$ is wrong in detail but right in spirit. The refinement is a small loop: use $W$ to build a synthetic dictionary, run Procrustes on the dictionary to update $W$, then use the updated $W$ to *rebuild* the dictionary and run Procrustes again. Repeat until the dictionary and $W$ stop moving.

**Step 1: Build the synthetic dictionary.** Take the most frequent $k$ source words (the paper uses ten thousand, on the assumption that frequent words are less ambiguous and less subject to the lexical-mismatch problems of section 3). For each source word $x$, compute $Wx$ and find its nearest neighbor in the target space. Compute the reverse direction too: for each candidate target $y$, find its nearest source neighbor. Keep only the pairs that agree in both directions. This **mutual nearest neighbor** filter discards the noisy matches and keeps the confident ones.

**Step 2: Solve Procrustes on the dictionary.** With the filtered pairs, stack the source vectors as $X$ and the target vectors as $Y$, form $M = Y^T X$, take its SVD, set $W = U V^T$. This is the same closed-form Procrustes solution from section 5, applied to the bootstrapped dictionary instead of a human-provided one.

**Step 3: Iterate.** Go back to step 1 with the new $W$. The cleaner $W$ produces a cleaner dictionary, because more mutual nearest neighbors now agree on the correct pair. The cleaner dictionary in turn produces a cleaner $W$, because Procrustes on better pairs gives a better closed-form answer. After about five passes the procedure converges: the dictionary stops changing, and Procrustes stops moving $W$. The GAN runs *once*, at the start; Procrustes runs *several times* on a dictionary that keeps improving. On language pairs where the geometric prior holds well, the final $W$ matches or exceeds what a fully supervised pipeline would have achieved with the same data.

The structural pattern is worth a name. Adversarial training is a heat-seeking missile: it can find the rough target from far out, but it cannot land precisely. Procrustes is a precision instrument: it can land exactly on a target it can see, but it needs an initial guess clean enough to bootstrap from. Composed together, each compensates for the other's weakness. Stochastic global search hands a synthetic dictionary to a closed-form local solver, which hands a refined $W$ back to the dictionary builder. Convergence is fast because the closed-form step is exact.

*Noisy global discovery followed by exact local refinement.* This pattern recurs all over modern ML, often without anyone noticing it's the same template.

</section>

<section data-vis="csls">

## 9. The hubness problem and the CSLS fix

One last piece, and it's the piece that lifts the system from "good in theory" to "good on the benchmark."

Once $W$ is trained, translation is supposed to be retrieval. For a query word $x$, find the target $y$ with highest $\cos(Wx, y)$. In low dimensions this is fine. In three hundred dimensions it fails in a specific way.

Certain target points become **hubs**: they end up being the nearest neighbor of disproportionately many source points, not because they're the right translation for any of them, but because they sit in a region of space where average cosine similarity to a typical query is unusually high. A handful of hubs hoover up wrong matches and top-1 accuracy collapses.

The paper introduces a calibrated similarity, **Cross-domain Similarity Local Scaling**, that suppresses hubs:

$$\text{CSLS}(x, y) = 2 \cos(W x, y) - r_T(W x) - r_S(y)$$

where $r_T(Wx)$ is the average cosine similarity of $Wx$ to its $K$ nearest neighbors in the target space, and $r_S(y)$ is the analogous quantity for $y$ on the source side. A target that is everyone's average friend (high $r_S$) gets penalized. A query in a dense neighborhood gets calibrated against that density.

Retrieve with CSLS instead of plain cosine. The visual on the right shows the contrast. The left panel uses nearest neighbor and funnels matches onto a hub. The right panel uses CSLS and spreads matches to the right targets.

CSLS isn't a heuristic patch. It corresponds to a principled density normalization, and the paper shows it improves results across every variant tested, supervised or not.

</section>

<section data-vis="closing">

## 10. Why this is still the paper to read

A monolingual corpus in language A. A monolingual corpus in language B. Train embeddings independently. Train a GAN to align the two clouds. Bootstrap a dictionary via mutual nearest neighbors. Solve Procrustes by SVD. Iterate. Retrieve with CSLS.

That is the entire pipeline. On English paired with Spanish, French, German, Italian, it rivals fully supervised baselines. On English-Chinese it falls short of the European-language results but still produces a usable lexicon from no parallel text. The gradient across language pairs is itself a contribution: it shows you, quantitatively, how isomorphic two languages are.

It is tempting in 2026 to wave this off. Multilingual LLMs do translation now. Subword tokenizers ate word-level representations. The frontier is somewhere else. But the lesson of this paper is not really about translation. It is the elegance of how four separate things fit together.

The geometry of embeddings is *just there*, a side effect of learning to predict context. The geometry of two languages is *almost the same*, a side effect of describing the same world. The SVD is a closed-form way to recover any rotation that's been contaminated with stretching. The GAN is a way to match distributions when correspondences are unknown. Each of these four pieces was discovered independently, for unrelated reasons. The paper sets them down next to each other, and they snap into a pipeline.

That kind of result has gotten rarer. There is no enormous model here. There is no compute moat. There are two clean ideas (a GAN, an SVD), a clever density correction, a linguistic prior that is honest about its scope, and a translation system that should have been impossible. Everything in this paper could have been written on a whiteboard.

Read slowly. The math is short. The geometry is gorgeous. The linguistics is doing more work than the linear algebra. And the lesson, that the right linear algebra applied to the right prior can outclass any amount of brute force, is one the field forgets and rediscovers on a five-year cycle.

</section>

<section data-vis="alignment">

## References

- Conneau, Lample, Ranzato, Denoyer, Jégou. *Word Translation Without Parallel Data.* ICLR 2018. [arXiv:1710.04087](https://arxiv.org/pdf/1710.04087)
- Goodfellow et al. *Generative Adversarial Networks.* NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Mikolov, Le, Sutskever. *Exploiting Similarities among Languages for Machine Translation.* 2013. [arXiv:1309.4168](https://arxiv.org/abs/1309.4168)
- Mikolov et al. *Distributed Representations of Words and Phrases.* NeurIPS 2013. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
- Schönemann. *A Generalized Solution of the Orthogonal Procrustes Problem.* Psychometrika 31, 1966.
- Søgaard, Ruder, Vulić. *On the Limitations of Unsupervised Bilingual Dictionary Induction.* ACL 2018. [aclanthology.org/P18-1072](https://aclanthology.org/P18-1072/)
- Winawer, Witthoft, Frank, Wu, Wade, Boroditsky. *Russian blues reveal effects of language on color discrimination.* PNAS 104(19), 2007. [pnas.org/doi/10.1073/pnas.0701644104](https://www.pnas.org/doi/10.1073/pnas.0701644104)

</section>
