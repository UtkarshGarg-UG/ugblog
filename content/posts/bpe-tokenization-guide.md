---
title: "Understanding BPE Tokenization: From Bytes to Tokens"
date: 2025-01-03
draft: true
math: true
summary: "A deep dive into how Byte-Pair Encoding transforms raw text into tokens for language models"
tags: ["NLP", "Tokenization", "BPE", "LLM", "Transformers"]
slug: "bpe-tokenization-guide"
description: "Learn how Byte-Pair Encoding (BPE) works: from Unicode and UTF-8 encoding to training your own tokenizer from scratch."
showToc: true
tocOpen: true
---

**TL;DR**

* **Text needs preprocessing:** Language models don't work directly with characters - they need tokens.
* **Byte-level BPE combines the best of both worlds:** No out-of-vocabulary issues (like character-level) with reasonable sequence lengths (like word-level).
* **BPE is data-driven:** The algorithm learns which byte sequences to merge based on frequency in your training data.
* **Training has three steps:** Initialize with 256 bytes, pre-tokenize text, then iteratively merge the most frequent pairs.
* **Encoding mirrors training:** Apply the same merges in the same order to convert text to token IDs.

*Modern LLMs like GPT-4, Llama, and Claude all use variants of BPE tokenization. Understanding how it works is fundamental to understanding how these models process text.*

---

## The Problem: How Do Language Models Read Text?

Imagine you're building a language model. Your first instinct might be to feed it raw text, character by character. But there's a problem: a 100-word paragraph might be 500+ characters long. Processing such long sequences is computationally expensive and creates long-term dependencies that are hard for models to learn.

"Okay," you think, "let's use words instead!" Now that same paragraph is just 100 tokens. Much better! But what happens when your model encounters "GPT-4" or "antiestablishmentarianism" or "🤖"? Your fixed vocabulary can't handle every possible word, especially rare ones, typos, or new coinages.

**This is the fundamental tension in tokenization:**
- Character-level: handles everything but creates very long sequences
- Word-level: shorter sequences but can't handle rare/new words (the "out-of-vocabulary" problem)

**Byte-Pair Encoding (BPE)** elegantly solves this by operating at the byte level while learning to merge frequent byte sequences into larger units. The result? You get word-like tokens for common words, but can still represent *any* text as a sequence of bytes.

---

## Foundation: Understanding Unicode and UTF-8

Before we dive into BPE, we need to understand how computers represent text.

### Unicode: The Universal Character Map

Unicode is a standard that assigns a unique number (called a "code point") to every character across all writing systems. As of Unicode 16.0 (September 2024), there are 154,998 characters across 168 scripts.

```python
>>> ord('s')  # Get Unicode code point
115
>>> ord('牛')  # Works for any character!
29275
>>> chr(29275)  # Convert back
'牛'
```

Unicode code points are typically written as `U+0073` (for 's') where `U+` is a prefix and `0073` is the hexadecimal representation of 115.

### Why Not Train Directly on Unicode Code Points?

You might wonder: why not just use these code points as our vocabulary? The problem is **size and sparsity**:
- 154,998+ possible characters is a huge vocabulary
- Most characters are extremely rare (ancient scripts, mathematical symbols, etc.)
- The vocabulary would be unnecessarily large and sparse

### Enter UTF-8 Encoding

Instead of using code points directly, we use **UTF-8 encoding**, which converts each Unicode character into a sequence of bytes (integers from 0-255). This gives us a manageable base vocabulary of just 256 items.

```python
>>> text = "hello! こんにちは!"
>>> encoded = text.encode("utf-8")
>>> print(encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'

>>> list(encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147,
 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
```

Notice:
- ASCII characters like 'h', 'e', 'l', 'l', 'o' map to single bytes (104, 101, 108, 108, 111)
- Japanese characters require multiple bytes (e.g., 'こ' becomes [227, 129, 147])
- The 13-character string becomes 23 bytes

**Important insight:** One Unicode character doesn't always equal one byte! UTF-8 is a variable-length encoding.

### Why UTF-8 Specifically?

UTF-8 is the dominant encoding for the internet (>98% of web pages). Other options like UTF-16 and UTF-32 exist, but:
- UTF-8 is more compact for English/Latin text (1 byte per character vs. 2-4 bytes)
- UTF-8 is backward compatible with ASCII
- UTF-8 is already what most training data uses

**Bottom line:** By using UTF-8 encoding, we can represent *any* text with just 256 base tokens (0-255), ensuring we never have out-of-vocabulary issues.

---

## The Subword Tokenization Sweet Spot

Pure byte-level tokenization solves the vocabulary problem but creates a different issue: **extremely long sequences**.

Consider this sentence: "The cat sat on the mat"
- **Word-level:** 6 tokens ✓ Short sequence ✗ Fixed vocabulary
- **Byte-level:** ~22 tokens ✗ Long sequence ✓ No OOV issues
- **BPE:** ~8-10 tokens ✓ Balanced! ✓ No OOV issues

**BPE is a compression algorithm** that learns to merge frequent byte sequences into single tokens. If the byte sequence `b'the'` appears frequently in your training data, BPE will assign it a single token ID, compressing those 3 bytes into 1 token.

The key advantages:
1. **Common words become single tokens** (compression)
2. **Rare words break into subwords** (still representable)
3. **Any text is representable** (fall back to bytes)

---

## How BPE Training Works

Training a BPE tokenizer has three main steps: vocabulary initialization, pre-tokenization, and merge computation.

### Step 1: Vocabulary Initialization

Start with the 256 possible byte values as your initial vocabulary:

```python
vocab = {i: bytes([i]) for i in range(256)}
# {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
```

Plus any special tokens you want (like `<|endoftext|>` for document boundaries):

```python
vocab[256] = b'<|endoftext|>'
```

### Step 2: Pre-tokenization

Before we start merging, we need to split the corpus into "pre-tokens" - rough chunks that we'll process independently. This serves two purposes:

1. **Efficiency:** Instead of scanning the entire corpus for each merge, we can work with pre-token frequencies
2. **Preventing unwanted merges:** We don't want to merge across natural boundaries (like punctuation)

Modern BPE (like GPT-2) uses a regex pattern for pre-tokenization:

```python
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = "some text that i'll pre-tokenize"
pre_tokens = re.findall(PAT, text)
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

This regex:
- Keeps common contractions together (`'ll`, `'ve`, `'re`)
- Preserves leading spaces on words
- Separates punctuation
- Handles numbers

After pre-tokenization, convert each pre-token to UTF-8 bytes:

```python
# If 'low' appears 5 times in corpus
pretokens = {(108, 111, 119): 5}  # b'low': 5
```

### Step 3: Compute BPE Merges

Now comes the core algorithm. Iteratively:

1. **Count all adjacent byte pairs** across all pre-tokens
2. **Find the most frequent pair** (break ties lexicographically)
3. **Merge that pair** into a new token
4. **Update all pre-tokens** with this merge
5. **Repeat** until you reach your desired vocabulary size

**Example:** Let's walk through a simplified example from the assignment.

**Corpus:**
```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

**Pre-tokenization** (simple whitespace splitting):
```python
{
    ('l','o','w'): 5,
    ('l','o','w','e','r'): 2,
    ('w','i','d','e','s','t'): 3,
    ('n','e','w','e','s','t'): 6
}
```

**Iteration 1:** Count all byte pairs:
```python
{
    ('l','o'): 7,   # from 'low'(5) + 'lower'(2)
    ('o','w'): 7,   # from 'low'(5) + 'lower'(2)
    ('w','e'): 8,   # from 'lower'(2) + 'newest'(6)
    ('e','r'): 2,
    ('w','i'): 3,
    ('i','d'): 3,
    ('d','e'): 3,
    ('e','s'): 9,   # from 'widest'(3) + 'newest'(6)
    ('s','t'): 9,   # from 'widest'(3) + 'newest'(6)
    ('n','e'): 6,
    ('e','w'): 6
}
```

Most frequent pairs: `('e','s')` and `('s','t')` both have count 9. Tie-break lexicographically → `('s','t')` wins!

**Merge** `('s','t')` → new token `st` (ID 257):
```python
{
    ('l','o','w'): 5,
    ('l','o','w','e','r'): 2,
    ('w','i','d','e','st'): 3,      # 't' was merged!
    ('n','e','w','e','st'): 6       # 't' was merged!
}
```

**Iteration 2:** Count pairs again:
```python
{
    ('l','o'): 7,
    ('o','w'): 7,
    ('w','e'): 8,
    ('e','r'): 2,
    ('w','i'): 3,
    ('i','d'): 3,
    ('d','e'): 3,
    ('e','st'): 9,  # ← 'st' is now a single token!
    ('n','e'): 6,
    ('e','w'): 6
}
```

Most frequent: `('e','st')` → merge to token `est` (ID 258)

Continue until you reach your target vocabulary size (e.g., 10,000 or 32,000).

**Final result:**
- **Vocabulary:** Mapping from token IDs to byte sequences
- **Merges:** Ordered list of (token1, token2) pairs that were merged

```python
vocab = {
    0: b'\x00', ..., 255: b'\xff',  # Original bytes
    256: b'<|endoftext|>',           # Special token
    257: b'st',                       # First merge
    258: b'est',                      # Second merge
    259: b'ow',
    260: b'low',
    261: b'west',
    262: b'ne',
    # ... continues
}

merges = [
    (b's', b't'),        # Merge 1
    (b'e', b'st'),       # Merge 2
    (b'o', b'w'),        # Merge 3
    (b'l', b'ow'),       # Merge 4
    (b'w', b'est'),      # Merge 5
    (b'n', b'e'),        # Merge 6
    # ... continues
]
```

### Key Implementation Details

**Tie-breaking:** When multiple pairs have the same frequency, choose the lexicographically greater pair:

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')  # Lexicographically greatest
```

**Efficiency:** The naive algorithm is slow because you rescan the entire corpus for each merge. Optimizations:
- Cache pair counts and update incrementally (only pairs adjacent to the merged pair change)
- Use priority queues/heaps for finding max
- Parallelize pre-tokenization (but merge step is inherently sequential)

**Special tokens:** Never split special tokens! Before pre-tokenization:

```python
# Split corpus on special tokens first
parts = text.split('<|endoftext|>')
# Then pre-tokenize each part independently
```

---

## How BPE Encoding Works

Once you've trained your BPE tokenizer, encoding new text mirrors the training process:

### Encoding Algorithm

**Input:** Raw text string
**Output:** List of token IDs

**Steps:**

1. **Encode to UTF-8 bytes**
   ```python
   text = "the cat ate"
   bytes_seq = text.encode("utf-8")
   # b'the cat ate'
   ```

2. **Pre-tokenize** using the same regex pattern
   ```python
   pre_tokens = ['the', ' cat', ' ate']
   ```

3. **For each pre-token:** Apply merges in the same order they were learned

   Start with: `['the']` → bytes `[116, 104, 101]` (t, h, e)

   Apply merge 1: `(116, 104)` → token 257 (if that was learned)
   Result: `[257, 101]`

   Apply merge 2: `(257, 101)` → token 258 (if `the` was learned)
   Result: `[258]`

4. **Convert to token IDs** and concatenate

### Example: Encoding "the cat ate"

Given vocabulary:
```python
{
    0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
    6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'
}
```

And merges:
```python
[
    (b't', b'h'),    # Creates token 6
    (b' ', b'c'),    # Creates token 7
    (b' ', b'a'),    # Creates token 8
    (b'th', b'e'),   # Creates token 9
    (b' a', b't')    # Creates token 10
]
```

**Encoding 'the':**
- Start: `[b't', b'h', b'e']` → `[5, 4, 3]`
- Apply merge 1: `(b't', b'h')` → `[b'th', b'e']` → `[6, 3]`
- Apply merge 4: `(b'th', b'e')` → `[b'the']` → `[9]`

**Encoding ' cat':**
- Start: `[b' ', b'c', b'a', b't']` → `[0, 2, 1, 5]`
- Apply merge 2: `(b' ', b'c')` → `[b' c', b'a', b't']` → `[7, 1, 5]`
- No more applicable merges → `[7, 1, 5]`

**Encoding ' ate':**
- Start: `[b' ', b'a', b't', b'e']` → `[0, 1, 5, 3]`
- Apply merge 3: `(b' ', b'a')` → `[b' a', b't', b'e']` → `[8, 5, 3]`
- Apply merge 5: `(b' a', b't')` → `[b' at', b'e']` → `[10, 3]`

**Final result:** `[9, 7, 1, 5, 10, 3]`

---

## How BPE Decoding Works

Decoding is straightforward: just reverse the process.

**Input:** List of token IDs
**Output:** UTF-8 string

```python
def decode(token_ids, vocab):
    # 1. Look up each token ID in vocabulary
    byte_sequences = [vocab[id] for id in token_ids]

    # 2. Concatenate all byte sequences
    full_bytes = b''.join(byte_sequences)

    # 3. Decode UTF-8 to string
    return full_bytes.decode('utf-8', errors='replace')
```

The `errors='replace'` parameter handles invalid UTF-8 sequences (e.g., if someone passes arbitrary token IDs) by replacing them with the Unicode replacement character `�` (U+FFFD).

**Example:**
```python
token_ids = [9, 7, 1, 5, 10, 3]  # "the cat ate"
vocab = {9: b'the', 7: b' c', 1: b'a', 5: b't', 10: b' at', 3: b'e'}

bytes_seq = b'the' + b' c' + b'a' + b't' + b' at' + b'e'
         = b'the cat ate'

text = bytes_seq.decode('utf-8')
     = "the cat ate"
```

---

## Practical Considerations

### Vocabulary Size

Typical vocabulary sizes:
- **Small models (GPT-2):** ~50K tokens
- **Modern LLMs:** 32K-100K+ tokens

Larger vocabularies:
- ✓ Better compression (shorter sequences)
- ✓ More efficient inference
- ✗ Larger embedding matrices
- ✗ More training data needed to learn good representations

### Pre-tokenization Patterns

Different BPE implementations use different pre-tokenization strategies:

**Original BPE (Sennrich et al., 2016):** Simple whitespace splitting
```python
text.split(' ')
```

**GPT-2 pattern (tiktoken):** Complex regex preserving spaces and contractions
```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

The pattern significantly affects what gets merged:
- Preserving leading spaces: `" the"` vs `"the"`
- Keeping contractions whole: `"don't"` vs `"don"` + `"'t"`

### Training Data Matters

BPE tokenizers are **data-driven** - they learn what to merge based on frequency. This means:

1. **Train on representative data:** A tokenizer trained on children's stories (TinyStories) will have different merges than one trained on web text (OpenWebText)

2. **Domain mismatch hurts:** Using a TinyStories tokenizer on technical documentation will give poor compression (rare technical terms split into many tokens)

3. **Common tokens reflect training data:** GPT-2's tokenizer has single tokens for common Python keywords because it was trained on code-heavy web data

### Memory-Efficient Loading

For large datasets that don't fit in RAM, use memory-mapping:

```python
import numpy as np

# Don't load entire array into memory
data = np.load('tokens.npy', mmap_mode='r')

# Only loads accessed portions
batch = data[start:end]  # Loads only this slice
```

---

## Real-World Example: Training on TinyStories

Let's look at realistic parameters for training a BPE tokenizer:

**Configuration:**
```python
vocab_size = 10_000  # 256 bytes + ~9,744 learned merges
special_tokens = ['<|endoftext|>']
```

**Dataset:** TinyStories (2.12M children's stories)

**Results after training:**
- Vocabulary contains common words: `b'the'`, `b' and'`, `b'Once'`
- Child-specific terms: `b'princess'`, `b'dinosaur'`, `b'friend'`
- Longest token might be something like `b' beautiful'` or `b'interesting'`

**Compression ratio:** ~3-4 bytes per token (compared to ~1 byte for pure byte-level)

**Sample tokenization:**
```python
text = "Once upon a time there was a little girl"
# Might tokenize as:
# ['Once', ' upon', ' a', ' time', ' there', ' was', ' a', ' little', ' girl']
# 9 tokens vs ~40+ bytes
```

---

## Common Pitfalls and Debugging

### Issue 1: Inconsistent Pre-tokenization

**Problem:** Training and inference use different pre-tokenization

**Symptom:** Model generates weird text, high perplexity

**Solution:** Save pre-tokenization pattern with your tokenizer

```python
# Save alongside vocab and merges
config = {
    'pretok_pattern': PAT,
    'special_tokens': ['<|endoftext|>'],
    'vocab_size': 10000
}
```

### Issue 2: Special Token Splitting

**Problem:** Special tokens get split during encoding

**Symptom:** `<|endoftext|>` becomes `['<', '|', 'endo', 'ft', 'ext', '|', '>']`

**Solution:** Handle special tokens before pre-tokenization

```python
# Split on special tokens first
parts = re.split('(<\|endoftext\|>)', text)
# Process each part, keeping special tokens whole
```

### Issue 3: UTF-8 Encoding Errors

**Problem:** Trying to decode bytes that aren't valid UTF-8

```python
# This can happen with arbitrary token sequences
bytes_seq = b'\xff\xfe'  # Invalid UTF-8
bytes_seq.decode('utf-8')  # Raises UnicodeDecodeError
```

**Solution:** Use `errors='replace'` for robustness

```python
bytes_seq.decode('utf-8', errors='replace')  # Returns '��'
```

### Issue 4: Poor Compression on New Domains

**Problem:** Tokenizer trained on English struggles with code/math

**Example:**
```python
# Tokenizer trained on stories
code = "def fibonacci(n):"

# Poor tokenization
# ['d', 'ef', ' ', 'f', 'ib', 'on', 'acci', '(', 'n', '):', ...]
# vs. ideal: ['def', ' fibonacci', '(', 'n', '):', ...]
```

**Solution:** Train domain-specific tokenizers or use larger, multilingual corpora

---

## BPE Variants and Modern Extensions

While the core BPE algorithm is widely used, modern LLMs often use variants:

### WordPiece (BERT)

Instead of merging based on frequency, WordPiece chooses merges that maximize the likelihood of the training data. Similar results but different optimization.

### Unigram (SentencePiece)

Starts with a large vocabulary and iteratively removes tokens. Allows for probabilistic tokenization (multiple valid tokenizations for the same text).

### BPE Extensions

- **tiktoken (OpenAI):** Optimized Rust implementation with special handling for code
- **SentencePiece:** Language-agnostic implementation (treats spaces as regular characters)
- **Byte-level BPE:** What we've described - operates directly on UTF-8 bytes

---

## Conclusion

BPE tokenization elegantly solves the fundamental tension in text preprocessing:
- Start with a minimal vocabulary (256 bytes) that can represent anything
- Learn data-driven merges to compress common sequences
- Get word-like tokens for common words, subwords for rare ones
- Never face out-of-vocabulary issues

**Key takeaways:**

1. **BPE is a compression algorithm** that learns from data frequency
2. **Pre-tokenization matters** - it defines merge boundaries
3. **Training order is crucial** - encoding must apply merges in the same order
4. **UTF-8 encoding ensures universality** - any text can be represented
5. **Vocabulary size is a hyperparameter** - balance compression vs. model capacity

When you see a language model with a "50K vocabulary," you now know:
- 256 of those are base UTF-8 bytes
- ~49,744 were learned by merging frequent pairs
- The model can still represent any text, even with "out-of-vocabulary" words

Understanding BPE is foundational to working with modern LLMs. Every time GPT-4 reads your prompt or Llama generates a response, BPE tokenization is happening behind the scenes, converting human-readable text into the token sequences these models actually process.

---

## References

1. Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Neural machine translation of rare words with subword units." *Proceedings of ACL*, 2016.

2. Wang, Changhan, Kyunghyun Cho, and Jiatao Gu. "Neural machine translation with byte-level subwords." *arXiv:1909.03341*, 2019.

3. Radford, Alec, et al. "Language models are unsupervised multitask learners." *OpenAI Blog*, 2019.

4. Gage, Philip. "A new algorithm for data compression." *C Users Journal* 12.2 (1994): 23-38.

---

*This post is based on the CS336 (Spring 2025) assignment on building Transformers from scratch. For hands-on practice, check out the full assignment at [github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics).*
