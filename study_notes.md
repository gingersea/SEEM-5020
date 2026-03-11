# SEEM5020 – Algorithms for Big Data: Complete Study Notes

> **Course:** SEEM5020 Algorithms for Big Data, Spring 2026, CUHK  
> **Instructor:** Sibo WANG  
> These notes cover every key concept from Lectures 1–6, with definitions, code, and examples. Nothing is skipped.

---

## Table of Contents

- [Lecture 1: Review of Probability Concepts & Concentration Bounds](#lecture-1-review-of-probability-concepts--concentration-bounds)
- [Lecture 2: Streaming Algorithms (I)](#lecture-2-streaming-algorithms-i)
- [Lecture 3: Streaming Algorithms (II)](#lecture-3-streaming-algorithms-ii)
- [Lecture 4: Nearest Neighbor Search — Locality Sensitive Hashing](#lecture-4-nearest-neighbor-search--locality-sensitive-hashing)
- [Lecture 5: Dimension Reduction — Johnson-Lindenstrauss Lemma](#lecture-5-dimension-reduction--johnson-lindenstrauss-lemma)
- [Lecture 6: Dimension Reduction — PCA and SVD](#lecture-6-dimension-reduction--pca-and-svd)

---

# Lecture 1: Review of Probability Concepts & Concentration Bounds

---

## 1.1 Basic Concepts of Discrete Probability

### Definition 1 — Sample Space
The **sample space** Ω is the set of **all possible outcomes** of an experiment. Individual outcomes are called **elementary events**.

### Definition 2 — Events
An **event** is a **subset of the sample space** Ω.
- Ω itself is the *certain event*; ∅ is the *null event*.
- Two events A and B are **disjoint (mutually exclusive)** if A ∩ B = ∅.

> **Example 1:** Flip two coins. Ω = {HH, HT, TH, TT}.  
> Event A = {HH, HT} and B = {TH, TT} are disjoint.

### Probability Axioms
A probability distribution P[ ] on Ω maps events to [0,1] and satisfies:
1. **Non-negativity:** P[A] ≥ 0 for any A ⊆ Ω.
2. **Unitarity:** P[Ω] = 1.
3. **Additivity:** If A and B are disjoint, P[A ∪ B] = P[A] + P[B] (extends to arbitrarily many pairwise disjoint events).

**Fact:** For any A and B:  
`P[A ∪ B] = P[A] + P[B] − P[A ∩ B]`

### Theorem 1 — Union Bound
For events A₁, A₂, …, Aₖ:  
`P[A₁ ∪ A₂ ∪ … ∪ Aₖ] ≤ P[A₁] + P[A₂] + … + P[Aₖ]`

### Definition 3 — Independence
Events A and B are **independent** if and only if P[A ∩ B] = P[A] · P[B].  
More generally, events A₁, …, Aₖ are **mutually independent** if for every subset I ⊆ {1,…,k}:  
`P[∩_{i∈I} Aᵢ] = ∏_{i∈I} P[Aᵢ]`

```python
# Example: verify independence numerically
p_A = 0.5   # P[head on coin 1]
p_B = 0.5   # P[head on coin 2]
p_AB = 0.25  # P[both heads]
print("Independent:", abs(p_AB - p_A * p_B) < 1e-9)  # True
```

---

## 1.2 Conditional Probability

### Definition 4 — Conditional Probability
`P[A | B] = P[A ∩ B] / P[B]`

> **Example 2:** Six-sided die. A = {1,3,5}, B = {1,2,3,4}.  
> P[A|B] = P[A∩B]/P[B] = P[{1,3}] / P[{1,2,3,4}] = (2/6)/(4/6) = **1/2**.

### Lemma 1 — Multiplication Rule
`P[∩_{i=1}^k Aᵢ] = P[A₁] · P[A₂|A₁] · P[A₃|A₁∩A₂] · … · P[Aₖ|A₁∩…∩Aₖ₋₁]`

> **Example 3:** 3 cards drawn without replacement from a 52-card deck.  
> P[none is heart] = (39/52) × (38/51) × (37/50).

### Theorem 2 — Law of Total Probability
If A₁, …, Aₖ are disjoint and ∪Aᵢ = Ω, then:  
`P[B] = Σᵢ P[B|Aᵢ] · P[Aᵢ]`

> **Example 4:** 3 bags, each with 8 balls and 3, 5, 2 green balls respectively.  
> P[green ball] = (1/3)(3/8) + (1/3)(5/8) + (1/3)(2/8) = (3+5+2)/24 = 10/24 = **5/12**.

### Theorem 3 — Bayes' Theorem
`P[Aⱼ | B] = P[B|Aⱼ]·P[Aⱼ] / Σᵢ P[B|Aᵢ]·P[Aᵢ]`

> **Example 5 (Drug test):** Test has 99% true positive and 99% true negative. 0.5% are drug users.  
> P[user | positive] = (0.99 × 0.005) / (0.99×0.005 + 0.01×0.995) ≈ **33%** (surprisingly low!).

```python
p_user = 0.005
p_pos_given_user = 0.99
p_pos_given_non = 0.01

p_pos = p_pos_given_user * p_user + p_pos_given_non * (1 - p_user)
p_user_given_pos = (p_pos_given_user * p_user) / p_pos
print(f"P[user|positive] = {p_user_given_pos:.4f}")  # ≈ 0.3322
```

---

## 1.3 Random Variables

A **random variable** X maps each elementary event s ∈ Ω to a real value. For discrete X:  
`P[X = a] = Σ_{s: X(s)=a} P[s]`

> **Example 6:** X = sum of two dice rolls.  
> P[X=4] = |{(1,3),(2,2),(3,1)}| / 36 = 3/36 = **1/12**.

---

## 1.4 Expectation

### Definition 5 — Expectation (Discrete)
`E[X] = Σᵢ i · P[X = i]`

### Theorem 4 — Linearity of Expectations
`E[X₁ + X₂ + … + Xₖ] = E[X₁] + E[X₂] + … + E[Xₖ]`
*(holds even for dependent variables!)*

> **Example 7 — Empty bins:** n balls, n bins, uniform random throws.  
> Let Xᵢ = 1 if bin i is empty. E[Xᵢ] = ((n−1)/n)^n.  
> Expected empty bins = n · ((n−1)/n)^n ≈ n/e ≈ 0.368n.

### Theorem 5 — Jensen's Inequality
If f is **convex** (f''(x) ≥ 0): `E[f(X)] ≥ f(E[X])`

> **Example 8:** E[X²] ≥ (E[X])² (since f(x)=x² is convex).

---

## 1.5 Conditional Expectation

### Definition 6
`E[Y | Z = z] = Σᵧ y · P[Y = y | Z = z]`

### Lemma 2 — Total Expectation (discrete)
`E[X] = Σᵧ P[Y = y] · E[X | Y = y]`

### Theorem 6 — Law of Total Expectation
`E[Y] = E[E[Y|Z]]`

---

## 1.6 Variance, Covariance, Moments

### Definition 8 — Variance
`Var[X] = E[(X − E[X])²] = E[X²] − (E[X])²`

Standard deviation: `σ[X] = √Var[X]`

### Definition 9 — Moment
The **k-th moment** of X: `E[Xᵏ]`

### Definition 10 — Covariance
`Cov(X, Y) = E[(X − E[X])(Y − E[Y])]`

### Theorem 7 — Variance of Independent Variables
If X and Y are independent: Cov(X,Y) = 0 and `Var[X + Y] = Var[X] + Var[Y]`

```python
import numpy as np

np.random.seed(42)
X = np.random.binomial(1, 0.5, 10000)  # fair coin flips
Y = np.random.binomial(1, 0.5, 10000)  # independent fair coin
print(f"Var[X+Y] = {np.var(X+Y):.4f}")
print(f"Var[X]+Var[Y] = {np.var(X)+np.var(Y):.4f}")  # approximately equal
```

---

## 1.7 Continuous Random Variables

### Definition 11 — Probability Density Function (PDF)
X is **continuous** if described by a non-negative PDF f_X such that:  
`P[X ∈ B] = ∫_B f_X(x) dx`  
`P[a ≤ X ≤ b] = ∫_a^b f_X(x) dx`

### Definition 12 — Expectation (Continuous)
`E[X] = ∫_{-∞}^{+∞} x · f_X(x) dx`

> **Example 13:** X uniform on [0,2] with f_X(x) = 3x²/8.  
> E[X] = ∫₀² x · (3x²/8) dx = (3/8) · [x⁴/4]₀² = (3/8)(4) = **3/2**.

---

## 1.8 Concentration Bounds

These are tools to bound how far a random variable deviates from its mean.

### Theorem 8 — Markov's Inequality
For non-negative X and any a > 0:  
`P[X ≥ a] ≤ E[X] / a`

### Theorem 9 — Chebyshev's Inequality
For any a > 0:  
`P[|X − E[X]| ≥ a] ≤ Var[X] / a²`

> **Exercise:** n fair coin flips. P[#heads ≥ 3n/4]:  
> By Markov: P[X ≥ 3n/4] ≤ (n/2)/(3n/4) = **2/3**.  
> By Chebyshev: Var[X] = n/4; P[|X−n/2| ≥ n/4] ≤ (n/4)/(n/4)² = **4/n**.

---

## 1.9 Moment Generating Functions (MGF)

### Definition 13 — MGF
`M_X(t) = E[e^{tX}]`

### Theorem 10 — Properties
1. `E[Xⁿ] = M_X^{(n)}(0)` (n-th derivative at 0)
2. If X and Y are independent: `M_{X+Y}(t) = M_X(t) · M_Y(t)`

---

## 1.10 Chernoff Bound

For independent Bernoulli(pᵢ) random variables X₁,…,Xₙ, let X = ΣXᵢ, μ = Σpᵢ.

**Key step:** M_{Xᵢ}(t) = 1 + pᵢ(eᵗ−1) ≤ e^{pᵢ(eᵗ−1)}  
→ M_X(t) ≤ e^{μ(eᵗ−1)}

### Theorem 11 — Chernoff Upper Tail
For δ > 0:  
`P[X ≥ (1+δ)μ] ≤ (eᵟ / (1+δ)^{1+δ})^μ`  
For 0 < δ ≤ 1:  
`P[X ≥ (1+δ)μ] ≤ e^{−μδ²/3}`

### Theorem 12 — Chernoff Lower Tail
For 0 < δ < 1:  
`P[X ≤ (1−δ)μ] ≤ (e^{−δ} / (1−δ)^{1−δ})^μ`  
`P[X ≤ (1−δ)μ] ≤ e^{−μδ²/2}`

> **Example (fair coin flips):** P[X ≥ 3n/4] with δ=1/2, μ=n/2:  
> ≤ e^{−(n/2)(1/4)/3} = e^{−n/24}. **Exponentially small!**

```python
import math

def chernoff_upper(mu, delta):
    """P[X >= (1+delta)*mu] <= e^(-mu*delta^2/3)  for 0 < delta <= 1"""
    return math.exp(-mu * delta**2 / 3)

n = 1000
mu = n / 2    # expected heads for n fair coins
delta = 0.5   # want P[X >= 750]
print(f"Chernoff upper bound: {chernoff_upper(mu, delta):.6f}")
```

---

# Lecture 2: Streaming Algorithms (I)

---

## 2.1 Data Stream Model

### Definition 1 — Data Stream Model
A sequence of n integers drawn from domain [m] = {1,2,…,m}.
- **Not random-accessible** — arrives element by element.
- At most a **small number of passes** (usually just one).
- Space limited to **O(log m)** or **O(polylog(m, n))** bits.

**Real applications:**
- Query/click streams (Google, Wikipedia)
- Social media trending topics (Twitter, TikTok)
- Network packet monitoring (DDoS detection)
- Sensor data anomaly detection

---

## 2.2 Sampling from Data Streams

### Problem 1 — Uniform Sampling
Sample k elements uniformly at random from a stream of unknown size n.

### Algorithm 1 — Reservoir Sampling
```
Initialize array A[1..k] with the first k elements.
For i = k+1, k+2, ...:
    Draw r uniformly from {1, ..., i}
    If r <= k:
        A[r] = a_i   (replace a randomly chosen position)
Return A
```
Space: O(k · log m) bits.

```python
import random

def reservoir_sampling(stream, k):
    """Sample k elements uniformly at random from a stream."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            r = random.randint(0, i)
            if r < k:
                reservoir[r] = item
    return reservoir

# Example
stream = list(range(1, 101))  # elements 1..100
sample = reservoir_sampling(stream, 5)
print("Reservoir sample:", sample)
```

### Theorem 1 — Correctness of Reservoir Sampling
Algorithm 1 returns each element with probability **k/n**.

**Proof (induction):** Base case n=k trivial. Inductive step: new element e_{n+1} enters with prob k/(n+1). Each existing element eᵢ stays with prob:  
`P[eᵢ ∈ A_{n+1}] = (k/n) · (n/(n+1)) = k/(n+1)`. ✓

### Fact 1 — Weighted Reservoir Sampling
Reservoir sampling is equivalent to k rounds of **sampling without replacement**.

### Definition 2 — Weighted Reservoir Sampling
Each element eᵢ has weight wᵢ. We want k rounds of **weighted sampling without replacement**.

### Algorithm 2 — A-Res Algorithm (Weighted)
```
Initialize a min-priority queue R (size k).
For each element (eᵢ, wᵢ):
    Draw Uᵢ ~ Uniform(0,1)
    Compute key Kᵢ = Uᵢ^(1/wᵢ)
    If i <= k: insert (Kᵢ, eᵢ) into R
    Else if Kᵢ > K_min:
        Remove element with K_min from R
        Insert (Kᵢ, eᵢ) into R
Return elements in R
```

```python
import heapq
import random
import math

def weighted_reservoir_sampling(items_weights, k):
    """A-Res algorithm: weighted sampling without replacement."""
    heap = []  # min-heap of (key, element)
    for item, weight in items_weights:
        u = random.random()
        key = u ** (1.0 / weight)
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))
    return [item for _, item in heap]

# Example: 5 items with different weights
items = [("A", 1), ("B", 5), ("C", 2), ("D", 3), ("E", 10)]
sample = weighted_reservoir_sampling(items, 2)
print("Weighted sample:", sample)  # B, E most likely
```

**Theorem 2:** Each element eᵢ has the largest key with probability **wᵢ / W**, where W = Σwⱼ.

---

## 2.3 Approximate Counting

### Problem 2 — Counting Problem
Count the number of events in a stream using **sub-linear space**.

Exact counting requires O(log n) bits (optimal). Can we do better with approximation?

### Definition 3 — (ε, δ)-approximation
Estimator μ̂ is an **(ε,δ)-approximation** of μ if:  
`P[|μ − μ̂| > ε·μ] ≤ δ`

### Algorithm 3 — Morris Algorithm
```
Initialize counter X = 0.
For each event:
    Increment X with probability 1/2^X
Return n̂ = 2^X − 1
```

```python
import random
import math

def morris_count(n):
    """Morris approximate counting algorithm."""
    X = 0
    for _ in range(n):
        if random.random() < 1.0 / (2 ** X):
            X += 1
    return 2**X - 1  # estimated count

# Test
n = 1000
estimates = [morris_count(n) for _ in range(1000)]
mean_est = sum(estimates) / len(estimates)
print(f"True count: {n}, Mean estimate: {mean_est:.1f}")
```

**Lemma 1:** `E[2^{X_n}] = n + 1`  
→ `E[n̂] = E[2^{X_n}] − 1 = n` (unbiased estimator)

**Lemma 2:** `E[2^{2X_n}] = (3n² + 3n + 2) / 2`  
→ `Var[n̂] = n(n−1)/2`

By Chebyshev: `P[|n̂−n| > ε·n] ≤ 1/(2ε²)` — too loose for small ε.

### Morris+ (Multiple Trials)
Take t independent Morris trials, average the results:  
`Var[average] = n(n−1)/(2t)` → Setting t = 3/(2ε²) gives δ = 1/3.

### Morris++ (Median Trick)
Run Morris+ s = 48·log(1/δ) times, take the **median**.  
Failure probability ≤ δ. Reduces dependency from O(1/δ) to **O(log(1/δ))**.

```python
import random
import statistics

def morris_plus(n, epsilon=0.1, delta=0.1):
    """Morris++ with mean-then-median trick."""
    t = int(3 / (2 * epsilon**2)) + 1
    s = int(48 * math.log(1.0 / delta)) + 1

    def morris_plus_single():
        estimates = [morris_count(n) for _ in range(t)]
        return sum(estimates) / t

    medians = [morris_plus_single() for _ in range(s)]
    return statistics.median(medians)

print(f"Morris++ estimate: {morris_plus(1000):.1f}")
```

**Space:** O(log(1/δ) · loglog n / ε²) bits — much better than O(log n)!

---

## 2.4 Distinct Element Counting

### Definition 4 — Distinct Element Counting (DEC)
Given a stream from [m], count the number of **distinct elements** (NDE).

> **Example 1:** Stream = 1,2,2,1,5,4,2,2,1 → NDE = **4** (elements 1,2,4,5).

Naïve: O(m) or O(NDE·log m) bits — both expensive.

---

### Algorithm 4 — MinHash Algorithm
Uses an idealized hash h: [n] → [0,1].
```
Initialize X_min = 1.
For each element eᵢ:
    Compute h(eᵢ)
    Update X_min = min(X_min, h(eᵢ))
Return 1/X_min − 1   ← estimate of NDE
```

**Lemma 4:** `E[X_min] = 1 / (NDE + 1)`  
So `E[1/X_min − 1] = NDE`. Unbiased!

**Variance:** `Var[X_min] = NDE / ((NDE+1)²(NDE+2))`

**MinHash+:** Apply s = 3/ε² independent hash functions, average:  
Space: O(log(1/δ)/ε²) bits.

```python
import random

def minhash_nde_estimate(stream, num_hashes=100):
    """Estimate distinct elements using MinHash."""
    distinct = set(stream)
    nde_true = len(distinct)

    min_vals = []
    for _ in range(num_hashes):
        # simulate random hash h: element -> [0,1]
        hashes = {e: random.random() for e in range(max(stream)+1)}
        x_min = min(hashes[e] for e in stream)
        min_vals.append(1.0/x_min - 1)

    estimate = sum(min_vals) / len(min_vals)
    return estimate, nde_true

stream = [1,2,2,1,5,4,2,2,1,3,3,5,6]
est, true = minhash_nde_estimate(stream)
print(f"True NDE: {true}, Estimated: {est:.1f}")
```

---

### Algorithm — Bottom-k Sketch
Maintain the **k smallest hash values** instead of just the minimum.

1. Set range b = n³ (avoid collisions with prob ≥ 1−1/n).
2. Maintain set Sₖ of k smallest hash values; track k-th smallest h_{kth}.
3. Return estimate: `b·k / h_{kth}`

### Definition 5 — k-wise Independent Hash Family
H mapping [a]→[b] is k-wise independent if for any j₁,…,jₖ ∈ [b] and distinct i₁,…,iₖ ∈ [a]:  
`P_{h∈H}[h(i₁)=j₁ ∧ … ∧ h(iₖ)=jₖ] = 1/bᵏ`

> **Example 2:** For prime P > b, choose a₀,…,aₖ₋₁ ∈ [P].  
> `H(v) = (a₀ + a₁v + … + aₖ₋₁vᵏ⁻¹) mod P mod b`  
> is a k-wise independent hash function.

**Analysis:** Setting k = ⌈12/ε²⌉: `P[(1−ε)NDE < bk/h_{kth} < (1+ε)NDE] ≥ 2/3`.  
With median trick: total space **O(log(1/δ)/ε²)**.

```python
def bottom_k_sketch(stream, k):
    """Bottom-k sketch for distinct element counting."""
    import hashlib

    def hash_to_float(x):
        # deterministic hash to [0,1]
        h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
        return h / (2**128)

    top_k = []  # max-heap (negate for min-heap of k smallest)
    seen = set()

    for e in stream:
        h = hash_to_float(e)
        if len(top_k) < k:
            heapq.heappush(top_k, -h)
        elif h < -top_k[0]:
            heapq.heapreplace(top_k, -h)

    h_kth = -top_k[0]  # k-th smallest hash value
    return k / h_kth - 1  # estimate

import heapq
stream = list(range(50)) * 3 + list(range(50, 100))
est = bottom_k_sketch(stream, 20)
print(f"True NDE: 100, Estimated: {est:.1f}")
```

---

### More Solutions for Distinct Element Counting

**FM Algorithm (Flajolet-Martin):**
- Hash each element; let r(e) = number of **trailing 0s** in binary of h(e).
- Keep R = max r(e) seen.
- Estimate NDE ≈ 2^R / 0.77351.

> **Example:** h(e) = 12 = (1100)₂ → r(e) = 2.

**HyperLogLog:** Extension of FM that splits stream into sub-streams and uses the **harmonic mean** for improved accuracy. Industry-standard in Redis, Presto, etc.

---

# Lecture 3: Streaming Algorithms (II)

---

## 3.1 Frequent Items in Data Streams

### Counter-based Algorithms

#### Algorithm 1 — Misra-Gries (Frequent) Algorithm
Deterministic; space O((1/γ)·log n); provides **γ·n-approximation**.

```
S = {} (set of <key, count> pairs, max size 1/γ − 1)
For each element e in stream:
    if <e, c> in S:
        update to <e, c+1>
    elif |S| < 1/γ − 1:
        add <e, 1>
    else:
        decrement ALL counters by 1; remove those reaching 0
Query: if <e, c> in S, return f̂(e) = c; else return 0
```

```python
def misra_gries(stream, gamma):
    """Misra-Gries frequent items algorithm."""
    max_size = int(1.0 / gamma) - 1
    S = {}  # {element: count}

    for e in stream:
        if e in S:
            S[e] += 1
        elif len(S) < max_size:
            S[e] = 1
        else:
            # Decrement all counters
            to_remove = [k for k, v in S.items() if v <= 1]
            for k in to_remove:
                del S[k]
            for k in S:
                S[k] -= 1

    return S

stream = ['B','A','B','C','B','D','A','D','D','E','E','E','E']
result = misra_gries(stream, gamma=1/4)
print("Misra-Gries:", result)
# Should contain B, D, E (frequent elements)
```

> **Example 1:** Stream = B,A,B,C,B,D,A,D,D,E,E,E,E with γ=1/4.  
> Final S = {B:1, D:1, E:3}.

**Theorem 1:** All elements with frequency > γ·n will be in S.  
`f̂(e) = c(e)` satisfies `f(e) − γn ≤ f̂(e) ≤ f(e)`.

---

#### Algorithm 2 — Space-Saving Algorithm
Key difference: when S is full, **replace** the element with minimum count.

```
S = {} (max size 1/γ)
For each element e:
    if <e, c> in S:
        update to <e, c+1>
    elif |S| < 1/γ:
        add <e, 1>
    else:
        Let <e_min, c_min> = pair with minimum count
        Replace with <e, c_min + 1>
```

```python
def space_saving(stream, gamma):
    """Space-Saving algorithm for frequent items."""
    max_size = int(1.0 / gamma)
    S = {}  # {element: count}

    for e in stream:
        if e in S:
            S[e] += 1
        elif len(S) < max_size:
            S[e] = 1
        else:
            # Find and replace min count element
            min_elem = min(S, key=S.get)
            min_count = S[min_elem]
            del S[min_elem]
            S[e] = min_count + 1

    return S

stream = ['B','A','B','C','B','D','A','D','D','E','E','E','E']
result = space_saving(stream, gamma=1/3)
print("Space-Saving:", result)
# Should be {B:3, D:4, E:6}
```

> **Example 2:** Same stream, γ=1/3, final S = {B:3, D:4, E:6}.

**Theorem 2:** Space-Saving provides γ·n-approximation.  
For e in S: `f(e) ≤ f̂(e) ≤ f(e) + c_min`.

**Theorem 3 — Equivalence:** Misra-Gries and Space-Saving are equivalent (offset by global c_min).

**Theorem 4 — Space Lower Bound:** Any deterministic γn-approximation algorithm needs Ω((1/γ)·log m) bits.

---

### Sketch-based Algorithms

#### Count-Min Sketch
Structure: d arrays, each of width w. Each array has its own pairwise-independent hash function hᵢ: [m] → [w].

**Update:** When element e arrives: `A[i][hᵢ(e)] += 1` for each i.  
**Query:** `f̂(e) = min_{i∈[d]} A[i][hᵢ(e)]`

```
Array 1: [-, -, h1(e)+1, -, ..., -]
Array 2: [-, h2(e)+1, -, -, ..., -]
...
Array d: [hd(e)+1, -, -, -, ..., -]
```

```python
import hashlib

class CountMinSketch:
    def __init__(self, epsilon, delta):
        """
        epsilon: error = epsilon * n
        delta:   failure probability
        """
        import math
        self.w = math.ceil(3.0 / epsilon)         # width
        self.d = math.ceil(math.log(1.0/delta, 3))  # depth
        self.table = [[0]*self.w for _ in range(self.d)]
        self.seeds = [i*1234567 for i in range(self.d)]

    def _hash(self, element, row):
        h = hash((element, self.seeds[row])) % self.w
        return h

    def update(self, element, count=1):
        for i in range(self.d):
            j = self._hash(element, i)
            self.table[i][j] += count

    def query(self, element):
        return min(self.table[i][self._hash(element, i)] for i in range(self.d))

# Example
cms = CountMinSketch(epsilon=0.1, delta=0.01)
stream = ['apple']*50 + ['banana']*30 + ['cherry']*15 + ['date']*5
for item in stream:
    cms.update(item)

print(f"apple freq (true=50): {cms.query('apple')}")
print(f"banana freq (true=30): {cms.query('banana')}")
```

**Lemma 4:** Setting w = ⌈3/ε⌉:  
`P[A[i][hᵢ(e)] − f(e) ≥ ε·n] ≤ 1/3`

**Lemma 5:** Setting d = ⌈log₃(1/δ)⌉:  
`P[f̂(e) − f(e) ≥ ε·n] ≤ δ`

**Total space:** O(log m · log(1/δ) / ε) bits.

---

#### Count-Min Extension: Range Queries

### Definition 2 — Range-based Frequency Query
`f(ℓ···r) = Σ_{i=ℓ}^r f(i)`

**Dyadic tree solution:** Build L = ⌈log₂ m⌉ Count-Min sketches for dyadic ranges [1..2], [3..4],..., [1..4], etc. Any query range [ℓ,r] decomposes into ≤ 2L disjoint dyadic ranges. Error: O(log m · ε · n).

---

#### Count-Min Extension: Turnstile Model

### Definition 3 — Turnstile Model
Stream of (eᵢ, cᵢ) pairs where cᵢ can be **negative** (deletions). Frequency = Σcᵢ for element e. Error now related to L₁-norm: `P[f̂(e) − f(e) ≥ ε·‖f‖₁] ≤ δ`.

---

#### Count-Sketch
Error relates to **L₂-norm** instead of L₁.  
Uses two hash functions per row: hᵢ: [m]→[w] and gᵢ: [m]→{+1,−1}.

**Update:** `A[i][hᵢ(e)] += gᵢ(e) · cₑ`  
**Query:** median over d rows of `gᵢ(e) · A[i][hᵢ(e)]`

```python
class CountSketch:
    def __init__(self, w, d):
        self.w = w
        self.d = d
        self.table = [[0]*w for _ in range(d)]

    def _hash_h(self, elem, row):
        return hash((elem, row, 'h')) % self.w

    def _hash_g(self, elem, row):
        return 1 if hash((elem, row, 'g')) % 2 == 0 else -1

    def update(self, elem, count=1):
        for i in range(self.d):
            j = self._hash_h(elem, i)
            s = self._hash_g(elem, i)
            self.table[i][j] += s * count

    def query(self, elem):
        import statistics
        estimates = []
        for i in range(self.d):
            j = self._hash_h(elem, i)
            s = self._hash_g(elem, i)
            estimates.append(s * self.table[i][j])
        return statistics.median(estimates)
```

---

## 3.2 Querying Over a Sliding Window

### Definition 4 — Sliding Window
Given stream with timestamps t₁ < t₂ < …, a **sliding window of size N** = elements {e_{c-N+1}, …, e_c} at current time tc.

**Challenge:** N so large it won't fit in memory; too many streams.

### Problem 1 — Count 1s in a Sliding Window
Given binary stream, estimate how many 1s in the last k ≤ N elements.

Exact solution requires storing the whole window — too expensive.

---

### Algorithm — DGIM Algorithm (Datar-Gionis-Indyk-Motwani)
Space: **O(log²N)** bits per stream. Provides **0.5-approximation** (can reduce to ε via O(log²N/ε) bits).

**Key concept: Buckets**  
Each bucket = (timestamp of last 1, size of bucket).

**Bucket constraints:**
1. Bucket size must be a **power of 2**.
2. At most **2 buckets of the same size**.
3. Buckets **do not overlap** in timestamps.
4. Sorted: earlier buckets never smaller than later.

> **Example buckets (N=70):**  
> Orange: ⟨16, 16⟩ (size 16)  
> Pink: ⟨31, 8⟩, ⟨45, 8⟩ (size 8 × 2)  
> Magenta: ⟨53, 4⟩, ⟨59, 4⟩ (size 4 × 2)  
> Cyan: ⟨65, 2⟩  
> Yellow: ⟨66, 1⟩, ⟨69, 1⟩

**Processing new bit:**
1. Drop oldest bucket if its timestamp falls outside window.
2. If new bit = 0: done.
3. If new bit = 1:
   - Create new size-1 bucket with current timestamp.
   - If now 3 buckets of size 1: merge oldest two → new size-2 bucket.
   - Repeat merging upward until constraints satisfied.

```python
from collections import deque

class DGIMStream:
    def __init__(self, N):
        self.N = N
        self.time = 0
        self.buckets = deque()  # (end_timestamp, size)

    def add(self, bit):
        self.time += 1
        # Step 1: Remove expired buckets
        while self.buckets and self.buckets[0][0] <= self.time - self.N:
            self.buckets.popleft()

        if bit == 0:
            return
        # Step 2: Create new size-1 bucket
        self.buckets.append((self.time, 1))
        # Step 3: Merge if constraint violated
        self._merge()

    def _merge(self):
        # Group buckets by size; if any size has >= 3, merge oldest 2
        while True:
            size_counts = {}
            for _, size in self.buckets:
                size_counts[size] = size_counts.get(size, 0) + 1
            merged = False
            for size, count in sorted(size_counts.items()):
                if count >= 3:
                    # merge the two oldest buckets of this size
                    new_buckets = deque()
                    merged_once = False
                    skip_next = False
                    for ts, sz in self.buckets:
                        if sz == size and not merged_once:
                            first_ts = ts
                            merged_once = True
                            skip_next = True
                            continue
                        if skip_next:
                            # merge this with first_ts
                            new_buckets.append((ts, size * 2))
                            skip_next = False
                            merged = True
                        else:
                            new_buckets.append((ts, sz))
                    self.buckets = new_buckets
                    break
            if not merged:
                break

    def estimate_count(self):
        if not self.buckets:
            return 0
        # Sum all except oldest (use half of oldest)
        buckets_list = list(self.buckets)
        total = sum(sz for _, sz in buckets_list[1:])
        total += buckets_list[0][1] // 2
        return total

# Example
dgim = DGIMStream(N=20)
bits = [1,0,1,0,1,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0]
for b in bits:
    dgim.add(b)
print(f"Estimated 1s in last 20: {dgim.estimate_count()}")
print(f"True 1s: {sum(bits[-20:])}")
```

**Querying (estimate of 1s in last k ≤ N):**
- Sum sizes of all buckets except the oldest.
- Add **half** the size of the oldest bucket.

**Why half?** We don't know how many 1s of the oldest bucket are within the window.

**Error Bound:**
- Let oldest bucket have size 2^r.
- Error from last bucket ≤ 2^{r-1}.
- At least 2^r 1s exist in window → error ratio ≤ 2^{r-1}/2^r = **0.5**. ✓

---

# Lecture 4: Nearest Neighbor Search — Locality Sensitive Hashing

---

## 4.1 Nearest Neighbor Search (NNS)

### Definition 1 — NNS
Given set P = {x₁,…,xₙ} ⊂ ℝᵈ and distance metric dist(·,·), for query q ∈ ℝᵈ, find the **point closest to q** in P.

### Definition 2 — r-Near Neighbor Search (r-NNS)
Return any x ∈ P with dist(q, x) ≤ r (if one exists).

**Exact solutions:**
- d=1: binary search, O(n) space, O(log n) query.
- d=2: Voronoi diagram, O(n) space, O(log n) query.
- d>2: Voronoi = O(n^{⌈d/2⌉}) space (too expensive!), linear scan = O(d·n).

### Definition 3 — c-Approximate Nearest Neighbor (c-ANNS)
Return x with `dist(q, x) ≤ c · dist(q, x*)`, where x* is the true nearest neighbor.

### Definition 4 — (c, r)-ANNS
If ∃ x with dist(q,x) ≤ r, return x' with dist(q,x') ≤ c·r.

---

## 4.2 Locality Sensitive Hashing (LSH)

**Intuition:** Hash nearby points into the **same bucket** with high probability.

### Definition 5 — (r, c·r, p₁, p₂)-sensitive Hash Family
Family H is (r, c·r, p₁, p₂)-sensitive (with p₁ > p₂, c > 1) if:
- `P[h(x)=h(y)] ≥ p₁` when dist(x,y) ≤ r (close points → high collision)
- `P[h(x)=h(y)] ≤ p₂` when dist(x,y) ≥ c·r (far points → low collision)

**Key quality parameter:** `ρ = log(p₁) / log(p₂)` (smaller is better)

---

### AND Operation (Reduce False Positives)
**Problem:** p₂·n far points collide with query in expectation ("false positives").

**Solution:** Concatenate k independent hash functions:  
`g(x) = ⟨h₁(x), h₂(x), …, hₖ(x)⟩`  
g(x)=g(y) iff all k functions agree.

**Lemma 1:** g is (r, c·r, p₁ᵏ, p₂ᵏ)-sensitive.

**Choose k:** Set p₂ᵏ = 1/n → k = log(n) / log(1/p₂)  
Then p₁ᵏ = 1/n^ρ.

### Multiple Hash Tables (Increase True Positive Rate)
Use L = n^ρ hash functions from the new (r,c·r,p₁ᵏ,p₂ᵏ)-family G.  
**Storage:** O(n·L) = O(n^{1+ρ}).

### Query Processing
```
For each of L hash functions gᵢ:
    Look up bucket gᵢ(q)
    For each retrieved point x:
        If dist(q, x) ≤ c·r: return x
Stop after examining 3L points (bound search cost)
Return failure if no point found
```
Search complexity: O(d·k·L).

### Theorem 1 — Success Probability
The LSH algorithm correctly returns a c-approximate r-near neighbor with probability ≥ **2/3 − 1/e**.

**Lemma 2:** P[r-near neighbor misses all L tables] ≤ (1 − 1/n^ρ)^L ≤ 1/e.  
**Lemma 3:** P[more than 3L far points examined] ≤ 1/3 (by Markov).

```python
import numpy as np
from collections import defaultdict

class LSH_Euclidean:
    """Simple E2LSH implementation for Euclidean distance."""
    def __init__(self, d, k, L, r=1.0):
        self.k = k   # concatenation length
        self.L = L   # number of hash tables
        self.r = r   # bucket width
        self.d = d
        # Random Gaussian vectors and offsets
        self.us = np.random.randn(L, k, d)
        self.bs = np.random.uniform(0, r, (L, k))
        self.tables = [defaultdict(list) for _ in range(L)]

    def _hash(self, x, table_idx):
        proj = self.us[table_idx] @ x          # shape (k,)
        bucket = tuple(np.ceil((proj + self.bs[table_idx]) / self.r).astype(int))
        return bucket

    def index(self, points):
        self.points = points
        for i, x in enumerate(points):
            for l in range(self.L):
                key = self._hash(x, l)
                self.tables[l][key].append(i)

    def query(self, q, c_r):
        candidates = set()
        for l in range(self.L):
            key = self._hash(q, l)
            candidates.update(self.tables[l].get(key, []))
        for idx in candidates:
            if np.linalg.norm(q - self.points[idx]) <= c_r:
                return self.points[idx], idx
        return None, None

# Example
np.random.seed(42)
d = 10
points = np.random.randn(1000, d)
lsh = LSH_Euclidean(d=d, k=5, L=20, r=2.0)
lsh.index(points)

q = points[42] + np.random.randn(d) * 0.1  # near point[42]
result, idx = lsh.query(q, c_r=4.0)
if result is not None:
    print(f"Found neighbor at index {idx}, dist={np.linalg.norm(q-result):.3f}")
```

---

## 4.3 LSH Families for Specific Distance Measures

### LSH for Euclidean Distance (E2LSH)

Hash function with random Gaussian vector **u** and offset b ∈ [0,r]:  
`h_{u,b}(x) = ⌈(u·x + b) / r⌉`

**u** is generated as d i.i.d. N(0,1) variables.

**Lemma 4 — Gaussian Sum:**  
Z = X + Y where X∼N(μ₁,σ₁²), Y∼N(μ₂,σ₂²) → Z∼N(μ₁+μ₂, σ₁²+σ₂²).

**Lemma 6:** For this LSH, `ρ = log(1/p₁)/log(1/p₂) ≤ 1/c`.

---

### LSH for Cosine Distance — SimHash

**Cosine similarity:** `cos(x,y) = (x·y) / (‖x‖₂·‖y‖₂)`  
**Cosine distance:** `dist_{cos}(x,y) = arccos(cos(x,y))` ∈ [0, π]

**SimHash** with random unit vector **u**:  
`h_u(x) = sign(u·x)` (outputs +1 or −1)

**Lemma 7:**  
`P[h_u(x) = h_u(y)] = 1 − dist_{cos}(x,y) / π`

So: p₁ = 1−r/π and p₂ = 1−cr/π. We can show ρ ≤ 1/c for c ≥ 2.

```python
def simhash(x, num_planes=128):
    """SimHash for cosine similarity."""
    d = len(x)
    planes = np.random.randn(num_planes, d)
    bits = (planes @ x >= 0).astype(int)
    return bits

# Two similar vectors
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1, 2, 3, 4, 6], dtype=float)

hash_x = simhash(x)
hash_y = simhash(y)
collision_rate = np.mean(hash_x == hash_y)
print(f"Collision rate: {collision_rate:.3f}")

# True cosine similarity
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
expected_collision = 1 - np.arccos(cos_sim) / np.pi
print(f"Expected collision (1 - angle/pi): {expected_collision:.3f}")
```

---

### LSH for Jaccard Distance — MinHash

**Jaccard similarity** between binary vectors x, y (sets X, Y):  
`J(x,y) = |X ∩ Y| / |X ∪ Y|`  
**Jaccard distance:** `dist_J(x,y) = 1 − J(x,y)`

> **Example 1:** x=(1,0,0,0,1,0,1,0,1,1), y=(0,0,0,0,1,0,1,0,0,0)  
> X={1,5,7,9,10}, Y={5,7} → J = 2/5 → dist_J = 3/5.

**MinHash:**  
Generate random permutation P of [1···d]. Define:  
`h(x) = min_{x(i)=1} P[i]`

**Key property:** `P[h(x) = h(y)] = J(x,y)`

So p₁ = 1−r (where r = Jaccard distance for close pairs) and ρ ≤ 1/c.

> **Example 2:** x=(1,0,0,0,1,0,1,0,1,1), y=(0,0,0,0,1,0,1,0,0,0).  
> Permutation [4,2,10,5,1,3,8,7,9,6].  
> h(x) = min{P[1],P[5],P[7],P[9],P[10]} = min{4,1,8,9,6} = **1**.  
> h(y) = min{P[5],P[7]} = min{1,8} = **1**.  
> h(x) = h(y) ✓ (consistent with J=2/5 > 0).

```python
def minhash_jaccard(set_x, set_y, d, num_perms=100):
    """Estimate Jaccard similarity using MinHash."""
    import random
    collisions = 0
    for _ in range(num_perms):
        perm = list(range(1, d+1))
        random.shuffle(perm)
        perm_dict = {i+1: perm[i] for i in range(d)}

        h_x = min(perm_dict[i] for i in set_x)
        h_y = min(perm_dict[i] for i in set_y)
        if h_x == h_y:
            collisions += 1

    estimated_jaccard = collisions / num_perms
    true_jaccard = len(set_x & set_y) / len(set_x | set_y)
    return estimated_jaccard, true_jaccard

X = {1, 5, 7, 9, 10}
Y = {5, 7}
est, true = minhash_jaccard(X, Y, d=10)
print(f"Estimated Jaccard: {est:.3f}, True: {true:.3f}")
```

---

# Lecture 5: Dimension Reduction — Johnson-Lindenstrauss Lemma

---

## 5.1 Dimension Reduction

### Definition 1 — Dimension Reduction
Given X = {x₁,…,xₙ} ⊂ ℝᵈ, find f: ℝᵈ → ℝᵏ with k < d that **preserves certain properties** (e.g., pairwise distances).

**Why?**
- **Efficiency:** Speed up algorithms.
- **Curse of dimensionality:** High-d spaces are counterintuitive.
- **Storage:** Reduce memory footprint.

**Applications:** Vector embeddings (YouTube, Airbnb, Alibaba), ML pipelines.

### Problem 1 — Distance-Preserving Reduction
Find f: ℝᵈ → ℝᵏ such that for all pairs i,j:  
`(1−ε)‖xᵢ−xⱼ‖₂ ≤ ‖f(xᵢ)−f(xⱼ)‖₂ ≤ (1+ε)‖xᵢ−xⱼ‖₂`

---

## 5.2 The Johnson-Lindenstrauss Lemma

### Lemma 1 — JL Lemma
There exists a **linear mapping** f that maps each of n points from ℝᵈ to ℝᵏ with **k = O(log n / ε²)** satisfying the distance-preservation property.

### Theorem 1 — (ε,δ)-JL Property
Let Π be a k×d random matrix with Πᵢⱼ ∼ (1/√k)·N(0,1). Then f(x) = Π·x satisfies:  
`(1−ε)‖x‖₂² ≤ ‖Πx‖₂² ≤ (1+ε)‖x‖₂²`  
with probability ≥ 1−δ, when **k = O(log(1/δ)/ε²)**.

*(Setting δ' = δ/C(n,2) and applying union bound ensures all pairs preserved.)*

```python
import numpy as np

def jl_projection(X, k, seed=42):
    """Johnson-Lindenstrauss random Gaussian projection."""
    np.random.seed(seed)
    n, d = X.shape
    Π = np.random.randn(k, d) / np.sqrt(k)
    return X @ Π.T  # shape: (n, k)

# Generate 100 points in 1000 dimensions
n, d, k = 100, 1000, 50
X = np.random.randn(n, d)
X_proj = jl_projection(X, k)

# Check distance preservation for all pairs
max_distortion = 0
for i in range(n):
    for j in range(i+1, n):
        orig_dist = np.linalg.norm(X[i] - X[j])
        proj_dist = np.linalg.norm(X_proj[i] - X_proj[j])
        if orig_dist > 1e-10:
            ratio = proj_dist / orig_dist
            max_distortion = max(max_distortion, abs(ratio - 1))

print(f"Max distortion: {max_distortion:.4f}")
# Should be small (e.g., < 0.3) for k=50
```

---

## 5.3 Analysis of JL Property

**Setup:** Let G = √k · Π (entries N(0,1)). Then:  
`‖Πx‖₂² = (1/k) Σᵢ₌₁ᵏ wᵢ²` where wᵢ = gᵢᵀ x.

**Key:** wᵢ = x₁gᵢ₁ + … + xₐgᵢₐ ~ N(0, ‖x‖₂²) (sum of independent Gaussians).

### Definition 2 — χ² Random Variable
`χₖ² = Σᵢ₌₁ᵏ zᵢ²` where zᵢ ∼ N(0,1) i.i.d.

### Theorem 2 — Concentration of χ²
`P[|Σzᵢ²/k − 1| ≥ ε] ≤ 2e^{−kε²/8}`  
Setting k = 8log(2/δ)/ε² gives δ-confidence.

### Lemma 4 — χ² Upper Tail (via MGF)
`P[Z > (1+ε)²k] ≤ e^{−3kε²/4}` where Z = Σzᵢ².  
*(Proof uses MGF of χ²: E[e^{tZ}] = (1−2t)^{−k/2}.)*

---

## 5.4 Achlioptas Construction (Sparse JL)

### Theorem 3 — Achlioptas
Let Πᵢⱼ i.i.d. from:
```
+√3/√k  with prob 1/6
   0    with prob 2/3
−√3/√k  with prob 1/6
```
Then Π satisfies (ε,δ)-JL property with k = O(log(1/δ)/ε²).

**Advantage:** ~2/3 of entries are 0 → faster matrix-vector multiplication!

```python
def achlioptas_projection(X, k):
    """Achlioptas sparse JL projection."""
    n, d = X.shape
    # Each entry: +sqrt(3/k) w.p. 1/6, 0 w.p. 2/3, -sqrt(3/k) w.p. 1/6
    scale = np.sqrt(3.0 / k)
    r = np.random.choice([scale, 0, -scale], size=(k, d), p=[1/6, 2/3, 1/6])
    return X @ r.T

X = np.random.randn(100, 1000)
X_proj = achlioptas_projection(X, k=50)
print(f"Projected shape: {X_proj.shape}")
print(f"Sparsity: {np.mean(X_proj == 0):.2f}")  # Many zeros but result is full
```

### Theorem 4 — Sparse JL (Kane-Nelson)
For k = O(log n/ε²), with only O(log n/ε) non-zeros per row, the (ε,δ)-JL property holds. **(Sparser = faster!)**

---

## 5.5 Applications of JL Lemma

1. **Approximate Nearest Neighbor Search:** Reduce dimension then run LSH.
2. **Approximate Linear Regression:**  
   min ‖Xa − y‖₂ over a ∈ ℝᵈ  
   → approximate as: min ‖ΠXa − Πy‖₂ (smaller system, faster to solve)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(0)
n, d = 500, 200
X = np.random.randn(n, d)
a_true = np.random.randn(d)
y = X @ a_true + 0.1 * np.random.randn(n)

# Direct regression
reg = LinearRegression(fit_intercept=False).fit(X, y)
err_direct = np.linalg.norm(X @ reg.coef_ - y)

# JL-compressed regression
k = 50
Π = np.random.randn(k, n) / np.sqrt(k)
X_proj = Π @ X   # (k x d)
y_proj = Π @ y   # (k,)
reg_jl = LinearRegression(fit_intercept=False).fit(X_proj, y_proj)
err_jl = np.linalg.norm(X @ reg_jl.coef_ - y)

print(f"Direct regression error: {err_direct:.3f}")
print(f"JL regression error:     {err_jl:.3f}")
```

---

# Lecture 6: Dimension Reduction — PCA and SVD

---

## 6.1 Warm-up: Basic Statistical Concepts

### Mean and Variance
For sequence S = (r₁, …, rₙ):  
`μ(S) = (1/n)Σrᵢ`,  
`Var(S) = (1/n)Σ(rᵢ − μ(S))²`

### Covariance
For sequences S and S':  
`cov(S, S') = (1/n)Σ(rᵢ−μ(S))(rᵢ'−μ(S'))`

> **Example 1:** S=(3,3,2,8), S'=(4,7,11,6).  
> μ(S)=4, μ(S')=7.  
> cov = (1/4)[(3−4)(4−7) + (3−4)(7−7) + (2−4)(11−7) + (8−4)(6−7)]  
> = (1/4)[3 + 0 − 8 − 4] = **−9/4**.

### Covariance Matrix
For n d-dimensional points, the **covariance matrix A** is d×d where:  
`Aᵢⱼ = cov(sequence of i-th coordinates, sequence of j-th coordinates)`

> **Example 2:** 3D points:  
> | Point | x₁ | x₂ | x₃ | x₄ |  
> |-------|----|----|----|----|  
> | d=1   | 1  | 0  | 2  | 1  |  
> | d=2   | 1  | 2  | 0  | 1  |  
> | d=3   | 1  | 1  | 1  | 1  |  
> A₁₁ = 1/2, A₁₂ = −1/2, A₁₃ = 0, A₂₂ = 1/2, etc.

```python
import numpy as np

# Example covariance matrix computation
X = np.array([
    [1, 0, 2, 1],   # dim 1
    [1, 2, 0, 1],   # dim 2
    [1, 1, 1, 1],   # dim 3
], dtype=float)  # shape: (3 dimensions x 4 points)

# Covariance matrix: (X - mu)(X - mu)^T / n
X_centered = X - X.mean(axis=1, keepdims=True)
A = (X_centered @ X_centered.T) / X.shape[1]
print("Covariance matrix:\n", A)
```

---

## 6.2 Eigenvectors and Eigenvalues

For d×d matrix A, if `A·v = λ·v` for unit vector v:  
- **v** is an **eigenvector** of A
- **λ** is the corresponding **eigenvalue**

### Eigen-Decomposition
For symmetric A: `A = V Λ Vᵀ`  
where V = [v₁,…,vₙ] (eigenvectors as columns) and Λ = diag(λ₁,…,λₙ).

**Property:** Aᵏ = V Λᵏ Vᵀ.

### Algorithm — Power Method
```
Initialize u₀ (random nonzero vector)
Repeat:
    u_k = A·u_{k-1} / ‖A·u_{k-1}‖₂
Until ‖u_k − u_{k-1}‖₂ < threshold
Eigenvector v₁ = u_k
Eigenvalue λ₁ = v₁ᵀ A v₁
Next: A ← A − λ₁v₁v₁ᵀ, repeat for (v₂, λ₂), etc.
```

```python
import numpy as np

def power_method(A, num_iters=1000, tol=1e-9):
    """Compute the top eigenvector using the Power Method."""
    n = A.shape[0]
    u = np.random.randn(n)
    u /= np.linalg.norm(u)

    for _ in range(num_iters):
        u_new = A @ u
        u_new /= np.linalg.norm(u_new)
        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new

    eigenvalue = u.T @ A @ u
    return eigenvalue, u

# Example from lecture slides
A = np.array([[3, 2], [2, 6]], dtype=float)
lam, v = power_method(A)
print(f"Eigenvalue: {lam:.3f}, Eigenvector: {v}")
# Expected: λ₁ ≈ 6.993, v₁ ≈ [0.447, 0.894]
```

---

## 6.3 Principal Component Analysis (PCA)

**Goal:** Project data to k dimensions that **maximize retained variance**.

### PCA Steps (5 Steps)

1. **Zero-mean each dimension.** Update xᵢ = xᵢ − μ (subtract dimension mean).
2. **Compute covariance matrix A** of centered points.
3. **Compute eigenvectors** of A; sort in **descending eigenvalue order**.
4. **Select top-k eigenvectors** v₁, v₂, …, vₖ.
5. **Project:** Map xᵢ to (v₁·xᵢ, v₂·xᵢ, …, vₖ·xᵢ).

### Theorem 1 — PCA Maximizes Variance
The eigenvector v₁ (largest eigenvalue) is the direction that maximizes:  
`max_{‖u‖=1} Var(projections) = max uᵀAu = λ₁`

**Proof:** Via Lagrange multipliers:  
`L(u,λ) = uᵀAu − λ(uᵀu−1) → ∂L/∂u = 2Au − 2λu = 0 → Au = λu`  
So u must be an eigenvector; variance = λ; maximize by taking largest λ.

**Lagrange Multiplier Method (Theorem 2):**  
To maximize f(x) subject to g(x) = 0:  
Define `L(x,λ) = f(x) − λg(x)`, then solve ∂L/∂x = 0 and g(x) = 0.

> **Example 3:** Maximize f(x,y)=x+y s.t. x²+y²=1.  
> L = x+y−λ(x²+y²−1). Setting derivatives to 0: x=y=1/(2λ).  
> Constraint → x=y=1/√2, max f = √2.

### Theorem 3 — PCA Optimality
Projecting onto the top-k eigenvectors maximizes total retained variance = Σᵢ₌₁ᵏ λᵢ.

```python
import numpy as np
import matplotlib.pyplot as plt

def pca(X, k):
    """
    PCA: reduce X (n x d) to k dimensions.
    Returns projected data and eigenvectors.
    """
    # Step 1: Center data
    mu = X.mean(axis=0)
    X_c = X - mu

    # Step 2: Covariance matrix
    A = X_c.T @ X_c / len(X_c)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # symmetric matrix

    # Step 4: Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Project onto top-k
    V_k = eigenvectors[:, :k]    # shape: (d, k)
    X_proj = X_c @ V_k           # shape: (n, k)

    return X_proj, V_k, eigenvalues

# Example: 2D data projected to 1D
np.random.seed(0)
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] * 2 + 0.5 * np.random.randn(100)  # correlated

X_1d, V_k, eigenvalues = pca(X, k=1)
print(f"Data shape: {X.shape} -> Projected: {X_1d.shape}")
print(f"Variance explained: {eigenvalues[0]/sum(eigenvalues)*100:.1f}%")
```

**Time complexity:** Eigendecomposition of d×d matrix = O(d³).

---

## 6.4 Singular Value Decomposition (SVD)

### Theorem 4 — SVD
Any m×n matrix A of rank r can be decomposed as:  
`A = U Σ Vᵀ`  
where:
- **U** (m×r): left singular vectors, orthonormal (UᵀU = Iᵣ)
- **Σ** (r×r): diagonal with **σ₁ ≥ σ₂ ≥ … ≥ σᵣ > 0** (singular values)
- **V** (n×r): right singular vectors, orthonormal (VᵀV = Iᵣ)

**Outer product form:**  
`A = Σᵢ₌₁ʳ σᵢ uᵢ vᵢᵀ`

```python
import numpy as np

# SVD example
A = np.array([[1,2,3],[4,5,6],[7,8,9],[1,0,1]], dtype=float)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U shape: {U.shape}, s: {s.round(3)}, V^T shape: {Vt.shape}")
print(f"Reconstruction error: {np.linalg.norm(A - U @ np.diag(s) @ Vt):.10f}")
```

### Finding Singular Vectors via Eigendecomposition
- **Left singular vectors:** eigenvectors of **AАᵀ** (m×m). σᵢ = √λᵢ.
- **Right singular vectors:** eigenvectors of **AᵀA** (n×n). Or V = AᵀU Σ⁻¹.

**Lemma 3:** If λ ≠ 0 is eigenvalue of AAᵀ, it is also eigenvalue of AᵀA.

---

## 6.5 Low-Rank Approximation

### Problem 1 — Best Rank-k Approximation
Find rank-k matrix B minimizing ‖A − B‖_F.

**Frobenius norm:** `‖A‖_F² = ΣΣ Aᵢⱼ² = Σᵢ σᵢ²`

### Theorem 5 — Truncated SVD
Define Aₖ = UₖΣₖVₖᵀ = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ. Then:  
`‖A − Aₖ‖_F = min_{rank(B)≤k} ‖A−B‖_F = √(Σᵢ₌ₖ₊₁ʳ σᵢ²)`

**Intuition:** Drop small singular values; little information is lost if σₖ₊₁,…,σᵣ ≈ 0.

```python
def truncated_svd(A, k):
    """Best rank-k approximation via truncated SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Keep only top-k components
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    A_k = U_k @ np.diag(s_k) @ Vt_k
    return A_k, s

# Example: Image compression-style
np.random.seed(1)
A = np.random.randn(20, 15)
A_k, s = truncated_svd(A, k=5)
error = np.linalg.norm(A - A_k, 'fro')
print(f"Frobenius reconstruction error (k=5): {error:.4f}")
print(f"Expected (sqrt of remaining singular values²): {np.sqrt(sum(s[5:]**2)):.4f}")

# Show variance captured
total_var = sum(s**2)
for k in [1, 3, 5, 10]:
    captured = sum(s[:k]**2) / total_var * 100
    print(f"  k={k}: {captured:.1f}% variance captured")
```

### Analysis of Truncated SVD (for k=1)
Goal: minimize `Σᵢ ‖aᵢ − xᵢy‖₂²` for unit vector y.  
→ Maximize `‖Ay‖₂² = yᵀAᵀAy` subject to ‖y‖=1.  
→ y = top eigenvector of AᵀA (= first right singular vector). ✓

---

## 6.6 Relationship Between PCA and SVD

When data matrix X has zero mean (after centering):  
**Covariance matrix = XᵀX / n** (or XXᵀ/n depending on convention).

PCA eigenvectors of XᵀX = **right singular vectors** of X.  
PCA eigenvalues λᵢ = σᵢ² / n.

So: **PCA is just SVD on the centered data matrix.**

```python
def pca_via_svd(X, k):
    """PCA using SVD (more numerically stable than eigendecomposition)."""
    X_c = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    # V columns = right singular vectors = PCA directions
    V_k = Vt[:k].T   # shape: (d, k)
    X_proj = X_c @ V_k
    return X_proj, V_k, s**2 / len(X_c)

X = np.random.randn(100, 10)
X_proj, directions, variances = pca_via_svd(X, k=3)
print(f"Projected shape: {X_proj.shape}")
print(f"Explained variance ratio: {variances[:3]/sum(variances)}")
```

**Time complexity of SVD:**  
Computing AᵀA: O(min(m²n, mn²)).  
Eigenvectors of AᵀA: O(min(n³, m³)).  
Total: O(min(m²n, mn²)).

---

## Summary Table

| Lecture | Topic | Key Algorithms | Space / Time |
|---------|-------|---------------|-------------|
| **Lec 1** | Probability & Concentration | Markov, Chebyshev, Chernoff | — |
| **Lec 2** | Streaming I | Reservoir, Morris++, MinHash, Bottom-k | O(log n / ε²) |
| **Lec 3** | Streaming II | Misra-Gries, Space-Saving, Count-Min, DGIM | O(1/γ·log n), O(log²N) |
| **Lec 4** | LSH / NNS | E2LSH, SimHash, MinHash | O(n^{1+ρ}) |
| **Lec 5** | JL Lemma | Gaussian proj., Achlioptas proj. | k = O(log n / ε²) |
| **Lec 6** | PCA & SVD | Power Method, Truncated SVD | O(min(m²n, mn²)) |

---

## Key Formula Quick Reference

| Concept | Formula |
|---------|---------|
| Bayes' Theorem | P[A\|B] = P[B\|A]·P[A] / P[B] |
| Markov | P[X ≥ a] ≤ E[X]/a |
| Chebyshev | P[\|X−μ\| ≥ a] ≤ Var[X]/a² |
| Chernoff (upper) | P[X ≥ (1+δ)μ] ≤ e^{−μδ²/3} (0<δ≤1) |
| Chernoff (lower) | P[X ≤ (1−δ)μ] ≤ e^{−μδ²/2} (0<δ<1) |
| Reservoir sample prob | k/n |
| Morris estimator | n̂ = 2^X − 1 |
| MinHash expectation | E[X_min] = 1/(NDE+1) |
| JL dimension | k = O(log n / ε²) |
| SimHash collision | P[h(x)=h(y)] = 1 − angle(x,y)/π |
| Jaccard MinHash | P[h(x)=h(y)] = J(x,y) |
| SVD decomposition | A = UΣVᵀ |
| Truncated SVD error | ‖A−Aₖ‖_F² = Σᵢ>k σᵢ² |
