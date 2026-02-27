# Micrograd: Building Neural Networks from Scratch
## A Comprehensive Guide Using the Feynman Technique

---

## Table of Contents
1. [Micrograd Overview](#1-micrograd-overview)
2. [Derivative of a Simple Function with One Input](#2-derivative-of-a-simple-function-with-one-input)
3. [Derivative of a Function with Multiple Inputs](#3-derivative-of-a-function-with-multiple-inputs)
4. [Starting the Core Value Object and Its Visualization](#4-starting-the-core-value-object-and-its-visualization)
5. [Manual Backpropagation Example #1: Simple Expression](#5-manual-backpropagation-example-1-simple-expression)
6. [Preview of a Single Optimization Step](#6-preview-of-a-single-optimization-step)
7. [Manual Backpropagation Example #2: A Neuron](#7-manual-backpropagation-example-2-a-neuron)
8. [Implementing the Backward Function for Each Operation](#8-implementing-the-backward-function-for-each-operation)
9. [Implementing the Backward Function for a Whole Expression Graph](#9-implementing-the-backward-function-for-a-whole-expression-graph)
10. [Breaking Up a tanh: Exercising with More Operations](#10-breaking-up-a-tanh-exercising-with-more-operations)
11. [Building Out a Neural Net Library (Multi-Layer Perceptron)](#11-building-out-a-neural-net-library-multi-layer-perceptron)
12. [Creating a Tiny Dataset and Writing the Loss Function](#12-creating-a-tiny-dataset-and-writing-the-loss-function)
13. [Collecting All Parameters of the Neural Net](#13-collecting-all-parameters-of-the-neural-net)
14. [Doing Gradient Descent Optimization Manually](#14-doing-gradient-descent-optimization-manually)
15. [Summary: What We Learned and Path to Modern Neural Nets](#15-summary-what-we-learned-and-path-to-modern-neural-nets)
16. [Conclusion](#16-conclusion)

---

## 1. Micrograd Overview

### What is Micrograd?

**Micrograd** is a tiny autograd engine that implements backpropagation (reverse-mode automatic differentiation) over a dynamically built computational graph.

**In simple terms**: It's a ~150 line program that can:
1. Do math operations (add, multiply, etc.)
2. **Automatically** figure out how to adjust inputs to get better outputs
3. This is the **core** of how neural networks learn!

### Why Build It?

Modern frameworks like PyTorch do this for you, but they're **black boxes**. By building micrograd from scratch, you'll understand:
- How neural networks actually learn
- What `.backward()` does in PyTorch
- Why gradients matter
- How to debug when things go wrong

### The Big Picture

```
You → Build a tiny autograd engine → Understand PyTorch → Build modern AI
```

Micrograd is **NOT** for production. It's for **learning**. Once you understand it, you'll use PyTorch - but you'll actually know what's happening under the hood.

### What You'll Build

By the end, you'll have:
- A `Value` class that tracks operations and computes gradients
- A simple neural network (Multi-Layer Perceptron)
- A training loop that learns from data

**All in ~150 lines of Python!**

---

## 2. Derivative of a Simple Function with One Input

### What is a Derivative? (The Intuitive Answer)

A derivative answers the question: **"If I nudge the input a tiny bit, how much does the output change?"**

### Example: A Simple Function

```python
def f(x):
    return 3*x**2 - 4*x + 5
```

Let's say `x = 3.0`:
```python
f(3.0) = 3*(3**2) - 4*3 + 5 = 27 - 12 + 5 = 20
```

### Finding the Derivative Numerically

**The Question**: If we increase `x` from 3.0 to 3.0001, how much does `f(x)` change?

```python
h = 0.0001  # tiny nudge
x = 3.0

f(x) = 20.0
f(x + h) = 20.0014  # slightly bigger

# The derivative (slope):
derivative = (f(x + h) - f(x)) / h
derivative ≈ 14.0
```

**What this means**: At `x=3`, if you increase `x` by 1, `f(x)` increases by about 14.

### The Calculus Way (Analytical Derivative)

Using calculus rules:
```
f(x) = 3x² - 4x + 5
f'(x) = 6x - 4

At x=3:
f'(3) = 6(3) - 4 = 18 - 4 = 14
```

Perfect match! But in neural networks with millions of operations, we can't do this by hand - we need **automatic** differentiation.

### Why This Matters for Neural Networks

Neural networks are just huge mathematical functions. The derivative tells us:
- Which direction to adjust weights
- How much to adjust them
- This is how networks **learn**!

---

## 3. Derivative of a Function with Multiple Inputs

### The Problem Gets More Interesting

What if your function has multiple inputs?

```python
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
# d = -6 + 10 = 4
```

Now we have **three** questions:
- How does `d` change if we nudge `a`?
- How does `d` change if we nudge `b`?
- How does `d` change if we nudge `c`?

These are called **partial derivatives**.

### Calculating ∂d/∂a (derivative of d with respect to a)

```python
h = 0.0001

# Original
d1 = a*b + c = 2*(-3) + 10 = 4

# Nudge 'a'
a_new = 2.0001
d2 = a_new*b + c = 2.0001*(-3) + 10 = 4.0003

# Slope
∂d/∂a = (d2 - d1) / h = -3.0
```

**Interpretation**: Increasing `a` by 1 decreases `d` by 3.

### Calculating ∂d/∂b

```python
# Nudge 'b'
b_new = -2.9999
d2 = a*b_new + c = 2*(-2.9999) + 10 = 4.0002

∂d/∂b = (d2 - d1) / h = 2.0
```

**Interpretation**: Increasing `b` by 1 increases `d` by 2.

### Calculating ∂d/∂c

```python
# Nudge 'c'
c_new = 10.0001
d2 = a*b + c_new = -6 + 10.0001 = 4.0001

∂d/∂c = (d2 - d1) / h = 1.0
```

**Interpretation**: Increasing `c` by 1 increases `d` by 1.

### Why These Values Make Sense

Looking at the expression `d = a*b + c`:
- `∂d/∂a = b = -3` (if you change `a`, it gets multiplied by `b`)
- `∂d/∂b = a = 2` (if you change `b`, it gets multiplied by `a`)
- `∂d/∂c = 1` (if you change `c`, it's just added directly)

**This is exactly what calculus tells us!**

### Neural Network Connection

A neural network might have **millions** of these variables. We need to know:
- How does the error change if we adjust weight #1?
- How about weight #2?
- ... weight #1,000,000?

Computing these derivatives is what **backpropagation** does automatically.

---

## 4. Starting the Core Value Object and Its Visualization

### The Problem We're Solving

We want to:
1. Do math operations (like `a*b + c`)
2. Automatically track how these operations relate to each other
3. Later, compute derivatives automatically

### Introducing the `Value` Class

Instead of using plain numbers, we wrap them in a `Value` object:

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data              # The actual number
        self.grad = 0.0               # The gradient (derivative)
        self._prev = set(_children)   # What created this value?
        self._op = _op                # Which operation created it?
        self.label = label            # A friendly name
        self._backward = lambda: None # How to compute gradients
```

### Using Value Objects

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b      # e = -6.0
d = e + c      # d = 4.0
```

### What Gets Tracked?

Each `Value` remembers:
- Its **data**: the actual number
- Its **parents**: what values created it
- The **operation**: was it +, *, etc.?

Example:
```python
e = a * b
# e.data = -6.0
# e._prev = {a, b}
# e._op = '*'
```

This creates a **computational graph** - a family tree of operations!

### Visualizing the Computational Graph

Using graphviz, we can draw this:

```
[a: 2.0] ──┐
           ├──[*]──[e: -6.0]──┐
[b: -3.0]──┘                  │
                              ├──[+]──[d: 4.0]
[c: 10.0]─────────────────────┘
```

**This visualization shows**:
- Which values fed into which operations
- How the final output was computed
- Later: how to trace gradients backward!

### The Power of This Approach

Now we can:
1. Do complex math
2. Keep track of every operation
3. Later, walk backward through the graph to compute derivatives

**This is the foundation of autograd!**

---

## 5. Manual Backpropagation Example #1: Simple Expression

### The Goal

We have: `L = (a*b + c) * f`

Let's manually compute how `L` changes with respect to each input:
- ∂L/∂a
- ∂L/∂b
- ∂L/∂c
- ∂L/∂f

### The Expression Graph

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b;           e.label = 'e'      # e = -6
d = e + c;         d.label = 'd'      # d = 4
f = Value(-2.0, label='f')
L = d * f;         L.label = 'L'      # L = -8
```

Graph:
```
a ──┐
    ├─[*]─ e ──┐
b ──┘          ├─[+]─ d ──┐
               │          ├─[*]─ L
c ─────────────┘          │
                          │
f ─────────────────────────┘
```

### Starting from the End: ∂L/∂L = 1.0

The gradient of a value with respect to itself is always 1.0.

```python
L.grad = 1.0  # "L affects L by a factor of 1"
```

### Working Backward: ∂L/∂d and ∂L/∂f

`L = d * f`

Using the chain rule:
- `∂L/∂d = f = -2.0`
- `∂L/∂f = d = 4.0`

```python
d.grad = f.data * L.grad = -2.0 * 1.0 = -2.0
f.grad = d.data * L.grad = 4.0 * 1.0 = 4.0
```

**Interpretation**:
- If `d` increases by 1, `L` changes by -2
- If `f` increases by 1, `L` changes by 4

### Next Level: ∂L/∂e and ∂L/∂c

`d = e + c`

Using the chain rule:
- `∂L/∂e = ∂L/∂d * ∂d/∂e = -2.0 * 1.0 = -2.0`
- `∂L/∂c = ∂L/∂d * ∂d/∂c = -2.0 * 1.0 = -2.0`

```python
e.grad = 1.0 * d.grad = 1.0 * (-2.0) = -2.0
c.grad = 1.0 * d.grad = 1.0 * (-2.0) = -2.0
```

### Final Level: ∂L/∂a and ∂L/∂b

`e = a * b`

Using the chain rule:
- `∂L/∂a = ∂L/∂e * ∂e/∂a = -2.0 * b = -2.0 * (-3.0) = 6.0`
- `∂L/∂b = ∂L/∂e * ∂e/∂b = -2.0 * a = -2.0 * 2.0 = -4.0`

```python
a.grad = b.data * e.grad = -3.0 * (-2.0) = 6.0
b.grad = a.data * e.grad = 2.0 * (-2.0) = -4.0
```

### Summary of All Gradients

```
L.grad = 1.0    # Output
f.grad = 4.0
d.grad = -2.0
c.grad = -2.0
e.grad = -2.0
b.grad = -4.0
a.grad = 6.0    # Inputs
```

### Verification with Numerical Gradient

Let's check `a.grad = 6.0`:

```python
# Original
L1 = (2.0 * -3.0 + 10.0) * -2.0 = -8.0

# Nudge 'a'
a_new = 2.0001
L2 = (2.0001 * -3.0 + 10.0) * -2.0 = -8.0006

# Gradient
(L2 - L1) / 0.0001 = 6.0 ✓
```

**Perfect match!**

### The Key Insight: The Chain Rule

Backpropagation is just applying the chain rule:
1. Start at the output (L.grad = 1.0)
2. Work backward through each operation
3. Multiply the "local gradient" by the "downstream gradient"

This is what makes neural networks learn!

---

## 6. Preview of a Single Optimization Step

### The Setup: We Have Gradients, Now What?

We computed:
```
a.grad = 6.0   # If a increases by 1, L increases by 6
b.grad = -4.0  # If b increases by 1, L decreases by 4
c.grad = -2.0
f.grad = 4.0
```

Current loss: `L = -8.0`

### The Goal: Make L Smaller (Optimize!)

Let's say we want to **minimize** `L` (make it more negative).

Looking at the gradients:
- `a.grad = 6.0` is positive → if we **decrease** `a`, `L` will decrease ✓
- `b.grad = -4.0` is negative → if we **increase** `b`, `L` will decrease ✓

### The Update Rule: Gradient Descent

```python
learning_rate = 0.01  # How big of a step to take

# Update each parameter in the opposite direction of its gradient
a.data += -learning_rate * a.grad
b.data += -learning_rate * b.grad
c.data += -learning_rate * c.grad
f.data += -learning_rate * f.grad
```

### Let's Do It!

```python
# Before
a.data = 2.0
b.data = -3.0
c.data = 10.0
f.data = -2.0

# Update
a.data += -0.01 * 6.0  = 2.0 - 0.06  = 1.94
b.data += -0.01 * -4.0 = -3.0 + 0.04 = -2.96
c.data += -0.01 * -2.0 = 10.0 + 0.02 = 10.02
f.data += -0.01 * 4.0  = -2.0 - 0.04 = -2.04
```

### Recompute L with New Values

```python
e = a * b = 1.94 * -2.96 = -5.7424
d = e + c = -5.7424 + 10.02 = 4.2776
L = d * f = 4.2776 * -2.04 = -8.726304
```

### The Result

```
Before: L = -8.0
After:  L = -8.726304
```

**L got more negative!** We successfully optimized in the right direction! 🎉

### The Big Picture

This is **exactly** how neural networks learn:

1. **Forward pass**: Compute the output and loss
2. **Backward pass**: Compute gradients (how each parameter affects loss)
3. **Update**: Adjust parameters in the opposite direction of gradients
4. **Repeat**: Do this thousands of times until loss is minimized

You just did one iteration of gradient descent by hand!

---

## 7. Manual Backpropagation Example #2: A Neuron

### What is a Neuron?

A neuron is the basic building block of a neural network. It:
1. Takes multiple inputs
2. Weighs them (multiplies by weights)
3. Adds a bias
4. Applies an activation function (like tanh)

### Building a Neuron Mathematically

```python
# Inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights (parameters we'll learn)
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Bias (another learnable parameter)
b = Value(6.8813735870195432, label='b')

# Compute weighted sum
x1w1 = x1*w1              # 2.0 * -3.0 = -6.0
x2w2 = x2*w2              # 0.0 * 1.0 = 0.0
x1w1x2w2 = x1w1 + x2w2    # -6.0 + 0.0 = -6.0
n = x1w1x2w2 + b          # -6.0 + 6.88... = 0.88...

# Apply activation function
o = n.tanh()              # tanh(0.88) ≈ 0.707
```

### The Computational Graph

```
x1 ──┐
     ├─[*]─ x1w1 ──┐
w1 ──┘             │
                   ├─[+]─ x1w1x2w2 ──┐
x2 ──┐             │                 │
     ├─[*]─ x2w2 ──┘                 ├─[+]─ n ──[tanh]─ o
w2 ──┘                               │
                                     │
b ───────────────────────────────────┘
```

This is a **single neuron**! All the complexity of neural networks comes from chaining many of these together.

### Manual Backpropagation Through the Neuron

Let's compute gradients starting from `o`:

#### Step 1: ∂o/∂o = 1.0
```python
o.grad = 1.0
```

#### Step 2: ∂o/∂n (through tanh)
The derivative of `tanh(x)` is `1 - tanh(x)²`:

```python
n.grad = (1 - o.data**2) * o.grad
       = (1 - 0.707**2) * 1.0
       = 0.5
```

#### Step 3: ∂o/∂b and ∂o/∂(x1w1x2w2) (through addition)
`n = x1w1x2w2 + b`, so both get the same gradient:

```python
b.grad = 1.0 * n.grad = 0.5
x1w1x2w2.grad = 1.0 * n.grad = 0.5
```

#### Step 4: ∂o/∂(x1w1) and ∂o/∂(x2w2) (through addition)
```python
x1w1.grad = 1.0 * x1w1x2w2.grad = 0.5
x2w2.grad = 1.0 * x1w1x2w2.grad = 0.5
```

#### Step 5: ∂o/∂x1, ∂o/∂w1 (through multiplication)
`x1w1 = x1 * w1`:

```python
x1.grad = w1.data * x1w1.grad = -3.0 * 0.5 = -1.5
w1.grad = x1.data * x1w1.grad = 2.0 * 0.5 = 1.0
```

#### Step 6: ∂o/∂x2, ∂o/∂w2 (through multiplication)
`x2w2 = x2 * w2`:

```python
x2.grad = w2.data * x2w2.grad = 1.0 * 0.5 = 0.5
w2.grad = x2.data * x2w2.grad = 0.0 * 0.5 = 0.0
```

### Summary of Gradients

```
o.grad = 1.0      # Output
n.grad = 0.5
b.grad = 0.5      # ← Learnable parameter
x1w1x2w2.grad = 0.5
x1w1.grad = 0.5
x2w2.grad = 0.5
w1.grad = 1.0     # ← Learnable parameter
w2.grad = 0.0     # ← Learnable parameter
x1.grad = -1.5    # Input (not learned)
x2.grad = 0.5     # Input (not learned)
```

### What These Gradients Mean

- `w1.grad = 1.0`: Increasing `w1` by 1 increases the output by 1
- `w2.grad = 0.0`: Changing `w2` doesn't affect output (because `x2 = 0`)
- `b.grad = 0.5`: Increasing bias increases output by 0.5

### Training This Neuron

To adjust the neuron's behavior:
```python
learning_rate = 0.01
w1.data += -learning_rate * w1.grad  # Adjust w1
w2.data += -learning_rate * w2.grad  # Adjust w2
b.data += -learning_rate * b.grad    # Adjust b
```

You just trained a neuron by hand! 🧠

---

## 8. Implementing the Backward Function for Each Operation

### The Goal

Instead of computing gradients manually, we want each operation to **know** how to compute its own gradients.

### The Pattern: Local Gradients

Each operation stores a `_backward` function that:
1. Knows the local derivative rule
2. Applies the chain rule (multiply by downstream gradient)
3. Updates the gradients of its inputs

### Addition: `out = a + b`

**Local derivatives**:
- `∂out/∂a = 1.0`
- `∂out/∂b = 1.0`

**Implementation**:
```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += 1.0 * out.grad   # Chain rule
        other.grad += 1.0 * out.grad  # Chain rule
    out._backward = _backward

    return out
```

**Why `+=` instead of `=`?**
A value might be used multiple times (like `c = a + a`). We need to **accumulate** gradients from all uses.

### Multiplication: `out = a * b`

**Local derivatives**:
- `∂out/∂a = b`
- `∂out/∂b = a`

**Implementation**:
```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad  # ∂out/∂self = other
        other.grad += self.data * out.grad  # ∂out/∂other = self
    out._backward = _backward

    return out
```

**Example**: If `out = 3 * 4 = 12`, and `out.grad = 2.0`:
- `3.grad += 4 * 2.0 = 8.0`
- `4.grad += 3 * 2.0 = 6.0`

### Tanh: `out = tanh(x)`

**Local derivative**:
- `∂tanh(x)/∂x = 1 - tanh(x)²`

**Implementation**:
```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out
```

### Power: `out = x**n`

**Local derivative**:
- `∂(x^n)/∂x = n * x^(n-1)`

**Implementation**:
```python
def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
```

### Exponential: `out = e^x`

**Local derivative**:
- `∂(e^x)/∂x = e^x`

**Implementation**:
```python
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad  # Derivative of e^x is e^x
    out._backward = _backward

    return out
```

### Helper Operations (Composition)

Division, subtraction, and negation can be built from what we have:

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __truediv__(self, other):
    return self * other**-1
```

These automatically work because they use operations we've already defined!

### The Beauty of This Design

Each operation is **self-contained**:
- It knows its forward computation
- It knows its backward (gradient) computation
- It connects to other operations automatically

This is exactly how PyTorch works internally!

---

## 9. Implementing the Backward Function for a Whole Expression Graph

### The Challenge

We've implemented `_backward()` for each operation, but how do we call them in the right order?

### The Problem: Order Matters!

Given: `L = (a*b + c) * f`

We **can't** just call `_backward()` on every node randomly. We must go in reverse topological order:

```
Correct order:  L → d → e → a,b,c,f
Wrong order:    a,b,c → e → d → L  ❌
```

**Why?** Child gradients must be computed before parent gradients (chain rule).

### The Solution: Topological Sort

A **topological sort** arranges nodes so that parents come after children.

#### Building the Topological Order

```python
def backward(self):
    # Step 1: Build topological order
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:  # Visit children first
                build_topo(child)
            topo.append(v)         # Then add self

    build_topo(self)
```

**Example**:
For `L = (a*b + c) * f`:

```
Visiting L:
  - Visit child d
    - Visit child e
      - Visit child a (leaf, add to topo)
      - Visit child b (leaf, add to topo)
      - Add e to topo
    - Visit child c (leaf, add to topo)
    - Add d to topo
  - Visit child f (leaf, add to topo)
  - Add L to topo

Result: topo = [a, b, e, c, d, f, L]
```

### Calling Backward Functions in Order

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # Step 2: Set output gradient to 1
    self.grad = 1.0

    # Step 3: Go backward through the graph
    for node in reversed(topo):
        node._backward()
```

### The Complete Flow: An Example

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b
d = e + c
f = Value(-2.0, label='f')
L = d * f

L.backward()
```

**What happens**:

1. **Build topo**: `[a, b, e, c, d, f, L]`
2. **Set**: `L.grad = 1.0`
3. **Reverse order**: `[L, f, d, c, e, b, a]`
4. **Call backward**:
   - `L._backward()`: does nothing (leaf)
   - `f._backward()`: computes `d.grad`
   - `d._backward()`: computes `e.grad` and `c.grad`
   - `c._backward()`: does nothing (leaf)
   - `e._backward()`: computes `a.grad` and `b.grad`
   - `b._backward()`: does nothing (leaf)
   - `a._backward()`: does nothing (leaf)

**Result**: All gradients computed automatically! 🎉

### The Magic

With just this `backward()` method, you can:
- Build arbitrarily complex expressions
- Automatically compute all gradients
- Never worry about the order or math

**This is the core of PyTorch's autograd!**

---

## 10. Breaking Up a tanh: Exercising with More Operations

### The Challenge

We implemented `tanh` as a single operation, but it can be broken down into simpler operations.

**Why break it down?**
1. Practice using more operations
2. Verify our implementation is correct
3. Understand how complex functions are built from simple ones

### The tanh Formula

```python
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

### Breaking It Down Step by Step

```python
# Instead of:
o = n.tanh()

# Do this:
e = (2*n).exp()        # e^(2x)
o = (e - 1) / (e + 1)  # (e^(2x) - 1) / (e^(2x) + 1)
```

### The Full Example

```python
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.88137, label='b')

# Weighted sum
x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b

# Manual tanh breakdown
two_n = 2*n           # 2*n
e = two_n.exp()       # e^(2n)
e_minus_1 = e - 1     # e^(2n) - 1
e_plus_1 = e + 1      # e^(2n) + 1
o = e_minus_1 / e_plus_1  # tanh(n)

o.backward()
```

### The Computational Graph (Expanded)

```
n ─[*2]─ two_n ─[exp]─ e ─┬─[-1]─ e_minus_1 ─┐
                          │                   ├─[/]─ o
                          └─[+1]─ e_plus_1 ───┘
```

**Much more complex!** But each operation is simple.

### Verifying the Gradients

Let's check that breaking down `tanh` gives the same gradients as using it directly:

**Method 1: Direct tanh**
```python
o = n.tanh()
o.backward()
print(n.grad)  # Should be 1 - tanh(n)^2 ≈ 0.5
```

**Method 2: Broken down tanh**
```python
e = (2*n).exp()
o = (e - 1) / (e + 1)
o.backward()
print(n.grad)  # Should also be ≈ 0.5
```

**Result**: Both give the same gradient! ✓

### What We Learned

1. **Complex operations are built from simple ones**
   - `tanh` is just `*, exp, -, +, /`
   - Neural networks are just compositions of simple operations

2. **Automatic differentiation handles everything**
   - No matter how we build the graph
   - The gradients are computed correctly

3. **More operations = same result, bigger graph**
   - Breaking down `tanh` creates more nodes
   - But the final gradients are identical
   - This is why frameworks often implement common operations efficiently

### PyTorch Connection

In PyTorch:
- High-level ops like `torch.tanh()` are optimized
- But under the hood, everything is composed of simple operations
- The autograd engine handles it all automatically

---

## 11. Building Out a Neural Net Library (Multi-Layer Perceptron)

### The Big Picture

We have:
- ✅ The `Value` class (does math and tracks gradients)
- ✅ Basic operations (+, *, tanh, etc.)

Now we build:
- 🧠 **Neuron**: Single computational unit
- 📊 **Layer**: Collection of neurons
- 🏗️ **MLP (Multi-Layer Perceptron)**: Stack of layers

### Building Block #1: Neuron

A neuron:
1. Has input weights (one per input)
2. Has a bias
3. Computes weighted sum + bias
4. Applies tanh activation

```python
class Neuron:
    def __init__(self, nin):
        # Create random weights and bias
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w*x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
```

**Example**:
```python
n = Neuron(3)  # Takes 3 inputs
x = [2.0, 3.0, -1.0]
output = n(x)  # Single Value output
```

**What it does**:
```
x[0]=2.0  ──[*w[0]]──┐
x[1]=3.0  ──[*w[1]]──┼──[+]──[+b]──[tanh]── output
x[2]=-1.0 ──[*w[2]]──┘
```

### Building Block #2: Layer

A layer is multiple neurons in parallel:

```python
class Layer:
    def __init__(self, nin, nout):
        # Create 'nout' neurons, each taking 'nin' inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Each neuron processes the same input
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

**Example**:
```python
layer = Layer(3, 4)  # 3 inputs → 4 outputs
x = [2.0, 3.0, -1.0]
outputs = layer(x)  # List of 4 Values
```

**What it does**:
```
         ┌─ Neuron1 ─ out[0]
         ├─ Neuron2 ─ out[1]
x ───────┼─ Neuron3 ─ out[2]
         └─ Neuron4 ─ out[3]
```

### Building Block #3: MLP (Multi-Layer Perceptron)

An MLP stacks layers:

```python
class MLP:
    def __init__(self, nin, nouts):
        # nouts is a list: [4, 4, 1] means 2 hidden layers of 4, output of 1
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

**Example**:
```python
mlp = MLP(3, [4, 4, 1])
# 3 inputs → 4 neurons → 4 neurons → 1 output

x = [2.0, 3.0, -1.0]
output = mlp(x)  # Single Value
```

**Architecture**:
```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output
(3 inputs)     (4 neurons)       (4 neurons)       (1 neuron)

x[0] ──┐
       ├──── [Neuron] ──┐
x[1] ──┤     [Neuron] ──┤
       ├──── [Neuron] ──┼──── [Neuron] ──┐
x[2] ──┘     [Neuron] ──┘     [Neuron] ──┤
                              [Neuron] ──┼──── [Neuron] ── output
                              [Neuron] ──┘
```

### Complete Example

```python
# Create network: 3 inputs, 2 hidden layers (4 neurons each), 1 output
n = MLP(3, [4, 4, 1])

# Forward pass
x = [2.0, 3.0, -1.0]
output = n(x)

print(output)  # Value(data=0.165...)
print(len(n.parameters()))  # 41 parameters total!
# Layer 1: 3*4 weights + 4 biases = 16
# Layer 2: 4*4 weights + 4 biases = 20
# Layer 3: 4*1 weights + 1 bias = 5
# Total: 16 + 20 + 5 = 41
```

### What We Just Built

You now have a **complete neural network**!
- Random initialization
- Forward pass computation
- Gradient tracking (through Value)
- Parameter access (for training)

**Next step**: Train it on data!

---

## 12. Creating a Tiny Dataset and Writing the Loss Function

### Creating a Simple Dataset

Let's create a tiny dataset for a classification problem:

```python
# Inputs: 3 features each
xs = [
    [2.0, 3.0, -1.0],   # Example 1
    [3.0, -1.0, 0.5],   # Example 2
    [0.5, 1.0, 1.0],    # Example 3
    [1.0, 1.0, -1.0],   # Example 4
]

# Desired outputs (labels)
ys = [1.0, -1.0, -1.0, 1.0]
```

**Interpretation**:
- Example 1 should output `1.0`
- Example 2 should output `-1.0`
- etc.

This is a binary classification problem (1 or -1).

### The Goal

Train the network so that:
```python
n(xs[0]) ≈ 1.0
n(xs[1]) ≈ -1.0
n(xs[2]) ≈ -1.0
n(xs[3]) ≈ 1.0
```

### Forward Pass: Making Predictions

```python
n = MLP(3, [4, 4, 1])

# Predict for each example
ypred = [n(x) for x in xs]

print(ypred)
# [Value(data=0.165), Value(data=-0.234), Value(data=0.891), Value(data=0.123)]
```

These predictions are **random** (network is untrained).

### Measuring Error: The Loss Function

We need a single number that measures "how wrong" our predictions are.

**Mean Squared Error (MSE)**:
```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

**Breaking it down**:
```python
# Example 1: predicted 0.165, target 1.0
error1 = (0.165 - 1.0)**2 = 0.697

# Example 2: predicted -0.234, target -1.0
error2 = (-0.234 - (-1.0))**2 = 0.586

# ... etc

# Total loss
loss = error1 + error2 + error3 + error4
```

**What this means**:
- `loss = 0`: Perfect predictions ✓
- `loss > 0`: Some error
- Bigger loss = worse predictions

### Computing the Loss

```python
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

print(loss)  # Value(data=3.456...)
```

### Why This Loss Function?

**Mean Squared Error** is popular because:
1. Always positive (errors don't cancel out)
2. Larger errors are penalized more (squared!)
3. Differentiable (we can compute gradients)
4. Simple and interpretable

### Alternative Loss Functions

For different problems:
- **Cross-Entropy**: For classification (better than MSE)
- **Absolute Error**: `|predicted - actual|`
- **Hinge Loss**: For support vector machines

### The Training Loop Preview

```python
# 1. Make predictions
ypred = [n(x) for x in xs]

# 2. Compute loss
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# 3. Compute gradients
loss.backward()

# 4. Update parameters (we'll do this next!)
```

We're almost ready to train!

---

## 13. Collecting All Parameters of the Neural Net

### The Problem

Our network has **41 parameters** scattered across:
- Layer 1: 16 parameters (weights and biases)
- Layer 2: 20 parameters
- Layer 3: 5 parameters

How do we access them all for gradient descent?

### The Solution: `.parameters()` Method

We already implemented this!

**Neuron level**:
```python
class Neuron:
    def parameters(self):
        return self.w + [self.b]  # All weights + bias
```

**Layer level**:
```python
class Layer:
    def parameters(self):
        # Flatten parameters from all neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

**MLP level**:
```python
class MLP:
    def parameters(self):
        # Flatten parameters from all layers
        return [p for layer in self.layers for p in layer.parameters()]
```

### Using It

```python
n = MLP(3, [4, 4, 1])

params = n.parameters()
print(len(params))  # 41

# Each parameter is a Value object
print(params[0])  # Value(data=0.234...)
print(params[0].data)  # 0.234... (the actual weight)
print(params[0].grad)  # 0.0 (gradient, not computed yet)
```

### Why This Design?

This **compositional structure** is beautiful:
- Each level knows its own parameters
- Higher levels just ask lower levels
- Adding new layer types is easy
- This is how PyTorch works too!

### Inspecting Parameters

```python
# Before training
for i, p in enumerate(n.parameters()):
    print(f"Param {i}: data={p.data:.3f}, grad={p.grad:.3f}")

# Param 0: data=0.234, grad=0.000
# Param 1: data=-0.567, grad=0.000
# ... (41 total)
```

### After Computing Gradients

```python
# Forward pass
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# Backward pass
loss.backward()

# Now gradients are filled!
for i, p in enumerate(n.parameters()):
    print(f"Param {i}: data={p.data:.3f}, grad={p.grad:.3f}")

# Param 0: data=0.234, grad=-0.145
# Param 1: data=-0.567, grad=0.234
# ... (41 total)
```

Each gradient tells us:
- Positive grad: Decrease this weight to reduce loss
- Negative grad: Increase this weight to reduce loss
- Large grad: This weight has big impact
- Small grad: This weight matters less

### Ready for Training!

Now we have:
- ✅ Network structure
- ✅ Dataset
- ✅ Loss function
- ✅ Access to all parameters and their gradients

**Next**: Actually update the parameters!

---

## 14. Doing Gradient Descent Optimization Manually

### The Training Loop

Training a neural network has 4 steps:

1. **Forward pass**: Make predictions
2. **Compute loss**: Measure error
3. **Backward pass**: Compute gradients
4. **Update**: Adjust parameters

Let's do it!

### Iteration 1: First Training Step

```python
# Step 1: Forward pass
ypred = [n(x) for x in xs]

# Step 2: Compute loss
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print(f"Loss: {loss.data}")  # 3.456...

# Step 3: Backward pass
loss.backward()

# Step 4: Update parameters
learning_rate = 0.1
for p in n.parameters():
    p.data += -learning_rate * p.grad

print(f"Loss after update: {loss.data}")  # Still 3.456 (old loss)
```

**Wait!** The loss didn't change because we're looking at the old value.

### Computing Loss Again

```python
# Need to re-run forward pass to see new loss
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print(f"New loss: {loss.data}")  # 3.123... (lower!)
```

**Success!** The loss decreased! 🎉

### Multiple Iterations

```python
for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    for p in n.parameters():
        p.grad = 0.0  # Important! Zero gradients
    loss.backward()

    # Update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
```

**Output**:
```
0 0.00206
1 0.00204
2 0.00203
3 0.00201
4 0.00199
5 0.00198
...
19 0.00179
```

**Loss is decreasing steadily!** 📉

### Why Zero Gradients?

```python
for p in n.parameters():
    p.grad = 0.0
```

Remember: our `_backward()` functions use `+=`:
```python
self.grad += other.data * out.grad
```

Without zeroing, gradients **accumulate** from previous iterations!

### Checking the Predictions

```python
print("Predictions:")
for i, (x, ygt, yout) in enumerate(zip(xs, ys, ypred)):
    print(f"Example {i}: target={ygt}, predicted={yout.data:.3f}")
```

**Output**:
```
Example 0: target=1.0, predicted=0.982
Example 1: target=-1.0, predicted=-0.986
Example 2: target=-1.0, predicted=-0.977
Example 3: target=1.0, predicted=0.973
```

**Amazing!** The network learned the pattern! 🧠✨

### What Just Happened?

1. **Started with random weights**: Network outputs were random
2. **Computed gradients**: Found which way to adjust each weight
3. **Updated weights**: Moved them slightly in the right direction
4. **Repeated 20 times**: Gradually improved predictions

### The Learning Curve

```
Iteration  Loss      Interpretation
0          3.456     Random, terrible
1          3.123     Slightly better
5          2.789     Improving
10         1.234     Much better
15         0.456     Getting close
20         0.002     Nearly perfect!
```

### Hyperparameters We Used

- **Learning rate**: `0.1` - how big each step is
  - Too small: Learning is slow
  - Too large: Might overshoot and diverge

- **Iterations**: `20` - how many update steps
  - More iterations → better fit (usually)
  - But can overfit small datasets

### Comparison to PyTorch

This is **exactly** what `optimizer.step()` does in PyTorch:

```python
# PyTorch equivalent
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()  # Zero gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
```

You just implemented SGD (Stochastic Gradient Descent) from scratch!

---

## 15. Summary: What We Learned and Path to Modern Neural Nets

### What We Built

In ~200 lines of code, we created:

1. **Autograd Engine** (`Value` class)
   - Wraps numbers and tracks operations
   - Automatically computes gradients via backpropagation
   - Implements chain rule through computational graphs

2. **Neural Network Components**
   - `Neuron`: Basic computational unit (weights, bias, activation)
   - `Layer`: Collection of neurons
   - `MLP`: Multi-layer perceptron (stack of layers)

3. **Training Infrastructure**
   - Loss function (MSE)
   - Gradient descent optimizer
   - Training loop (forward, backward, update)

### Core Concepts We Learned

#### 1. **Derivatives Tell Us Direction**
- Derivative = "how much output changes when input changes"
- Positive gradient → decrease parameter to reduce loss
- Negative gradient → increase parameter to reduce loss

#### 2. **Computational Graphs Track Everything**
- Each operation creates a node
- Nodes remember their parents (inputs)
- Graph structure enables automatic differentiation

#### 3. **Backpropagation is Just Chain Rule**
- Start at output (loss)
- Work backward through operations
- Each operation knows its local derivative
- Multiply local derivatives to get total derivative

#### 4. **Gradient Descent is Simple**
```python
parameter = parameter - learning_rate * gradient
```
That's it! Repeat thousands of times.

#### 5. **Neural Networks are Function Approximators**
- Input → layers of neurons → output
- Each layer transforms the data
- Non-linearity (tanh) enables learning complex patterns

### Limitations of Our Implementation

What we **didn't** include (but PyTorch has):

1. **Efficiency**
   - We use Python scalars (slow)
   - PyTorch uses tensors (GPU-accelerated)

2. **Operations**
   - We have +, *, tanh, exp, **
   - PyTorch has hundreds (conv, attention, etc.)

3. **Optimizers**
   - We use basic SGD
   - PyTorch has Adam, RMSprop, etc. (smarter updates)

4. **Features**
   - No batching (we process one example at a time)
   - No regularization (prevent overfitting)
   - No learning rate scheduling
   - No model saving/loading

### Path to Modern Neural Networks

#### From Micrograd → PyTorch

**What stays the same**:
- Computational graphs
- Autograd (automatic differentiation)
- Backpropagation
- Gradient descent

**What changes**:
- **Scalars → Tensors**: Multi-dimensional arrays (vectors, matrices)
- **CPU → GPU**: Parallel computation
- **Simple → Complex**: More layer types, optimizers, loss functions

#### The Journey

```
Micrograd (you are here)
    ↓
Understanding Tensors
    ↓
PyTorch Basics
    ↓
Convolutional Networks (CNNs)
    ↓
Recurrent Networks (RNNs, LSTMs)
    ↓
Transformers (Attention)
    ↓
Modern Architectures (GPT, BERT, etc.)
```

### Key Insights

#### 1. **Simple Components, Complex Behavior**
- Neurons are simple: `tanh(w*x + b)`
- But thousands of them learn amazing patterns
- Emergence through composition

#### 2. **Learning is Optimization**
- Define a loss function (what's "wrong"?)
- Compute gradients (which direction to improve?)
- Update parameters (take a step)
- Repeat until good enough

#### 3. **Gradients are Everything**
- Without gradients, no learning
- Backpropagation makes gradients automatic
- This enabled the deep learning revolution

#### 4. **It's All Differentiable Math**
- Neural networks = differentiable functions
- Training = optimizing functions
- Success requires: good architecture + good data + good optimization

### Skills You Now Have

✅ **Understand PyTorch internals**
- What `requires_grad=True` means
- Why `loss.backward()` works
- What `optimizer.step()` does

✅ **Debug gradient problems**
- Vanishing gradients (derivatives → 0)
- Exploding gradients (derivatives → ∞)
- Gradient flow issues

✅ **Design custom operations**
- Implement forward pass
- Implement backward pass
- Integrate with autograd

✅ **Think in computational graphs**
- Visualize operations
- Understand dependencies
- Optimize graph structure

### What's Next?

1. **Study PyTorch tutorials**
   - Now you'll understand what's happening!
   - Everything maps to concepts you learned

2. **Explore makemore series** (next in this course)
   - Character-level language models
   - More complex architectures
   - Practical applications

3. **Build projects**
   - Image classification (MNIST, CIFAR)
   - Text generation
   - Reinforcement learning

4. **Deepen understanding**
   - Read papers on optimization
   - Study different architectures
   - Experiment with hyperparameters

### The Bottom Line

**You built a neural network from scratch using only:**
- Basic Python
- Math operations
- ~200 lines of code

**You now understand:**
- How neural networks learn
- What backpropagation really is
- Why frameworks like PyTorch work

This foundation is **invaluable**. Most people use PyTorch without understanding it. You're different - you know what's happening under the hood.

**That makes you dangerous.** 🚀

---

## 16. Conclusion

### What We Accomplished

We started with a simple question: **"How do neural networks learn?"**

We answered it by building everything from scratch:

1. **Automatic differentiation** - Computing gradients automatically
2. **Neural network components** - Neurons, layers, networks
3. **Training algorithms** - Gradient descent optimization
4. **A working model** - That actually learns from data!

### The Journey

**Started with**: Understanding derivatives
```python
h = 0.0001
slope = (f(x+h) - f(x)) / h
```

**Ended with**: A trained neural network
```python
n = MLP(3, [4, 4, 1])
# ... training ...
# Network makes accurate predictions!
```

### Key Takeaways

#### 1. **Simplicity and Power**
The core of deep learning isn't complicated:
- Operations (add, multiply, etc.)
- Chain rule (calculus)
- Gradient descent (optimization)

But these simple pieces combine to create systems that can:
- Recognize images
- Understand language
- Play games
- Generate art

#### 2. **Understanding Beats Using**
You could have just used PyTorch. But by building micrograd:
- You **understand** what's happening
- You can **debug** problems
- You can **innovate** new ideas
- You're not limited by the framework

#### 3. **Math is Your Friend**
Neural networks are just math:
- Derivatives tell you direction
- Optimization finds solutions
- Composition creates complexity

Don't fear the math - embrace it!

#### 4. **Start Simple, Scale Up**
Our network:
- 3 inputs, 4-4-1 architecture
- 4 training examples
- 41 parameters

Modern networks (GPT-4, etc.):
- Millions of inputs
- Billions of parameters
- Trillions of training examples

**Same principles!** Just bigger.

### The Bigger Picture

You learned more than just micrograd. You learned:

**The Scientific Method for AI**:
1. Hypothesis (network architecture)
2. Experiment (training)
3. Measure (loss function)
4. Iterate (gradient descent)

**Problem-Solving Skills**:
- Break complex problems into simple parts
- Build from first principles
- Test and verify incrementally
- Visualize to understand

**A New Perspective**:
- Intelligence can be learned
- Learning is optimization
- Optimization is automatic with the right tools

### Where This Leads

This is **Lecture 1** of Neural Networks: Zero to Hero.

**What's next in the series**:
- **Makemore Part 1**: Bigram language models
- **Makemore Part 2**: Multi-layer perceptrons for text
- **Makemore Part 3**: Batch normalization
- **Makemore Part 4**: Manual backpropagation through complex networks
- **Makemore Part 5**: WaveNet (convolutional networks)
- **Beyond**: Transformers, attention, modern architectures

Each builds on what you learned here.

### Final Thoughts

Building micrograd isn't about creating a production framework. It's about **understanding deeply**.

**Andrej Karpathy's philosophy**:
> "What I cannot create, I do not understand." - Richard Feynman

You created it. Now you understand it.

Modern AI is built on these foundations:
- Every PyTorch model uses autograd (you built it)
- Every optimizer uses gradients (you computed them)
- Every neural network has layers (you designed them)

### Your Superpower

You now have a superpower: **understanding**.

When others see PyTorch as magic, you see:
- Computational graphs
- Automatic differentiation
- Gradient flow
- Optimization loops

This understanding will:
- Make you a better ML engineer
- Help you debug faster
- Enable you to innovate
- Give you confidence to tackle harder problems

### One Last Thing

Save your micrograd code. Look back at it in 6 months, 1 year, 5 years.

Each time, you'll:
- Appreciate it more
- Understand it deeper
- See new connections
- Remember where you started

This is your foundation. Build on it.

---

## Appendix: Quick Reference

### The Value Class (Simplified)

```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        # Build topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Backpropagate
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

### Training Loop Template

```python
for iteration in range(num_iterations):
    # Forward
    predictions = [model(x) for x in inputs]
    loss = loss_function(predictions, targets)

    # Backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Update
    for p in model.parameters():
        p.data += -learning_rate * p.grad
```

### Key Formulas

**Gradient Descent**:
```
θ_new = θ_old - α * ∂L/∂θ
```

**Chain Rule**:
```
∂L/∂x = ∂L/∂y * ∂y/∂x
```

**Mean Squared Error**:
```
L = Σ(predicted - actual)²
```

**tanh Derivative**:
```
∂tanh(x)/∂x = 1 - tanh(x)²
```

---

**Congratulations!** You've completed the micrograd comprehensive guide. You're ready to build neural networks! 🎉🧠🚀
