# The `Value` Class - Explained Using the Feynman Technique

Think of the `Value` class as a **smart number that remembers its family tree**.

---

## Variables (The Memory)

### `self.data`
**What it is**: The actual number (like 5, -3.2, 100)

**Why we need it**: This is the "value" part of `Value`. When you do math like `2 + 3`, you need to store that `5` somewhere.

**Analogy**: Like writing down your answer on paper. If you calculate 2+3, you write down 5.

---

### `self.grad` (gradient)
**What it is**: A number that tells us "if I change this value a tiny bit, how much does my final answer change?"

**Why we need it**: This is the WHOLE POINT of the class! In machine learning, we need to know "if I make this weight a bit bigger, does my prediction get better or worse?"

**Analogy**: Imagine you're trying to reach the top of a hill in the fog. The gradient tells you which direction is uphill. If `grad = 5`, going up a tiny step here makes you go up 5 units on the final output. If `grad = -2`, going up here actually makes you go DOWN 2 units at the end.

**Why it starts at 0.0**: We haven't calculated it yet! It gets filled in during backpropagation.

---

### `self._prev` (previous nodes)
**What it is**: A set containing the "parent" values that created this value

**Why we need it**: To remember how we got this number. If `d = a + b`, then `d._prev = {a, b}` because `a` and `b` made `d`.

**Analogy**: Your family tree! You have parents, they have parents, etc. Each Value knows who its parents are.

**Why it's a set**: We only care WHO the parents are, not the order, and we don't want duplicates.

```python
# Example:
a = Value(2.0)
b = Value(3.0)
c = a + b  # c._prev = {a, b}
```

---

### `self._op` (operation)
**What it is**: A string like `'+'`, `'*'`, or `'tanh'` that says WHAT operation created this value

**Why we need it**: For visualization and debugging. When you draw the computational graph, you want to see "oh, this node was created by multiplication"

**Analogy**: Like keeping a receipt. "How did you get this result?" "Oh, I multiplied two numbers!"

```python
c = a + b  # c._op = '+'
d = a * b  # d._op = '*'
```

---

### `self._backward`
**What it is**: A function that knows how to propagate gradients backward to this value's parents

**Why we need it**: Different operations have different gradient rules:
- For addition: gradient passes through equally to both parents
- For multiplication: gradient gets multiplied by the OTHER parent's value

**Analogy**: Each operation has its own "instruction manual" for sending blame backward. If the final answer is wrong, this function tells the parents "here's how much YOU contributed to the error"

**Why it starts as `lambda: None`**: Leaf nodes (inputs like weights) don't have parents, so they don't need to pass gradients backward to anyone.

---

### `self.label`
**What it is**: A friendly name like `'w1'` or `'loss'`

**Why we need it**: Pure convenience for debugging and visualization. Instead of seeing "Value at 0x7f9b..." you see "weight1"

**Analogy**: Name tags at a conference

---

## Functions (The Actions)

### `__init__` (constructor)
**What it does**: Creates a new Value object

**Why each parameter**:
- `data`: The actual number we're wrapping
- `_children=()`: Who created this? (empty if it's an input)
- `_op=''`: How was it created? (empty if it's an input)
- `label=''`: Optional name for debugging

---

### `__repr__`
**What it does**: Shows how to print this object nicely

```python
print(Value(5.0))  # Output: Value(data=5.0)
```

**Why we need it**: For debugging in Jupyter notebooks

---

### `__add__` (addition: a + b)
**What it does**: Adds two Values and creates a new Value

```python
a = Value(2.0)
b = Value(3.0)
c = a + b  # c.data = 5.0
```

**The backward function inside**:
```python
def _backward():
  self.grad += 1.0 * out.grad
  other.grad += 1.0 * out.grad
```

**Why this math**: Calculus tells us: if `out = self + other`, then:
- `d(out)/d(self) = 1.0`
- `d(out)/d(other) = 1.0`

**Analogy**: If you're adding ingredients to a pot, both ingredients contribute equally to the final amount. Adding 1 more cup of either ingredient increases the total by 1 cup.

**Why `+=` not `=`**: A value might be used multiple times! Like `b = a + a`. When gradients flow back, we need to ADD them up, not replace them.

---

### `__mul__` (multiplication: a * b)
**What it does**: Multiplies two Values

```python
a = Value(2.0)
b = Value(3.0)
c = a * b  # c.data = 6.0
```

**The backward function**:
```python
def _backward():
  self.grad += other.data * out.grad
  other.grad += self.data * out.grad
```

**Why this math**: Calculus says: if `out = self * other`, then:
- `d(out)/d(self) = other.data` (if I increase self by 1, out increases by the value of other)
- `d(out)/d(other) = self.data` (if I increase other by 1, out increases by the value of self)

**Analogy**: If you're calculating area (length × width):
- Making the length 1 unit bigger increases area by `width` square units
- Making the width 1 unit bigger increases area by `length` square units

---

### `tanh` (activation function)
**What it does**: Applies hyperbolic tangent, squashing any number to between -1 and +1

**Why we need it**: Neural networks need nonlinearity! Without it, stacking layers is useless.

**The backward function**:
```python
def _backward():
  self.grad += (1 - t**2) * out.grad
```

**Why this math**: The derivative of tanh(x) is `1 - tanh(x)²`. This is from calculus.

---

### `backward` (THE MAIN EVENT)
**What it does**: Calculates gradients for EVERY value in the computation graph

**The algorithm**:
1. **Build topological order**: Visit all nodes starting from this one, going backward to inputs
2. **Set this node's gradient to 1.0**: "I am the output, I have 100% influence on myself"
3. **Go backward through nodes**: Each node distributes its gradient to its parents using its `_backward` function

**Why topological sort**: You must compute gradients in the right order! You can't calculate parent gradients until you've calculated the child's gradient.

**Analogy**:
Imagine a chain of blame for a mistake:
1. The final output (the mistake) has gradient 1.0 (100% to blame for itself)
2. It tells its parents "you're partially to blame"
3. Those parents tell THEIR parents
4. This continues until we reach the inputs (the original decisions)

Now we know exactly how much each input contributed to the final mistake!

---

## The Big Picture

```python
# Forward pass: build the graph
a = Value(2.0)
b = Value(-3.0)
c = a * b  # c = -6.0

# Backward pass: calculate gradients
c.backward()

# Now we know:
print(a.grad)  # -3.0 (if a increases by 1, c changes by -3)
print(b.grad)  #  2.0 (if b increases by 1, c changes by 2)
```

This lets neural networks learn! They can compute how changing EACH weight affects the final loss, then adjust all weights in the right direction.

---

## Key Insights

1. **Forward pass builds the graph**: Each operation creates a new Value node and remembers its parents
2. **Backward pass calculates gradients**: Starting from the output, we propagate gradients backward using the chain rule
3. **Each operation knows its own derivative**: The `_backward` function encodes the local gradient rule
4. **Gradients accumulate with `+=`**: Because values can be reused multiple times in a computation
5. **Topological sort is crucial**: Ensures we calculate gradients in the correct order

This ~50 line class is essentially a miniature version of PyTorch's autograd engine!
