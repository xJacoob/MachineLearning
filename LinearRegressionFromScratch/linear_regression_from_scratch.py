import numpy as np
import matplotlib.pyplot as plt

# Random data
gen = np.random.default_rng(seed=42)
X = gen.uniform(0, 10, 1000)
noise = gen.normal(0, 1, 1000)
y = 3 * X + 5 + noise

# Chart
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color="blue", label="Training data")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

# Initialization of the parameters of the linear equation
w = 0.0
b = 0.0

loss_history = []
for epoch in range(10000):
    # Prediction
    y_pred = w * X + b

    # Loss function
    loss = np.sum((y - y_pred) ** 2) / len(X)
    loss_history.append(loss)

    # Learning rate, derivative of loss w.r.t. (with respect to) w and derivative of loss w.r.t. b
    lr = 0.001
    dw = -(2/len(X)) * np.sum(X * (y - y_pred))
    db = -(2/len(X)) * np.sum(y - y_pred)

    # Update parameters of slope - w and bias - b
    w = w - lr * dw
    b = b - lr * db

plt.figure(figsize=(8, 5))
plt.plot(loss_history, color="red", label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
print(f"w = {w}, b = {b}")

'''
## Experiment: learning rate too small vs too large

- lr=0.1: loss diverges immediately — grows exponentially every epoch (472 → 15,652 
  → 25 quadrillion+ within 10 epochs) instead of decreasing. On a linear-scale plot, 
  this divergence is invisible until the very last epochs, where values become 
  astronomically large — the loss was never actually near zero, the plot's scale 
  just hid the early exponential growth.
- lr=0.0001: loss decreases smoothly and never diverges, but converges extremely 
  slowly — after 20,000 epochs, w reached ~3.26 (true value 3) while b was still 
  only at ~3.25 (true value 5). w and b converge at very different speeds, because 
  the gradient for w is scaled by X (which ranges 0-10), making the loss surface 
  much steeper along the w-axis than along the b-axis. A single shared learning 
  rate that suits the steep w-direction is far too conservative for the flatter 
  b-direction.
- lr=0.001: a good middle ground — after 10,000 epochs, w≈3.02, b≈4.79, loss≈1.02 
  (close to the noise variance of 1, meaning the model recovered nearly all the 
  learnable signal).

Root cause of the too-large case: X is unscaled (range 0-10), so the gradient 
step for w (which involves multiplying by X) can massively overshoot the true 
minimum on a single update, causing the parameters to oscillate with ever-growing 
magnitude instead of settling down.

Root cause of the slow-b case: unscaled X creates an elongated, unevenly-curved 
loss surface — steep along w, flat along b — so no single learning rate is ideal 
for both parameters simultaneously.

Consequence: feature scaling (StandardScaler) isn't just useful for regularized 
models like Ridge/Lasso — it also directly speeds up and stabilizes gradient 
descent by making the loss surface more symmetric across parameters, allowing a 
much larger, safer learning rate and faster convergence for all parameters at 
once.
'''

