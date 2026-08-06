import numpy as np
import matplotlib.pyplot as plt

# Random data
gen = np.random.default_rng(seed=42)
X0 = gen.uniform(0, 10, 1000)
X1 = gen.uniform(0, 5, 1000)
noise = gen.normal(0, 1, 1000)
y = 3 * X1 + 2 * X0 + 5 + noise

# Chart
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X0, X1, c=y, cmap="viridis", alpha=0.7)
ax.set_xlabel("X0")
ax.set_ylabel("X1")
ax.set_zlabel("y")
fig.colorbar(scatter, ax=ax, label="Wartość y", shrink=0.5)
plt.title("3D Chart: y with respect to X0 and X1")
plt.show()

# Initialization of the parameters of the linear equation
w0 = 0.0
w1 = 0.0
b = 0.0

# Batch size and permutation
batch = 32

loss_history = []
for epoch in range(300):
    idxs = gen.permutation(len(X0))
    X0_shuffled = X0[idxs]
    X1_shuffled = X1[idxs]
    y_shuffled = y[idxs]
    for i in range(0, len(X0), batch):
        X0_batch = X0_shuffled[i : i + batch]
        X1_batch = X1_shuffled[i : i + batch]
        y_batch = y_shuffled[i : i + batch]

    # Prediction
        y_pred = w1 * X1_batch + w0 * X0_batch + b

    # Loss function
        loss = np.sum((y_batch - y_pred) ** 2) / len(X0_batch)
        loss_history.append(loss)

    # Learning rate, derivative of loss w.r.t. (with respect to) w and derivative of loss w.r.t. b
        lr = 0.001
        dw0 = -(2/len(X0_batch)) * np.sum(X0_batch * (y_batch - y_pred))
        dw1 = -(2/len(X0_batch)) * np.sum(X1_batch * (y_batch - y_pred))
        db = -(2/len(X0_batch)) * np.sum(y_batch - y_pred)

    # Update parameters of slope - w and bias - b
        w0 = w0 - lr * dw0
        w1 = w1 - lr * dw1
        b = b - lr * db

plt.figure(figsize=(8, 5))
plt.plot(loss_history[-500:], color="red", label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
print(f"w1 = {w1}, w0 = {w0}, b = {b}")

'''
--- Part 1: single feature ---

Experiment: learning rate too small vs too large

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

'''
--- Part 2: two feature with mini-batches ---
## Experiment: full-batch vs mini-batch gradient descent

- Observation: full-batch gradient descent (one gradient computed over all 1000 
  points per update) produces a smooth, monotonically decreasing loss curve — 
  each step reliably moves toward the minimum. Mini-batch gradient descent 
  (32 points per update, ~31 updates per epoch) produces a noisy, jagged curve — 
  after convergence, loss oscillates roughly between 0.5 and 2.0 instead of 
  settling smoothly near the noise floor (~1.0).
- Interpretation: full-batch computes the exact average gradient over the entire 
  dataset, so every update points in a well-defined, consistent direction. 
  Mini-batch computes the gradient on a small, randomly shuffled subset each 
  time — this is only a noisy approximation of the true gradient, since different 
  batches contain different points (some batches may over-represent points the 
  model currently fits poorly, others may be easier). Each individual update can 
  therefore overshoot, undershoot, or move slightly off-direction relative to the 
  true minimum, which shows up as the jagged curve.
- Consequence: mini-batch trades per-step precision for speed and update 
  frequency — it makes ~31 parameter updates per epoch instead of 1, using far 
  less memory per step, which matters enormously once datasets become too large 
  to fit in memory or compute a full gradient over efficiently (the case for most 
  real neural network training). The added noise isn't purely a downside either — 
  it's the same mechanism mentioned earlier when discussing local minima: a noisy 
  gradient occasionally pushes the parameters out of shallow local dips that a 
  perfectly smooth, deterministic gradient descent could get stuck in.
'''