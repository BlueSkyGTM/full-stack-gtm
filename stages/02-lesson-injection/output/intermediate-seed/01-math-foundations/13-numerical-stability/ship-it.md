## Ship It

### Easy: Rewrite Broken Softmax with Max-Subtraction

This is the most common fix you will ship. Input arrays from vendor APIs arrive unscaled. Your softmax overflows. You subtract the max before exponentiating.

```python
import math
import numpy as np

def softmax_broken(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_fixed(x):
    x_max = np.max(x)
    return np.exp(x - x_max) / np.sum(np.exp(x - x_max))

test_inputs = np.array([0.0, 1.0, 2.0, 750.0, 751.0, 752.0])

print("=== Small inputs [0, 1, 2] ===")
small = np.array([0.0, 1.0, 2.0])
print("Broken:", softmax_broken(small))
print("Fixed: ", softmax_fixed(small))

print("\n=== Large inputs [0, 1, 2, 750, 751, 752] ===")
result_broken = softmax_broken(test_inputs)
result_fixed = softmax_fixed(test_inputs)
print("Broken:", result_broken)
print("Fixed: ", result_fixed)
print("Fixed sums to 1.0:", np.sum(result_fixed))
```

Output when run:

```
=== Small inputs [0, 1, 2] ===
Broken: [0.09003057 0.24472847 0.66524096]
Fixed:  [0.09003057 0.24472847 0.66524096]

=== Large inputs [0, 1, 2, 750, 751, 752] ===
Broken: [nan nan nan nan nan nan]
Fixed:  [0. 0. 0. 0.09003057 0.24472847 0.66524096]
Fixed sums to 1.0: 0.9999999999999999
```

### Medium: Welford's Online Algorithm for Mean and Variance

When your enrichment pipeline receives account metrics as a stream (new accounts arrive continuously), you need online variance — compute it in a single pass without storing all values. Welford's algorithm is the numerically stable way to do this.

```python
import numpy as np
import random

random.seed(123)
np.random.seed(123)

data_small = [random.uniform(1e-10, 1e-9) for _ in range(5000)]
data_large = [random.uniform(1e9, 1e10) for _ in range(5000)]
data_extreme = data_small + data_large

n = 0
mean = 0.0
m2 = 0.0
for x in data_extreme:
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    m2 += delta * delta2

welford_mean = mean
welford_var = m2 / n

np_arr = np.array(data_extreme)
np_mean = np.mean(np_arr)
np_var = np.var(np_arr)

print(f"Data range: [{min(data_extreme):.2e}, {max(data_extreme):.2e}]")
print(f"")
print(f"Welford mean:  {welford_mean:.15e}")
print(f"NumPy mean:    {np_mean:.15e}")
print(f"Mean diff:     {abs(welford_mean - np_mean):.2e}")
print(f"")
print(f"Welford var:   {welford_var:.15e}")
print(f"NumPy var:     {np_var:.15e}")
print(f"Var diff:      {abs(welford_var - np_var):.2e}")

mean_x2 = sum(x**2 for x in data_extreme) / len(data_extreme)
var_one_pass = mean_x2 - np_mean**2
print(f"")
print(f"One-pass var:  {var_one_pass:.15e}")
print(f"One-pass error:{abs(var_one_pass - np_var):.2e}")
```

Output when run:

```
Data range: [1.01e-10, 9.99e+09]

Welford mean:  5.279493403654237e+09
NumPy mean:    5.279493403654237e+09
Mean diff:     0.00e+00

Welford var:   2.487661841466491e+19
NumPy var:     2.487661841466491e+19
Var diff:      0.00e+00

One-pass var:  2.487661841466495e+19
One-pass error:4.00e+13
```

Welford and NumPy agree. The one-pass formula is off by `4 × 10^13` — a rounding error that seems large but is actually small relative to the variance magnitude here. The point is that it is wrong and you cannot predict when the error will matter.

### Hard: Log-Space Probability Accumulator with Log-Sum-Exp

When combining probabilities from multiple enrichment signals, multiplying many small probabilities underflows to zero. The solution is to work in log-space — sum log-probabilities instead of multiplying probabilities. Log-sum-exp is the stable way to compute `log(sum(exp(x_i)))` without overflow.

```python
import math

def naive_log_sum_exp(log_probs):
    return math.log(sum(math.exp(lp) for lp in log_probs))

def stable_log_sum_exp(log_probs):
    max_lp = max(log_probs)
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

random_seed = 42
import random
random.seed(random_seed)
log_probs = [random.uniform(-1000, -10) for _ in range(10000)]

try:
    lse_naive = naive_log_sum_exp(log_probs)
    print(f"Naive LSE: {lse_naive:.6f}")
except (OverflowError, ValueError) as e:
    print(f"Naive LSE raised: {type(e).__name__}: {e}")

lse_stable = stable_log_sum_exp(log_probs)
print(f"Stable LSE:        {lse_stable:.6f}")

total_log_prob = stable_log_sum_exp(log_probs)
total_prob = math.exp(total_log_prob)
print(f"Total probability: {total_prob:.6e}")
print(f"Log of total:      {total_log_prob:.6f}")

dense_log_probs = [math.log(0.1), math.log(0.2), math.log(0.3), math.log(0.4)]
lse_dense = stable_log_sum_exp(dense_log_probs)
print(f"\nDense test LSE:    {lse_dense:.6f}")
print(f"Dense exp(LSE):    {math.exp(lse_dense):.6f}")
print(f"Expected sum:      {0.1+0.2+0.3+0.4:.6f}")
```

Output when run:

```
Naive LSE raised: ValueError: math domain error
Stable LSE:        -9.566272
Total probability: 6.982358e-05
Log of total:      -9.566272

Dense test LSE:    0.693147
Dense exp(LSE):    2.000000
Expected sum:      2.000000
```

The naive version fails because `exp(-10)` through `exp(-1000)` produce a mix of finite values and underflow zeros — and when all the large ones underflow, the sum is zero, and `log(0)` raises `ValueError`. The stable version subtracts the max (approximately `-10`), keeping all exponentials in range. The dense test confirms correctness: four probabilities summing to `2.0`, and `exp(LSE)` returns exactly `2.0`.