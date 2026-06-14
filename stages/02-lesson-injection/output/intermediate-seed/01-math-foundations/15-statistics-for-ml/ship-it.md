## Ship It

Build a Python script that takes a CSV of account scores with conversion labels, runs statistical tests to determine whether the scoring model has signal, and prints a go/no-go recommendation. This is the validation step that precedes any predictive scoring workflow — the Python equivalent of the enrichment environment where you will run Clay webhooks and API calls in Zone 2 GTM work.

```python
import csv
import math
import random

random.seed(42)

def generate_account_scores_csv(filename, n_accounts=500):
    random.seed(42)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['account_id', 'score', 'converted'])
        for i in range(n_accounts):
            if random.random() < 0.15:
                score = max(0.0, min(1.0, random.gauss(0.68, 0.12)))
                converted = 1 if random.random() < 0.45 else 0
            else:
                score = max(0.0, min(1.0, random.gauss(0.35, 0.14)))
                converted = 1 if random.random() < 0.08 else 0
            writer.writerow([f"ACC-{i:04d}", f"{score:.4f}", converted])
    return filename

def load_scores(filename):
    converted = []
    not_converted = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row['score'])
            if int(row['converted']) == 1:
                converted.append(score)
            else:
                not_converted.append(score)
    return converted, not_converted

def mean(data):
    return sum(data) / len(data)

def median(data):
    s = sorted(data)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]

def variance(data, ddof=1):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)

def std_dev(data, ddof=1):
    return math.sqrt(variance(data, ddof))

def cohens_d(group1, group2):
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std_dev(group1), std_dev(group2)
    pooled_sd = math.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled_sd if pooled_sd > 0 else 0

def welch_t_test(group1, group2):
    m1, m2 = mean(group1), mean(group2)
    v1, v2 = variance(group1), variance(group2)
    n1, n2 = len(group1), len(group2)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 0 else 0
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else 1
    return t_stat, df

def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def permutation_test(sample_a, sample_b, num_permutations=10000):
    observed_diff = abs(mean(sample_a) - mean(sample_b))
    combined = sample_a + sample_b
    n_a = len(sample_a)
    count_extreme = 0
    for _ in range(num_permutations):
        shuffled = combined[:]
        random.shuffle(shuffled)
        perm_diff = abs(mean(shuffled[:n_a]) - mean(shuffled[n_a:]))
        if perm_diff >= observed_diff:
            count_extreme += 1
    return count_extreme / num_permutations

def bootstrap_ci(data, num_bootstrap=5000, confidence=0.95):
    n = len(data)
    boot_means = []
    for _ in range(num_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = 1 - confidence
    return boot_means[int((alpha / 2) * num_bootstrap)], boot_means[int((1 - alpha / 2) * num_bootstrap)]

def evaluate_scoring_model(filename):
    converted, not_converted = load_scores(filename)

    print("=" * 72)
    print("ACCOUNT SCORING MODEL VALIDATION")
    print("=" * 72)

    print(f"\n1. SAMPLE SUMMARY")
    print(f"   Converted accounts:     n={len(converted)}")
    print(f"     Mean score:   {mean(converted):.4f}")
    print(f"     Median score: {median(converted):.4f}")
    print(f"     Std dev:      {std_dev(converted):.4f}")
    print(f"   Non-converted accounts: n={len(not_converted)}")
    print(f"     Mean score:   {mean(not_converted):.4f}")
    print(f"     Median score: {median(not_converted):.4f}")
    print(f"     Std dev:      {std_dev(not_converted):.4f}")

    print(f"\n2. WELCH'S T-TEST (does not assume equal variance)")
    t_stat, df = welch_t_test(converted, not_converted)
    p_two_tailed = 2 * (1 - normal_cdf(abs(t_stat)))
    print(f"   t-statistic:      {t_stat:.4f}")
    print(f"   degrees of freedom: {df:.1f}")
    print(f"   Approximate p-value: {p_two_tailed:.6f}")
    print(f"   Significant at alpha=0.05? {'YES' if p_two_tailed < 0.05 else 'NO'}")

    print(f"\n3. PERMUTATION TEST (distribution-free validation)")
    perm_p = permutation_test(converted, not_converted, num_permutations=5000)
    print(f"   p-value: {perm_p:.4f}")
    print(f"   Significant at alpha=0.05? {'YES' if perm_p < 0.05 else 'NO'}")

    print(f"\n4. EFFECT SIZE (practical significance)")
    d = cohens_d(converted, not_converted)
    print(f"   Cohen's d: {d:.4f}")
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"
    print(f"   Interpretation: {interp}")
    print(f"   (Even if p-value is significant, d < 0.2 means the")
    print(f"    difference is too small to matter operationally)")

    print(f"\n5. BOOTSTRAP 95% CI FOR MEAN DIFFERENCE")
    combined_diffs = []
    for _ in range(5000):
        boot_conv = [random.choice(converted) for _ in range(len(converted))]
        boot_non = [random.choice(not_converted) for _ in range(len(not_converted))]
        combined_diffs.append(mean(boot_conv) - mean(boot_non))
    combined_diffs.sort()
    lo = combined_diffs[int(0.025 * 5000)]
    hi = combined_diffs[int(0.975 * 5000)]
    print(f"   Mean difference: {mean(converted) - mean(not_converted):.4f}")
    print(f"   95% CI:          [{lo:.4f}, {hi:.4f}]")
    print(f"   CI excludes 0?   {'YES — signal is real' if lo > 0 or hi < 0 else 'NO — cannot rule out no difference'}")

    print(f"\n{'=' * 72}")
    print("GO / NO-GO RECOMMENDATION")
    print(f"{'=' * 72}")

    checks = {
        "t-test significant": p_two_tailed < 0.05,
        "permutation test significant": perm_p < 0.05,
        "effect size >= small (d >= 0.2)": abs(d) >= 0.2,
        "bootstrap CI excludes 0": lo > 0 or hi < 0,
    }

    passed = sum(checks.values())
    total = len(checks)

    for check, result in checks.items():
        status =