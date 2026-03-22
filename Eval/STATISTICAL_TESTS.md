# Statistical Testing Methodology

This document explains how all statistical tests and corrections in `feedback_statistical_test.py` are computed.

---

## Overview

The analysis compares **4 therapist modes** (`simple`, `trained_agent`, `complex`, `multi_agent`) across **10 MI criteria** using a two-stage procedure:

1. **Friedman test** — omnibus test: is there *any* difference across the 4 modes?
2. **Pairwise Wilcoxon tests** — post-hoc: *which* pairs differ?

Both stages apply multiple testing corrections.

---

## Step 1 — Data Shaping (per criterion)

For each criterion (e.g. *Empathy*), the raw session-level data is pivoted into a matrix of **N rows × 4 columns**, where:
- Each **row** is one patient
- Each **column** is a therapist mode
- Each **cell** is the mean score across all folds that patient had in that mode

$$
\text{pivot}[i, m] = \frac{1}{K}\sum_{k=1}^{K} \text{score}_{i,m,k}
$$

This produces **matched/paired samples** — the same patients appear in all 4 modes, which is required by both the Friedman and Wilcoxon tests.

---

## Step 2 — Friedman Test (`Friedman_Chi2`, `p_value`)

The Friedman test is a non-parametric alternative to repeated-measures ANOVA. It ranks mode scores **within each patient row**, then tests whether the rank sums differ more than chance.

**For each patient $i$**, rank the 4 scores (1 = lowest, 4 = highest):

$$
r_{i,m} \in \{1, 2, 3, 4\}
$$

**Sum ranks per mode** across all N patients:

$$
R_m = \sum_{i=1}^{N} r_{i,m}
$$

**Test statistic** (N patients, k = 4 modes):

$$
\chi^2_F = \frac{12}{Nk(k+1)} \sum_{m=1}^{k} R_m^2 - 3N(k+1)
$$

This follows a $\chi^2$ distribution with $k - 1 = 3$ degrees of freedom. The p-value is:

$$
p = P\!\left(\chi^2_3 \geq \chi^2_F\right)
$$

---

## Step 3 — Multiple Testing Corrections on Friedman Tests

Since 10 Friedman tests run simultaneously (one per criterion), two FDR corrections are applied:

### 3a — BH-FDR (`p_fdr_bh`, `Significant_fdr_bh`)

**Benjamini-Hochberg** controls the False Discovery Rate assuming *positive regression dependence* (PRDS) among the test statistics — appropriate here because MI criteria are positively correlated.

1. Sort the 10 raw p-values ascending: $p_{(1)} \leq \cdots \leq p_{(10)}$
2. Compute adjusted p-values:

$$
p^{BH}_{(j)} = p_{(j)} \times \frac{10}{j}
$$

3. Enforce monotonicity from the largest rank downward.

### 3b — BY-FDR (`p_fdr_by`, `Significant_fdr_by`)

**Benjamini-Yekutieli** controls the False Discovery Rate under *arbitrary or unknown dependence* — a more conservative robustness check. It multiplies by the harmonic number $c(10) = \sum_{i=1}^{10} \frac{1}{i} \approx 2.93$:

$$
p^{BY}_{(j)} = p_{(j)} \times \frac{10 \cdot c(10)}{j}
$$

**Interpretation**: If results survive BY-FDR, they are robust even if the dependence structure between criteria is unknown or adversarial. Including both BH and BY allows reporting that findings hold under both the standard and the most conservative FDR procedure.

---

## Step 4 — Pairwise Wilcoxon Signed-Rank Tests (`Wilcoxon_Statistic`, `p_value`)

For each of the $\binom{4}{2} = 6$ mode pairs and each criterion, the Wilcoxon signed-rank test compares **paired** patient scores:

1. Compute signed differences per patient:

$$
d_i = \text{score}_{i,\text{mode}_1} - \text{score}_{i,\text{mode}_2}
$$

2. Discard ties ($d_i = 0$)
3. Rank the absolute differences $|d_i|$
4. Sum ranks of positive differences ($W^+$) and negative differences ($W^-$):

$$
W = \min(W^+,\ W^-)
$$

Under $H_0$ (no difference between modes), each observed difference $d_i$ is equally likely to be positive or negative — i.e. the sign of each $d_i$ is an independent fair coin flip. The $n$ non-zero ranks $\{1, 2, \ldots, n\}$ are fixed by their absolute values, but each rank is assigned a $+$ or $-$ sign with probability $\tfrac{1}{2}$. Therefore $W^+$ is the sum of a random subset of $\{1, \ldots, n\}$, giving the **exact null distribution**:

$$
E[W^+] = \frac{n(n+1)}{4}, \qquad \text{Var}(W^+) = \frac{n(n+1)(2n+1)}{24}
$$

For small $n$ the exact distribution is enumerated; for large $n$ (as in this study), a normal approximation is used:

$$
z = \frac{W^+ - \dfrac{n(n+1)}{4}}{\sqrt{\dfrac{n(n+1)(2n+1)}{24}}} \;\xrightarrow{d}\; \mathcal{N}(0,1)
$$

A two-sided test is used (`alternative='two-sided'`), so $p = 2 \cdot P(Z \geq |z|)$.

**The 6 pairs tested** (ordered by expected effect direction):

| Pair | Description |
|---|---|
| `multi_agent` vs `trained_agent` | Most complex vs second-most complex |
| `multi_agent` vs `complex` | Most complex vs middle |
| `multi_agent` vs `simple` | Most complex vs baseline |
| `trained_agent` vs `complex` | Trained vs middle |
| `complex` vs `simple` | Middle vs baseline |
| `trained_agent` vs `simple` | Trained vs baseline |

---

## Step 5 — Bonferroni Correction for 6 Pairs (`p_bonferroni`, `Significant_bonferroni`)

There are $\binom{4}{2} = 6$ pairwise comparisons (post-hoc family). Bonferroni multiplies each raw p-value by 6, capped at 1:

$$
p^{\text{Bonf}} = \min(p \times 6,\ 1)
$$

**Significance threshold**: $\alpha^{\text{Bonf}} = 0.05 / 6 \approx 0.0083$

**Interpretation**: Bonferroni controls the *Family-Wise Error Rate* (FWER) — the probability of making *any* false positive across the 6 comparisons. It is the standard post-hoc correction after a significant omnibus Friedman test.

---

## Step 6 — FDR Corrections Across 10 Criteria Within Each Pair

Separately from the Bonferroni correction across pairs, both FDR methods are also applied **within each pair** across the 10 criteria:

### 6a — BH-FDR (`p_fdr_bh`, `Significant_fdr_bh`)

$$
p^{BH}_{(j)} = p_{(j)} \times \frac{10}{j}
$$

Appropriate when MI criteria are positively correlated (less conservative).

### 6b — BY-FDR (`p_fdr_by`, `Significant_fdr_by`)

$$
p^{BY}_{(j)} = p_{(j)} \times \frac{10 \cdot c(10)}{j}, \quad c(10) \approx 2.93
$$

Valid under arbitrary dependence (more conservative robustness check).

---

## Step 7 — Effect Size (`Effect_Size`)

Cohen's *d* for paired samples measures the **practical magnitude** of the difference:

$$
d = \frac{\bar{d}}{\text{SD}(d_i)}
$$

where:

$$
d_i = \text{score}_{i,\text{mode}_1} - \text{score}_{i,\text{mode}_2}, \qquad \bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i
$$

and $\text{SD}(d_i)$ is the sample standard deviation of those differences (ddof=1).

**Conventional thresholds** (Cohen, 1988):

| Effect size `d` | Interpretation |
|:---:|:---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |
| > 1.0 | Very large |

---

## Summary of Corrections Applied

| Test | Family | Correction | Controls | Assumption |
|---|---|---|---|---|
| Friedman (10 criteria) | 10 omnibus tests | BH-FDR | False Discovery Rate | Positive dependence (PRDS) |
| Friedman (10 criteria) | 10 omnibus tests | BY-FDR | False Discovery Rate | Arbitrary dependence |
| Wilcoxon post-hoc (6 pairs) | 6 pairwise comparisons | Bonferroni (×6) | Family-Wise Error Rate | None |
| Wilcoxon (10 criteria per pair) | 10 criteria within each pair | BH-FDR | False Discovery Rate | Positive dependence (PRDS) |
| Wilcoxon (10 criteria per pair) | 10 criteria within each pair | BY-FDR | False Discovery Rate | Arbitrary dependence |

---

## Output Files

| File | Contents |
|---|---|
| `csv/friedman_test_results.csv` | Friedman $\chi^2$, raw p, BH-FDR p, BY-FDR p, means/SDs per criterion |
| `csv/wilcoxon_test_multi_agent_vs_trained_agent.csv` | Wilcoxon W, raw p, Bonferroni p, BH-FDR p, BY-FDR p, effect size |
| `csv/wilcoxon_test_multi_agent_vs_complex.csv` | Same for multi_agent vs complex |
| `csv/wilcoxon_test_multi_agent_vs_simple.csv` | Same for multi_agent vs simple |
| `csv/wilcoxon_test_trained_agent_vs_complex.csv` | Same for trained_agent vs complex |
| `csv/wilcoxon_test_complex_vs_simple.csv` | Same for complex vs simple |
| `csv/wilcoxon_test_trained_agent_vs_simple.csv` | Same for trained_agent vs simple |
