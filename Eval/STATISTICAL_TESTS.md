# Statistical Testing Methodology

This document explains how all statistical tests and corrections in `feedback_statistical_test.py` are computed.

---

## Overview

The analysis compares 3 therapist modes (**simple**, **complex**, **multi_agent**) across 10 MI criteria using a two-stage procedure:

1. **Friedman test** — omnibus test: is there *any* difference across the 3 modes?
2. **Pairwise Wilcoxon tests** — post-hoc: *which* pairs differ?

Both stages apply multiple testing corrections.

---

## Step 1 — Data Shaping (per criterion)

For each criterion (e.g. *Empathy*), the raw session-level data is pivoted into a matrix of **N = 133 rows × 3 columns**, where:
- Each **row** is one patient
- Each **column** is a therapist mode
- Each **cell** is the mean score across all 5 folds that patient had in that mode

$$
\text{pivot}[i, m] = \frac{1}{K}\sum_{k=1}^{K} \text{score}_{i,m,k}, \quad K = 5 \text{ folds}
$$

This produces **matched/paired samples** — the same 133 patients appear in all 3 modes, which is required by both the Friedman and Wilcoxon tests.

---

## Step 2 — Friedman Test (`Friedman_Chi2`, `p_value`)

The Friedman test is a non-parametric alternative to repeated-measures ANOVA. It ranks mode scores **within each patient row**, then tests whether the rank sums differ more than chance.

**For each patient $i$**, rank the 3 scores (1 = lowest, 3 = highest):

$$
r_{i,m} \in \{1, 2, 3\}
$$

**Sum ranks per mode** across all N patients:

$$
R_m = \sum_{i=1}^{N} r_{i,m}
$$

**Test statistic** (N = 133 patients, k = 3 modes):

$$
\chi^2_F = \frac{12}{Nk(k+1)} \sum_{m=1}^{k} R_m^2 - 3N(k+1)
$$

This follows a $\chi^2$ distribution with $k - 1 = 2$ degrees of freedom. The p-value is:

$$
p = P\!\left(\chi^2_2 \geq \chi^2_F\right)
$$

**Example** — *Empathy*: $\chi^2_F = 195.28$, $p \approx 0$ → strong evidence that the 3 modes differ on Empathy.

---

## Step 3 — BH-FDR Correction on Friedman Tests (`p_fdr_bh`, `Significant_fdr`)

Since 10 Friedman tests run simultaneously (one per criterion), we correct for multiple comparisons using the **Benjamini-Hochberg False Discovery Rate** procedure:

1. Sort the 10 raw p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(10)}$
2. Compute adjusted p-values:

$$
p^{BH}_{(j)} = p_{(j)} \times \frac{10}{j}
$$

3. Enforce monotonicity by taking the cumulative minimum from the largest rank downward.

**Interpretation**: BH-FDR controls the *False Discovery Rate* at 5% — among all criteria declared significant, at most 5% are expected to be false positives. It is less conservative than Bonferroni and appropriate when testing multiple related outcomes (MI criteria).

**Result**: All 10 criteria survive BH-FDR correction (p-values are all ~0).

---

## Step 4 — Pairwise Wilcoxon Signed-Rank Tests (`Wilcoxon_Statistic`, `p_value`)

For each of the 3 mode pairs and each criterion, the Wilcoxon signed-rank test compares **paired** patient scores:

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

Under $H_0$ (no difference), $W$ follows the Wilcoxon signed-rank distribution. A two-sided test is used (`alternative='two-sided'`).

**The 3 pairs tested**:
- `multi_agent` vs `complex`
- `complex` vs `simple`
- `multi_agent` vs `simple`

---

## Step 5 — Bonferroni Correction for 3 Pairs (`p_bonferroni`, `Significant_bonferroni`)

There are $\binom{3}{2} = 3$ pairwise comparisons (post-hoc family). Bonferroni multiplies each raw p-value by 3, capped at 1:

$$
p^{\text{Bonf}} = \min(p \times 3,\ 1)
$$

**Significance threshold**: $\alpha^{\text{Bonf}} = 0.05 / 3 \approx 0.0167$

**Interpretation**: Bonferroni controls the *Family-Wise Error Rate* (FWER) — the probability of making *any* false positive across the 3 comparisons. It is the standard post-hoc correction after a significant omnibus test.

**Example** — *Direction* (multi_agent vs complex): $p = 0.035 \rightarrow p^{\text{Bonf}} = 0.106$ → not significant after correction (and notably, complex slightly outperforms multi_agent on this criterion).

---

## Step 6 — BH-FDR Correction Across 10 Criteria Within Each Pair (`p_fdr_bh`, `Significant_fdr`)

Separately from the Bonferroni correction across pairs, BH-FDR is also applied **within each pair** across the 10 criteria simultaneously:

$$
p^{BH}_{(j)} = p_{(j)} \times \frac{10}{j}
$$

This corrects for testing all 10 MI criteria in a single pairwise comparison and is less conservative than Bonferroni.

---

## Step 7 — Effect Size (`Effect_Size`)

Cohen's *d* for paired samples measures the **practical magnitude** of the difference:

$$
d = \frac{\bar{d}}{\text{SD}(d_i)}
$$

where $\bar{d}$ is the mean of individual differences $d_i = \text{score}_{i,\text{mode}_1} - \text{score}_{i,\text{mode}_2}$, and $\text{SD}(d_i)$ is their standard deviation.

**Conventional thresholds** (Cohen, 1988):

| Effect size `d` | Interpretation |
|:---:|:---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |
| > 1.0 | Very large |

**Examples**:

| Criterion | Comparison | d | Interpretation |
|:---|:---|:---:|:---|
| Empathy | multi_agent vs complex | 0.74 | Large |
| Evocation | multi_agent vs simple | 2.51 | Very large |
| Direction | multi_agent vs complex | -0.13 | Negligible (complex slightly higher) |

---

## Summary of Corrections Applied

| Test | Family | Correction | Controls |
|---|---|---|---|
| Friedman (10 criteria) | 10 simultaneous omnibus tests | BH-FDR | False Discovery Rate |
| Wilcoxon post-hoc (3 pairs) | 3 pairwise comparisons | Bonferroni (×3) | Family-Wise Error Rate |
| Wilcoxon (10 criteria per pair) | 10 criteria within each pair | BH-FDR | False Discovery Rate |

---

## Output Files

| File | Contents |
|---|---|
| `csv/friedman_test_results.csv` | Friedman $\chi^2$, raw p, BH-FDR p, means/SDs per criterion |
| `csv/wilcoxon_test_multi_agent_vs_complex.csv` | Wilcoxon W, raw p, Bonferroni p, BH-FDR p, effect size |
| `csv/wilcoxon_test_complex_vs_simple.csv` | Same for complex vs simple |
| `csv/wilcoxon_test_multi_agent_vs_simple.csv` | Same for multi_agent vs simple |
