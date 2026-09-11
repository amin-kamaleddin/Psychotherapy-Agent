"""Paired Wilcoxon: full multi-agent (judge) vs proposer-only (single-call self-selection).

Applies multiple-testing corrections consistent with feedback_statistical_test.py:
- BH-FDR (Benjamini-Hochberg) across the criteria within this pair (assumes positive
  dependence / PRDS), matching the existing pipeline.
- BY-FDR (Benjamini-Yekutieli) as a more conservative robustness check.
"""
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

CRITERIA = [
    'Acceptance', 'Empathy', 'Direction', 'Collaboration', 'Autonomy Support',
    'Evocation', 'Partnership', 'Cultivating Change Talk', 'Softening Sustain Talk',
    'Overall Average',
]

df = pd.read_csv('Eval/csv/feedback_scores_by_criterion.csv')
df = df[df['therapist_mode'].isin(['multi_agent', 'proposer_only'])]

MODE1 = 'multi_agent'
MODE2 = 'proposer_only'

rows = []
for c in CRITERIA:
    d = df[df['criterion'] == c]
    p = d.pivot_table(index='patient_id', columns='therapist_mode', values='score')
    if MODE1 not in p.columns or MODE2 not in p.columns:
        continue
    p = p[[MODE1, MODE2]].dropna()
    if len(p) < 2:
        continue
    s1 = p[MODE1].values
    s2 = p[MODE2].values
    try:
        stat, pval = wilcoxon(s1, s2)
    except Exception:
        stat, pval = float('nan'), float('nan')
    diff = s1 - s2
    std = diff.std(ddof=1) if len(diff) > 1 else 0
    es = diff.mean() / std if std > 0 else 0
    rows.append({
        'Criterion': c,
        'N': len(p),
        f'{MODE1}_mean': round(s1.mean(), 3),
        f'{MODE2}_mean': round(s2.mean(), 3),
        'Mean_Difference': round(diff.mean(), 3),
        'Effect_Size': round(es, 4),
        'p_value': pval,
        'Significant': 'Yes' if pval < 0.05 else 'No',
    })

res = pd.DataFrame(rows)

if not res.empty:
    pvals = res['p_value'].values
    # BH-FDR across the criteria (matching the existing pipeline)
    reject_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res['p_fdr_bh'] = p_bh
    res['Significant_fdr_bh'] = ['Yes' if r else 'No' for r in reject_bh]
    # BY-FDR robustness check
    reject_by, p_by, _, _ = multipletests(pvals, alpha=0.05, method='fdr_by')
    res['p_fdr_by'] = p_by
    res['Significant_fdr_by'] = ['Yes' if r else 'No' for r in reject_by]

print(res.to_string(index=False))