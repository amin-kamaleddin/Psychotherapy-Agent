"""Compare proposer_only (single-call self-selection) against all 4 manuscript modes.

Data sources:
- simple / trained_agent / complex / multi_agent: restored from origin/data-backup
  (133 patients x 5 folds, manuscript run).
- proposer_only: run on main (fold 1 per patient; averaged if multiple folds).

Method (consistent with Eval/feedback_statistical_test.py):
1. Per-patient mean across folds per mode -> matched samples.
2. Friedman test over the 5 modes (needs >=3 modes).
3. Pairwise Wilcoxon signed-rank tests (primarily multi_agent vs proposer_only,
   plus each baseline vs proposer_only), with Bonferroni + BH/BY-FDR corrections.

Outputs: Eval/csv/ablation_5mode_friedman.csv, Eval/csv/ablation_5mode_wilcoxon_*.csv
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

CRITERIA = [
    'Acceptance', 'Empathy', 'Direction', 'Collaboration', 'Autonomy Support',
    'Evocation', 'Partnership', 'Cultivating Change Talk', 'Softening Sustain Talk',
    'Overall Average',
]

BACKUP_FILES = {
    'simple':        'Session_Metadata/results_simple_backup.csv',
    'trained_agent': 'Session_Metadata/results_trained_agent_backup.csv',
    'complex':       'Session_Metadata/results_complex_backup.csv',
    'multi_agent':   'Session_Metadata/results_multi_agent_backup.csv',
}

OUT_DIR = 'Eval/csv'
os.makedirs(OUT_DIR, exist_ok=True)


def find_proposer_only_csv():
    """Find proposer_only results CSV. Prefer the full 133-patient fold-5 run (matches the
    backup's fold-5 baselines), falling back to the fold-1 133-patient run, then any results."""
    pref = 'Session_Metadata/proposer133_proposer_only_patients_0_133_fold5_20260911_104343/results_proposer_only.csv'
    if os.path.exists(pref):
        return pref
    fallback = 'Session_Metadata/proposer133_proposer_only_patients_0_133_20260911_102939/results_proposer_only.csv'
    if os.path.exists(fallback):
        return fallback
    cands = sorted(glob.glob('Session_Metadata/*proposer*/results_proposer_only.csv'))
    return cands[-1] if cands else None


def load_mode_csv(path, mode):
    df = pd.read_csv(path)
    df['mode'] = mode
    return df


def main():
    frames = [load_mode_csv(p, m) for m, p in BACKUP_FILES.items()]

    prop_path = find_proposer_only_csv()
    if not prop_path:
        print('ERROR: proposer_only results CSV not found. Run proposer_only first.')
        return
    print(f'Loading proposer_only from: {prop_path}')
    frames.append(load_mode_csv(prop_path, 'proposer_only'))

    data = pd.concat(frames, ignore_index=True)

    # Averaging across folds per patient per mode -> matched samples
    pivot_by = data.groupby(['Patient', 'mode'])[CRITERIA].mean().reset_index()

    print('\n=== Per-mode patient counts ===')
    print(pivot_by.groupby('mode')['Patient'].nunique())

    print('\n=== Per-mode mean +/- std (averaged across patients) ===')
    for c in ['Overall Average'] + CRITERIA:
        g = pivot_by.groupby('mode')[c]
        print(f'{c:34s} ' + '  '.join(f'{m}: {g.get_group(m).mean():.3f}±{g.get_group(m).std():.3f}'
                                      for m in g.groups))

    # 1) Friedman over modes that have matched patients
    modes = sorted(pivot_by['mode'].unique())
    common_patients = None
    for m in modes:
        pats = set(pivot_by[pivot_by['mode'] == m]['Patient'])
        common_patients = pats if common_patients is None else (common_patients & pats)
    common_patients = sorted(common_patients)
    print(f'\nMatched patients across {len(modes)} modes: {len(common_patients)}')

    friedman_rows = []
    if len(modes) >= 3 and len(common_patients) >= 3:
        piv = pivot_by.pivot_table(index='Patient', columns='mode', values=CRITERIA[0])
        # use all criteria
        for c in CRITERIA:
            cols = {m: pivot_by[(pivot_by['mode'] == m) &
                                (pivot_by['Patient'].isin(common_patients))].set_index('Patient')[c]
                    for m in modes}
            mdf = pd.DataFrame(cols).dropna()
            if len(mdf) < 3 or len(mdf.columns) < 3:
                continue
            stat, p = friedmanchisquare(*[mdf[m].values for m in mdf.columns])
            avail = list(mdf.columns)
            friedman_rows.append({
                'Criterion': c, 'N': len(mdf),
                'Friedman_Chi2': round(stat, 4), 'p_value': p,
                'Significant': 'Yes' if p < 0.05 else 'No',
                'Modes': '+'.join(avail),
                **{f'{m}_mean': round(mdf[m].mean(), 3) for m in avail},
            })
    if friedman_rows:
        fr = pd.DataFrame(friedman_rows)
        pvals = fr['p_value'].values
        rb, pb, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        fr['p_fdr_bh'] = pb
        fr['Significant_fdr_bh'] = ['Yes' if r else 'No' for r in rb]
        out = os.path.join(OUT_DIR, 'ablation_5mode_friedman.csv')
        fr.to_csv(out, index=False)
        print(f'\n[FRIEDMAN] saved -> {out}')
        print(fr.to_string(index=False))

    # 2) Pairwise Wilcoxon (focus: multi_agent vs proposer_only, plus each vs proposer_only)
    pairs = [
        ('multi_agent', 'proposer_only'),
        ('complex', 'proposer_only'),
        ('trained_agent', 'proposer_only'),
        ('simple', 'proposer_only'),
        ('multi_agent', 'complex'),
        ('multi_agent', 'simple'),
    ]
    for m1, m2 in pairs:
        if m1 not in modes or m2 not in modes:
            continue
        rows = []
        for c in CRITERIA:
            a = pivot_by[(pivot_by['mode'] == m1) & (pivot_by['Patient'].isin(common_patients))].set_index('Patient')[c]
            b = pivot_by[(pivot_by['mode'] == m2) & (pivot_by['Patient'].isin(common_patients))].set_index('Patient')[c]
            j = pd.concat([a, b], axis=1, keys=[m1, m2]).dropna()
            if len(j) < 2:
                continue
            s1, s2 = j[m1].values, j[m2].values
            try:
                stat, p = wilcoxon(s1, s2)
            except Exception:
                stat, p = np.nan, np.nan
            diff = s1 - s2
            std = diff.std(ddof=1) if len(diff) > 1 else 0
            es = diff.mean() / std if std > 0 else 0
            rows.append({
                'Criterion': c, 'N': len(j),
                f'{m1}_mean': round(s1.mean(), 3), f'{m2}_mean': round(s2.mean(), 3),
                'Mean_Difference': round(diff.mean(), 3), 'Effect_Size': round(es, 4),
                'p_value': p, 'Significant': 'Yes' if p < 0.05 else 'No',
            })
        if rows:
            rr = pd.DataFrame(rows)
            pv = rr['p_value'].values
            n_pairs = len(pairs)
            rr['p_bonferroni'] = np.minimum(pv * n_pairs, 1.0)
            rr['Significant_bonferroni'] = ['Yes' if x < 0.05 else 'No' for x in rr['p_bonferroni']]
            rbh, pbh, _, _ = multipletests(pv, alpha=0.05, method='fdr_bh')
            rr['p_fdr_bh'] = pbh
            rr['Significant_fdr_bh'] = ['Yes' if r else 'No' for r in rbh]
            out = os.path.join(OUT_DIR, f'ablation_wilcoxon_{m1}_vs_{m2}.csv')
            rr.to_csv(out, index=False)
            print(f'\n[WILCOXON] {m1} vs {m2} -> {out}')
            print(rr.to_string(index=False))


if __name__ == '__main__':
    main()