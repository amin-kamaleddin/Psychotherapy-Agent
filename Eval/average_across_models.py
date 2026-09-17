"""Average feedback scores across an arbitrary set of models.

Reads each model's `{model}_feedback_scores_by_criterion.csv` from Eval/csv (produced by
feedback_analysis.py when PSYCH_MODEL=<model> is set) and averages the per-patient score
across models for each (patient_id, therapist_mode, criterion).

Usage:
    python Eval/average_across_models.py --models gpt-5-mini qwen/qwen3.8-27b deepseek/deepseek-v4-flash
    python Eval/average_across_models.py   # auto-detect all <model>_feedback_scores_by_criterion.csv

Outputs:
    - Eval/csv/avg_feedback_scores_by_criterion.csv   (per-model + average rows)
    - Eval/csv/avg_feedback_metrics_{mode}.csv        (averaged mean/std per mode & criterion)
    - Eval/csv/avg_comparison.csv                     (mode means from the average, for plotting)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

CRITERIA = [
    'Acceptance', 'Empathy', 'Direction', 'Collaboration', 'Autonomy Support',
    'Evocation', 'Partnership', 'Cultivating Change Talk', 'Softening Sustain Talk',
    'Overall Average',
]

CSV_DIR = 'Eval/csv'
PREFIX = '_feedback_scores_by_criterion.csv'


def auto_detect_models():
    models = []
    for f in glob.glob(os.path.join(CSV_DIR, f'*{PREFIX}')):
        base = os.path.basename(f)[:-len(PREFIX)]
        if base:  # ignore the untagged 'feedback_scores_by_criterion.csv'
            models.append(base)
    return sorted(models)


def model_tag(model_id):
    """Sanitize a model id into a file-prefix tag (no '/', which would create nested paths)."""
    return model_id.replace("/", "-") if model_id else ""


def main():
    ap = argparse.ArgumentParser(description='Average feedback scores across N models')
    ap.add_argument('--models', nargs='*', default=None,
                    help='Model ids/tags to average (e.g. gpt-5-mini qwen/qwen3.8-27b). '
                         'Defaults to auto-detected tags.')
    args = ap.parse_args()

    # Map each provided model id to its sanitized file tag.
    models = [model_tag(m) for m in args.models] if args.models else auto_detect_models()
    if not models:
        print('No model-tagged score files found. Run feedback_analysis.py with PSYCH_MODEL set first.')
        return
    print('Averaging across models (tags):', models)

    frames = {}
    for m in models:
        path = os.path.join(CSV_DIR, f'{m}{PREFIX}')
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found, skipping {m}')
            continue
        df = pd.read_csv(path)
        df['model'] = m
        frames[m] = df
        print(f'  loaded {m}: {len(df)} rows')

    if not frames:
        print('No valid model files to average.')
        return

    all_df = pd.concat(frames.values(), ignore_index=True)
    key = ['patient_id', 'therapist_mode', 'criterion']

    # Per-model wide table of score
    wide = all_df.pivot_table(index=key, columns='model', values='score', aggfunc='first').reset_index()

    # Average row
    model_cols = [m for m in models if m in wide.columns]
    wide['avg_score'] = wide[model_cols].mean(axis=1)
    wide['num_models'] = wide[model_cols].notna().sum(axis=1)

    out = os.path.join(CSV_DIR, 'avg_feedback_scores_by_criterion.csv')
    wide.to_csv(out, index=False)
    print(f'\nSaved averaged per-patient scores -> {out} ({len(wide)} rows)')

    # Pipeline-compatible long format (matches feedback_scores_by_criterion.csv schema used by
    # feedback_statistical_test.py / plot_metrics.py): patient_id, therapist_mode, criterion, score.
    pipe = wide[['patient_id', 'therapist_mode', 'criterion', 'avg_score']].copy()
    pipe = pipe.rename(columns={'avg_score': 'score'})
    pipe_out = os.path.join(CSV_DIR, 'avg_feedback_pipeline.csv')
    pipe.to_csv(pipe_out, index=False)
    print(f'Saved pipeline-compatible averaged scores -> {pipe_out} ({len(pipe)} rows)')

    # Per-mode averaged metrics (mean/std over patients of avg_score)
    metrics_list = []
    for mode, g in wide.groupby('therapist_mode'):
        piv = g.pivot_table(index='criterion', values='avg_score', aggfunc=['mean', 'std', 'count'])
        piv = piv.reset_index()
        piv.columns = ['criterion', 'mean', 'std', 'count']
        piv['therapist_mode'] = mode
        metrics_list.append(piv)
    metrics = pd.concat(metrics_list, ignore_index=True)
    metrics['mean'] = metrics['mean'].round(3)
    metrics['std'] = metrics['std'].round(3)
    for mode in metrics['therapist_mode'].unique():
        mm = metrics[metrics['therapist_mode'] == mode].copy()
        outm = os.path.join(CSV_DIR, f'avg_feedback_metrics_{mode}.csv')
        mm.to_csv(outm, index=False)
        print(f'  avg_feedback_metrics_{mode}.csv -> {len(mm)} criteria')

    # Comparison table: mode x criterion mean + Overall Average
    comp_rows = []
    for mode, g in wide.groupby('therapist_mode'):
        for c in CRITERIA:
            sub = g[g['criterion'] == c]['avg_score']
            if len(sub):
                comp_rows.append({'therapist_mode': mode, 'criterion': c,
                                  'mean': sub.mean(), 'std': sub.std()})
    comp = pd.DataFrame(comp_rows)
    comp['mean'] = comp['mean'].round(3)
    comp['std'] = comp['std'].round(3)
    comp_out = os.path.join(CSV_DIR, 'avg_feedback_metrics_all.csv')
    comp.to_csv(comp_out, index=False)
    print(f'  avg_feedback_metrics_all.csv -> {len(comp)} rows')

    print('\n=== Overall Average by mode (averaged across models) ===')
    oa = comp[comp['criterion'] == 'Overall Average'].sort_values('mean', ascending=False)
    if not oa.empty:
        print(oa.to_string(index=False))


if __name__ == '__main__':
    main()