"""Average LSM / sentiment / word metrics across multiple models.

After running lsm_analysis.py / sentiment_analysis.py / word_analysis.py per model with
PSYCH_MODEL set, this script reads the model-tagged outputs and averages them across models.

Per-model outputs expected:
  - {model}_lsm_metrics_{mode}.csv               (turn_number, avg_lsm, std_lsm, count)
  - {model}_sentiment_dataset_by_turn.csv        (patient_id, therapist_mode, turn_number, prop_pos, prop_neu, prop_neg)
  - {model}_metrics_output_{mode}.csv            (turn, hdr: counselor/... )

Usage:
    python Eval/average_lang_metrics.py --models gpt-5-mini qwen-qwen3.8-27b z-ai-glm-5.3-flash
"""
import argparse, glob, os
import pandas as pd

CSV_DIR = 'Eval/csv'


def model_tag(mid):
    return mid.replace("/", "-") if mid else ""


def avg_lsm(models):
    """Average per-mode, per-turn LSM (avg_lsm) across models."""
    frames = []
    for m in models:
        for mode in ['simple', 'trained_agent', 'complex', 'multi_agent', 'proposer_only']:
            p = os.path.join(CSV_DIR, f'{m}_lsm_metrics_{mode}.csv')
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            df['model'] = m
            df['therapist_mode'] = mode
            frames.append(df)
    if not frames:
        print('No model-tagged lsm_metrics files found. Run lsm_analysis per model with PSYCH_MODEL set.')
        return
    all_df = pd.concat(frames, ignore_index=True)
    g = all_df.groupby(['therapist_mode', 'turn_number'])[['avg_lsm']].mean().reset_index()
    g = g.melt(id_vars=['therapist_mode', 'turn_number'], value_name='score').rename(columns={'variable': 'criterion'})
    for mode in g['therapist_mode'].unique():
        sub = g[g['therapist_mode'] == mode]
        sub.to_csv(os.path.join(CSV_DIR, f'avg_lsm_metrics_{mode}.csv'), index=False)
    all_out = os.path.join(CSV_DIR, 'avg_lsm_metrics_all.csv')
    g.to_csv(all_out, index=False)
    print(f'Averaged LSM -> {all_out} ({len(g)} rows)')


def avg_sentiment(models):
    """Average per-mode, per-turn cumulative sentiment proportions across models."""
    frames = []
    for m in models:
        p = os.path.join(CSV_DIR, f'{m}_sentiment_dataset_by_turn.csv')
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df['model'] = m
        frames.append(df)
    if not frames:
        print('No model-tagged sentiment files found. Run sentiment_analysis per model with PSYCH_MODEL set.')
        return
    all_df = pd.concat(frames, ignore_index=True)
    key = ['therapist_mode', 'turn_number']
    for col in ['prop_pos', 'prop_neu', 'prop_neg']:
        agg = all_df.groupby(key)[col].mean().reset_index()
        agg = agg.rename(columns={col: f'avg_{col}'})
        if col == 'prop_pos':
            out = agg
        else:
            out = out.merge(agg, on=key, how='outer')
    all_out = os.path.join(CSV_DIR, 'avg_sentiment_by_turn.csv')
    out.to_csv(all_out, index=False)
    print(f'Averaged sentiment -> {all_out} ({len(out)} rows)')


def avg_word(models):
    """Average per-mode, per-turn word metrics across models (assuming same numeric columns)."""
    frames = []
    for m in models:
        for mode in ['simple', 'trained_agent', 'complex', 'multi_agent', 'proposer_only']:
            p = os.path.join(CSV_DIR, f'{m}_metrics_output_{mode}.csv')
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            # Normalize the turn column: word_analysis emits 'Turn_Pair'
            df = df.rename(columns={'Turn_Pair': 'turn'})
            df['model'] = m
            df['therapist_mode'] = mode
            frames.append(df)
    if not frames:
        print('No model-tagged metrics_output files found. Run word_analysis per model with PSYCH_MODEL set.')
        return
    all_df = pd.concat(frames, ignore_index=True)
    # numeric columns excluding identifiers
    num_cols = [c for c in all_df.columns if c not in ('model', 'therapist_mode', 'turn')]
    g = all_df.groupby(['therapist_mode', 'turn'])[num_cols].mean().reset_index()
    all_out = os.path.join(CSV_DIR, 'avg_metrics_output_all.csv')
    g.to_csv(all_out, index=False)
    print(f'Averaged word metrics -> {all_out} ({len(g)} rows)')


def main():
    ap = argparse.ArgumentParser(description='Average LSM/sentiment/word metrics across models')
    ap.add_argument('--models', nargs='*', default=None)
    args = ap.parse_args()
    models = [model_tag(m) for m in args.models] if args.models else \
        sorted({os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(CSV_DIR, '*_lsm_metrics_*.csv'))})
    print('Averaging across models:', models)
    avg_lsm(models)
    avg_sentiment(models)
    avg_word(models)
    print('\nDone.')


if __name__ == '__main__':
    main()