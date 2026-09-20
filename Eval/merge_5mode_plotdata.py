"""Merge 4 baseline modes (from a baseline worktree) + proposer_only (from current run)
into all-5-mode files so plots show all 5 modes.

Sources (configurable via environment variables, with defaults below):
- Baselines: $BASELINE_DIR/Eval/csv/{feedback_scores_by_criterion,sentiment_dataset_by_turn}.csv
- proposer_only: Eval/csv/{feedback_scores_by_criterion,sentiment_dataset_by_turn}.csv (current, fold-5 run)

Writes all-5-mode versions to Eval/csv/ for plotting.

Usage:
    $env:BASELINE_DIR="D:/path/to/baseline/worktree"; python Eval/merge_5mode_plotdata.py
"""
import os
import pandas as pd

BASELINE_DIR = os.environ.get('BASELINE_DIR', 'D:/db-check')
BASELINE_FB = os.path.join(BASELINE_DIR, 'Eval/csv/feedback_scores_by_criterion.csv')
BASELINE_SENT = os.path.join(BASELINE_DIR, 'Eval/csv/sentiment_dataset_by_turn.csv')
CUR_FB = 'Eval/csv/feedback_scores_by_criterion.csv'
CUR_SENT = 'Eval/csv/sentiment_dataset_by_turn.csv'
OUT_FB = 'Eval/csv/feedback_scores_by_criterion.csv'
OUT_SENT = 'Eval/csv/sentiment_dataset_by_turn.csv'

BASE_MODES = ['simple', 'trained_agent', 'complex', 'multi_agent']

# --- Feedback scores ---
base_fb = pd.read_csv(BASELINE_FB)
cur_fb = pd.read_csv(CUR_FB)
prop_fb = cur_fb[cur_fb['therapist_mode'] == 'proposer_only']
merged_fb = pd.concat([base_fb[base_fb['therapist_mode'].isin(BASE_MODES)], prop_fb], ignore_index=True)
print('feedback modes:', sorted(merged_fb['therapist_mode'].unique()))
print('feedback rows:', len(merged_fb))
merged_fb.to_csv(OUT_FB, index=False)

# --- Sentiment ---
base_st = pd.read_csv(BASELINE_SENT)
cur_st = pd.read_csv(CUR_SENT)
prop_st = cur_st[cur_st['therapist_mode'] == 'proposer_only']
merged_st = pd.concat([base_st[base_st['therapist_mode'].isin(BASE_MODES)], prop_st], ignore_index=True)
print('sentiment modes:', sorted(merged_st['therapist_mode'].unique()))
print('sentiment rows:', len(merged_st))
merged_st.to_csv(OUT_SENT, index=False)

print('\nDone. All-5-mode files written.')