"""
Generate plots comparing conversation metrics across different therapist modes.

Creates seven plots:
1. Average words per turn by counselors (PSYCHIATRIST) as conversation progresses
2. Average words per turn by clients (PATIENT) as conversation progresses
3. Word ratio between counselors and clients as conversation progresses
4. Distribution of positive, negative, and neutral sentiment by counseling quality
5. Distribution of positive sentiment over 20% splits of the conversation
6. Linguistic style matching (LSM) across conversation progress
7. Feedback scores by criterion and therapist mode
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib

# Global plot settings: Arial font, minimum font size 18
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, 'csv')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

# Ensure directories exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEEDBACK_CSV = os.path.join(CSV_DIR, 'avg_feedback_pipeline.csv')
FEEDBACK_CSV_FALLBACK = os.path.join(CSV_DIR, 'feedback_scores_by_criterion.csv')
# Per-model per-patient sentiment files (source of per-patient values for the box plot).
# The same 133 patients are scored by each model, so the box plot pools the per-patient
# proportions across all models (maintaining patient-level distributional info).
MODEL_SENTIMENT_CSVS = [
    os.path.join(CSV_DIR, f'{m}_sentiment_dataset_by_turn.csv')
    for m in ['gpt-5-mini', 'qwen-qwen3.8-27b', 'z-ai-glm-5.3-flash']
]
# Per-model per-patient LSM / word files (sources of per-patient values). The same
# patients are scored by each model, so plots pool the patient-level values across models.
MODEL_NAMES = ['gpt-5-mini', 'qwen-qwen3.8-27b', 'z-ai-glm-5.3-flash']
MODEL_LSM_PER_PATIENT = [
    os.path.join(CSV_DIR, f'{m}_lsm_per_patient.csv') for m in MODEL_NAMES
]
MODEL_WORDS_PER_PATIENT = [
    os.path.join(CSV_DIR, f'{m}_words_per_patient.csv') for m in MODEL_NAMES
]

# Mode labels
MODE_LABELS = {
    'simple': 'simple',
    'trained_agent': 'trained agent',
    'complex': 'complex',
    'proposer_only': 'proposer-only',
    'multi_agent': 'multi-agent',
}

# Colors and styles
COLORS = {
    'simple': '#FF8C00',        # Orange
    'trained_agent': '#2ECC71', # Emerald green
    'complex': '#4169E1',       # Royal blue
    'multi_agent': '#1E90FF',   # Dodger blue
    'proposer_only': '#91003f', # Magenta
}

STYLES = {
    'simple': '-',          # Solid
    'trained_agent': '--',  # Dashed
    'complex': '-',         # Solid
    'multi_agent': '-',     # Solid
    'proposer_only': ':',   # Dotted
}

MARKERS = {
    'simple': 'o',          # Circle
    'trained_agent': '^',   # Triangle
    'complex': 's',         # Square
    'multi_agent': 'x',     # X
    'proposer_only': 'D',   # Diamond
}

ALL_MODES = ['simple', 'trained_agent', 'complex', 'proposer_only', 'multi_agent']


def load_per_patient_lsm():
    """Load and concatenate per-patient LSM values across models.

    Returns a long DataFrame with columns patient_id, therapist_mode, turn_number,
    avg_lsm containing one row per patient per model per turn. If unavailable, returns None.
    """
    frames = []
    for path in MODEL_LSM_PER_PATIENT:
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
        else:
            print(f"Warning: {path} not found (per-patient LSM)")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_per_patient_words():
    """Load and concatenate per-patient word-metric values across models.

    Returns a long DataFrame with columns patient_id, therapist_mode, Turn_Pair,
    counselor, client, ratio. If unavailable, returns None.
    """
    frames = []
    for path in MODEL_WORDS_PER_PATIENT:
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
        else:
            print(f"Warning: {path} not found (per-patient words)")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_feedback_data():
    """Load feedback scores dataset (prefer the 3-model averaged per-patient scores)."""
    source = FEEDBACK_CSV if os.path.exists(FEEDBACK_CSV) else FEEDBACK_CSV_FALLBACK
    if os.path.exists(source):
        df = pd.read_csv(source)
        print(f"Loaded feedback data: {len(df)} rows")
        return df
    else:
        print(f"Warning: {source} not found")
        return None


def _patient_curve(df_pat, mode, turn_col, val_col, max_turn):
    """Compute per-turn mean and standard error of a per-patient value for a mode.

    The same 133 patients are scored by all models, so the 3 per-model values for a
    patient are averaged FIRST (clustered by patient); the SE is then the standard error
    across the 133 patient-means (std / sqrt(n_patients)). Pooling the raw (patient, model)
    rows instead would treat the 3 model runs as independent and understate uncertainty.
    Returns (x, mean, se).
    """
    sub = df_pat[df_pat['therapist_mode'] == mode]
    turns = sorted(sub[turn_col].unique())
    xs, means, ses = [], [], []
    for t in turns:
        tt = sub[sub[turn_col] == t]
        pat_means = tt.groupby('patient_id')[val_col].mean().dropna().values
        if len(pat_means) == 0:
            continue
        mean = np.mean(pat_means)
        sd = np.std(pat_means, ddof=1) if len(pat_means) > 1 else 0.0
        xs.append((t / max_turn) * 100)
        means.append(mean)
        ses.append(sd / np.sqrt(len(pat_means)))
    return np.array(xs), np.array(means), np.array(ses)


def _words_max_turn(df_pat):
    return int(df_pat['Turn_Pair'].max()) if df_pat is not None and len(df_pat) else 10


def plot_counselor_words(df_pat=None):
    """Plot 1: Average words per turn by counselors (mean +/- SE across patients, 3 models pooled)."""
    if df_pat is None or df_pat.empty:
        print("Warning: no per-patient word data for counselor plot")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title('Average words per turn by counselors (per-patient SE)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(14, 7))
    mx = _words_max_turn(df_pat)
    for mode in ALL_MODES:
        xs, means, ses = _patient_curve(df_pat, mode, 'Turn_Pair', 'counselor', mx)
        if len(xs):
            ax.fill_between(xs, means - ses, means + ses, color=COLORS[mode], alpha=0.2)
            ax.plot(xs, means, label=MODE_LABELS.get(mode, mode), color=COLORS[mode],
                    linestyle=STYLES[mode], marker=MARKERS[mode], linewidth=2, markersize=8)
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=18)
    ax.set_ylabel('Average number of words per turn', fontsize=18)
    ax.set_title('Average words per turn by counselors (per-patient SE)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])
    plt.tight_layout()
    return fig


def plot_client_words(df_pat=None):
    """Plot 2: Average words per turn by clients (mean +/- SE across patients, 3 models pooled)."""
    if df_pat is None or df_pat.empty:
        print("Warning: no per-patient word data for client plot")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title('Average words per turn by clients (per-patient SE)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(14, 7))
    mx = _words_max_turn(df_pat)
    for mode in ALL_MODES:
        xs, means, ses = _patient_curve(df_pat, mode, 'Turn_Pair', 'client', mx)
        if len(xs):
            ax.fill_between(xs, means - ses, means + ses, color=COLORS[mode], alpha=0.2)
            ax.plot(xs, means, label=MODE_LABELS.get(mode, mode), color=COLORS[mode],
                    linestyle=STYLES[mode], marker=MARKERS[mode], linewidth=2, markersize=8)
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=18)
    ax.set_ylabel('Average number of words per turn', fontsize=18)
    ax.set_title('Average words per turn by clients (per-patient SE)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])
    plt.tight_layout()
    return fig


def plot_word_ratio(df_pat=None):
    """Plot 3: Word ratio (counselor words / client words) per turn (mean +/- SE across patients)."""
    if df_pat is None or df_pat.empty:
        print("Warning: no per-patient word data for ratio plot")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title('Word ratio by turn (per-patient SE)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(14, 7))
    mx = _words_max_turn(df_pat)
    for mode in ALL_MODES:
        xs, means, ses = _patient_curve(df_pat, mode, 'Turn_Pair', 'ratio', mx)
        if len(xs):
            ax.fill_between(xs, means - ses, means + ses, color=COLORS[mode], alpha=0.2)
            ax.plot(xs, means, label=MODE_LABELS.get(mode, mode), color=COLORS[mode],
                    linestyle=STYLES[mode], marker=MARKERS[mode], linewidth=2, markersize=8)
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=18)
    ax.set_ylabel('Ratio', fontsize=18)
    ax.set_title('Word ratio by turn between counselors and clients (per-patient SE)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])
    plt.tight_layout()
    return fig


def plot_sentiment_distribution():
    """Plot 4: Distribution of positive, negative, and neutral sentiment by counseling quality.

    Box-and-whisker plot at turn 10 with the 3 model scores averaged within each patient
    (so each of the 133 patients contributes one value per mode), preserving the
    patient-level distribution instead of collapsing to a single mean.
    """
    # Collect per-patient values for turn 10 across all per-model files, averaging the
    # 3 model scores within each patient (so each patient contributes one value per mode).
    col_map = {'negative': 'prop_neg', 'neutral': 'prop_neu', 'positive': 'prop_pos'}
    rows = []
    for path in MODEL_SENTIMENT_CSVS:
        if not os.path.exists(path):
            print(f"Warning: {path} not found; box plot will be missing a model")
            continue
        df = pd.read_csv(path)
        t10 = df[df['turn_number'] == 10]
        rows.append(t10[['patient_id', 'therapist_mode'] + list(col_map.values())])
    if not rows:
        print("Warning: no per-patient sentiment data for distribution plot")
        return None
    all_t10 = pd.concat(rows, ignore_index=True)
    all_t10 = all_t10[all_t10['therapist_mode'].isin(ALL_MODES)]
    # Average the 3 model scores within each (patient, mode).
    patient_means = all_t10.groupby(['patient_id', 'therapist_mode'])[list(col_map.values())].mean().reset_index()

    val_map = {s: {m: [] for m in ALL_MODES} for s in col_map}
    for _, row in patient_means.iterrows():
        mode = row['therapist_mode']
        for sent, col in col_map.items():
            v = row[col]
            if v == v:  # not NaN
                val_map[sent][mode].append(v)

    total_vals = sum(len(v) for v in val_map['positive'].values())
    if total_vals == 0:
        print("Warning: no per-patient sentiment data for distribution plot")
        return None

    n_modes = len(ALL_MODES)
    sentiment_types = ['negative', 'neutral', 'positive']
    n_sentiments = len(sentiment_types)
    fig, ax = plt.subplots(figsize=(20, 8))
    group_width = 2.5
    # Boxes sit 1.0 apart between adjacent modes; width ~0.62 fills the slot (matches reference).
    box_width = 0.62

    positions = []
    data_for_plot = []
    colors_for_boxes = []
    for i, sent in enumerate(sentiment_types):
        base_pos = i * (n_modes + 1)
        for j, mode in enumerate(ALL_MODES):
            pos = base_pos + j + 1
            positions.append(pos)
            values = val_map[sent][mode]
            data_for_plot.append(values if values else [0])  # avoid empty lists for boxplot
            colors_for_boxes.append(COLORS[mode])

    bp = ax.boxplot(data_for_plot, positions=positions, widths=box_width,
                    patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], colors_for_boxes):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # One tick per sentiment group, centered on that group's box midpoint.
    tick_positions = [sum(positions[i*n_modes:(i+1)*n_modes]) / n_modes for i in range(n_sentiments)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(sentiment_types)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['simple'], alpha=0.7, label='simple'),
        Patch(facecolor=COLORS['trained_agent'], alpha=0.7, label='trained agent'),
        Patch(facecolor=COLORS['complex'], alpha=0.7, label='complex'),
        Patch(facecolor=COLORS['proposer_only'], alpha=0.7, label='proposer-only'),
        Patch(facecolor=COLORS['multi_agent'], alpha=0.7, label='multi-agent'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.set_xlabel('Sentiment', fontsize=18)
    ax.set_ylabel('Proportion', fontsize=18)
    ax.set_title('Sentiment distribution (per-patient, pooled across 3 models, turn 10) by counseling quality',
                 fontsize=18, fontweight='bold')
    # Proportions are bounded 0-1; use the full scale for the box plot.
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_positive_sentiment_over_time():
    """Plot 5: Positive sentiment over 20% splits, pooled per-patient across the 3 models.

    Computes the per-patient cumulative positive proportion at turns 2/4/6/8/10 from the
    per-model per-patient files, then renders a grouped bar chart of the per-patient mean
    with SEM (= std/sqrt(n) across pooled patients) error bars.
    """
    # Define 20% splits: turn 2 (20%), turn 4 (40%), turn 6 (60%), turn 8 (80%), turn 10 (100%)
    splits = {20: 2, 40: 4, 60: 6, 80: 8, 100: 10}

    # Pool the per-patient 'prop_pos' values (cumulative positive proportion) per mode/turn
    # across all models. Columns: patient_id, therapist_mode, turn_number, prop_pos ...
    frames = []
    for path in MODEL_SENTIMENT_CSVS:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            print(f"Warning: {path} not found (positive sentiment over time)")
    if not frames:
        print("Warning: No per-patient sentiment data for positive sentiment over time plot")
        return None
    all_sent = pd.concat(frames, ignore_index=True)

    results = {mode: {'splits': [], 'means': [], 'ses': []} for mode in ALL_MODES}
    for percent, turn_num in splits.items():
        split_data = all_sent[all_sent['turn_number'] == turn_num]
        for mode in ALL_MODES:
            mode_data = split_data[split_data['therapist_mode'] == mode]
            pat_means = mode_data.groupby('patient_id')['prop_pos'].mean().dropna().values
            if len(pat_means) == 0:
                continue
            results[mode]['splits'].append(percent)
            results[mode]['means'].append(np.mean(pat_means))
            results[mode]['ses'].append(np.std(pat_means, ddof=1) / np.sqrt(len(pat_means)) if len(pat_means) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.array([20, 40, 60, 80, 100])
    width = 3  # Bar width so 5 grouped bars fit cleanly between 20%-spaced x ticks
    n_modes = len(ALL_MODES)
    for i, mode in enumerate(ALL_MODES):
        if results[mode]['splits']:
            means = np.array(results[mode]['means'])
            ses = np.array(results[mode]['ses'])
            offset = (i - (n_modes - 1) / 2) * width
            x_pos_adjusted = x_pos + offset
            ax.bar(x_pos_adjusted, means, width, yerr=ses,
                   label=MODE_LABELS.get(mode, mode),
                   color=COLORS[mode], alpha=0.7, capsize=3)

    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=18)
    ax.set_ylabel('Proportion positive', fontsize=18)
    ax.set_title('Positive sentiment over 20% splits of the conversation (per-patient, 3 models pooled)', fontsize=18, fontweight='bold')
    ax.set_xticks([20, 40, 60, 80, 100])
    max_top = max(v + e for mode in results.values() for v, e in zip(mode['means'], mode['ses'])) if any(results[m]['ses'] for m in results) else 1.0
    ax.set_ylim(0, max(0.7, max_top * 1.2))
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_lsm_over_time(lsm_patient=None):
    """Plot 6: Linguistic style matching across conversation progress (mean +/- SE across patients)."""
    if lsm_patient is None or len(lsm_patient) == 0:
        print("Warning: no per-patient LSM data for LSM plot")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title('Linguistic style matching across conversation progress (per-patient SE)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(16, 8))
    mx = int(lsm_patient['turn_number'].max())

    for mode in ALL_MODES:
        sub = lsm_patient[lsm_patient['therapist_mode'] == mode]
        turns = sorted(sub['turn_number'].unique())
        xs, means, ses = [], [], []
        for t in turns:
            tt = sub[sub['turn_number'] == t]
            pat_means = tt.groupby('patient_id')['avg_lsm'].mean().dropna().values
            if len(pat_means) == 0:
                continue
            mean = np.mean(pat_means)
            sd = np.std(pat_means, ddof=1) if len(pat_means) > 1 else 0.0
            xs.append((t / mx) * 100)
            means.append(mean)
            ses.append(sd / np.sqrt(len(pat_means)))
        xs, means, ses = np.array(xs), np.array(means), np.array(ses)
        if len(xs):
            ax.fill_between(xs, means - ses, means + ses, color=COLORS[mode], alpha=0.2)
            ax.plot(xs, means, label=MODE_LABELS.get(mode, mode), color=COLORS[mode],
                    linestyle=STYLES[mode], marker=MARKERS[mode], linewidth=2, markersize=8)

    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=18)
    ax.set_ylabel('Average LSM', fontsize=18)
    ax.set_title('Linguistic style matching across conversation progress (per-patient SE)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])

    plt.tight_layout()
    return fig


def plot_feedback_scores(feedback_df):
    """Plot feedback scores by criterion and therapist mode - bar chart with standard error bars."""
    if feedback_df is None or feedback_df.empty:
        print("Warning: No feedback data available for feedback scores plot")
        return None
    
    # Define the order of criteria (MI criteria + Overall Average)
    criteria_order = [
        'Acceptance',
        'Empathy',
        'Direction',
        'Collaboration',
        'Autonomy Support',
        'Evocation',
        'Partnership',
        'Cultivating Change Talk',
        'Softening Sustain Talk',
        'Overall Average'
    ]
    
    modes = ALL_MODES
    
    # Group by criterion and mode, calculate mean and std
    grouped = feedback_df.groupby(['criterion', 'therapist_mode'])['score'].agg(['mean', 'std', 'count']).reset_index()
    
    # Prepare data for plotting
    plot_data = {}
    for mode in modes:
        plot_data[mode] = {
            'means': [],
            'ses': [],  # Changed from 'stds' to 'ses' for standard errors
            'counts': []
        }
        mode_data = grouped[grouped['therapist_mode'] == mode]
        for criterion in criteria_order:
            criterion_data = mode_data[mode_data['criterion'] == criterion]
            if not criterion_data.empty:
                mean = criterion_data['mean'].iloc[0]
                std = criterion_data['std'].iloc[0]
                count = int(criterion_data['count'].iloc[0])
                se = std / np.sqrt(count) if count > 0 else 0  # Calculate standard error
                
                plot_data[mode]['means'].append(mean)
                plot_data[mode]['ses'].append(se)
                plot_data[mode]['counts'].append(count)
            else:
                plot_data[mode]['means'].append(0)
                plot_data[mode]['ses'].append(0)
                plot_data[mode]['counts'].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(22, 9))
    
    # Set up bar positions
    n_criteria = len(criteria_order)
    n_modes = len(modes)
    x = np.arange(n_criteria)
    width = 0.18  # Width of bars (reduced for 5 modes)
    
    # Plot bars for each mode
    for i, mode in enumerate(modes):
        means = np.array(plot_data[mode]['means'])
        ses = np.array(plot_data[mode]['ses'])  # Changed from 'stds' to 'ses'
        offset = (i - (n_modes - 1) / 2) * width  # Center the bars for n modes
        bars = ax.bar(x + offset, means, width, yerr=ses, 
                     label=MODE_LABELS.get(mode, mode),
                     color=COLORS[mode], alpha=0.7, capsize=5)
    
    # Customize plot
    ax.set_xlabel('Score Categories', fontsize=18)
    ax.set_ylabel('Average Score', fontsize=18)
    ax.set_title('Feedback Scores by Criterion and Therapist Mode', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_order, rotation=45, ha='right')
    ax.set_ylim(0, 6)  # Scores are 1-5, but allow some space above
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def main():
    """Main function to generate all plots."""
    print("Loading data...")
    feedback_df = load_feedback_data()
    lsm_patient = load_per_patient_lsm()
    words_patient = load_per_patient_words()

    print("\nGenerating plots...")

    # Generate all plots (per-patient, pooled across models)
    fig1 = plot_counselor_words(words_patient)
    fig2 = plot_client_words(words_patient)
    fig3 = plot_word_ratio(words_patient)
    fig4 = plot_sentiment_distribution()
    fig5 = plot_positive_sentiment_over_time()
    fig6 = plot_lsm_over_time(lsm_patient)
    fig7 = plot_feedback_scores(feedback_df)
    
    # Save plots (SVG + PNG for each)
    fig1.savefig(os.path.join(PLOTS_DIR, 'counselor_words_per_turn.svg'), format='svg', bbox_inches='tight')
    fig1.savefig(os.path.join(PLOTS_DIR, 'counselor_words_per_turn.png'), format='png', bbox_inches='tight')
    fig2.savefig(os.path.join(PLOTS_DIR, 'client_words_per_turn.svg'), format='svg', bbox_inches='tight')
    fig2.savefig(os.path.join(PLOTS_DIR, 'client_words_per_turn.png'), format='png', bbox_inches='tight')
    fig3.savefig(os.path.join(PLOTS_DIR, 'word_ratio.svg'), format='svg', bbox_inches='tight')
    fig3.savefig(os.path.join(PLOTS_DIR, 'word_ratio.png'), format='png', bbox_inches='tight')
    
    if fig4:
        fig4.savefig(os.path.join(PLOTS_DIR, 'sentiment_distribution.svg'), format='svg', bbox_inches='tight')
        fig4.savefig(os.path.join(PLOTS_DIR, 'sentiment_distribution.png'), format='png', bbox_inches='tight')
    if fig5:
        fig5.savefig(os.path.join(PLOTS_DIR, 'positive_sentiment_over_time.svg'), format='svg', bbox_inches='tight')
        fig5.savefig(os.path.join(PLOTS_DIR, 'positive_sentiment_over_time.png'), format='png', bbox_inches='tight')
    if fig6:
        fig6.savefig(os.path.join(PLOTS_DIR, 'lsm_over_time.svg'), format='svg', bbox_inches='tight')
        fig6.savefig(os.path.join(PLOTS_DIR, 'lsm_over_time.png'), format='png', bbox_inches='tight')
    if fig7:
        fig7.savefig(os.path.join(PLOTS_DIR, 'feedback_scores_by_criterion.svg'), format='svg', bbox_inches='tight')
        fig7.savefig(os.path.join(PLOTS_DIR, 'feedback_scores_by_criterion.png'), format='png', bbox_inches='tight')
    
    print(f"\nPlots saved to {PLOTS_DIR}:")
    print("  - counselor_words_per_turn.svg")
    print("  - client_words_per_turn.svg")
    print("  - word_ratio.svg")
    if fig4:
        print("  - sentiment_distribution.svg")
    if fig5:
        print("  - positive_sentiment_over_time.svg")
    if fig6:
        print("  - lsm_over_time.svg")
    if fig7:
        print("  - feedback_scores_by_criterion.svg")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

