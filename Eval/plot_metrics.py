"""
Generate plots comparing conversation metrics across different therapist modes.

Creates five plots:
1. Average words per turn by counselors (PSYCHIATRIST) as conversation progresses
2. Average words per turn by clients (PATIENT) as conversation progresses
3. Word ratio between counselors and clients as conversation progresses
4. Distribution of positive, negative, and neutral sentiment by counseling quality
5. Distribution of positive sentiment over 20% splits of the conversation
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMPLE_CSV = os.path.join(SCRIPT_DIR, 'metrics_output_simple.csv')
COMPLEX_CSV = os.path.join(SCRIPT_DIR, 'metrics_output_complex.csv')
MULTI_AGENT_CSV = os.path.join(SCRIPT_DIR, 'metrics_output_multi_agent.csv')
SENTIMENT_CSV = os.path.join(SCRIPT_DIR, 'sentiment_dataset_by_turn.csv')
LSM_CSV = {
    'simple': os.path.join(SCRIPT_DIR, 'lsm_metrics_simple.csv'),
    'complex': os.path.join(SCRIPT_DIR, 'lsm_metrics_complex.csv'),
    'multi_agent': os.path.join(SCRIPT_DIR, 'lsm_metrics_multi_agent.csv')
}

# Mode labels
MODE_LABELS = {
    'simple': 'simple',
    'complex': 'complex',
    'multi_agent': 'multi-agent'
}

# Colors and styles
COLORS = {
    'simple': '#FF8C00',  # Orange
    'complex': '#4169E1',  # Royal blue
    'multi_agent': '#1E90FF'  # Dodger blue
}

STYLES = {
    'simple': '-',  # Solid
    'complex': '-',  # Solid
    'multi_agent': '-'  # Solid
}

MARKERS = {
    'simple': 'o',  # Circle
    'complex': 's',  # Square
    'multi_agent': 'x'  # X
}


def load_data():
    """Load all three CSV files."""
    data = {}
    
    for mode, filepath in [('simple', SIMPLE_CSV), ('complex', COMPLEX_CSV), ('multi_agent', MULTI_AGENT_CSV)]:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[mode] = df
            print(f"Loaded {mode}: {len(df)} rows")
        else:
            print(f"Warning: {filepath} not found")
    
    return data


def load_sentiment_data():
    """Load sentiment dataset."""
    if os.path.exists(SENTIMENT_CSV):
        df = pd.read_csv(SENTIMENT_CSV)
        print(f"Loaded sentiment data: {len(df)} rows")
        return df
    else:
        print(f"Warning: {SENTIMENT_CSV} not found")
        return None


def load_lsm_data():
    """Load LSM metrics for each mode."""
    data = {}
    for mode, filepath in LSM_CSV.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[mode] = df
            print(f"Loaded LSM {mode}: {len(df)} rows")
        else:
            print(f"Warning: {filepath} not found")
    return data


def convert_turn_to_percentage(df):
    """Convert turn numbers to percentage of conversation."""
    max_turn = df['Turn_Pair'].max()
    df['Percent'] = (df['Turn_Pair'] / max_turn) * 100
    return df


def convert_lsm_turn_to_percentage(df):
    """Convert LSM turn numbers to percentage of conversation."""
    max_turn = df['turn_number'].max()
    df['Percent'] = (df['turn_number'] / max_turn) * 100
    return df


def plot_counselor_words(data):
    """Plot 1: Average words per turn by counselors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mode in ['simple', 'complex', 'multi_agent']:
        if mode in data:
            df = data[mode].copy()
            df = convert_turn_to_percentage(df)
            
            ax.plot(
                df['Percent'],
                df['Counselor_Acc_Avg'],
                label=MODE_LABELS.get(mode, mode),
                color=COLORS[mode],
                linestyle=STYLES[mode],
                marker=MARKERS[mode],
                linewidth=2,
                markersize=8
            )
    
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=12)
    ax.set_ylabel('Average number of words per turn', fontsize=12)
    ax.set_title('Average words per turn by counselors as the conversation progresses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])
    
    plt.tight_layout()
    return fig


def plot_client_words(data):
    """Plot 2: Average words per turn by clients."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mode in ['simple', 'complex', 'multi_agent']:
        if mode in data:
            df = data[mode].copy()
            df = convert_turn_to_percentage(df)
            
            ax.plot(
                df['Percent'],
                df['Client_Acc_Avg'],
                label=MODE_LABELS.get(mode, mode),
                color=COLORS[mode],
                linestyle=STYLES[mode],
                marker=MARKERS[mode],
                linewidth=2,
                markersize=8
            )
    
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=12)
    ax.set_ylabel('Average number of words per turn', fontsize=12)
    ax.set_title('Average words per turn by clients as the conversation progresses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])
    
    plt.tight_layout()
    return fig


def plot_word_ratio(data):
    """Plot 3: Word ratio between counselors and clients."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mode in ['simple', 'complex', 'multi_agent']:
        if mode in data:
            df = data[mode].copy()
            df = convert_turn_to_percentage(df)
            
            ax.plot(
                df['Percent'],
                df['Acc_Ratio'],
                label=MODE_LABELS.get(mode, mode),
                color=COLORS[mode],
                linestyle=STYLES[mode],
                marker=MARKERS[mode],
                linewidth=2,
                markersize=8
            )
    
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=12)
    ax.set_ylabel('Ratio', fontsize=12)
    ax.set_title('Word ratio by turn between counselors and clients as the conversation progresses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])
    
    plt.tight_layout()
    return fig


def plot_sentiment_distribution(sentiment_df):
    """Plot 4: Distribution of positive, negative, and neutral sentiment by counseling quality."""
    if sentiment_df is None or sentiment_df.empty:
        print("Warning: No sentiment data available for distribution plot")
        return None
    
    # Use turn 10 data (final cumulative proportions)
    final_turns = sentiment_df[sentiment_df['turn_number'] == 10]
    
    if final_turns.empty:
        print("Warning: No turn 10 data available for sentiment distribution plot")
        return None
    
    # Prepare data organized by sentiment type and mode
    sentiment_types = ['negative', 'neutral', 'positive']
    modes = ['simple', 'complex', 'multi_agent']
    
    # Organize data: sentiment_type -> mode -> values
    plot_data = {sent: {mode: [] for mode in modes} for sent in sentiment_types}
    
    for _, row in final_turns.iterrows():
        mode = row['therapist_mode']
        if mode in modes:
            plot_data['negative'][mode].append(row['prop_neg'])
            plot_data['neutral'][mode].append(row['prop_neu'])
            plot_data['positive'][mode].append(row['prop_pos'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate positions for grouped box plots
    n_sentiments = len(sentiment_types)
    n_modes = len(modes)
    group_width = 0.8
    box_width = group_width / n_modes
    
    positions = []
    data_for_plot = []
    colors_for_boxes = []
    
    for i, sent_type in enumerate(sentiment_types):
        base_pos = i * (n_modes + 1)  # Add gap between groups
        for j, mode in enumerate(modes):
            pos = base_pos + j + 1
            positions.append(pos)
            values = plot_data[sent_type][mode]
            data_for_plot.append(values if values else [0])  # Avoid empty lists
            colors_for_boxes.append(COLORS[mode])
    
    # Create box plots
    bp = ax.boxplot(data_for_plot, positions=positions, widths=box_width, 
                     patch_artist=True, showfliers=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_for_boxes):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis labels
    tick_positions = [sum(positions[i*n_modes:(i+1)*n_modes]) / n_modes 
                     for i in range(n_sentiments)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(sentiment_types)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['simple'], alpha=0.7, label='simple'),
        Patch(facecolor=COLORS['complex'], alpha=0.7, label='complex'),
        Patch(facecolor=COLORS['multi_agent'], alpha=0.7, label='multi-agent')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Distribution of positive, negative, and neutral sentiment by counseling quality', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_positive_sentiment_over_time(sentiment_df):
    """Plot 5: Distribution of positive sentiment over 20% splits of the conversation."""
    if sentiment_df is None or sentiment_df.empty:
        print("Warning: No sentiment data available for positive sentiment over time plot")
        return None
    
    # Define 20% splits: turn 2 (20%), turn 4 (40%), turn 6 (60%), turn 8 (80%), turn 10 (100%)
    splits = {
        20: 2,
        40: 4,
        60: 6,
        80: 8,
        100: 10
    }
    
    # Collect data for each mode and split
    results = {mode: {'splits': [], 'means': [], 'stds': []} 
               for mode in ['simple', 'complex', 'multi_agent']}
    
    for percent, turn_num in splits.items():
        split_data = sentiment_df[sentiment_df['turn_number'] == turn_num]
        
        for mode in ['simple', 'complex', 'multi_agent']:
            mode_split_data = split_data[split_data['therapist_mode'] == mode]
            if not mode_split_data.empty:
                prop_pos_values = mode_split_data['prop_pos'].values
                results[mode]['splits'].append(percent)
                results[mode]['means'].append(prop_pos_values.mean())
                results[mode]['stds'].append(prop_pos_values.std())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.array([20, 40, 60, 80, 100])
    width = 5  # Width of bars
    
    for i, mode in enumerate(['simple', 'complex', 'multi_agent']):
        if results[mode]['splits']:
            means = np.array(results[mode]['means'])
            stds = np.array(results[mode]['stds'])
            splits = np.array(results[mode]['splits'])
            
            # Adjust x position for grouped bars
            offset = (i - 1) * width
            x_pos_adjusted = x_pos + offset
            
            ax.bar(x_pos_adjusted, means, width, yerr=stds, 
                   label=MODE_LABELS.get(mode, mode),
                   color=COLORS[mode], alpha=0.7, capsize=5)
    
    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Distribution of positive sentiment over 20% splits of the conversation', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_lsm_over_time(lsm_data):
    """Plot 6: Linguistic style matching across conversation progress."""
    if not lsm_data:
        print("Warning: No LSM data available for LSM plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode in ['simple', 'complex', 'multi_agent']:
        if mode in lsm_data:
            df = lsm_data[mode].copy()
            df = convert_lsm_turn_to_percentage(df)

            ax.plot(
                df['Percent'],
                df['avg_lsm'],
                label=MODE_LABELS.get(mode, mode),
                color=COLORS[mode],
                linestyle=STYLES[mode],
                marker=MARKERS[mode],
                linewidth=2,
                markersize=8
            )

    ax.set_xlabel('Portion of the conversation (% of turns)', fontsize=12)
    ax.set_ylabel('Average LSM', fontsize=12)
    ax.set_title('Linguistic style matching across conversation progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])

    plt.tight_layout()
    return fig


def main():
    """Main function to generate all plots."""
    print("Loading data...")
    data = load_data()
    sentiment_df = load_sentiment_data()
    lsm_data = load_lsm_data()
    
    if not data:
        print("Error: No metrics data files found!")
        return
    
    print("\nGenerating plots...")
    
    # Generate all five plots
    fig1 = plot_counselor_words(data)
    fig2 = plot_client_words(data)
    fig3 = plot_word_ratio(data)
    fig4 = plot_sentiment_distribution(sentiment_df)
    fig5 = plot_positive_sentiment_over_time(sentiment_df)
    fig6 = plot_lsm_over_time(lsm_data)
    
    # Save plots
    output_dir = SCRIPT_DIR
    fig1.savefig(os.path.join(output_dir, 'counselor_words_per_turn.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'client_words_per_turn.png'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'word_ratio.png'), dpi=300, bbox_inches='tight')
    
    if fig4:
        fig4.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    if fig5:
        fig5.savefig(os.path.join(output_dir, 'positive_sentiment_over_time.png'), dpi=300, bbox_inches='tight')
    if fig6:
        fig6.savefig(os.path.join(output_dir, 'lsm_over_time.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved to {output_dir}:")
    print("  - counselor_words_per_turn.png")
    print("  - client_words_per_turn.png")
    print("  - word_ratio.png")
    if fig4:
        print("  - sentiment_distribution.png")
    if fig5:
        print("  - positive_sentiment_over_time.png")
    if fig6:
        print("  - lsm_over_time.png")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

