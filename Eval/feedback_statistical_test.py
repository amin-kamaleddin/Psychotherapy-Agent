"""
Perform Friedman test and pairwise Wilcoxon tests on feedback scores to compare the 3 counselor modes.
Tests for significant differences across simple, complex, and multi_agent modes
for each criterion (9 MI criteria + Overall Average).

The Friedman test tests for overall differences across all 3 modes.
The Wilcoxon signed-rank test performs pairwise comparisons between each pair of modes.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
FEEDBACK_CSV = os.path.join(CSV_DIR, 'feedback_scores_by_criterion.csv')

# Define the 10 criteria (9 MI criteria + Overall Average)
CRITERIA = [
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

MODES = ['simple', 'complex', 'multi_agent']


def perform_friedman_test(feedback_df):
    """
    Perform Friedman test for each criterion across the 3 counselor modes.
    
    The Friedman test is a non-parametric statistical test used to detect 
    differences in treatments across multiple test attempts. It's appropriate
    when we have matched samples (same patients evaluated across different modes).
    
    Returns:
        DataFrame with test results for each criterion
    """
    results = []
    
    for criterion in CRITERIA:
        criterion_data = feedback_df[feedback_df['criterion'] == criterion].copy()
        
        if criterion_data.empty:
            print(f"Warning: No data found for criterion '{criterion}'")
            continue
        
        # Pivot data to have modes as columns and patient_id as index
        # This creates matched samples (same patient across different modes)
        # Use patient_id only (not timestamp) since patients are tested at different times in each mode
        pivot_data = criterion_data.pivot_table(
            index='patient_id',
            columns='therapist_mode',
            values='score',
            aggfunc='mean'  # In case of duplicates (same patient in multiple sessions), take mean
        )
        
        # Check if we have all 3 modes
        available_modes = [mode for mode in MODES if mode in pivot_data.columns]
        if len(available_modes) < 3:
            print(f"Warning: Criterion '{criterion}' only has data for {available_modes}")
            continue
        
        # Keep only rows where we have scores for all 3 modes (complete cases)
        pivot_data = pivot_data[MODES].dropna()
        
        if len(pivot_data) < 3:
            print(f"Warning: Criterion '{criterion}' has insufficient matched samples ({len(pivot_data)} < 3)")
            continue
        
        # Extract scores for each mode as arrays
        mode_scores = [pivot_data[mode].values for mode in MODES]
        
        # Perform Friedman test
        # scipy.stats.friedmanchisquare takes arrays as arguments
        try:
            statistic, p_value = friedmanchisquare(*mode_scores)
            
            # Calculate descriptive statistics
            n_samples = len(pivot_data)
            means = {mode: pivot_data[mode].mean() for mode in MODES}
            stds = {mode: pivot_data[mode].std() for mode in MODES}
            
            # Determine if significant (using alpha = 0.05)
            significant = p_value < 0.05
            
            results.append({
                'Criterion': criterion,
                'N': n_samples,
                'Friedman_Chi2': round(statistic, 4),
                'p_value': round(p_value, 6),
                'Significant': 'Yes' if significant else 'No',
                'simple_mean': round(means['simple'], 3),
                'complex_mean': round(means['complex'], 3),
                'multi_agent_mean': round(means['multi_agent'], 3),
                'simple_std': round(stds['simple'], 3),
                'complex_std': round(stds['complex'], 3),
                'multi_agent_std': round(stds['multi_agent'], 3)
            })
        except Exception as e:
            print(f"Error performing Friedman test for '{criterion}': {e}")
            continue
    
    return pd.DataFrame(results)


def perform_pairwise_wilcoxon(feedback_df, mode1, mode2):
    """
    Perform pairwise Wilcoxon signed-rank test for each criterion between two modes.
    
    The Wilcoxon signed-rank test is a non-parametric test for paired samples.
    It's appropriate when comparing the same patients across two different modes.
    
    Args:
        feedback_df: DataFrame with feedback scores
        mode1: First mode name (e.g., 'simple')
        mode2: Second mode name (e.g., 'complex')
    
    Returns:
        DataFrame with test results for each criterion
    """
    results = []
    
    for criterion in CRITERIA:
        criterion_data = feedback_df[feedback_df['criterion'] == criterion].copy()
        
        if criterion_data.empty:
            print(f"Warning: No data found for criterion '{criterion}'")
            continue
        
        # Pivot data to have modes as columns and patient_id as index
        pivot_data = criterion_data.pivot_table(
            index='patient_id',
            columns='therapist_mode',
            values='score',
            aggfunc='mean'  # In case of duplicates, take mean
        )
        
        # Check if we have both modes
        if mode1 not in pivot_data.columns or mode2 not in pivot_data.columns:
            print(f"Warning: Criterion '{criterion}' missing data for {mode1} or {mode2}")
            continue
        
        # Keep only rows where we have scores for both modes (complete pairs)
        pair_data = pivot_data[[mode1, mode2]].dropna()
        
        if len(pair_data) < 2:
            print(f"Warning: Criterion '{criterion}' has insufficient paired samples ({len(pair_data)} < 2)")
            continue
        
        # Extract scores for each mode
        scores1 = pair_data[mode1].values
        scores2 = pair_data[mode2].values
        
        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
            
            # Calculate descriptive statistics
            n_samples = len(pair_data)
            mean1 = scores1.mean()
            mean2 = scores2.mean()
            std1 = scores1.std()
            std2 = scores2.std()
            
            # Calculate effect size (Cohen's d for paired samples)
            # d = mean_diff / std_diff (standard deviation of differences)
            diff = scores1 - scores2
            std_diff = diff.std() if len(diff) > 1 else 0
            effect_size = (mean1 - mean2) / std_diff if std_diff > 0 else 0
            
            # Determine if significant (using alpha = 0.05)
            significant = p_value < 0.05
            
            # Determine direction of difference
            if mean1 > mean2:
                direction = f"{mode1} > {mode2}"
            elif mean2 > mean1:
                direction = f"{mode2} > {mode1}"
            else:
                direction = "Equal"
            
            results.append({
                'Criterion': criterion,
                'N': n_samples,
                'Wilcoxon_Statistic': round(statistic, 4),
                'p_value': round(p_value, 6),
                'Significant': 'Yes' if significant else 'No',
                f'{mode1}_mean': round(mean1, 3),
                f'{mode2}_mean': round(mean2, 3),
                f'{mode1}_std': round(std1, 3),
                f'{mode2}_std': round(std2, 3),
                'Mean_Difference': round(mean1 - mean2, 3),
                'Direction': direction,
                'Effect_Size': round(effect_size, 4)
            })
        except Exception as e:
            print(f"Error performing Wilcoxon test for '{criterion}' ({mode1} vs {mode2}): {e}")
            continue
    
    return pd.DataFrame(results)


def main():
    """Main function to perform Friedman tests and export results."""
    
    # Load feedback data
    if not os.path.exists(FEEDBACK_CSV):
        print(f"Error: {FEEDBACK_CSV} not found!")
        print("Please run feedback_analysis.py first to generate the feedback scores data.")
        return
    
    print(f"Loading feedback data from {FEEDBACK_CSV}...")
    feedback_df = pd.read_csv(FEEDBACK_CSV)
    print(f"Loaded {len(feedback_df)} records")
    
    # Check available criteria and modes
    available_criteria = feedback_df['criterion'].unique()
    available_modes = feedback_df['therapist_mode'].unique()
    print(f"\nAvailable criteria: {sorted(available_criteria)}")
    print(f"Available modes: {sorted(available_modes)}")
    
    # Perform Friedman tests
    print("\nPerforming Friedman tests for each criterion...")
    results_df = perform_friedman_test(feedback_df)
    
    if results_df.empty:
        print("Error: No valid test results generated!")
        return
    
    # Export results to CSV
    output_path = os.path.join(CSV_DIR, 'friedman_test_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("FRIEDMAN TEST RESULTS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    n_significant = len(results_df[results_df['Significant'] == 'Yes'])
    print(f"\nSummary:")
    print(f"  - Total criteria tested: {len(results_df)}")
    print(f"  - Significant differences (p < 0.05): {n_significant}")
    print(f"  - Non-significant: {len(results_df) - n_significant}")
    
    # Perform pairwise Wilcoxon tests
    print("\n" + "="*80)
    print("PERFORMING PAIRWISE WILCOXON TESTS")
    print("="*80)
    
    # Define the 3 pairs
    pairs = [
        ('simple', 'complex'),
        ('simple', 'multi_agent'),
        ('complex', 'multi_agent')
    ]
    
    for mode1, mode2 in pairs:
        print(f"\nPerforming Wilcoxon test: {mode1} vs {mode2}...")
        wilcoxon_results = perform_pairwise_wilcoxon(feedback_df, mode1, mode2)
        
        if wilcoxon_results.empty:
            print(f"Warning: No valid test results for {mode1} vs {mode2}")
            continue
        
        # Export results to CSV
        output_path = os.path.join(CSV_DIR, f'wilcoxon_test_{mode1}_vs_{mode2}.csv')
        wilcoxon_results.to_csv(output_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"WILCOXON TEST RESULTS: {mode1.upper()} vs {mode2.upper()}")
        print(f"{'='*80}")
        print(wilcoxon_results.to_string(index=False))
        print(f"\nResults saved to: {output_path}")
        
        # Summary for this pair
        n_sig = len(wilcoxon_results[wilcoxon_results['Significant'] == 'Yes'])
        print(f"\nSummary ({mode1} vs {mode2}):")
        print(f"  - Total criteria tested: {len(wilcoxon_results)}")
        print(f"  - Significant differences (p < 0.05): {n_sig}")
        print(f"  - Non-significant: {len(wilcoxon_results) - n_sig}")


if __name__ == "__main__":
    main()

