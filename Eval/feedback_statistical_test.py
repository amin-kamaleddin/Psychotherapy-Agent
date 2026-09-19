"""
Perform Friedman test and pairwise Wilcoxon tests on feedback scores to compare the 5 counselor modes.
Tests for significant differences across simple, trained_agent, complex, multi_agent, and proposer_only modes
for each criterion (9 MI criteria + Overall Average).

The Friedman test tests for overall differences across all 5 modes.
The Wilcoxon signed-rank test performs pairwise comparisons between each pair of modes (C(5,2)=10 pairs).

Data source: prefers the cross-model averaged per-patient scores (avg_feedback_pipeline.csv,
produced by average_across_models.py) so the tests reflect the 3-model average, falling back to
the single/untagged feedback_scores_by_criterion.csv when no averaged file is present.
"""

import os
import sys
from math import comb
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
# Prefer the cross-model averaged per-patient scores; fall back to the untagged file.
FEEDBACK_CSV = os.path.join(CSV_DIR, 'avg_feedback_pipeline.csv')
FEEDBACK_CSV_FALLBACK = os.path.join(CSV_DIR, 'feedback_scores_by_criterion.csv')

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

MODES = ['simple', 'trained_agent', 'complex', 'multi_agent', 'proposer_only']


def perform_friedman_test(feedback_df):
    """
    Perform Friedman test for each criterion across the 5 counselor modes.
    
    The Friedman test is a non-parametric statistical test used to detect 
    differences in treatments across multiple test attempts. It's appropriate
    when we have matched samples (same patients evaluated across different modes).
    Multiple testing correction:
        BH-FDR (Benjamini-Hochberg) correction is applied across all 10 criteria,
        since 10 simultaneous Friedman tests are performed (one per criterion).
        Both raw and FDR-corrected p-values are reported.
    
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
        
        # Check if we have all 5 modes
        available_modes = [mode for mode in MODES if mode in pivot_data.columns]
        if len(available_modes) < 2:
            print(f"Warning: Criterion '{criterion}' only has data for {available_modes}")
            continue
        
        # Keep only rows where we have scores for all available modes (complete cases)
        pivot_data = pivot_data[available_modes].dropna()
        
        if len(pivot_data) < 3:
            print(f"Warning: Criterion '{criterion}' has insufficient matched samples ({len(pivot_data)} < 3)")
            continue
        
        # Extract scores for each mode as arrays
        mode_scores = [pivot_data[mode].values for mode in available_modes]
        
        # Perform Friedman test
        # scipy.stats.friedmanchisquare takes arrays as arguments
        try:
            statistic, p_value = friedmanchisquare(*mode_scores)
            
            # Calculate descriptive statistics
            n_samples = len(pivot_data)
            means = {mode: pivot_data[mode].mean() for mode in available_modes}
            stds = {mode: pivot_data[mode].std() for mode in available_modes}
            
            # Determine if significant (using alpha = 0.05)
            significant = p_value < 0.05
            
            results.append({
                'Criterion': criterion,
                'N': n_samples,
                'Friedman_Chi2': round(statistic, 4),
                'p_value': p_value,
                'Significant': 'Yes' if significant else 'No',
                **{f'{mode}_mean': round(means[mode], 3) for mode in available_modes},
                **{f'{mode}_std': round(stds[mode], 3) for mode in available_modes}
            })
        except Exception as e:
            print(f"Error performing Friedman test for '{criterion}': {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        p_values = results_df['p_value'].values

        # 1. BH-FDR: controls FDR assuming positive dependence (PRDS) between criteria.
        #    Less conservative; appropriate because MI criteria are positively correlated.
        reject_bh, p_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        results_df['p_fdr_bh'] = p_bh
        results_df['Significant_fdr_bh'] = ['Yes' if r else 'No' for r in reject_bh]

        # 2. BY-FDR: controls FDR under arbitrary/unknown dependence structure.
        #    More conservative than BH (multiplies by harmonic number ~2.93 for N=10);
        #    provides a robustness check that doesn't assume positive dependence.
        reject_by, p_by, _, _ = multipletests(p_values, alpha=0.05, method='fdr_by')
        results_df['p_fdr_by'] = p_by
        results_df['Significant_fdr_by'] = ['Yes' if r else 'No' for r in reject_by]

    return results_df


def perform_pairwise_wilcoxon(feedback_df, mode1, mode2, n_pairs=10):
    """
    Perform pairwise Wilcoxon signed-rank test for each criterion between two modes.
    
    The Wilcoxon signed-rank test is a non-parametric test for paired samples.
    It's appropriate when comparing the same patients across two different modes.
    
    Multiple testing corrections applied:
        1. Bonferroni (N = n_pairs): p × n_pairs, the standard post-hoc family
           correction after Friedman (C(5,2)=10 pairwise comparisons). Controls FWER.
        2. BH-FDR across the 10 criteria within this pair: controls False
           Discovery Rate for testing all MI criteria in a single comparison.
    
    Args:
        feedback_df: DataFrame with feedback scores
        mode1: First mode name (e.g., 'simple')
        mode2: Second mode name (e.g., 'complex')
        n_pairs: Total number of pairwise mode comparisons (for Bonferroni).
    
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
            std1 = scores1.std(ddof=1)
            std2 = scores2.std(ddof=1)
            
            # Calculate effect size (Cohen's d for paired samples)
            # d = mean_diff / std_diff (sample standard deviation of differences, ddof=1)
            diff = scores1 - scores2
            std_diff = diff.std(ddof=1) if len(diff) > 1 else 0
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
                'p_value': p_value,
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
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        p_values = results_df['p_value'].values

        # 1. Bonferroni correction for all pairwise comparisons (standard post-hoc family)
        #    Controls FWER across the mode comparisons.
        p_bonf = np.minimum(p_values * n_pairs, 1.0)
        results_df['p_bonferroni'] = p_bonf
        results_df['Significant_bonferroni'] = ['Yes' if p < 0.05 else 'No' for p in p_bonf]

        # 2. BH-FDR correction across 10 criteria within this pair
        #    Controls FDR assuming positive dependence (PRDS); less conservative.
        reject_bh, p_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        results_df['p_fdr_bh'] = p_bh
        results_df['Significant_fdr_bh'] = ['Yes' if r else 'No' for r in reject_bh]

        # 3. BY-FDR correction across 10 criteria within this pair
        #    Controls FDR under arbitrary dependence; more conservative robustness check.
        reject_by, p_by, _, _ = multipletests(p_values, alpha=0.05, method='fdr_by')
        results_df['p_fdr_by'] = p_by
        results_df['Significant_fdr_by'] = ['Yes' if r else 'No' for r in reject_by]

    return results_df


def main():
    """Main function to perform Friedman tests and export results."""
    
    # Load feedback data: prefer the 3-model averaged per-patient scores, falling
    # back to the untagged feedback_scores_by_criterion.csv if not yet generated.
    source = FEEDBACK_CSV if os.path.exists(FEEDBACK_CSV) else FEEDBACK_CSV_FALLBACK
    if not os.path.exists(source):
        print(f"Error: {source} not found!")
        print("Please run feedback_analysis.py per model and average_across_models.py first.")
        return
    
    print(f"Loading feedback data from {source}...")
    feedback_df = pd.read_csv(source)
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
    n_significant     = len(results_df[results_df['Significant'] == 'Yes'])
    n_significant_bh  = len(results_df[results_df['Significant_fdr_bh'] == 'Yes']) if 'Significant_fdr_bh' in results_df.columns else 'N/A'
    n_significant_by  = len(results_df[results_df['Significant_fdr_by'] == 'Yes']) if 'Significant_fdr_by' in results_df.columns else 'N/A'
    print(f"\nSummary:")
    print(f"  - Total criteria tested: {len(results_df)}")
    print(f"  - Significant (raw p < 0.05):            {n_significant}")
    print(f"  - Significant after BH-FDR correction:   {n_significant_bh}  (assumes positive dependence)")
    print(f"  - Significant after BY-FDR correction:   {n_significant_by}  (arbitrary dependence; more conservative)")
    print(f"  NOTE: Both FDR methods correct for {len(results_df)} simultaneous Friedman tests (one per criterion).") 
    
    # Perform pairwise Wilcoxon tests
    print("\n" + "="*80)
    print("PERFORMING PAIRWISE WILCOXON TESTS")
    print("="*80)
    
    # Define all pairs C(n,2) over the modes that actually have data.
    # List ordering keeps the most relevant comparisons (multi_agent vs baselines) first.
    preferred_order = ['multi_agent', 'proposer_only', 'trained_agent', 'complex', 'simple']
    available_modes = [m for m in preferred_order if m in set(feedback_df['therapist_mode'].unique())]
    pairs = []
    for i in range(len(available_modes)):
        for j in range(i + 1, len(available_modes)):
            pairs.append((available_modes[i], available_modes[j]))
    
    for mode1, mode2 in pairs:
        print(f"\nPerforming Wilcoxon test: {mode1} vs {mode2}...")
        wilcoxon_results = perform_pairwise_wilcoxon(feedback_df, mode1, mode2, n_pairs=len(pairs))
        
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
        n_sig_raw  = len(wilcoxon_results[wilcoxon_results['Significant'] == 'Yes'])
        n_sig_bonf = len(wilcoxon_results[wilcoxon_results['Significant_bonferroni'] == 'Yes']) if 'Significant_bonferroni' in wilcoxon_results.columns else 'N/A'
        n_sig_bh   = len(wilcoxon_results[wilcoxon_results['Significant_fdr_bh'] == 'Yes']) if 'Significant_fdr_bh' in wilcoxon_results.columns else 'N/A'
        n_sig_by   = len(wilcoxon_results[wilcoxon_results['Significant_fdr_by'] == 'Yes']) if 'Significant_fdr_by' in wilcoxon_results.columns else 'N/A'
        print(f"\nSummary ({mode1} vs {mode2}):")
        print(f"  - Total criteria tested: {len(wilcoxon_results)}")
        print(f"  - Significant (raw p < 0.05):                          {n_sig_raw}")
        print(f"  - Significant after Bonferroni (x{len(pairs)} pairs):       {n_sig_bonf}  [a = {0.05/len(pairs):.4f}]")
        print(f"  - Significant after BH-FDR (10 criteria):              {n_sig_bh}  (assumes positive dependence)")
        print(f"  - Significant after BY-FDR (10 criteria):              {n_sig_by}  (arbitrary dependence; more conservative)")
        print(f"  NOTE: Bonferroni corrects for C({len(available_modes)},2)={len(pairs)} post-hoc pairwise comparisons;")
        print(f"        BH/BY-FDR correct for {len(wilcoxon_results)} criteria tested within this pair.")


if __name__ == "__main__":
    main()

