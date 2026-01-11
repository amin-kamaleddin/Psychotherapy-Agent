# PsychotherapyAgent Comparison Guide: Simple vs Complex vs Multi-Agent System

This guide explains how to compare the performance between three different therapist configurations:
1. Simple prompt single-agent
2. Complex prompt single-agent (with detailed MI techniques)
3. Multi-agent system (proposer + judge)

## Overview

The system supports three modes with **two-stage aggregation** for robust statistical analysis:

1. **Simple Mode** (`therapist_mode="simple"`):
   - Uses `PsychotherapySingleAgent` with a simple prompt from `Therapist_Prompts/therapist_prompts.csv`
   - Single-agent system that generates one therapist response
   - Response is sent directly to the patient

2. **Complex Mode** (`therapist_mode="complex"`):
   - Uses `PsychotherapySingleAgent` with a complex, MI-focused prompt from `Therapist_Prompts/therapist_prompts.csv`
   - Single-agent system with more sophisticated prompt
   - Response is sent directly to the patient

3. **Multi-Agent Mode** (`therapist_mode="multi_agent"`):
   - Uses `PsychotherapyMultiAgent` with proposer and judge agents
   - Proposer agent generates 3 candidate therapist responses
   - Judge agent selects the best response from the 3 candidates
   - Selected response is sent to the patient
   - This is the new enhanced system

## Data Aggregation Method

The evaluation system uses a **two-stage aggregation approach** to ensure statistical reliability:

### Stage 1: Within-Patient Aggregation (Fold Averaging)
- When using `--fold N`, each patient is run N times
- All N sessions for each patient are **averaged together**
- Result: One score per patient per mode (reduces session-to-session noise)

### Stage 2: Cross-Patient Aggregation (Population Statistics)
- Averaged scores from all patients are aggregated
- Mean and standard deviation calculated across all patients
- Result: Population-level statistics with reduced variance

**Example with fold=5 and 133 patients:**
```
Patient 42, Simple Mode:
  Session 1: Empathy = 4.2
  Session 2: Empathy = 4.0
  Session 3: Empathy = 4.3
  Session 4: Empathy = 4.1
  Session 5: Empathy = 4.4
  → Stage 1 Average: 4.2

All 133 patients → Stage 2: Mean = 3.87, Std = 0.33
```

This approach provides **more stable and reliable comparisons** between modes.

## Evaluation Script Commands

The `run_automated_test.py` script supports all three modes with flexible options:

1. **Run Simple Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode simple --run-id test1
   ```

2. **Run Complex Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode complex --run-id test2
   ```

3. **Run Multi-Agent Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode multi_agent --run-id test3
   ```

4. **Run All Three Modes + Aggregation (Comparison)**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --run-id exp1
   ```

5. **Run All Three Modes WITHOUT Aggregation**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --no-aggregate --run-id exp1
   ```

6. **Run Aggregation Only (after sessions completed)**:
   ```bash
   python run_automated_test.py --aggregate
   ```

7. **Specify Custom Parameters**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id full_test \
     --max-workers 20 --skip-sentiment
   ```

### Key Parameters

- `--fold N`: Run each patient N times (default: 1)
- `--compare`: Run all 3 modes and generate comparison
- `--no-aggregate`: Skip evaluation/aggregation scripts
- `--aggregate`: Only run aggregation (no new sessions)
- `--max-workers N`: Parallel workers for processing (default: 20)
- `--skip-sentiment`: Skip sentiment analysis (fastest part to skip)

## Running Comparisons

### Step 1: Run Comparison Experiment

The `--compare` flag runs all three modes sequentially and generates a comparison:

```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id exp1
```

This will:
1. Run all patients with simple prompt mode (133 patients × 5 folds = 665 sessions)
2. Run all patients with complex prompt mode (665 sessions)
3. Run all patients with multi-agent mode (665 sessions)
4. Automatically run all evaluation/aggregation scripts
5. Generate comparison statistics and plots

**For faster execution without aggregation:**
```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --no-aggregate --run-id exp1
# Then run aggregation separately when ready:
python run_automated_test.py --aggregate
```

### Step 2: Review Results
#### Session-Level Data
Stored in `Saved_Sessions/` directory:
- Individual session folders with transcripts and feedback
- Format: `session_export_patient_{id}_{timestamp}_{uuid}/`

#### Aggregated Results
Stored in `Session_Metadata/` directory:

**Per-Mode Folders:**
- `{run_id}_simple_patients_{start}_{end}_fold{N}_{timestamp}/`
  - `metadata.json` - Run configuration
  - `results_simple.csv` - Individual session results (before aggregation)
  
- `{run_id}_complex_patients_{start}_{end}_fold{N}_{timestamp}/`
- `{run_id}_multi_agent_patients_{start}_{end}_fold{N}_{timestamp}/`

**Evaluation Results** (in `Eval/csv/` directory):
- `feedback_scores_by_criterion.csv` - Per-patient scores (133 patients × 3 modes)
- `feedback_metrics_{mode}.csv` - Aggregated statistics per mode
- `sentiment_dataset_by_turn.csv` - Sentiment analysis per turn
- `lsm_metrics_{mode}.csv` - Language Style Matching scores
- `metrics_output_{mode}.csv` - Word count and ratio metrics
- `comparison_results.csv` - Three-way comparison summary
- `friedman_test_results.csv` - Statistical significance tests
- `wilcoxon_test_{mode1}_vs_{mode2}.csv` - Pairwise comparisons

**Plots** (in `Eval/plots/` directory):
- `feedback_scores_by_criterion.png`
- `counselor_words_per_turn.png`
- `client_words_per_turn.png`
- `word_ratio.png`
- `sentiment_distribution.png`
- `lsm_over_time.png`e results
  - `results_multi_agent.csv` - Individual patient scores for multi-agent mode

### Step 3: Analyze Comparison

The comparison CSV includes:

- **Category**: Evaluation category (Empathy, Acceptance, etc.)
- **Simple_Mean**: Average score for simple prompt mode
- **Simple_Std**: Standard deviation for simple prompt mode
- **Complex_Mean**: Average score for complex prompt mode
- **Complex_Std**: Standard deviation for complex prompt mode
- **MultiAgent_Mean**: Average score for multi-agent mode
- **MultiAgent_Std**: Standard deviation for multi-agent mode
- **Best_Mode**: Which mode performed best for this category
- **Simple_vs_Complex**: Difference (Complex - Simple)
- **Simple_vs_MultiAgent**: Difference (Multi-Agent - Simple)
- **Complex_vs_MultiAgent**: Difference (Multi-Agent - Complex)

## Evaluation Categories

The system evaluates on these MI (Motivational Interviewing) criteria:

1. Cultivating Change Talk
2. Softening Sustain Talk
3. Partnership
4. Empathy
5. Acceptance
6. Direction
7. Autonomy Support
8. Collaboration
9. Evocation
10. Overall Average

## Example Output

```
THREE-WAY COMPARISON SUMMARY
================================================================================
Category                    Simple_Mean  Complex_Mean  MultiAgent_Mean  Best_Mode
Cultivating Change Talk     3.2         3.5          3.7              Multi-Agent
Empathy  fold parameter for reliability**: Use `--fold 5` to run each patient 5 times for more stable measurements
2. **Use the same patient set**: Run all modes on the same patients for fair comparison
3. **Parallel execution**: Use `--max-workers` to speed up processing (default: 20)
4. **Batch processing**: You can run patients in batches (e.g., 0-50, 50-100, 100-133) - the aggregation will correctly merge them
5. **Check aggregated metrics**: The evaluation scripts automatically handle fold averaging and patient aggregation
6. **Statistical tests**: Review Friedman and Wilcoxon test results for significance
7. **Review plots**: Visual comparisons in `Eval/plots/` provide intuitive insights

## Understanding the Count Column

- **Count mismatch**: If counts are not 133, regenerate CSVs by running `python run_automated_test.py --aggregate`
- **Multiple test runs**: The system correctly handles patients appearing in multiple batches by grouping on `(patient_id, mode)` only
- **Memory issues**: Use `--skip-sentiment` to skip the most memory-intensive evaluation step
- **Stale data**: Always regenerate evaluation CSVs after code changes using `--aggregate`
All aggregated CSV files show **count = 133** (one per patient):
- Each patient's multiple folds are averaged in Stage 1
- Statistics are calculated across 133 unique patients in Stage 2
- If you ran patients in multiple batches, the same patient across batches is merged automatically
================================================================================
Simple Prompt:     3.500 ± 0.200
Complex Prompt:    3.600 ± 0.180
Multi-Agent:       3.700 ± 0.150

Best Performing: Multi-Agent

Differences:
  Complex vs Simple:      +0.100 (+2.86%)
  Multi-Agent vs Simple: +0.200 (+5.71%)
  Multi-Agent vs Complex: +0.100 (+2.78%)
```

## Best Practices

1. **Use the same patient set**: Run both modes on the same patients for fair comparison
2. **Run multiple times**: Consider running multiple experiments to account for variability
3. **Check individual scores**: Look at individual patient results, not just averages
4. **Review feedback files**: Check the detailed feedback files in saved sessions for qualitative insights

## Troubleshooting

- **Import errors**: Make sure you're using the correct classes from the `PsychotherapyAgent` package:
  - `from PsychotherapyAgent import PsychotherapySingleAgent, PsychotherapyMultiAgent`
- **Missing feedback files**: Check that sessions completed successfully
- **Score extraction issues**: Verify that feedback files contain properly formatted scores

## Architecture Notes

### Multi-Agent Flow

1. **PsychotherapyMultiAgent** initialized with proposer (index 2) and judge (index 3) from `Therapist_Prompts/therapist_prompts.csv`
2. **Proposer Agent**: Generates 3 candidate responses based on conversation
3. **Judge Agent**: Evaluates and selects best response based on:
   - Therapeutic appropriateness
   - MI principles adherence
   - Empathy and rapport
   - Clinical effectiveness
4. Selected response sent to patient
5. Patient responds

### Single-Agent Flow (Simple & Complex Modes)

1. **PsychotherapySingleAgent** initialized with therapist profile from `Therapist_Prompts/therapist_prompts.csv`:
   - Simple mode: uses index 0 (0_Simple_Agent)
   - Complex mode: uses index 1 (1_Complex_Agent)
2. Agent generates response directly
3. Response sent to patient
4. Patient responds

