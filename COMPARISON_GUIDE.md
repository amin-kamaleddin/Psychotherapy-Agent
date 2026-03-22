# PsychotherapyAgent Comparison Guide: Simple vs Trained Agent vs Complex vs Multi-Agent System

This guide explains how to compare the performance between four different therapist configurations:
1. Simple prompt single-agent (no MI, no patient profile)
2. Trained agent single-agent (MI prompt, no patient profile)
3. Complex prompt single-agent (MI prompt + patient profile)
4. Multi-agent system (proposer + judge)

## Overview

The system supports four modes with **two-stage aggregation** for robust statistical analysis:

1. **Simple Mode** (`therapist_mode="simple"`):
   - Uses `PsychotherapySingleAgent` with a naive counselor prompt (CSV index 0: `0_Simple_Agent`)
   - No MI-specific language or rubric; patient profile is not provided
   - Response is sent directly to the patient

2. **Trained Agent Mode** (`therapist_mode="trained_agent"`):
   - Uses `PsychotherapySingleAgent` with a full MI-focused prompt (CSV index 1: `1_Trained_Agent`)
   - Patient profile is intentionally withheld to isolate the effect of MI prompting alone
   - Response is sent directly to the patient

3. **Complex Mode** (`therapist_mode="complex"`):
   - Uses `PsychotherapySingleAgent` with the same MI-focused prompt AND full patient profile context (CSV index 2: `2_Complex_Agent`)
   - Single-agent system with sophisticated MI prompt and patient history
   - Response is sent directly to the patient

4. **Multi-Agent Mode** (`therapist_mode="multi_agent"`):
   - Uses `PsychotherapyMultiAgent` with proposer (CSV index 3: `3_MultiAgent_Proposer`) and judge (CSV index 4: `4_MultiAgent_Judge`) agents
   - Proposer agent generates 3 candidate therapist responses
   - Judge agent selects the best response from the 3 candidates based on MI criteria
   - Selected response is sent to the patient

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

The `run_automated_test.py` script supports all four modes with flexible options:

1. **Run Simple Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode simple --run-id test1
   ```

2. **Run Trained Agent Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode trained_agent --run-id test2
   ```

3. **Run Complex Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode complex --run-id test3
   ```

4. **Run Multi-Agent Mode Only**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode multi_agent --run-id test4
   ```

5. **Run All Four Modes + Aggregation (Comparison)**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --run-id exp1
   ```

6. **Run All Four Modes WITHOUT Aggregation**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --no-aggregate --run-id exp1
   ```

7. **Run Aggregation Only (after sessions completed)**:
   ```bash
   python run_automated_test.py --aggregate
   ```

8. **Specify Custom Parameters**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id full_test \
     --max-workers 20 --skip-sentiment
   ```

### Key Parameters

- `--fold N`: Run each patient N times (default: 1)
- `--compare`: Run all 4 modes and generate comparison
- `--no-aggregate`: Skip evaluation/aggregation scripts
- `--aggregate`: Only run aggregation (no new sessions)
- `--max-workers N`: Parallel workers for processing (default: 20)
- `--skip-sentiment`: Skip sentiment analysis (fastest part to skip)

## Running Comparisons

### Step 1: Run Comparison Experiment

The `--compare` flag runs all four modes sequentially and generates a comparison:

```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id exp1
```

This will:
1. Run all patients with simple prompt mode (133 patients × 5 folds = 665 sessions)
2. Run all patients with trained agent mode (665 sessions)
3. Run all patients with complex prompt mode (665 sessions)
4. Run all patients with multi-agent mode (665 sessions)
5. Automatically run all evaluation/aggregation scripts
6. Generate comparison statistics and plots

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
  
- `{run_id}_trained_agent_patients_{start}_{end}_fold{N}_{timestamp}/`
- `{run_id}_complex_patients_{start}_{end}_fold{N}_{timestamp}/`
- `{run_id}_multi_agent_patients_{start}_{end}_fold{N}_{timestamp}/`

**Evaluation Results** (in `Eval/csv/` directory):
- `feedback_scores_by_criterion.csv` - Per-patient scores (133 patients × 4 modes)
- `feedback_metrics_{mode}.csv` - Aggregated statistics per mode
- `sentiment_dataset_by_turn.csv` - Sentiment analysis per turn
- `lsm_metrics_{mode}.csv` - Language Style Matching scores
- `metrics_output_{mode}.csv` - Word count and ratio metrics
- `comparison_results.csv` - Four-way comparison summary
- `friedman_test_results.csv` - Statistical significance tests
- `wilcoxon_test_{mode1}_vs_{mode2}.csv` - Pairwise comparisons

**Plots** (in `Eval/plots/` directory):
- `feedback_scores_by_criterion.png`
- `counselor_words_per_turn.png`
- `client_words_per_turn.png`
- `word_ratio.png`
- `sentiment_distribution.png`
- `lsm_over_time.png`

### Step 3: Analyze Comparison

The comparison CSV includes:

- **Category**: Evaluation category (Empathy, Acceptance, etc.)
- **Simple_Mean** / **Simple_Std**: Score statistics for simple mode
- **TrainedAgent_Mean** / **TrainedAgent_Std**: Score statistics for trained agent mode
- **Complex_Mean** / **Complex_Std**: Score statistics for complex mode
- **MultiAgent_Mean** / **MultiAgent_Std**: Score statistics for multi-agent mode
- **Best_Mode**: Which mode performed best for this category
- **Simple_vs_TrainedAgent**: Difference (Trained Agent - Simple)
- **Simple_vs_Complex**: Difference (Complex - Simple)
- **Simple_vs_MultiAgent**: Difference (Multi-Agent - Simple)
- **TrainedAgent_vs_Complex**: Difference (Complex - Trained Agent)
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
FOUR-WAY COMPARISON SUMMARY
================================================================================
Category                  Simple_Mean  TrainedAgent_Mean  Complex_Mean  MultiAgent_Mean  Best_Mode
Cultivating Change Talk   3.2          3.4                3.5           3.7              Multi-Agent
Empathy                   3.5          3.6                3.7           3.9              Multi-Agent
```

## Best Practices

1. **Use the `--fold` parameter for reliability**: Use `--fold 5` to run each patient 5 times for more stable measurements
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

### Example Overall Comparison Summary

```text
Simple Prompt: 3.500 ± 0.200
Trained Agent: 3.580 ± 0.190
Complex Prompt: 3.600 ± 0.180
Multi-Agent: 3.700 ± 0.150

Best Performing: Multi-Agent

Differences:
Trained Agent vs Simple: +0.080 (+2.29%)
Complex vs Simple: +0.100 (+2.86%)
Multi-Agent vs Simple: +0.200 (+5.71%)
Complex vs Trained Agent: +0.020 (+0.56%)
Multi-Agent vs Complex: +0.100 (+2.78%)
```

## Troubleshooting

- **Import errors**: Make sure you're using the correct classes from the `PsychotherapyAgent` package:
  - `from PsychotherapyAgent import PsychotherapySingleAgent, PsychotherapyMultiAgent`
- **Missing feedback files**: Check that sessions completed successfully
- **Score extraction issues**: Verify that feedback files contain properly formatted scores

## Architecture Notes

### Multi-Agent Flow

1. **PsychotherapyMultiAgent** initialized with proposer (index 3: `3_MultiAgent_Proposer`) and judge (index 4: `4_MultiAgent_Judge`) from `Therapist_Prompts/therapist_prompts.csv`
2. **Proposer Agent**: Generates 3 candidate responses based on conversation
3. **Judge Agent**: Evaluates and selects best response based on:
   - Therapeutic appropriateness
   - MI principles adherence
   - Empathy and rapport
   - Clinical effectiveness
4. Selected response sent to patient
5. Patient responds

### Single-Agent Flow (Simple, Trained Agent & Complex Modes)

1. **PsychotherapySingleAgent** initialized with therapist profile from `Therapist_Prompts/therapist_prompts.csv`:
   - Simple mode: uses index 0 (`0_Simple_Agent`) — no patient profile
   - Trained Agent mode: uses index 1 (`1_Trained_Agent`) — MI prompt, patient profile withheld
   - Complex mode: uses index 2 (`2_Complex_Agent`) — MI prompt with patient profile
2. Agent generates response directly
3. Response sent to patient
4. Patient responds

