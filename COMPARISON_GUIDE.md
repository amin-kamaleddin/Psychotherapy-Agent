# LearningBuddy Comparison Guide: Simple vs Complex vs Multi-Agent System

This guide explains how to compare the performance between three different therapist configurations:
1. Simple prompt single-agent
2. Complex prompt single-agent (with detailed MI techniques)
3. Multi-agent system (proposer + judge)

## Overview

The system now supports three modes:

1. **Simple Mode** (`therapist_mode="simple"`):
   - Uses `VirtualTherapist` with a simple prompt from `therapist_prompts.csv`
   - Single-agent system that generates one therapist response
   - Response is sent directly to the patient

2. **Complex Mode** (`therapist_mode="complex"`):
   - Uses `VirtualTherapist` with a complex, MI-focused prompt from `therapist_prompts.csv`
   - Single-agent system with more sophisticated prompt
   - Response is sent directly to the patient

3. **Multi-Agent Mode** (`therapist_mode="multi_agent"`):
   - Proposer agent generates 3 candidate therapist responses
   - Judge agent selects the best response from the 3 candidates
   - Selected response is sent to the patient
   - This is the new enhanced system

## Implementation Details

### LearningBuddy Changes

The `LearningBuddy` class in `LearningBuddy.py` accepts a `use_multi_agent` parameter:

```python
# Single-agent mode (simple or complex)
lb = LearningBuddy(api_key, use_multi_agent=False)

# Multi-agent mode
lb = LearningBuddy(api_key, use_multi_agent=True)
```

### Evaluation Script Changes

The `eval_tester.py` script has been updated to support all three modes and comparison:

1. **Run Simple Mode Only**:
   ```bash
   python eval_tester.py --start-patient 0 --end-patient 5 --run-id simple_test --mode simple
   ```

2. **Run Complex Mode Only**:
   ```bash
   python eval_tester.py --start-patient 0 --end-patient 5 --run-id complex_test --mode complex
   ```

3. **Run Multi-Agent Mode Only**:
   ```bash
   python eval_tester.py --start-patient 0 --end-patient 5 --run-id multiagent_test --mode multi_agent
   ```

4. **Run Comparison (All Three Modes)**:
   ```bash
   python eval_tester.py --start-patient 0 --end-patient 5 --run-id comparison_test --compare
   ```

5. **Specify Therapist Profile Indices**:
   ```bash
   python eval_tester.py --start-patient 0 --end-patient 5 --run-id comparison_test --compare \
     --simple-therapist 0 --complex-therapist 1
   ```

## Running Comparisons

### Step 1: Run Comparison Experiment

The `--compare` flag runs all three modes sequentially and generates a comparison:

```bash
python eval_tester.py --start-patient 0 --end-patient 10 --run-id exp1 --compare
```

Or with specific therapist profile indices:

```bash
python eval_tester.py --start-patient 0 --end-patient 10 --run-id exp1 --compare \
  --simple-therapist 0 --complex-therapist 1
```

This will:
1. Run all patients with simple prompt mode
2. Run all patients with complex prompt mode
3. Run all patients with multi-agent mode
4. Compare all three and generate statistics

### Step 2: Review Results

Results are saved in `saved_tests/` directory:

- `comparison_{run_id}_{timestamp}/` - Comparison results folder
  - `comparison_results.csv` - Detailed 3-way comparison statistics
  - Contains mean, std, differences between all pairs, and best performing mode for each category

- `{run_id}_simple_patients_{start}_{end}_{timestamp}/` - Simple mode results
  - `results_simple.csv` - Individual patient scores for simple prompt mode

- `{run_id}_complex_patients_{start}_{end}_{timestamp}/` - Complex mode results
  - `results_complex.csv` - Individual patient scores for complex prompt mode

- `{run_id}_multi_agent_patients_{start}_{end}_{timestamp}/` - Multi-agent mode results
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
Empathy                     3.8         4.0          4.1              Multi-Agent
Overall Average             3.5         3.6          3.7              Multi-Agent

OVERALL AVERAGE COMPARISON
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

- **Import errors**: Make sure you're using `LearningBuddy.py` (not `LearningBuddy_Agent`)
- **Missing feedback files**: Check that sessions completed successfully
- **Score extraction issues**: Verify that feedback files contain properly formatted scores

## Architecture Notes

### Multi-Agent Flow

1. User/VirtualTherapist generates draft response
2. **Proposer Agent**: Generates 3 candidate responses based on conversation
3. **Judge Agent**: Evaluates and selects best response based on:
   - Therapeutic appropriateness
   - MI principles adherence
   - Empathy and rapport
   - Clinical effectiveness
4. Selected response sent to patient
5. Patient responds

### Original Flow

1. VirtualTherapist generates response
2. Response sent directly to patient
3. Patient responds

