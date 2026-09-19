# PsychotherapyAgent Comparison Guide: Simple vs Trained Agent vs Complex vs Multi-Agent System vs Proposer-Only

This guide explains how to compare the performance between five different therapist configurations:
1. Simple prompt single-agent (no MI, no patient profile)
2. Trained agent single-agent (MI prompt, no patient profile)
3. Complex prompt single-agent (MI prompt + patient profile)
4. Multi-agent system (proposer + judge)
5. Proposer-only ablation (single call: proposer generates AND selects, matched 1-call cost)

## Overview

The system supports five modes with **two-stage aggregation** for robust statistical analysis:

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

5. **Proposer-Only Mode** (`therapist_mode="proposer_only"`):
   - Ablation control using `PsychotherapyMultiAgent` in self-selection mode (CSV index 5: `5_Proposer_SelfSelect`)
   - The single proposer call both generates the 3 candidates AND picks the best one
   - Matches the 1-call cost of the full system, isolating the separate-judge contribution

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

## Multi-Model Analysis (Per-Patient Means Averaged Across Models)

When comparing **multiple models** (e.g. `gpt-5-mini`, `qwen/qwen3.8-27b`, `z-ai/glm-5.3-flash`), each model independently scores the same **133 patients** across all 5 modes. This raises a subtle statistical question: *how should the 3 model scores for each patient be combined into a single number?*

### Why pool within patient, not across all rows?

Each patient produces **3 scores per metric** (one from each model). There are two naive ways to combine them, and only one is correct here:

1. **Pool all 133 × 3 = 399 rows together** and compute $\text{std}/\sqrt{399}$.
   ❌ This treats the 3 model runs as *independent* observations. They are not — the same patient is being measured 3 times, so the scores are **correlated** (a patient who is verbose under one model tends to be verbose under all three). Inflating $n$ from 133 to 399 spuriously **narrows the error bars** by ~$\sqrt{3}$, making differences look more significant than they are.

2. **Average the 3 model scores *within* each patient first**, then compute statistics across the 133 patient-means.
   ✅ This is the correct approach for **matched samples**. It collapses the 3 model measurements into one value per patient ($\bar{X}_p$), then treats those 133 values as the independent sample. The standard error is $\text{std}(\bar{X}_p)/\sqrt{133}$, which honestly reflects the between-patient variability.

**Workflow:**
1. Run `feedback_analysis.py`, `lsm_analysis.py`, `word_analysis.py`, `sentiment_analysis.py` **per model** with `PSYCH_MODEL` set, producing model-tagged per-patient CSVs.
2. Run `plot_metrics.py`, which loads the per-model per-patient files, averages them within patient, and plots the patient-level mean ± SE.

### Notation

- $M$ = number of models (here 3)
- $N$ = 133 patients in a mode
- $X^{(j)}_{p,k}$ = the value that model $j$ assigns to patient $p$ at turn $k$
- $\text{turn}_k$ = the $k$-th therapist–patient turn

### Step 1 — within-patient average (collapse the 3 models)

For each patient $p$, mode, and turn $k$, the 3 model scores are averaged into a single patient-level value:

$$\bar{X}_{p,k} = \frac{1}{M}\sum_{j=1}^{M} X^{(j)}_{p,k}$$

### Step 2 — population statistics across patients

For each mode and turn $k$, the patient-level values are summarized:

$$\overline{X}_k = \frac{1}{N}\sum_{p=1}^{N} \bar{X}_{p,k}
\qquad \mathrm{SE}_k = \frac{\mathrm{std}_p(\bar{X}_{p,k})}{\sqrt{N}}$$

where $\mathrm{std}_p(\cdot)$ is the sample standard deviation (ddof = 1) across the 133 patients.

The shaded bands and error bars in the plots show $\overline{X}_k \pm \mathrm{SE}_k$. The sentiment **distribution** plot instead draws a box-and-whisker of the per-patient means $\bar{X}_{p,10}$.

### Worked example

Suppose the *client word count* at turn 1 for patient #42 is measured by the three models as:

| Model | Client words |
|---|---|
| gpt-5-mini | 80.5 |
| qwen-qwen3.8-27b | 69.1 |
| z-ai-glm-5.3-flash | 90.6 |

- **Step 1** (within patient): $\bar{X}_{42,1} = (80.5 + 69.1 + 90.6)/3 = 80.1$
- **Step 2** (across patients): this $\bar{X}_{42,1}$ becomes one of the 133 values whose mean and SE are plotted.

If instead all 399 rows were pooled, patient #42 would contribute 3 data points instead of 1, and the SE would be artificially deflated by $\sqrt{3}$.

### Feedback Scores (handled separately via `average_across_models.py`)

The MI feedback chart is the one exception: it uses `average_across_models.py`, which averages the **per-patient score** across models for each `(patient_id, therapist_mode, criterion)` before computing population statistics, rather than pooling per-patient values:

$$\mathrm{avgscore}_{p} = \frac{1}{M}\sum_{j=1}^{M} \mathrm{score}^{(j)}_{p}$$

Per-mode summary across patients:

$$\overline{X} = \frac{1}{N_{mode}}\sum_{p=1}^{N_{mode}} \mathrm{avgscore}_p
\qquad \mathrm{std} = \sqrt{\tfrac{1}{N_{mode}-1}\sum_{p}(\mathrm{avgscore}_p - \overline{X})^2}$$

### LSM (`lsm_analysis.py` → `<model>_lsm_per_patient.csv`)

**Per-turn LSM score** (counselor vs client utterance at turn $k$), across $C$ word categories:

$$LSM_k = \frac{1}{C}\sum_{c=1}^{C}\left(1 - \frac{|p_{1,c} - p_{2,c}|}{p_{1,c} + p_{2,c} + 0.0001}\right),$$

where $p_{1,c}$ is the proportion of category-$c$ words in the counselor's utterance and $p_{2,c}$ the client's.

**Cumulative per-patient** value at turn $k$ (running mean over all turns $\le k$):

$$\mathrm{avglsm}_{p,k} = \frac{1}{k}\sum_{i=1}^{k} LSM_{p,i}$$

**Plot curve** (within-patient model-average, then across patients) for each `(therapist_mode, turn_number)`:

$$\bar{\mathrm{lsm}}_{p,k} = \frac{1}{M}\sum_{j=1}^{M} \mathrm{avglsm}^{(j)}_{p,k}, \qquad
\overline{\text{LSM}}_{k} = \frac{1}{N}\sum_{p=1}^{N} \bar{\mathrm{lsm}}_{p,k}, \qquad
\mathrm{SE}_k = \frac{\mathrm{std}_p(\bar{\mathrm{lsm}}_{p,k})}{\sqrt{N}}$$

### Word Metrics (`word_analysis.py` → `<model>_words_per_patient.csv`)

**Per-turn cumulative running average** of words per turn (counselor $c$ and client $cl$):

$$\mathrm{caccavg}_{p,k} = \frac{T^c_{p,k}}{n^c_{p,k}}, \qquad \mathrm{claccavg}_{p,k} = \frac{T^{cl}_{p,k}}{n^{cl}_{p,k}}$$

where $T_{p,k}$ = cumulative word count for patient $p$ and $n_k$ = number of turns up to turn $k$.

**Counselor:client ratio** (accumulated words):

$$r_{p,k} = \frac{T^c_{p,k}}{T^{cl}_{p,k}} \quad (0 \text{ if } T^{cl}_{p,k} = 0)$$

**Plot curve** (within-patient model-average, then across patients) for each `(therapist_mode, turn)`:

$$\bar{X}_{p,k} = \frac{1}{M}\sum_{j=1}^{M} X^{(j)}_{p,k}, \qquad
\overline{X}_{k} = \frac{1}{N}\sum_{p=1}^{N} \bar{X}_{p,k}, \qquad
\mathrm{SE}_k = \frac{\mathrm{std}_p(\bar{X}_{p,k})}{\sqrt{N}}$$

### Sentiment (`sentiment_analysis.py` → `<model>_sentiment_dataset_by_turn.csv`)

**Per-patient cumulative proportions** (positive / neutral / negative), counting sentiment classifications up to turn $k$:

$$\mathrm{proppos}_{p,k} = \frac{\sum_{i=1}^{k} \mathrm{pos}_{p,i}}{\sum_{i=1}^{k} n_{p,i}}, \quad
\mathrm{propneu}_{p,k} = \frac{\sum_{i=1}^{k} \mathrm{neu}_{p,i}}{\sum_{i=1}^{k} n_{p,i}}, \quad
\mathrm{propneg}_{p,k} = \frac{\sum_{i=1}^{k} \mathrm{neg}_{p,i}}{\sum_{i=1}^{k} n_{p,i}}$$

Note $\mathrm{proppos}_{p,k} + \mathrm{propneu}_{p,k} + \mathrm{propneg}_{p,k} = 1$.

- **Distribution plot** (turn 10): a **box-and-whisker** plot of the per-patient means $\bar{\mathrm{proppos}}_{p,10}$ (model-averaged within patient).
- **Positive-sentiment-over-time plot**: per-turn patient-level means with SEM error bars:

$$\bar{\mathrm{proppos}}_{p,k} = \frac{1}{M}\sum_{j=1}^{M} \mathrm{proppos}^{(j)}_{p,k}, \qquad
\overline{\mathrm{proppos}}_{k} = \frac{1}{N}\sum_{p=1}^{N} \bar{\mathrm{proppos}}_{p,k}, \qquad
\mathrm{SE}_k = \frac{\mathrm{std}_p(\bar{\mathrm{proppos}}_{p,k})}{\sqrt{N}}$$

> **Summary:** every plot metric is a patient-level mean $\bar{X}_{p,k}$ (the $M$ model scores averaged *within* each patient) followed by the population mean and SE across the $N$=133 patients, computed **after** the per-model fold-averaging (Stage 1). There is no separate cross-model averaged CSV — the plots aggregate the per-model per-patient files directly.

## Evaluation Script Commands

The `run_automated_test.py` script supports all five modes with flexible options:

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

5. **Run Proposer-Only Mode**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --mode proposer_only --run-id test5
   ```

6. **Run All Five Modes + Aggregation (Comparison)**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --run-id exp1
   ```

7. **Run All Five Modes WITHOUT Aggregation**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --no-aggregate --run-id exp1
   ```

8. **Run Aggregation Only (after sessions completed)**:
   ```bash
   python run_automated_test.py --aggregate
   ```

9. **Specify Custom Parameters**:
   ```bash
   python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id full_test \
     --max-workers 20 --skip-sentiment
   ```

### Key Parameters

- `--fold N`: Run each patient N times (default: 1)
- `--compare`: Run all 5 modes and generate comparison
- `--no-aggregate`: Skip evaluation/aggregation scripts
- `--aggregate`: Only run aggregation (no new sessions)
- `--max-workers N`: Parallel workers for processing (default: 20)
- `--skip-sentiment`: Skip sentiment analysis (fastest part to skip)

## Running Comparisons

### Step 1: Run Comparison Experiment

The `--compare` flag runs all five modes sequentially and generates a comparison:

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

**Evaluation Results** (in `Eval/csv/` directory, per model when `PSYCH_MODEL` is set):
- `feedback_scores_by_criterion.csv` - Per-patient scores (133 patients × 5 modes)
- `feedback_metrics_{mode}.csv` - Aggregated statistics per mode
- `{model}_sentiment_dataset_by_turn.csv` - Per-patient sentiment by turn (pooled in plots)
- `{model}_lsm_per_patient.csv` - Per-patient LSM scores per turn
- `{model}_words_per_patient.csv` - Per-patient word count and ratio per turn
- `comparison_results.csv` - Five-way comparison summary
- `friedman_test_results.csv` - Statistical significance tests
- `wilcoxon_test_{mode1}_vs_{mode2}.csv` - Pairwise comparisons

**Plots** (in `Eval/plots/` directory). Each metric is split into two mode groupings:
- `{metric}__4modes.png` — the 4 counselor configurations (`simple`, `trained_agent`, `complex`, `multi_agent`)
- `{metric}__proposer_vs_multi.png` — the proposer-only ablation vs the full multi-agent system

where `{metric}` is one of `counselor_words_per_turn`, `client_words_per_turn`, `word_ratio`, `sentiment_distribution`, `positive_sentiment_over_time`, `lsm_over_time`, `feedback_scores_by_criterion`.

### Step 3: Analyze Comparison

The comparison CSV (`comparison_results.csv`) includes:

- **Category**: Evaluation category (Empathy, Acceptance, etc.)
- `Simple_Mean` / `Simple_Std`: Score statistics for simple mode
- `TrainedAgent_Mean` / `TrainedAgent_Std`: Score statistics for trained agent mode
- `Complex_Mean` / `Complex_Std`: Score statistics for complex mode
- `MultiAgent_Mean` / `MultiAgent_Std`: Score statistics for multi-agent mode
- `ProposerOnly_Mean` / `ProposerOnly_Std`: Score statistics for proposer-only mode
- `Best_Mode`: Which mode performed best for this category
- `{ModeA}_vs_{ModeB}`: Mean difference (`ModeB - ModeA`) for each of the C(5,2)=10 mode pairs

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
FIVE-WAY COMPARISON SUMMARY
================================================================================
Category                  Simple_Mean  TrainedAgent_Mean  Complex_Mean  ProposerOnly_Mean  MultiAgent_Mean  Best_Mode
Cultivating Change Talk   3.2          3.4                3.5           3.6                3.7              Multi-Agent
Empathy                   3.5          3.6                3.7           3.8                3.9              Multi-Agent
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

