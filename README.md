# Psychotherapy-Agent

An AI-powered virtual simulation system for performing Motivational Interviewing (MI) therapy techniques. This proof-of-concept demonstrates that a multi-agent architecture (Therapist + Evaluator-in-the-loop) achieves higher MI fidelity than naive or single-agent approaches.

## Research Goal

Design an MI Therapist agent and demonstrate that a multi-agent setup (Therapist + Evaluator-in-the-loop) produces higher MI fidelity than:
1. A naive counselor prompt
2. A single-agent MI prompt

Using existing Patient profiles (based on AnnoMI dataset) and an MI Evaluator, we focus exclusively on Motivational Interviewing to leverage our deepest existing assets.

## Main Research Questions

### 1. MI Fidelity
Can an MI-specific Therapist agent, using our existing Evaluator, conduct brief sessions that achieve MI rubric scores comparable to human MI sessions from AnnoMI?

### 2. Multi-Agent Gain
Does using the Evaluator in-the-loop (as a critic/selector at each turn) improve MI fidelity compared with:
- A naive "supportive counselor" LLM
- A single-agent MI Therapist (no online Evaluator)

## System and Experimental Conditions

We test five therapist conditions, all focused on MI:

### 1. Naive Counselor (Simple Mode)
- **Prompt**: "Be an empathetic, supportive mental health counselor"
- **Characteristics**: No explicit MI language or rubric, no patient profile, no Evaluator involvement
- **Purpose**: Baseline for general supportive counseling without MI training

### 2. MI Therapist - Trained Agent (Trained Agent Mode)
- **Prompt**: Grounded in MI spirit and skills (OARS, change talk, autonomy support, avoiding confrontation/advice)
- **Characteristics**: Explicit MI techniques, but patient profile is intentionally withheld
- **Purpose**: Isolate the effect of MI-specific prompting without access to patient history

### 3. MI Therapist - Single Agent (Complex Mode)
- **Prompt**: Same MI Therapist prompt as trained agent
- **Characteristics**: Explicit MI techniques WITH full patient profile context, no Evaluator-in-the-loop
- **Evaluation**: Evaluator used only after the session for scoring
- **Purpose**: Test if combining MI prompting with patient profile improves fidelity

### 4. Multi-Agent MI Therapist (Multi-Agent Mode)
- **Prompt**: Same MI Therapist prompt as single-agent modes
- **Architecture**: At each turn, Therapist proposes 2-3 candidate replies; Evaluator scores candidates on MI criteria (empathy, reflection:question balance, autonomy support) and selects the best one
- **Patient Experience**: Patient sees only the selected response
- **Purpose**: Test if Evaluator-in-the-loop improves MI fidelity beyond single-agent

### 5. Proposer-Only Ablation (Proposer-Only Mode)
- **Prompt**: Dedicated single-call self-select prompt (`5_Proposer_SelfSelect`)
- **Architecture**: The single proposer call both generates the 3 candidate responses AND picks the best one (no separate judge/evaluator step)
- **Cost match**: 1 LLM call per turn, matching the full multi-agent system's 1 proposer call, isolating the separate-judge contribution
- **Purpose**: Ablation control to isolate the effect of the separate judge/Evaluator-in-the-loop module

## Key Capabilities

- **Comprehensive MI Evaluation**: 9 MI criteria scores (Empathy, Partnership, Autonomy Support, Collaboration, Evocation, etc.)
- **Linguistic Analysis**: Sentiment analysis, Linguistic Style Matching (LSM), word count and ratio metrics
- **Statistical Rigor**: Friedman and Wilcoxon tests for significance
- **Robust Design**: Two-stage aggregation (fold averaging → patient-level statistics)
- **Scalability**: Parallel processing, support for multiple runs per patient
- **Reproducibility**: Session metadata tracking and complete conversation logs

## Reproducing the Reported Experiment

The manuscript evaluates 133 patient profiles × 5 repeated runs × 5 therapist modes = **3,325 total sessions**. To reproduce these results exactly, run:

```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id manuscript_exp --max-workers 20
```

This command:
- Covers all 133 patient profiles (`--start-patient 0 --end-patient 133`)
- Runs each patient 5 times per mode (`--fold 5`), matching the fold=5 setting reported in the manuscript
- Executes all five therapist modes sequentially (`--compare`): `simple`, `trained_agent`, `complex`, `proposer_only`, and `multi_agent`
- After all sessions complete, automatically runs all evaluation scripts (`Eval/`) to produce `feedback_scores_by_criterion.csv`, statistical test results, word metrics, sentiment summaries, and LSM scores

> **Note**: At 20 parallel workers this run is compute- and cost-intensive. For a smaller-scale pilot, replace `--end-patient 133` with a smaller value and adjust `--fold` accordingly.

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- OpenRouter API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amin-kamaleddin/Psychotherapy-Agent.git
   cd Psychotherapy-Agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   OPENROUTER_API_KEY=your_api_key_here
   ```

### Basic Usage

**Run a simple comparison experiment**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 10 --fold 5 --compare --run-id demo
```

This command will:
- Run patients 0-10 with all five therapist modes
- Execute 5 sessions per patient per mode (5 folds)
- Generate evaluation metrics and visualizations
- Save results to `Session_Metadata/` and `Eval/`

## Usage Guide

### Command-Line Options

```bash
python run_automated_test.py [OPTIONS]
```

**Key Parameters**:
- `--start-patient N`: Start from patient index N (default: 0)
- `--end-patient N`: End at patient index N (default: 133)
- `--fold N`: Run each patient N times for robustness (default: 5)
- `--mode {simple|trained_agent|complex|multi_agent|proposer_only}`: Run a specific mode
- `--compare`: Run all five modes and generate comparisons
- `--no-aggregate`: Skip evaluation scripts (run sessions only)
- `--aggregate`: Run only evaluation scripts (no new sessions)
- `--max-workers N`: Parallel processing workers (default: 20)
- `--skip-sentiment`: Skip sentiment analysis (saves time)
- `--run-id ID`: Identifier for this experiment run

### Usage Examples

**1. Run only Simple Mode**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 10 --fold 3 --mode simple --run-id test1
```

**2. Run only Trained Agent Mode**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 10 --fold 3 --mode trained_agent --run-id test2
```

**3. Run all five modes for full comparison**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id full_exp
```

**4. Run sessions without evaluation (faster)**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 50 --fold 5 --compare --no-aggregate --run-id batch1
```

**5. Run evaluation only on existing sessions**:
```bash
python run_automated_test.py --aggregate
```

## Project Structure

```
Psychotherapy-Agent/
├── PsychotherapyAgent/          # Core agent modules
│   ├── psychotherapy_single_agent.py    # Single-agent therapist
│   ├── psychotherapy_multi_agent.py     # Multi-agent therapist
│   ├── psychotherapy_core.py            # Patient simulation core
│   ├── prompt_manager.py                # Prompt handling
│   ├── llm_manager.py                   # LLM interaction
│   └── workflows/                       # LangGraph workflows
├── Patient_Profiles/MI/         # Patient case studies (133 profiles)
├── Therapist_Prompts/           # Therapist system prompts
│   └── therapist_prompts.csv    # 5 therapist profiles (simple, trained, complex, proposer, judge, self-select)
├── Eval/                        # Evaluation scripts and results
│   ├── feedback_analysis.py     # MI criteria scoring
│   ├── sentiment_analysis.py    # Turn-level sentiment
│   ├── lsm_analysis.py          # Linguistic style matching
│   ├── word_analysis.py         # Word count metrics
│   ├── run_lang_per_model.py    # Run LSM/sentiment/word per model (model-tagged per-patient CSVs)
│   ├── plot_metrics.py          # Visualization generation (per-patient, pooled across models)
│   ├── feedback_statistical_test.py  # Statistical tests
│   ├── csv/                     # Generated metrics (after evaluation)
│   └── plots/                   # Generated visualizations
├── Saved_Sessions/              # Individual session transcripts
├── Session_Metadata/            # Experiment run metadata
├── run_automated_test.py        # Main experiment runner
├── requirements.txt             # Python dependencies
└── COMPARISON_GUIDE.md          # Detailed comparison methodology
```

## Configuration

### Therapist Prompts

Therapist behaviors are defined in `Therapist_Prompts/therapist_prompts.csv`:
- **Row 0** (`0_Simple_Agent`): Naive counselor — "be empathetic and supportive", no MI training, no patient profile
- **Row 1** (`1_Trained_Agent`): MI Therapist prompt without patient profile (used by `trained_agent` mode)
- **Row 2** (`2_Complex_Agent`): MI Therapist prompt with full patient profile context (used by `complex` mode)
- **Row 3** (`3_MultiAgent_Proposer`): Multi-agent proposer — same MI Therapist prompt, generates 3 candidate responses
- **Row 4** (`4_MultiAgent_Judge`): Multi-agent judge/evaluator — scores candidates on MI criteria and selects best
- **Row 5** (`5_Proposer_SelfSelect`): Proposer-only self-selection — single call generates 3 candidates AND picks the best (used by `proposer_only` mode)

### Patient Profiles

133 patient case studies are stored in `Patient_Profiles/MI/`, derived from the AnnoMI dataset. Each profile includes:
- Background and demographics
- Primary concerns and goals
- Stage of change (precontemplation, contemplation, preparation, action, maintenance)
- Emotional state and readiness for change

These profiles enable testing MI techniques across diverse patient presentations and change readiness levels.

## Evaluation Metrics

The system evaluates therapy sessions across multiple dimensions:

### 1. MI Criteria Scores (1-5 scale)
- Cultivating Change Talk
- Softening Sustain Talk
- Partnership
- Empathy
- Acceptance
- Autonomy Support
- Collaboration
- Evocation
- Direction

These 9 criteria are based on the Motivational Interviewing Treatment Integrity (MITI) framework and MI spirit principles.

### 2. Linguistic Metrics
- **LSM (Linguistic Style Matching)**: Coordination between therapist and patient language
- **Word Counts**: Average words per turn for therapist and patient
- **Word Ratio**: Therapist-to-patient word ratio

### 3. Sentiment Analysis
- Turn-level sentiment classification (positive/negative/neutral)
- Distribution analysis across modes

### 4. Statistical Tests
- **Friedman Test**: Five-way comparison across modes
- **Wilcoxon Signed-Rank Test**: Pairwise comparisons between modes

## Output Files

After running experiments, you'll find:

### Session Data
- `Saved_Sessions/session_export_patient_{id}_{timestamp}_{uuid}/`
  - `conversation.txt`: Full conversation transcript
  - `feedback.txt`: MI criteria evaluation
  - `session_info.json`: Session metadata

### Aggregated Results (`Eval/csv/`)
- `feedback_scores_by_criterion.csv`: Per-patient MI scores
- `feedback_metrics_{mode}.csv`: Aggregated statistics per mode
- `{model}_sentiment_dataset_by_turn.csv`: Per-patient turn-level sentiment
- `{model}_lsm_per_patient.csv`: Per-patient language style matching scores
- `{model}_words_per_patient.csv`: Per-patient word count and ratio metrics
- `friedman_test_results.csv`: Five-way statistical tests
- `wilcoxon_test_{mode1}_vs_{mode2}.csv`: Pairwise comparisons

### Visualizations (`Eval/plots/`)
Each metric is split into two mode groupings:
- `{metric}__4modes.png`: the 4 counselor configurations (`simple`, `trained_agent`, `complex`, `multi_agent`)
- `{metric}__proposer_vs_multi.png`: the proposer-only ablation vs the full multi-agent system

Metrics:
- `counselor_words_per_turn`: Therapist verbosity (per-patient mean ± SE)
- `client_words_per_turn`: Patient engagement (per-patient mean ± SE)
- `word_ratio`: Communication balance (per-patient mean ± SE)
- `sentiment_distribution`: Emotional tone box plot (per-patient, pooled across models)
- `positive_sentiment_over_time`: Positive sentiment trajectory
- `lsm_over_time`: Language coordination trends (per-patient mean ± SE)
- `feedback_scores_by_criterion`: MI criteria comparison

## Two-Stage Aggregation

The system uses a robust two-stage aggregation method:

### Stage 1: Fold Averaging (Within-Patient)
When using `--fold N`, each patient runs N times. All N sessions are averaged together to produce one score per patient, reducing measurement noise.

### Stage 2: Population Statistics (Cross-Patient)
Fold-averaged scores from all patients are aggregated to calculate mean and standard deviation, providing population-level statistics.

**Example**:
```
Patient 42, Simple Mode with fold=5:
  Run 1: Empathy = 4.2
  Run 2: Empathy = 4.0
  Run 3: Empathy = 4.3
  Run 4: Empathy = 4.1
  Run 5: Empathy = 4.4
  → Stage 1: Patient average = 4.2

All 133 patients → Stage 2: Population mean = 3.87, std = 0.33
```

This approach provides more reliable comparisons with reduced variance.

### Multi-Model Analysis (Linguistic & Sentiment Plots)

The linguistic (LSM, word count/ratio) and sentiment analyses are run **per model** (with `PSYCH_MODEL` set), producing model-tagged per-patient CSVs (e.g. `{model}_lsm_per_patient.csv`, `{model}_words_per_patient.csv`, `{model}_sentiment_dataset_by_turn.csv`). Because the same 133 patients are scored by every model, `plot_metrics.py` **averages the model scores within each patient first**, then plots the patient-level mean ± standard error (`SE = std/√133`, i.e. the standard error across the 133 patient-means, not across the 133×M raw rows). This avoids double-counting the same patient across models and keeps the error bars honest. See `COMPARISON_GUIDE.md` for the full formulas.

## Programmatic Usage

You can also use the agents programmatically:

```python
from PsychotherapyAgent import PsychotherapySingleAgent, PsychotherapyMultiAgent

# Initialize single-agent therapist
agent = PsychotherapySingleAgent(
    api_key="your-api-key",
    therapist_index=0  # 0=simple, 1=complex
)

# Start session with patient profile
state = agent.start_session(case_study="path/to/patient_profile.txt")

# Conduct conversation
response = agent.send_message("Hello, I'm here to support you today.")
print(f"Patient: {response['patient_response']}")

# Initialize multi-agent system
multi_agent = PsychotherapyMultiAgent(api_key="your-api-key")
multi_agent.set_multi_agent_therapist_profiles(
    proposer_index=3,  # Proposer (CSV row 3: 3_MultiAgent_Proposer)
    judge_index=4      # Judge (CSV row 4: 4_MultiAgent_Judge)
)

# Same usage pattern
state = multi_agent.start_session(case_study="path/to/patient_profile.txt")
response = multi_agent.send_message("How are you feeling today?")
```

## Troubleshooting

**Import errors**:
```python
from PsychotherapyAgent import PsychotherapySingleAgent, PsychotherapyMultiAgent
```

**Count mismatch in CSV files**:
If aggregated CSV files show counts other than 133, regenerate them:
```bash
python run_automated_test.py --aggregate
```

**Memory issues**:
Use `--skip-sentiment` to reduce memory usage:
```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --skip-sentiment
```

**Stale data**:
Always regenerate evaluation CSVs after code changes using `--aggregate`.

## Documentation

- **[COMPARISON_GUIDE.md](COMPARISON_GUIDE.md)**: Detailed methodology for comparing the five therapist modes
- **Code Comments**: Inline documentation in all Python modules
- **Session Logs**: Each session generates detailed logs for debugging

---

**Note**: This system is designed for research and training purposes. It should not be used as a replacement for professional mental health care.
