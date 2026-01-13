# Psychotherapy-Agent

An AI-powered virtual patient simulation system for evaluating Motivational Interviewing (MI) therapy techniques. This proof-of-concept demonstrates that a multi-agent architecture (Therapist + Evaluator-in-the-loop) achieves higher MI fidelity than naive or single-agent approaches.

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

We test three therapist conditions, all focused on MI:

### 1. Naive Counselor (Simple Mode)
- **Prompt**: "Be an empathetic, supportive mental health counselor"
- **Characteristics**: No explicit MI language or rubric, no Evaluator involvement
- **Purpose**: Baseline for general supportive counseling without MI training

### 2. MI Therapist - Single Agent (Complex Mode)
- **Prompt**: Grounded in MI spirit and skills (OARS, change talk, autonomy support, avoiding confrontation/advice)
- **Characteristics**: Explicit MI techniques, but no Evaluator-in-the-loop
- **Evaluation**: Evaluator used only after the session for scoring
- **Purpose**: Test if MI-specific prompting alone improves fidelity

### 3. Multi-Agent MI Therapist (Multi-Agent Mode)
- **Prompt**: Same MI Therapist prompt as single-agent
- **Architecture**: At each turn, Therapist proposes 2-3 candidate replies; Evaluator scores candidates on MI criteria (empathy, reflection:question balance, autonomy support) and selects the best one
- **Patient Experience**: Patient sees only the selected response
- **Purpose**: Test if Evaluator-in-the-loop improves MI fidelity beyond single-agent

## Key Capabilities

- **Comprehensive MI Evaluation**: 9 MI criteria scores (Empathy, Partnership, Autonomy Support, Collaboration, Evocation, etc.)
- **Linguistic Analysis**: Sentiment analysis, Linguistic Style Matching (LSM), word count and ratio metrics
- **Statistical Rigor**: Friedman and Wilcoxon tests for significance
- **Robust Design**: Two-stage aggregation (fold averaging → patient-level statistics)
- **Scalability**: Parallel processing, support for multiple runs per patient
- **Reproducibility**: Session metadata tracking and complete conversation logs

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
- Run patients 0-10 with all three therapist modes
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
- `--fold N`: Run each patient N times for robustness (default: 1)
- `--mode {simple|complex|multi_agent}`: Run a specific mode
- `--compare`: Run all three modes and generate comparisons
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

**2. Run all modes for full comparison**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 133 --fold 5 --compare --run-id full_exp
```

**3. Run sessions without evaluation (faster)**:
```bash
python run_automated_test.py --start-patient 0 --end-patient 50 --fold 5 --compare --no-aggregate --run-id batch1
```

**4. Run evaluation only on existing sessions**:
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
│   └── therapist_prompts.csv    # 4 therapist profiles (simple, complex, proposer, judge)
├── Eval/                        # Evaluation scripts and results
│   ├── feedback_analysis.py     # MI criteria scoring
│   ├── sentiment_analysis.py    # Turn-level sentiment
│   ├── lsm_analysis.py          # Linguistic style matching
│   ├── word_analysis.py         # Word count metrics
│   ├── plot_metrics.py          # Visualization generation
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
- **Row 0**: Naive counselor ("be empathetic and supportive" - no MI training)
- **Row 1**: MI Therapist (grounded in MI spirit: OARS, change talk, autonomy support)
- **Row 2**: Multi-agent proposer (same MI Therapist prompt, generates 3 candidate responses)
- **Row 3**: Multi-agent judge/evaluator (scores candidates on MI criteria, selects best)

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
- **Friedman Test**: Three-way comparison across modes
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
- `sentiment_dataset_by_turn.csv`: Turn-level sentiment data
- `lsm_metrics_{mode}.csv`: Language style matching scores
- `metrics_output_{mode}.csv`: Word count and ratio metrics
- `friedman_test_results.csv`: Three-way statistical tests
- `wilcoxon_test_{mode1}_vs_{mode2}.csv`: Pairwise comparisons

### Visualizations (`Eval/plots/`)
- `feedback_scores_by_criterion.png`: MI criteria comparison
- `counselor_words_per_turn.png`: Therapist verbosity
- `client_words_per_turn.png`: Patient engagement
- `word_ratio.png`: Communication balance
- `sentiment_distribution.png`: Emotional tone analysis
- `lsm_over_time.png`: Language coordination trends

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
    proposer_index=2,  # Proposer
    judge_index=3      # Judge
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

- **[COMPARISON_GUIDE.md](COMPARISON_GUIDE.md)**: Detailed methodology for comparing the three therapist modes
- **Code Comments**: Inline documentation in all Python modules
- **Session Logs**: Each session generates detailed logs for debugging

---

**Note**: This system is designed for research and training purposes. It should not be used as a replacement for professional mental health care.
