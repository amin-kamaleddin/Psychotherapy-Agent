import os
import re
import time
import json
import argparse
import logging
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Suppress langchain verbose deprecation warning
warnings.filterwarnings("ignore", message=".*Importing verbose from langchain root module.*", category=UserWarning)

# Thread-safe print lock
print_lock = Lock()

import pandas as pd

# Import PsychotherapyAgent classes from modular version
from PsychotherapyAgent import PsychotherapyMultiAgent, PsychotherapySingleAgent
from utils.api_key_loader import load_openai_api_key

# =============================
# === Experiment Constants ====
# =============================
THERAPIST_CSV_PATH = "Therapist_Prompts/therapist_prompts.csv"
PATIENT_DIR = "Patient_Profiles/MI/"

API_KEY = load_openai_api_key()


def model_runs_dir(model: str):
    """Resolve the Model_Runs/<model>/ root directory (sessions + metadata).

    Args:
        model: Model identifier, e.g. 'gpt-5-mini', 'qwen/qwen3.8-27b', 'deepseek/deepseek-v4-flash'.
               Use None/'' for the legacy root layout (Saved_Sessions, Session_Metadata).
    """
    if not model:
        return None
    # Sanitize model id (e.g. 'qwen/qwen3.8-27b') into a safe folder name
    folder = model.replace("/", "-")
    base = os.path.join("Model_Runs", folder)
    os.makedirs(base, exist_ok=True)
    return base


# =============================
# === Core Helper Functions ===
# =============================

def get_patient_files(start_idx=None, end_idx=None):
    """Get patient files in alphabetical order, optionally sliced by range."""

    all_files = sorted([
        os.path.join(PATIENT_DIR, f)
        for f in os.listdir(PATIENT_DIR)
        if os.path.isfile(os.path.join(PATIENT_DIR, f))
    ])

    if start_idx is not None and end_idx is not None:
        return all_files[start_idx:end_idx]
    return all_files


def run_test(patient_profile_path, therapist_profile_number,
             patient_number=None, therapist_feedback_csv="", api_key_input=None,
             therapist_mode="simple"):
    """Run one simulated therapy session between therapist and patient.

    Args:
        patient_profile_path: Path to patient profile file
        therapist_profile_number: Index of therapist profile to use (for simple/complex modes)
        patient_number: Optional patient identifier for folder naming
        therapist_feedback_csv: Optional CSV file with therapist feedback prompts
        api_key_input: Optional API key
        therapist_mode: One of "simple", "trained_agent", "complex", "multi_agent", or "proposer_only"
                        - "simple": Single-agent with 0_Simple_Agent prompt (index 0, no patient profile)
                        - "trained_agent": Single-agent with 1_Trained_Agent prompt (index 1, MI principles,
                                           no patient profile — distinct from the complex agent prompt)
                        - "complex": Single-agent with 2_Complex_Agent prompt (index 2, MI principles +
                                     explicit profile access, includes patient profile at runtime)
                        - "multi_agent": Multi-agent system (proposer index 3 + judge index 4)
                        - "proposer_only": Ablation control — SAME proposer model but the selection
                                           happens inside the single proposer LLM call (self-selection),
                                           not via a separate judge. Matched 1-call cost vs the full
                                           system's 2 calls, isolating the separate judge module.
    """
    api_key_input = API_KEY or api_key_input or "key"

    # Prepare session metadata for folder naming
    session_metadata = {}
    if patient_number is not None:
        session_metadata['patient_number'] = patient_number
    session_metadata['therapist_type'] = f'therapist_{therapist_mode}'

    add_on = ""
    if not therapist_feedback_csv == "":
        df = pd.read_csv(therapist_feedback_csv, quotechar='"', skipinitialspace=True)
        add_on = df.loc[therapist_profile_number, "Prompt"]

    # Determine if using multi-agent system
    use_multi_agent = (therapist_mode in ("multi_agent", "proposer_only"))

    # Initialize appropriate session manager based on mode
    if use_multi_agent:
        # Multi-agent / proposer_only mode: use PsychotherapyMultiAgent
        lb = PsychotherapyMultiAgent(api_key_input)
        
        # Set proposer and judge to use CSV rows 5 and 6 (indices 3 and 4)
        # CSV structure: Row 1=Header, Row 2=0_Simple_Agent (index 0), Row 3=1_Trained_Agent (index 1),
        #                Row 4=2_Complex_Agent (index 2), Row 5=3_MultiAgent_Proposer (index 3),
        #                Row 6=4_MultiAgent_Judge (index 4)
        proposer_index = 3  # Row 5 in CSV: 3_MultiAgent_Proposer
        judge_index = 4     # Row 6 in CSV: 4_MultiAgent_Judge
        lb.set_multi_agent_therapist_profiles(proposer_index, judge_index)
        if therapist_mode == "proposer_only":
            # Ablation control: same proposer model, but selection happens INSIDE the single
            # proposer LLM call (self-selection) instead of a separate judge agent.
            # This is a matched-cost single-agent control: 1 call vs the full system's 2 calls.
            lb.set_selection_mode("self")
            logging.info("proposer_only mode: single-call self-selection (proposer generates AND selects)")
        else:
            lb.set_selection_mode("judge")
        logging.info(f"Multi-agent mode: Proposer index {proposer_index}, Judge index {judge_index}")
    elif therapist_mode == "trained_agent":
        # Trained agent: 1_Trained_Agent prompt (index 1) — full MI principles, patient profile withheld at runtime
        lb = PsychotherapySingleAgent(
            api_key_input,
            therapist_index=therapist_profile_number,
            prompt_addon=add_on,
            include_patient_profile=False
        )
        logging.info(f"trained_agent mode: Using 1_Trained_Agent prompt (index {therapist_profile_number}), no patient profile")
    elif therapist_mode == "simple":
        # Simple agent: basic prompt (index 0), patient profile withheld
        lb = PsychotherapySingleAgent(
            api_key_input,
            therapist_index=therapist_profile_number,
            prompt_addon=add_on,
            include_patient_profile=False
        )
        logging.info(f"simple mode: Using 0_Simple_Agent prompt (index {therapist_profile_number}), no patient profile")
    else:  # complex
        # Complex agent: 2_Complex_Agent prompt (index 2) — MI principles + explicit patient profile access
        lb = PsychotherapySingleAgent(
            api_key_input,
            therapist_index=therapist_profile_number,
            prompt_addon=add_on,
            include_patient_profile=True
        )
        logging.info(f"complex mode: Using 2_Complex_Agent prompt (index {therapist_profile_number}), with patient profile")

    # Load patient profile and start session with metadata
    case = lb.load_patient_profile(patient_profile_path)

    # Pass metadata to start_session
    session_result = lb.start_session(case_study=case, session_metadata=session_metadata)

    # Run 10 exchanges between therapist and patient
    for exchange_num in range(10):
        # Both modes now use send_message() which handles therapist + patient response automatically
        # Multi-agent: proposer/judge generates therapist response internally
        # Single-agent: PsychotherapySingleAgent generates therapist response internally
        session_result = lb.send_message()

    # End session and get feedback
    session_result = lb.end_session()
    final_state = lb.get_session_state()

    # Get folder name - try both possible keys
    folder_name = final_state.get("export_folder") or final_state.get("folder_name", "")

    return {
        "session_result": session_result,
        "folder_name": folder_name,
        "mode": therapist_mode
    }


def extract_scores(eval_file_paths):
    """Extract numerical scores from feedback text files.
    
    With structured output, scores should always be in consistent format.
    This function handles both structured and legacy unstructured formats.
    """

    # Define categories in exact order expected
    categories = [
        "Acceptance",
        "Empathy",
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Direction",
        "Autonomy Support",
        "Collaboration",
        "Evocation"
    ]        

    scores = {cat: [] for cat in categories}

    for file_path in eval_file_paths:
        text = Path(file_path).read_text(encoding="utf-8")
        
        # Try to detect and skip example section if present
        # Look for marker or Linda-specific text
        if "[END OF EXAMPLE" in text:
            # Only parse text after the marker
            parts = text.split("[END OF EXAMPLE")
            if len(parts) > 1:
                text = parts[-1]
        elif "Linda's ambivalence" in text or "Linda's attachment to smoking" in text:
            # Likely has example - try to find where actual eval starts
            # Look for second occurrence of score sections
            lines = text.split('\n')
            acceptance_indices = [i for i, line in enumerate(lines) if '### 1. Acceptance' in line or '1. Acceptance' in line]
            if len(acceptance_indices) > 1:
                # Use text from second Acceptance onward
                text = '\n'.join(lines[acceptance_indices[1]:])

        for cat in categories:
            # Match pattern: "Score: X/5" or "**Score: X/5**"
            pattern = re.compile(
                rf"{re.escape(cat)}.*?Score:\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*5\**",
                re.IGNORECASE | re.DOTALL
            )
            matches = pattern.findall(text)
            if matches:
                # Use last match (in case there are still duplicates)
                score_value = float(matches[-1])
            else:
                # Try alternative pattern without "/5"
                pattern_alt = re.compile(
                    rf"{re.escape(cat)}.*?Score:\s*\**\s*(\d+(?:\.\d+)?)\**",
                    re.IGNORECASE | re.DOTALL
                )
                matches_alt = pattern_alt.findall(text)
                if matches_alt:
                    score_value = float(matches_alt[-1])
                else:
                    score_value = 0.0
            scores[cat].append(score_value)

    num_files = len(eval_file_paths)
    overall_averages = []
    for i in range(num_files):
        file_scores = [scores[cat][i] for cat in categories]
        overall_avg = round(sum(file_scores) / len(file_scores), 3)
        overall_averages.append(overall_avg)

    scores["Overall Average"] = overall_averages
    return scores



# =============================
# === Experiment Pipeline ====
# =============================

def run_parallel_experiments(args, therapist_mode="simple", simple_therapist_num=None, complex_therapist_num=None):
    """Main function for parallel execution mode.
    
    Args:
        args: Command line arguments
        therapist_mode: One of "simple", "trained_agent", "complex", or "multi_agent"
        simple_therapist_num: Therapist profile index for simple mode (default: 0, CSV index 0/row 2)
        complex_therapist_num: Therapist profile index for complex mode (default: 2, CSV index 2/row 4)
    """
    start_time = time.time()
    
    # Determine which therapist profile number to use
    # CSV structure: 0=simple (row 2), 1=trained_agent (row 3), 2=complex (row 4),
    #                3=multi_agent proposer (row 5), 4=multi_agent judge (row 6)
    if therapist_mode == "simple":
        therapist_num = simple_therapist_num if simple_therapist_num is not None else 0  # CSV index 0 (row 2)
    elif therapist_mode == "trained_agent":
        therapist_num = 1  # CSV index 1 (row 3): 1_Trained_Agent — complex MI, no patient profile
    elif therapist_mode == "complex":
        therapist_num = complex_therapist_num if complex_therapist_num is not None else 2  # CSV index 2 (row 4)
    else:  # multi_agent / proposer_only
        therapist_num = 3  # multi-agent uses proposer index 3 (row 5) and judge index 4 (row 6)

    # Get patient files for this range
    patient_files = get_patient_files(args.start_patient, args.end_patient)
    num_patients = len(patient_files)
    
    # Get fold parameter (default to 1 if not provided)
    fold = getattr(args, 'fold', 1)
    total_runs = num_patients * fold

    print(f"Processing patients {args.start_patient} to {args.end_patient} ({num_patients} patients)")
    print(f"Fold: {fold} (Total runs: {total_runs})")
    print(f"Mode: {therapist_mode}")
    print(f"Therapist Profile Index: {therapist_num}")

    # Create save folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_str = f"_fold{fold}" if fold > 1 else ""
    # Model-aware metadata root (Model_Runs/<model>/Session_Metadata) when --model given
    meta_root = os.environ.get("PSYCH_META_DIR", "Session_Metadata")
    save_folder = os.path.join(
        meta_root,
        f"{args.run_id}_{therapist_mode}_patients_{args.start_patient}_{args.end_patient}{fold_str}_{timestamp}"
    )
    os.makedirs(save_folder, exist_ok=True)

    # Save metadata
    metadata = {
        "run_id": args.run_id,
        "start_patient": args.start_patient,
        "end_patient": args.end_patient,
        "num_patients": num_patients,
        "fold": fold,
        "total_runs": total_runs,
        "timestamp": timestamp,
        "therapist_mode": therapist_mode,
        "therapist_profile_number": therapist_num
    }
    with open(os.path.join(save_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Initialize storage for all results
    all_results = []

    df = pd.read_csv(THERAPIST_CSV_PATH, quotechar='"', skipinitialspace=True)

    def process_single_patient(patient_data):
        """Process a single patient test - designed for parallel execution."""
        fold_idx, patient_idx, patient_file = patient_data
        # Patient number reflects the actual patient profile (same across all folds)
        # Uniqueness across folds is provided by timestamp in folder name
        patient_number = args.start_patient + patient_idx
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with print_lock:
                    print(f"\n{'=' * 60}")
                    print(f"Processing Patient {patient_number} (Fold {fold_idx+1}/{fold}): {os.path.basename(patient_file)}")
                    print(f"Mode: {therapist_mode}")
                    print(f"{'=' * 60}")
                    print(f"  [STEP 1] Running test ({therapist_mode} mode)...")
                
                # Run the test
                results = run_test(
                    patient_file,
                    therapist_profile_number=therapist_num,
                    patient_number=patient_number,
                    therapist_mode=therapist_mode
                )
                
                # Look for the feedback file
                folder_name = results["folder_name"]
                possible_names = ["detailed_feedback.txt", "feedback.txt"]
                
                feedback_path = None
                for name in possible_names:
                    potential_path = os.path.join(folder_name, name)
                    if os.path.exists(potential_path):
                        feedback_path = potential_path
                        break
                
                if feedback_path is None:
                    with print_lock:
                        print(f"  Warning: Could not find feedback file in {folder_name}")
                    feedback_path = os.path.join(folder_name, "detailed_feedback.txt")
                
                scores = extract_scores([feedback_path])
                
                # Store therapist result
                row = {
                    "Therapist": df.iloc[therapist_num]["Name"],
                    "Patient": os.path.basename(patient_file),
                    "Mode": therapist_mode,
                    "Session": os.path.basename(folder_name)
                }
                for key, values in scores.items():
                    row[key] = values[0] if values else None
                
                with print_lock:
                    print(f"  Completed Patient {patient_number}: {os.path.basename(patient_file)}")
                
                return row
            
            except Exception as e:
                with print_lock:
                    print(f"  Attempt {attempt}/{max_retries} failed for Patient {patient_number} ({os.path.basename(patient_file)}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)  # backoff: 5s, 10s, 15s
        
        with print_lock:
            print(f"  All {max_retries} attempts failed for Patient {patient_number} ({os.path.basename(patient_file)}), skipping.")
        return None

    # Determine number of parallel workers
    max_workers = args.max_workers if hasattr(args, 'max_workers') else 20
    
    with print_lock:
        print(f"\nStarting parallel processing with {max_workers} workers...")
        print(f"Processing {num_patients} patients x {fold} folds = {total_runs} total runs in {therapist_mode} mode\n")
    
    # Process patients in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks - create fold copies of each patient
        patient_data_list = []
        for fold_idx in range(fold):
            for patient_idx, patient_file in enumerate(patient_files):
                patient_data_list.append((fold_idx, patient_idx, patient_file))
        
        future_to_patient = {executor.submit(process_single_patient, data): data for data in patient_data_list}
        
        # Collect results as they complete
        # timeout=600 per task (10 min) prevents a single hung API call from blocking forever
        completed = 0
        for future in as_completed(future_to_patient, timeout=None):
            completed += 1
            try:
                result = future.result(timeout=600)  # 10-minute per-task timeout
                if result is not None:
                    all_results.append(result)
            except TimeoutError:
                patient_data = future_to_patient[future]
                with print_lock:
                    print(f"  WARNING: Task timed out after 600s, skipping: {patient_data}")
            except Exception as e:
                patient_data = future_to_patient[future]
                with print_lock:
                    print(f"  WARNING: Task raised exception, skipping: {patient_data} — {e}")
            
            with print_lock:
                print(f"Progress: {completed}/{total_runs} runs completed")
    
    # Sort results by patient filename to maintain order
    patient_file_basenames = {os.path.basename(f): idx for idx, f in enumerate(patient_files)}
    all_results.sort(key=lambda x: patient_file_basenames.get(x["Patient"], 999))

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(save_folder, f"results_{therapist_mode}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save all results to CSV files
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    print(f"\nParallel run complete! Results saved to {save_folder}")
    return save_folder, csv_path


def run_comparison_experiment(args):
    """Run all five therapist modes (simple, trained_agent, complex, multi-agent, proposer_only) and compare."""  
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON EXPERIMENT: Simple vs Trained Agent vs Complex vs Proposer-Only vs Multi-Agent")
    print("=" * 80)
    
    # Get fold parameter
    fold = getattr(args, 'fold', 1)
    if fold > 1:
        print(f"Running with {fold} folds per patient")
    
    # Get therapist profile numbers from args
    # CSV structure: 0=simple (row 2), 1=trained_agent (row 3), 2=complex (row 4), 3=proposer (row 5), 4=judge (row 6)
    simple_num = args.simple_therapist if hasattr(args, 'simple_therapist') and args.simple_therapist is not None else 0
    complex_num = args.complex_therapist if hasattr(args, 'complex_therapist') and args.complex_therapist is not None else 2  # CSV index 2
    
    # Run simple mode
    print("\n[PHASE 1] Running SIMPLE prompt mode...")
    simple_folder, simple_csv = run_parallel_experiments(
        args, 
        therapist_mode="simple",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Run trained_agent mode
    print("\n[PHASE 2] Running TRAINED AGENT mode (1_Trained_Agent prompt, index 1, no patient profile)...")
    trained_agent_folder, trained_agent_csv = run_parallel_experiments(
        args,
        therapist_mode="trained_agent",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Run complex mode
    print("\n[PHASE 3] Running COMPLEX prompt mode...")
    complex_folder, complex_csv = run_parallel_experiments(
        args,
        therapist_mode="complex",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )

    # Run proposer_only mode
    print("\n[PHASE 4] Running PROPOSER-ONLY mode (self-selection ablation)...")
    proposer_only_folder, proposer_only_csv = run_parallel_experiments(
        args,
        therapist_mode="proposer_only",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Run multi-agent mode
    print("\n[PHASE 5] Running MULTI-AGENT mode...")
    multiagent_folder, multiagent_csv = run_parallel_experiments(
        args,
        therapist_mode="multi_agent",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Compare results (5-way)
    print("\n[PHASE 6] Comparing results...")
    comparison_results = compare_results_five_way(
        simple_csv, complex_csv, multiagent_csv, trained_agent_csv, proposer_only_csv, args
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    
    # Run evaluation scripts after comparison completes (unless --no-aggregate is set)
    if not getattr(args, 'no_aggregate', False):
        # Use the same max_workers setting for sentiment analysis
        max_workers = args.max_workers if hasattr(args, 'max_workers') else 20
        aggregate_metrics(skip_sentiment=args.skip_sentiment, max_workers=max_workers)
    else:
        print("\nSkipping aggregation as requested (--no-aggregate flag)")
    
    return {
        "simple_folder": simple_folder,
        "complex_folder": complex_folder,
        "multiagent_folder": multiagent_folder,
        "trained_agent_folder": trained_agent_folder,
        "proposer_only_folder": proposer_only_folder,
        "simple_csv": simple_csv,
        "complex_csv": complex_csv,
        "multiagent_csv": multiagent_csv,
        "trained_agent_csv": trained_agent_csv,
        "proposer_only_csv": proposer_only_csv,
        "comparison": comparison_results
    }


def compare_results_three_way(simple_csv_path, complex_csv_path, multiagent_csv_path, args):
    """Compare results from simple, complex, and multi-agent modes."""
    simple_df = pd.read_csv(simple_csv_path)
    complex_df = pd.read_csv(complex_csv_path)
    multiagent_df = pd.read_csv(multiagent_csv_path)
    
    # Ensure all have the same patients
    simple_df = simple_df.sort_values("Patient").reset_index(drop=True)
    complex_df = complex_df.sort_values("Patient").reset_index(drop=True)
    multiagent_df = multiagent_df.sort_values("Patient").reset_index(drop=True)
    
    # Score categories
    categories = [
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Empathy",
        "Acceptance",
        "Direction",
        "Autonomy Support",
        "Collaboration",
        "Evocation",
        "Overall Average"
    ]
    
    # Calculate statistics
    comparison_data = []
    for cat in categories:
        if cat in simple_df.columns and cat in complex_df.columns and cat in multiagent_df.columns:
            simple_scores = simple_df[cat].dropna()
            complex_scores = complex_df[cat].dropna()
            multi_scores = multiagent_df[cat].dropna()
            
            if len(simple_scores) > 0 and len(complex_scores) > 0 and len(multi_scores) > 0:
                simple_mean = simple_scores.mean()
                complex_mean = complex_scores.mean()
                multi_mean = multi_scores.mean()
                
                # Find best performing mode
                best_score = max(simple_mean, complex_mean, multi_mean)
                best_mode = None
                if best_score == simple_mean:
                    best_mode = "Simple"
                elif best_score == complex_mean:
                    best_mode = "Complex"
                else:
                    best_mode = "Multi-Agent"
                
                comparison_data.append({
                    "Category": cat,
                    "Simple_Mean": simple_mean,
                    "Simple_Std": simple_scores.std(),
                    "Complex_Mean": complex_mean,
                    "Complex_Std": complex_scores.std(),
                    "MultiAgent_Mean": multi_mean,
                    "MultiAgent_Std": multi_scores.std(),
                    "Best_Mode": best_mode,
                    "Simple_vs_Complex": complex_mean - simple_mean,
                    "Simple_vs_MultiAgent": multi_mean - simple_mean,
                    "Complex_vs_MultiAgent": multi_mean - complex_mean
                })
    
    comparison_df = pd.DataFrame(comparison_data)

    # Round all float columns to 4 dp before saving
    float_cols = comparison_df.select_dtypes(include='float').columns
    comparison_df[float_cols] = comparison_df[float_cols].round(4)

    # Save comparison CSV to Eval/csv directory
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Eval")
    csv_dir = os.path.join(eval_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Save comparison CSV
    comparison_csv = os.path.join(csv_dir, "comparison_results.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("THREE-WAY COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_csv}")
    
    # Calculate overall statistics
    if "Overall Average" in comparison_df["Category"].values:
        overall_row = comparison_df[comparison_df["Category"] == "Overall Average"].iloc[0]
        print(f"\n{'=' * 80}")
        print("OVERALL AVERAGE COMPARISON")
        print(f"{'=' * 80}")
        print(f"Simple Prompt:     {overall_row['Simple_Mean']:.3f} ± {overall_row['Simple_Std']:.3f}")
        print(f"Complex Prompt:     {overall_row['Complex_Mean']:.3f} ± {overall_row['Complex_Std']:.3f}")
        print(f"Multi-Agent:       {overall_row['MultiAgent_Mean']:.3f} ± {overall_row['MultiAgent_Std']:.3f}")
        print(f"\nBest Performing: {overall_row['Best_Mode']}")
        print(f"\nDifferences:")
        print(f"  Complex vs Simple:     {overall_row['Simple_vs_Complex']:+.3f} ({overall_row['Simple_vs_Complex']/overall_row['Simple_Mean']*100:+.2f}%)")
        print(f"  Multi-Agent vs Simple: {overall_row['Simple_vs_MultiAgent']:+.3f} ({overall_row['Simple_vs_MultiAgent']/overall_row['Simple_Mean']*100:+.2f}%)")
        print(f"  Multi-Agent vs Complex: {overall_row['Complex_vs_MultiAgent']:+.3f} ({overall_row['Complex_vs_MultiAgent']/overall_row['Complex_Mean']*100:+.2f}%)")
    
    return {
        "comparison_df": comparison_df,
        "comparison_csv": comparison_csv
    }


def compare_results_four_way(simple_csv_path, complex_csv_path, multiagent_csv_path, trained_agent_csv_path, args):
    """Compare results from simple, trained_agent, complex, and multi-agent modes."""
    simple_df = pd.read_csv(simple_csv_path)
    complex_df = pd.read_csv(complex_csv_path)
    multiagent_df = pd.read_csv(multiagent_csv_path)
    trained_agent_df = pd.read_csv(trained_agent_csv_path)

    # Ensure all have the same patients
    for df in [simple_df, complex_df, multiagent_df, trained_agent_df]:
        df.sort_values("Patient", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Score categories
    categories = [
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Empathy",
        "Acceptance",
        "Direction",
        "Autonomy Support",
        "Collaboration",
        "Evocation",
        "Overall Average"
    ]

    all_cols = set(simple_df.columns) & set(complex_df.columns) & set(multiagent_df.columns) & set(trained_agent_df.columns)

    comparison_data = []
    for cat in categories:
        if cat not in all_cols:
            continue
        simple_scores = simple_df[cat].dropna()
        complex_scores = complex_df[cat].dropna()
        multi_scores = multiagent_df[cat].dropna()
        trained_scores = trained_agent_df[cat].dropna()

        if not (len(simple_scores) > 0 and len(complex_scores) > 0 and
                len(multi_scores) > 0 and len(trained_scores) > 0):
            continue

        simple_mean = simple_scores.mean()
        complex_mean = complex_scores.mean()
        multi_mean = multi_scores.mean()
        trained_mean = trained_scores.mean()

        best_score = max(simple_mean, complex_mean, multi_mean, trained_mean)
        if best_score == simple_mean:
            best_mode = "Simple"
        elif best_score == trained_mean:
            best_mode = "Trained Agent"
        elif best_score == complex_mean:
            best_mode = "Complex"
        else:
            best_mode = "Multi-Agent"

        comparison_data.append({
            "Category": cat,
            "Simple_Mean": simple_mean,
            "Simple_Std": simple_scores.std(),
            "TrainedAgent_Mean": trained_mean,
            "TrainedAgent_Std": trained_scores.std(),
            "Complex_Mean": complex_mean,
            "Complex_Std": complex_scores.std(),
            "MultiAgent_Mean": multi_mean,
            "MultiAgent_Std": multi_scores.std(),
            "Best_Mode": best_mode,
            "Simple_vs_TrainedAgent": trained_mean - simple_mean,
            "Simple_vs_Complex": complex_mean - simple_mean,
            "Simple_vs_MultiAgent": multi_mean - simple_mean,
            "TrainedAgent_vs_Complex": complex_mean - trained_mean,
            "Complex_vs_MultiAgent": multi_mean - complex_mean,
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Round all float columns to 4 dp before saving
    float_cols = comparison_df.select_dtypes(include='float').columns
    comparison_df[float_cols] = comparison_df[float_cols].round(4)

    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Eval")
    csv_dir = os.path.join(eval_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    comparison_csv = os.path.join(csv_dir, "comparison_results.csv")
    comparison_df.to_csv(comparison_csv, index=False)

    print("\n" + "=" * 80)
    print("FOUR-WAY COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_csv}")

    if "Overall Average" in comparison_df["Category"].values:
        overall_row = comparison_df[comparison_df["Category"] == "Overall Average"].iloc[0]
        print(f"\n{'=' * 80}")
        print("OVERALL AVERAGE COMPARISON")
        print(f"{'=' * 80}")
        print(f"Simple Prompt:         {overall_row['Simple_Mean']:.3f} ± {overall_row['Simple_Std']:.3f}")
        print(f"Trained Agent:         {overall_row['TrainedAgent_Mean']:.3f} ± {overall_row['TrainedAgent_Std']:.3f}")
        print(f"Complex Prompt:        {overall_row['Complex_Mean']:.3f} ± {overall_row['Complex_Std']:.3f}")
        print(f"Multi-Agent:           {overall_row['MultiAgent_Mean']:.3f} ± {overall_row['MultiAgent_Std']:.3f}")
        print(f"\nBest Performing: {overall_row['Best_Mode']}")

    return {
        "comparison_df": comparison_df,
        "comparison_csv": comparison_csv
    }


def compare_results_five_way(simple_csv_path, complex_csv_path, multiagent_csv_path,
                             trained_agent_csv_path, proposer_only_csv_path, args):
    """Compare results from simple, trained_agent, complex, multi-agent, and proposer_only modes."""
    simple_df = pd.read_csv(simple_csv_path)
    complex_df = pd.read_csv(complex_csv_path)
    multiagent_df = pd.read_csv(multiagent_csv_path)
    trained_agent_df = pd.read_csv(trained_agent_csv_path)
    proposer_only_df = pd.read_csv(proposer_only_csv_path)

    for df in [simple_df, complex_df, multiagent_df, trained_agent_df, proposer_only_df]:
        df.sort_values("Patient", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Score categories
    categories = [
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Empathy",
        "Acceptance",
        "Direction",
        "Autonomy Support",
        "Collaboration",
        "Evocation",
        "Overall Average"
    ]

    all_cols = (set(simple_df.columns) & set(complex_df.columns) & set(multiagent_df.columns)
                & set(trained_agent_df.columns) & set(proposer_only_df.columns))

    modes_map = {
        "Simple": simple_df, "TrainedAgent": trained_agent_df,
        "Complex": complex_df, "MultiAgent": multiagent_df, "ProposerOnly": proposer_only_df,
    }

    comparison_data = []
    for cat in categories:
        if cat not in all_cols:
            continue
        means = {name: df[cat].dropna() for name, df in modes_map.items() if cat in df.columns}
        if not all(len(s) > 0 for s in means.values()):
            continue
        mean_vals = {name: s.mean() for name, s in means.items()}
        best_name = max(mean_vals, key=mean_vals.get)
        best_mode = {"Simple": "Simple", "TrainedAgent": "Trained Agent", "Complex": "Complex",
                     "MultiAgent": "Multi-Agent", "ProposerOnly": "Proposer-Only"}[best_name]

        row = {"Category": cat, "Best_Mode": best_mode}
        for name, s in means.items():
            row[f"{name}_Mean"] = s.mean()
            row[f"{name}_Std"] = s.std()
        for a in list(means.items()):
            for b in list(means.items()):
                if a[0] < b[0]:
                    row[f"{a[0]}_vs_{b[0]}"] = means[b[0]].mean() - means[a[0]].mean()
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    float_cols = comparison_df.select_dtypes(include='float').columns
    comparison_df[float_cols] = comparison_df[float_cols].round(4)

    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Eval")
    csv_dir = os.path.join(eval_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    comparison_csv = os.path.join(csv_dir, "comparison_results.csv")
    comparison_df.to_csv(comparison_csv, index=False)

    print("\n" + "=" * 80)
    print("FIVE-WAY COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_csv}")

    if "Overall Average" in comparison_df["Category"].values:
        overall_row = comparison_df[comparison_df["Category"] == "Overall Average"].iloc[0]
        print(f"\n{'=' * 80}")
        print("OVERALL AVERAGE COMPARISON")
        print(f"{'=' * 80}")
        for name in ["Simple", "TrainedAgent", "Complex", "ProposerOnly", "MultiAgent"]:
            if f"{name}_Mean" in overall_row:
                print(f"{name:<14} {overall_row[f'{name}_Mean']:.3f} ± {overall_row[f'{name}_Std']:.3f}")
        print(f"\nBest Performing: {overall_row['Best_Mode']}")

    return {
        "comparison_df": comparison_df,
        "comparison_csv": comparison_csv
    }


def regenerate_comparison_results():
    """Regenerate comparison_results.csv by combining ALL batch results from Session_Metadata.

    Fixes the issue where compare_results_four_way was previously called inside each
    batch run with only that batch's CSV (e.g., 12 patients instead of all 133), causing
    comparison_results.csv to reflect only the last batch executed.
    """
    import glob as _glob

    # Model-aware metadata root (default: root Session_Metadata, or PSYCH_META_DIR for Model_Runs).
    session_metadata_dir = os.getenv("PSYCH_META_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Session_Metadata")
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Eval")
    csv_dir = os.path.join(eval_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    print("\n[regenerate_comparison_results] Collecting all batch results from Session_Metadata...")

    combined = {}
    for mode in ["simple", "trained_agent", "complex", "multi_agent", "proposer_only"]:
        dfs = []
        for csv_path in _glob.glob(os.path.join(session_metadata_dir, "*", f"results_{mode}.csv")):
            try:
                dfs.append(pd.read_csv(csv_path))
            except Exception as e:
                print(f"  Warning: Could not read {csv_path}: {e}")
        if dfs:
            combined[mode] = pd.concat(dfs, ignore_index=True)
            print(f"  {mode}: {len(combined[mode])} total fold-rows across all batches")
        else:
            print(f"  Warning: No results CSVs found for mode '{mode}'.")
            # Only fail if one of the original 3 modes is missing
            if mode in ("simple", "complex", "multi_agent"):
                print(f"  Skipping regeneration (missing required mode '{mode}').")
                return

    # Write combined CSVs to temp files, then run the appropriate comparison
    tmp_paths = {}
    for mode, df in combined.items():
        tmp_path = os.path.join(csv_dir, f"_tmp_all_results_{mode}.csv")
        df.to_csv(tmp_path, index=False)
        tmp_paths[mode] = tmp_path

    class _Args:
        pass

    # Proposer_only available -> 5-way comparison; else 4-way; else 3-way.
    if "proposer_only" in tmp_paths and "trained_agent" in tmp_paths:
        compare_results_five_way(
            tmp_paths["simple"], tmp_paths["complex"], tmp_paths["multi_agent"],
            tmp_paths["trained_agent"], tmp_paths["proposer_only"], _Args()
        )
    elif "trained_agent" in tmp_paths:
        compare_results_four_way(
            tmp_paths["simple"], tmp_paths["complex"],
            tmp_paths["multi_agent"], tmp_paths["trained_agent"], _Args()
        )
    else:
        compare_results_three_way(tmp_paths["simple"], tmp_paths["complex"], tmp_paths["multi_agent"], _Args())

    for tmp_path in tmp_paths.values():
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    print("[regenerate_comparison_results] Done.")


def aggregate_metrics(skip_sentiment=False, max_workers=30):
    """Run all Python scripts in Eval folder in the correct order.
    
    Args:
        skip_sentiment: If True, skip sentiment_analysis.py
        max_workers: Number of parallel workers for sentiment analysis (default: 20)
    """
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Eval")

    # Regenerate comparison_results.csv from ALL batch results before running eval scripts.
    # This ensures comparison_results.csv covers every patient, not just the last batch.
    regenerate_comparison_results()

    # Define the order of scripts to run (starting with sentiment_analysis, ending with plot_metrics)
    # Only include scripts that can be executed (have if __name__ == "__main__" block)
    eval_scripts = [
        "sentiment_analysis.py",
        "word_analysis.py",  # Calculates word metrics (counselor/client words, ratios)
        "lsm_analysis.py",
        "feedback_analysis.py",
        "feedback_statistical_test.py",
        "plot_metrics.py"
    ]
    
    # Skip sentiment analysis if requested
    if skip_sentiment:
        eval_scripts = [s for s in eval_scripts if s != "sentiment_analysis.py"]
        print("Skipping sentiment analysis as requested")
    
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION SCRIPTS")
    print("=" * 80)
    
    python_exe = sys.executable  # Use the same Python interpreter
    
    for script_name in eval_scripts:
        script_path = os.path.join(eval_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"Warning: Script {script_name} not found, skipping...")
            continue
        
        print(f"\n[RUNNING] {script_name}...")
        print("-" * 80)
        
        try:
            # Build command with additional arguments for sentiment_analysis.py
            cmd = [python_exe, script_path]
            if script_name == "sentiment_analysis.py":
                cmd.extend(["--max-workers", str(max_workers)])
                print(f"  Using {max_workers} parallel workers for sentiment analysis")
            
            # Run the script using subprocess
            result = subprocess.run(
                cmd,
                cwd=eval_dir,
                capture_output=False,  # Show output in real-time
                text=True,
                check=True
            )
            print(f"{script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: {e}")
            # Continue with other scripts even if one fails
            continue
        except Exception as e:
            print(f"Unexpected error running {script_name}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("ALL EVALUATION SCRIPTS COMPLETED")
    print("=" * 80)


# =============================
# === Main Entry Point ====
# =============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Testing Pipeline for MI")

    parser.add_argument("--start-patient", type=int,
                        help="Start patient index (for parallel execution)")
    parser.add_argument("--end-patient", type=int,
                        help="End patient index (for parallel execution)")
    parser.add_argument("--run-id", type=str, default="exp1",
                        help="Run identifier for grouping parallel runs")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results from parallel runs")
    parser.add_argument("--mode", type=str, choices=["simple", "trained_agent", "complex", "multi_agent", "proposer_only"],
                        help="Therapist mode: simple, trained_agent (complex MI no profile), complex, multi_agent, or proposer_only (ablation control)")
    parser.add_argument("--simple-therapist", type=int, default=None,
                        help="Therapist profile index for simple mode")
    parser.add_argument("--complex-therapist", type=int, default=None,
                        help="Therapist profile index for complex mode")
    parser.add_argument("--compare", action="store_true",
                        help="Run all five modes (simple, trained_agent, complex, proposer_only, multi-agent) and compare results")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Number of parallel workers for processing patients (default: 20)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment analysis when running evaluation scripts")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip aggregation/evaluation scripts (only run therapy sessions)")
    parser.add_argument("--fold", type=int, default=5,
                        help="Number of times to run each patient (e.g., fold=5 with 10 patients = 50 total runs)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model identifier for Model_Runs/<model>/ layout and therapist model "
                             "(e.g. gpt-5-mini, qwen/qwen3.8-27b, deepseek/deepseek-v4-flash). "
                             "Omit for legacy root layout + default gpt-5-mini.")

    args = parser.parse_args()

    # Configure model-aware output directories (Model_Runs/<model>/Saved_Sessions + Session_Metadata).
    # The env vars are picked up by FileManager (PSYCH_SAVE_DIR) and the aggregation scripts.

    # PSYCH_MODEL is used as a FILE PREFIX/tag, so we must sanitize it (no '/', which would
    # otherwise create nested paths). PSYCH_THERAPIST_MODEL keeps the raw OpenRouter id for routing.
    def model_tag(model_id):
        return model_id.replace("/", "-") if model_id else ""

    model_root = model_runs_dir(args.model)
    if model_root:
        os.environ["PSYCH_SAVE_DIR"] = os.path.join(model_root, "Saved_Sessions")
        os.environ["PSYCH_META_DIR"] = os.path.join(model_root, "Session_Metadata")
        os.environ["PSYCH_MODEL"] = model_tag(args.model)
        # Point the therapist/proposer/judge at the configured model (OpenRouter for ids with '/').
        if args.model:
            os.environ["PSYCH_THERAPIST_MODEL"] = args.model
        print(f"[model] Using Model_Runs layout: {model_root} (model={args.model}, tag={model_tag(args.model)})")
    else:
        os.environ["PSYCH_SAVE_DIR"] = os.environ.get("PSYCH_SAVE_DIR", "Saved_Sessions")
        os.environ["PSYCH_META_DIR"] = os.environ.get("PSYCH_META_DIR", "Session_Metadata")
        os.environ["PSYCH_MODEL"] = os.environ.get("PSYCH_MODEL", "")

    if args.aggregate:
        # Run only the evaluation/aggregation scripts
        print("\n" + "=" * 80)
        print("RUNNING EVALUATION SCRIPTS ONLY")
        print("=" * 80)
        max_workers = args.max_workers if hasattr(args, 'max_workers') else 20
        aggregate_metrics(skip_sentiment=args.skip_sentiment, max_workers=max_workers)
    elif args.compare:
        # Run 4-way comparison experiment
        run_comparison_experiment(args)
    else:
        # Run single mode
        therapist_mode = args.mode if args.mode else "simple"
        run_parallel_experiments(
            args, 
            therapist_mode=therapist_mode,
            simple_therapist_num=args.simple_therapist,
            complex_therapist_num=args.complex_therapist
        )