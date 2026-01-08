import csv
import os
import random
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

import matplotlib.pyplot as plt
import numpy as np
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
        therapist_mode: One of "simple", "complex", or "multi_agent"
                        - "simple": Single-agent with simple prompt (uses therapist_profile_number)
                        - "complex": Single-agent with complex MI prompt (uses therapist_profile_number)
                        - "multi_agent": Multi-agent system (proposer+judge)
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
    use_multi_agent = (therapist_mode == "multi_agent")

    # Initialize appropriate session manager based on mode
    if use_multi_agent:
        # Multi-agent mode: use PsychotherapyMultiAgent
        lb = PsychotherapyMultiAgent(api_key_input)
        
        # Set proposer and judge to use CSV rows 4 and 5 (indices 2 and 3)
        # CSV structure: Row 1=Header, Row 2=0_Simple_Agent (index 0), Row 3=1_Complex_Agent (index 1),
        #                Row 4=2_MultiAgent_Proposer (index 2), Row 5=3_MultiAgent_Judge (index 3)
        proposer_index = 2  # Row 4 in CSV: 2_MultiAgent_Proposer
        judge_index = 3     # Row 5 in CSV: 3_MultiAgent_Judge
        lb.set_multi_agent_therapist_profiles(proposer_index, judge_index)
        logging.info(f"Multi-agent mode: Proposer index {proposer_index}, Judge index {judge_index}")
    else:
        # Single-agent modes (simple/complex): use PsychotherapySingleAgent
        lb = PsychotherapySingleAgent(
            api_key_input,
            therapist_index=therapist_profile_number,
            prompt_addon=add_on
        )

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

def run_experiment(therapist_num, patient_files, therapist_csv_path, label, save_folder):
    """Run experiment for one therapist configuration."""
    df = pd.read_csv(therapist_csv_path, quotechar='"', skipinitialspace=True)
    all_results = []
    saved_data_directory = []

    # Run tests for each patient
    for patient_idx, patient in enumerate(patient_files):
        # Extract patient number from filename or use index
        patient_basename = os.path.basename(patient)
        patient_number = patient_idx + 1  # Or extract from filename if it contains a number

        results = run_test(
            patient,
            therapist_num,
            patient_number=patient_number,
            therapist_type=label
        )
        feedback_path = results["folder_name"] + f"/detailed_feedback_{label}.txt"
        saved_data_directory.append(feedback_path)


    return saved_data_directory


def run_parallel_experiments(args, therapist_mode="simple", simple_therapist_num=None, complex_therapist_num=None):
    THERAPIST_CSV_PATH = "Therapist_Prompts/therapist_prompts.csv"
    """Main function for parallel execution mode.
    
    Args:
        args: Command line arguments
        therapist_mode: One of "simple", "complex", or "multi_agent"
        simple_therapist_num: Therapist profile index for simple mode (default: 0, CSV index 0/row 2)
        complex_therapist_num: Therapist profile index for complex mode (default: 1, CSV index 1/row 3)
    """
    start_time = time.time()
    
    # Determine which therapist profile number to use
    # THERAPIST_NUM maps to mode: 0=simple (CSV index 0, row 2), 1=complex (CSV index 1, row 3), 2=multi-agent (uses indices 2&3, rows 4&5)
    if therapist_mode == "simple":
        therapist_num = simple_therapist_num if simple_therapist_num is not None else 0  # Mode 0 = CSV index 0 (row 2)
    elif therapist_mode == "complex":
        therapist_num = complex_therapist_num if complex_therapist_num is not None else 1  # Mode 1 = CSV index 1 (row 3)
    else:  # multi_agent
        therapist_num = 2  # Mode 2 = multi-agent (proposer uses CSV index 2/row 4, judge uses CSV index 3/row 5)

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
    save_folder = os.path.join(
        "Session_Metadata",
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
                print(f"  ✅ Completed Patient {patient_number}: {os.path.basename(patient_file)}")
            
            return row
        
        except Exception as e:
            with print_lock:
                print(f"  ❌ Error processing Patient {patient_number} ({os.path.basename(patient_file)}): {str(e)}")
            return None

    # Determine number of parallel workers
    max_workers = args.max_workers if hasattr(args, 'max_workers') else 3
    
    with print_lock:
        print(f"\n🚀 Starting parallel processing with {max_workers} workers...")
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
        completed = 0
        for future in as_completed(future_to_patient):
            completed += 1
            result = future.result()
            if result is not None:
                all_results.append(result)
            
            with print_lock:
                print(f"📊 Progress: {completed}/{total_runs} runs completed")
    
    # Sort results by patient filename to maintain order
    patient_file_basenames = {os.path.basename(f): idx for idx, f in enumerate(patient_files)}
    all_results.sort(key=lambda x: patient_file_basenames.get(x["Patient"], 999))

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(save_folder, f"results_{therapist_mode}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    # Save all results to CSV files
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    print(f"\n✅ Parallel run complete! Results saved to {save_folder}")
    return save_folder, csv_path


def run_comparison_experiment(args):
    """Run all three therapist modes (simple, complex, multi-agent) and compare results."""
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON EXPERIMENT: Simple vs Complex vs Multi-Agent")
    print("=" * 80)
    
    # Get fold parameter
    fold = getattr(args, 'fold', 1)
    if fold > 1:
        print(f"Running with {fold} folds per patient")
    
    # Get therapist profile numbers from args
    # THERAPIST_NUM maps to mode: 0=simple (CSV index 0, row 2), 1=complex (CSV index 1, row 3), 2=multi-agent
    simple_num = args.simple_therapist if hasattr(args, 'simple_therapist') and args.simple_therapist is not None else 0  # Mode 0
    complex_num = args.complex_therapist if hasattr(args, 'complex_therapist') and args.complex_therapist is not None else 1  # Mode 1
    
    # Run simple mode
    print("\n[PHASE 1] Running SIMPLE prompt mode...")
    simple_folder, simple_csv = run_parallel_experiments(
        args, 
        therapist_mode="simple",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Run complex mode
    print("\n[PHASE 2] Running COMPLEX prompt mode...")
    complex_folder, complex_csv = run_parallel_experiments(
        args,
        therapist_mode="complex",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Run multi-agent mode
    print("\n[PHASE 3] Running MULTI-AGENT mode...")
    multiagent_folder, multiagent_csv = run_parallel_experiments(
        args,
        therapist_mode="multi_agent",
        simple_therapist_num=simple_num,
        complex_therapist_num=complex_num
    )
    
    # Compare results
    print("\n[PHASE 4] Comparing results...")
    comparison_results = compare_results_three_way(simple_csv, complex_csv, multiagent_csv, args)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    
    # Run evaluation scripts after comparison completes
    # Use the same max_workers setting for sentiment analysis
    max_workers = args.max_workers if hasattr(args, 'max_workers') else 20
    aggregate_metrics(skip_sentiment=args.skip_sentiment, max_workers=max_workers)
    
    return {
        "simple_folder": simple_folder,
        "complex_folder": complex_folder,
        "multiagent_folder": multiagent_folder,
        "simple_csv": simple_csv,
        "complex_csv": complex_csv,
        "multiagent_csv": multiagent_csv,
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


def aggregate_metrics(skip_sentiment=False, max_workers=30):
    """Run all Python scripts in Eval folder in the correct order.
    
    Args:
        skip_sentiment: If True, skip sentiment_analysis.py
        max_workers: Number of parallel workers for sentiment analysis (default: 20)
    """
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Eval")
    
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
        print("⚠️  Skipping sentiment analysis as requested")
    
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
            print(f"✅ {script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_name}: {e}")
            # Continue with other scripts even if one fails
            continue
        except Exception as e:
            print(f"❌ Unexpected error running {script_name}: {e}")
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
    parser.add_argument("--mode", type=str, choices=["simple", "complex", "multi_agent"],
                        help="Therapist mode: simple (simple prompt), complex (complex MI prompt), or multi_agent")
    parser.add_argument("--simple-therapist", type=int, default=None,
                        help="Therapist profile index for simple mode")
    parser.add_argument("--complex-therapist", type=int, default=None,
                        help="Therapist profile index for complex mode")
    parser.add_argument("--compare", action="store_true",
                        help="Run all three modes (simple, complex, multi-agent) and compare results")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Number of parallel workers for processing patients (default: 20)")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment analysis when running evaluation scripts")
    parser.add_argument("--fold", type=int, default=1,
                        help="Number of times to run each patient (e.g., fold=5 with 10 patients = 50 total runs)")

    args = parser.parse_args()

    if args.aggregate:
        # Run only the evaluation/aggregation scripts
        print("\n" + "=" * 80)
        print("RUNNING EVALUATION SCRIPTS ONLY")
        print("=" * 80)
        max_workers = args.max_workers if hasattr(args, 'max_workers') else 20
        aggregate_metrics(skip_sentiment=args.skip_sentiment, max_workers=max_workers)
    elif args.compare:
        # Run 3-way comparison experiment
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