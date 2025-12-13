import csv
import os
import random
import re
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import LearningBuddy from modular version
from LearningBuddy_Modular import LearningBuddy
import VirtualTherapist
from utils.api_key_loader import load_openai_api_key

# =============================
# === Experiment Constants ====
# =============================
THERAPIST_CSV_PATH = "Therapist_Prompts/therapist_prompts.csv"
PATIENT_DIR = "Patient_Profiles/MI/"
NUM_PATIENTS = 1

# THERAPIST_NUM maps to mode: 0=simple, 1=complex, 2=multi-agent
# This maps to CSV indices: 0=row 2 (Simple), 1=row 3 (Complex), 2=row 4 (Proposer), 3=row 5 (Judge)
THERAPIST_NUM = 0  # Default to simple mode

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

    # Initialize LearningBuddy with appropriate mode
    lb = LearningBuddy(api_key_input, use_multi_agent=use_multi_agent)
    
    # For multi-agent mode, set proposer and judge to use CSV rows 4 and 5 (indices 2 and 3)
    # CSV structure: Row 1=Header, Row 2=0_Simple_Agent (index 0), Row 3=1_Complex_Agent (index 1),
    #                Row 4=2_MultiAgent_Proposer (index 2), Row 5=3_MultiAgent_Judge (index 3)
    if use_multi_agent:
        proposer_index = 2  # Row 4 in CSV: 2_MultiAgent_Proposer
        judge_index = 3     # Row 5 in CSV: 3_MultiAgent_Judge
        lb.set_multi_agent_therapist_profiles(proposer_index, judge_index)
        logging.info(f"Multi-agent mode: Proposer index {proposer_index}, Judge index {judge_index}")
    
    # Initialize VirtualTherapist only for single-agent modes (simple/complex)
    therapist = None
    if therapist_mode in ["simple", "complex"]:
        # Single-agent modes: use VirtualTherapist directly
        therapist = VirtualTherapist.VirtualTherapist(
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
        current_state = lb.get_session_state()
        interview_messages = current_state.get("interview_messages", [])
        
        if therapist_mode == "multi_agent":
            # Multi-agent mode: LearningBuddy's proposer/judge system generates therapist response
            # The proposer generates 3 candidate responses based on conversation history,
            # judge selects best one, then patient responds
            # We use the message_workflow directly to generate therapist response
            
            # Access the message workflow's proposer and judge methods
            message_workflow = lb.message_workflow
            
            # Create state for proposer/judge
            temp_state = {"interview_messages": interview_messages}
            
            # Generate 3 candidate therapist responses using proposer
            proposer_result = message_workflow._propose_therapist_responses(temp_state)
            candidates = proposer_result.get("therapist_candidates", [])
            
            if candidates:
                # Use judge to select best response
                judge_state = {**temp_state, "therapist_candidates": candidates}
                judge_result = message_workflow._judge_therapist_responses(judge_state)
                selected_response = judge_result.get("selected_therapist_response", candidates[0])
            else:
                # Fallback if proposer fails
                selected_response = "How are you feeling today?" if exchange_num == 0 else "Can you tell me more about that?"
            
            # Send the selected therapist response to trigger patient response
            session_result = lb.send_message(selected_response)
        else:
            # Single-agent modes (simple/complex): VirtualTherapist generates response, send directly
            therapist_utterance = therapist.get_response(interview_messages)
            session_result = lb.send_message(therapist_utterance)

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
    """Extract numerical scores from feedback text files."""

    # Define categories
    categories = [
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Empathy",
        "Acceptance",
        "Direction",
        "Autonomy Support",
        "Collaboration",
        "Evocation"
    ]        

    scores = {cat: [] for cat in categories}

    for file_path in eval_file_paths:
        text = Path(file_path).read_text(encoding="utf-8")

        for cat in categories:
            pattern = re.compile(
                rf"{re.escape(cat)}.*?Score:\s*\**\s*(\d+(\.\d+)?)\**",
                re.IGNORECASE | re.DOTALL
            )
            matches = pattern.findall(text)
            if matches:
                score_value = float(matches[-1][0])
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

    print(f"Processing patients {args.start_patient} to {args.end_patient} ({num_patients} patients)")
    print(f"Mode: {therapist_mode}")
    print(f"Therapist Profile Index: {therapist_num}")

    # Create save folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(
        "saved_tests",
        f"{args.run_id}_{therapist_mode}_patients_{args.start_patient}_{args.end_patient}_{timestamp}"
    )
    os.makedirs(save_folder, exist_ok=True)

    # Save metadata
    metadata = {
        "run_id": args.run_id,
        "start_patient": args.start_patient,
        "end_patient": args.end_patient,
        "num_patients": num_patients,
        "timestamp": timestamp,
        "therapist_mode": therapist_mode,
        "therapist_profile_number": therapist_num
    }
    with open(os.path.join(save_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Initialize storage for all results
    all_results = []

    df = pd.read_csv(THERAPIST_CSV_PATH, quotechar='"', skipinitialspace=True)

    # Process each patient
    for patient_idx, patient_file in enumerate(patient_files):
        print(f"\n{'=' * 60}")
        print(f"Processing Patient {patient_idx + 1}/{num_patients}: {os.path.basename(patient_file)}")
        print(f"Mode: {therapist_mode}")
        print(f"{'=' * 60}")

        patient_number = patient_idx + 1

        print(f"  [STEP 1] Running test ({therapist_mode} mode)...")
        results = run_test(
            patient_file,
            therapist_profile_number=therapist_num,
            patient_number=patient_number,
            therapist_mode=therapist_mode
        )
        # Look for the feedback file - it might be named differently
        folder_name = results["folder_name"]

        # Try to find the feedback file (check different possible names)
        possible_names = [
            "detailed_feedback.txt",
            "feedback.txt"
        ]

        feedback_path = None
        for name in possible_names:
            potential_path = os.path.join(folder_name, name)
            if os.path.exists(potential_path):
                feedback_path = potential_path
                break

        if feedback_path is None:
            # List all files in the folder to see what's actually there
            print(f"  Warning: Could not find feedback file in {folder_name}")
            print(
                f"  Files in folder: {os.listdir(folder_name) if os.path.exists(folder_name) else 'folder does not exist'}")
            # Use a default pattern - adjust based on what you see
            feedback_path = os.path.join(folder_name, "detailed_feedback.txt")
        
        scores = extract_scores([feedback_path])

        # Store therapist result
        row = {
            "Therapist": df.iloc[therapist_num]["Name"],
            "Patient": os.path.basename(patient_file),
            "Mode": therapist_mode
        }
        for key, values in scores.items():
            row[key] = values[0] if values else None
        all_results.append(row)

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
    
    # Create comparison folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_folder = os.path.join(
        "saved_tests",
        f"comparison_{args.run_id}_{timestamp}"
    )
    os.makedirs(comparison_folder, exist_ok=True)
    
    # Save comparison
    comparison_csv = os.path.join(comparison_folder, "comparison_results.csv")
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
        "comparison_csv": comparison_csv,
        "comparison_folder": comparison_folder
    }

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
                        help="Therapist profile index for simple mode (default: THERAPIST_NUM)")
    parser.add_argument("--complex-therapist", type=int, default=None,
                        help="Therapist profile index for complex mode (default: THERAPIST_NUM)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all three modes (simple, complex, multi-agent) and compare results")

    args = parser.parse_args()

    if args.compare:
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