import csv
import os
import random
import re
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LearningBuddy_Agent import LearningBuddy
import VirtualStudent
from utils.api_key_loader import load_openai_api_key

# =============================
# === Experiment Constants ====
# =============================
STUDENT_CSV_PATH = "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv"
PATIENT_DIR = "Patient_Profiles/MI/"
NUM_PATIENTS = 1

STUDENT_NUM = 0

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


def run_test(patient_profile_path, student_profile_number,
             patient_number=None, student_feedback_csv="", api_key_input=None):
    """Run one simulated therapy session between student and patient.

    Args:
        therapy_mode: 'MI' or 'CBT'
        patient_profile_path: Path to patient profile file
        student_profile_number: Index of student profile to use
        patient_number: Optional patient identifier for folder naming
        student_type: Optional student type ('good', 'bad_1st', 'bad_feedback') for folder naming
        api_key_input: Optional API key
    """
    api_key_input = API_KEY or "key"

    # Prepare session metadata for folder naming
    session_metadata = {}
    if patient_number is not None:
        session_metadata['patient_number'] = patient_number
    session_metadata['student_type'] = 'therapist'

    add_on = ""
    if not student_feedback_csv == "":
        df = pd.read_csv(student_feedback_csv, quotechar='"', skipinitialspace=True)
        add_on = df.loc[student_profile_number, "Prompt"]

    # Initialize LearningBuddy with appropriate mode
    lb = LearningBuddy(api_key_input)
    student = VirtualStudent.VirtualStudent(
        api_key_input,
        VirtualStudent.VirtualStudent.MI,
        student_index=student_profile_number,
        prompt_addon=add_on
    )

    # Load patient profile and start session with metadata
    case = lb.load_patient_profile(patient_profile_path)

    # Pass metadata to start_session
    session_result = lb.start_session(case_study=case, session_metadata=session_metadata)

    # Run 10 exchanges between student and patient
    for _ in range(10):
        current_state = lb.get_session_state()
        interview_messages = current_state.get("interview_messages", [])
        student_utterance = student.get_response(interview_messages)
        session_result = lb.send_message(student_utterance)

    # End session and get feedback
    session_result = lb.end_session()
    final_state = lb.get_session_state()

    # Get folder name - try both possible keys
    folder_name = final_state.get("export_folder") or final_state.get("folder_name", "")

    return {
        "session_result": session_result,
        "folder_name": folder_name
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

def run_experiment(student_num, patient_files, student_csv_path, label, save_folder):
    """Run experiment for one student configuration."""
    df = pd.read_csv(student_csv_path, quotechar='"', skipinitialspace=True)
    all_results = []
    saved_data_directory = []

    # Run tests for each patient
    for patient_idx, patient in enumerate(patient_files):
        # Extract patient number from filename or use index
        patient_basename = os.path.basename(patient)
        patient_number = patient_idx + 1  # Or extract from filename if it contains a number

        results = run_test(
            patient,
            student_num,
            patient_number=patient_number,
            student_type=label
        )
        feedback_path = results["folder_name"] + f"/detailed_feedback_{label}.txt"
        saved_data_directory.append(feedback_path)


    return saved_data_directory


def run_parallel_experiments(args):
    STUDENT_CSV_PATH = "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv"
    """Main function for parallel execution mode."""
    start_time = time.time()

    # Get patient files for this range
    patient_files = get_patient_files(args.start_patient, args.end_patient)
    num_patients = len(patient_files)

    print(f"Processing patients {args.start_patient} to {args.end_patient} ({num_patients} patients)")

    # Create save folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(
        "saved_tests",
        f"{args.run_id}_patients_{args.start_patient}_{args.end_patient}_{timestamp}"
    )
    os.makedirs(save_folder, exist_ok=True)

    # Save metadata
    metadata = {
        "run_id": args.run_id,
        "start_patient": args.start_patient,
        "end_patient": args.end_patient,
        "num_patients": num_patients,
        "timestamp": timestamp
    }
    with open(os.path.join(save_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Initialize storage for all results
    all_results = []

    df = pd.read_csv(STUDENT_CSV_PATH, quotechar='"', skipinitialspace=True)

    # Process each patient with all three student types
    for patient_idx, patient_file in enumerate(patient_files):
        print(f"\n{'=' * 60}")
        print(f"Processing Patient {patient_idx + 1}/{num_patients}: {os.path.basename(patient_file)}")
        print(f"{'=' * 60}")

        patient_number = patient_idx + 1

        # 1️⃣ student for this patient
        print(f"  [STEP 1] Running Student...")
        results = run_test(
            patient_file,
            student_profile_number=STUDENT_NUM,
            patient_number=patient_number,
        )
        # Look for the feedback file - it might be named differently
        folder_name = results["folder_name"]

        # Try to find the feedback file (check different possible names)
        possible_names = [
            "detailed_feedback_good_student.txt",
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

        # Store student result
        row = {
            "Student": df.iloc[STUDENT_NUM]["Name"],
            "Patient": os.path.basename(patient_file)
        }
        for key, values in scores.items():
            row[key] = values[0] if values else None
        all_results.append(row)



    # Save all results to CSV files
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    print(f"\n✅ Parallel run complete! Results saved to {save_folder}")
    return save_folder

# =============================
# === Main Entry Point ====
# =============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Testing Pipeline for MI/CBT")

    parser.add_argument("--start-patient", type=int,
                        help="Start patient index (for parallel execution)")
    parser.add_argument("--end-patient", type=int,
                        help="End patient index (for parallel execution)")
    parser.add_argument("--run-id", type=str, default="exp1",
                        help="Run identifier for grouping parallel runs")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results from parallel runs")

    args = parser.parse_args()

    run_parallel_experiments(args)