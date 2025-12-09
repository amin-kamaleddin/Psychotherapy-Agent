import json
import os
import random
import re
import time
from datetime import datetime
import pandas as pd

from LearningBuddy_Agent import LearningBuddy, MI_MODE
import VirtualStudent
from plot import *
from utils.api_key_loader import load_openai_api_key

API_KEY = load_openai_api_key()


# =============================
# === Core Helper Functions ===
# =============================

def run_test(therapy_mode, patient_profile_path, student_profile_number):
    """
    Run a single test with the selected student and patient profile.

    Args:
        therapy_mode: MI_MODE
        patient_profile_path: Path to patient profile file
        student_profile_number: Index of student profile to use

    Returns:
        Dictionary with session results and folder name
    """
    # Initialize LearningBuddy with appropriate mode
    if therapy_mode == MI_MODE:
        lb = LearningBuddy(API_KEY, "null", MI_MODE)
        student = VirtualStudent.VirtualStudent(
            api_key=API_KEY,
            therapy_type="MI",
            student_index=student_profile_number
        )
    else:
        raise ValueError(f"Invalid therapy mode: {therapy_mode}")

    # Load patient profile and start session
    case = lb.load_patient_profile(patient_profile_path)
    session_result = lb.start_session(case_study=case)

    # Loop for student agent and patient agent to talk to each other (limit of 2 in this version)
    for i in range(0, 2):
        # Get current state for student
        current_state = lb.get_session_state()
        interview_messages = current_state.get("interview_messages", [])

        # Student generates response
        student_utterance = student.get_response(interview_messages)

        # Send student's message to virtual patient
        session_result = lb.send_message(student_utterance)

    # Generate feedback
    session_result = lb.end_session()

    # Get export folder from final state
    final_state = lb.get_session_state()
    export_folder = final_state.get("export_folder", "")

    return {
        "session_result": session_result,
        "folder_name": export_folder
    }


def count_files(directory):
    """Count number of files in a directory."""
    return sum(
        1 for entry in os.scandir(directory)
        if entry.is_file()
    )


def extract_scores_from_text(text):
    """
    Extract MI scores from detailed feedback text.

    Args:
        text: Feedback text content

    Returns:
        Dictionary of category scores
    """
    categories = [
        "Acceptance",
        "Empathy",
        "Spirit of MI",
        "Cultivating Change Talk",
        "Softening Sustain Talk",
        "Partnership",
        "Open Questions",
        "Reflections",
        "Affirmations",
        "Persuade with Permission",
        "Emphasize Autonomy",
        "Engaging",
        "Focusing",
        "Evoking",
        "Planning",
        "Summarizing"
    ]

    # Normalize line endings and spacing
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    results = {}

    for cat in categories:
        # Markdown heading pattern, e.g.:
        # ### 1. Acceptance
        # ### Acceptance
        # ### **Acceptance**
        heading_pattern = re.compile(
            rf"^#+\s*(?:\d+\.\s*)?\*?\*?{re.escape(cat)}\*?\*?.*$",
            re.IGNORECASE | re.MULTILINE
        )
        match = heading_pattern.search(text)

        if not match:
            # Some headings might include parentheses or extra symbols
            print(f"Could not extract {cat}")
            results[cat] = 1.0
            continue

        # Extract text starting after this heading until the next ### or end of file
        section_start = match.end()
        next_heading = re.search(r"^###\s", text[section_start:], re.MULTILINE)
        section_end = section_start + next_heading.start() if next_heading else len(text)
        section_text = text[section_start:section_end]

        # Look for Score line – matches patterns like:
        # **Score: 2 (blah)**
        # **Score:** **2**
        # **Score: 2/5**
        score_match = re.search(
            r"\*\*Score[:\s]*\**\s*(\d+(?:\.\d+)?)(?:\s*/\s*\d+)?",
            section_text,
            re.IGNORECASE
        )

        if score_match:
            results[cat] = float(score_match.group(1))
        else:
            # Fall back to first standalone number if no explicit Score line
            num_match = re.search(r"(\d+(?:\.\d+)?)", section_text)
            results[cat] = float(num_match.group(1)) if num_match else 1.0

    # Add Overall Average
    numeric_scores = [v for k, v in results.items() if k in categories]
    results["Overall Average"] = (
        round(sum(numeric_scores) / len(numeric_scores), 2)
        if numeric_scores else 1.0
    )

    return results


def extract_all_scores_from_folder(parent_folder, filename="detailed_feedback.txt"):
    """
    Extract scores from all feedback files in a folder structure.

    Args:
        parent_folder: Parent directory containing session subdirectories
        filename: Name of feedback file to extract from

    Returns:
        DataFrame with all extracted scores
    """
    all_results = []
    for subdir in os.listdir(parent_folder):
        subpath = os.path.join(parent_folder, subdir)
        if os.path.isdir(subpath):
            file_path = os.path.join(subpath, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                scores = extract_scores_from_text(text)
                scores["session_folder"] = subdir
                all_results.append(scores)
            else:
                print(f"⚠️ File not found in: {subpath}")
    return pd.DataFrame(all_results)


# =============================
# === Main Testing Pipeline ===
# =============================

def run_all_tests(num_patients, student_profiles):
    """
    Run all tests in a loop for multiple students and patients.

    Args:
        num_patients: Number of patient profiles to test with
        student_profiles: List of student profile indices to test
    """
    start = time.time()

    # Select Patient Profiles
    directory = "Patient_Profiles/MI/"
    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    random_files = random.sample(all_files, num_patients)

    mode = MI_MODE
    all_results = []
    df = pd.read_csv(
        'Student_Profiles/AK_StudentAgent_10Profiles - Prompts for CBT Students.csv',
        quotechar='"',
        skipinitialspace=True
    )

    # Create save folder
    base_folder = "saved_tests"
    os.makedirs(base_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(base_folder, f"run_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_path = os.path.join(save_folder, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        completed_pairs = set(tuple(x) for x in checkpoint["completed_pairs"])
        all_file_paths = checkpoint["all_file_paths"]
        print(f"📂 Loaded checkpoint with {len(completed_pairs)} completed pairs")
    else:
        completed_pairs = set()
        all_file_paths = [[] for _ in student_profiles]
        print("🆕 Starting fresh run (no checkpoint found)")

    num_tests = len(student_profiles) * len(random_files)
    count = 1

    student_number = 0

    # Run all tests
    for student in student_profiles:
        print(f"\n{'=' * 60}")
        print(f"Testing Student Profile #{student}")
        print(f"{'=' * 60}")

        for patient in random_files:
            pair_key = (student, patient)

            # Skip if already completed
            if pair_key in completed_pairs:
                print(f"⏭️  Skipping completed pair: Student {student}, Patient {os.path.basename(patient)}")
                count += 1
                continue

            print(f"\n[{count}/{num_tests}] Running test...")
            print(f"  Student: {student}")
            print(f"  Patient: {os.path.basename(patient)}")

            try:
                # Run the test
                results = run_test(mode, patient, student)
                feedback_path = results["folder_name"] + "/detailed_feedback.txt"
                all_file_paths[student_number].append(feedback_path)

                # Mark as completed
                completed_pairs.add(pair_key)

                # Save checkpoint after each successful test
                checkpoint_data = {
                    "completed_pairs": list(completed_pairs),
                    "all_file_paths": all_file_paths
                }
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)

                print(f"  ✅ Test completed successfully")

            except Exception as e:
                print(f"  ❌ Error during test: {e}")
                import traceback
                traceback.print_exc()
                continue

            count += 1

        student_number += 1

    # Once all tests complete, extract scores
    print(f"\n{'=' * 60}")
    print("Extracting scores from feedback files...")
    print(f"{'=' * 60}")

    parent_folder = "Saved_Sessions"
    df = extract_all_scores_from_folder(parent_folder)

    # Add "Student" column
    half = len(df) // 2
    df['Student'] = ['good student'] * half + ['bad student'] * (len(df) - half)

    # Save results to CSV
    output_csv = os.path.join(save_folder, "extracted_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ Scores saved to: {output_csv}")

    # Save elapsed time
    end = time.time()
    elapsed_minutes = (end - start) / 60
    elapsed_time_path = os.path.join(save_folder, "elapsed_time.txt")
    with open(elapsed_time_path, "w") as f:
        f.write(f"Elapsed time: {elapsed_minutes:.2f} minutes\n")
    print(f"✅ Elapsed time: {elapsed_minutes:.2f} minutes")

    print(f"\n{'=' * 60}")
    print(f"Run complete! All results saved in: {save_folder}")
    print(f"{'=' * 60}")


# =============================
# === Main Execution =========
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("Automated Testing Pipeline")
    print("=" * 60)

    # Configuration
    # num_patients = count_files("Patient_Profiles/MI/")
    num_patients = 1
    student_profiles = [11]

    print(f"\nConfiguration:")
    print(f"  Number of patients: {num_patients}")
    print(f"  Student profiles: {student_profiles}")
    print(f"  Mode: MI")
    print()

    # Run tests
    run_all_tests(num_patients, student_profiles)