import csv
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LearningBuddy_Agent import LearningBuddy, MI_MODE
import VirtualStudent
from utils.api_key_loader import load_openai_api_key

# =============================
# === Experiment Constants ====
# =============================

STUDENT_CSV_PATH = "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv"
PATIENT_DIR = "Patient_Profiles/MI/"
NUM_PATIENTS = 10

GOOD_STUDENT_NUM = 11
BAD_STUDENT_NUM = 10

API_KEY = load_openai_api_key()


# =============================
# === Core Helper Functions ===
# =============================

def run_test(therapy_mode, patient_profile_path, student_profile_number, api_key_input=None):
    """Run one simulated therapy session between student and patient."""
    api_key_input = API_KEY or "key"

    # Initialize LearningBuddy with appropriate mode
    if therapy_mode == MI_MODE:
        lb = LearningBuddy(api_key_input, "null", MI_MODE)
        student = VirtualStudent.VirtualStudent(
            api_key_input,
            VirtualStudent.VirtualStudent.MI,
            student_index=student_profile_number
        )
    else:
        raise ValueError("Invalid therapy mode selected")

    # Load patient profile and start session
    case = lb.load_patient_profile(patient_profile_path)
    session_result = lb.start_session(case_study=case)

    # Run 10 exchanges between student and patient
    for _ in range(10):
        # Get current interview messages from state
        current_state = lb.get_session_state()
        interview_messages = current_state.get("interview_messages", [])

        # Student generates response
        student_utterance = student.get_response(interview_messages)

        # Send student's message to virtual patient
        session_result = lb.send_message(student_utterance)

    # End session and get feedback
    session_result = lb.end_session()

    # Get export folder path from state
    final_state = lb.get_session_state()
    folder_name = final_state.get("folder_name", "")

    return {
        "session_result": session_result,
        "folder_name": folder_name
    }


def extract_cbt_scores(eval_file_paths):
    """Extract numerical CBT scores from feedback text files."""
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
                # Each match is a tuple like ('4.5', '.5'), so take matches[-1][0]
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


def extract_recommendations(feedback_files):
    """Extract 'Recommendations' section from feedback files."""
    recommendations = []

    for path in feedback_files:
        text = Path(path).read_text(encoding="utf-8")

        pattern = re.compile(r"\s*Recommendations.*", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)

        if matches:
            # take the last occurrence
            recommendations.append(matches[-1].strip())

    print(recommendations)
    return recommendations


def augment_student_prompt(student_csv, student_num, feedback_texts, save_as="Student_Profiles/temp_student.csv"):
    """Append combined feedback to student's prompt and save as new CSV."""
    df = pd.read_csv(student_csv, quotechar='"', skipinitialspace=True)
    combined_feedback = "\n\n---\n\n".join(feedback_texts)
    improvement_intro = (
        "Now, reflect on the weaknesses of the previous instructions and strive to enhance your MI skills and performance during the therapy session with the patient by strictly incorporating the list of recommendations. "
        "End the conversation after exactly 10 clinician–patient exchanges. \n\n"
    )

    # improvement_intro = (
    #     "You have just completed your first session and received supervision. "
    #     "Now you have received some feedback to help you improve. "
    #     "You are now attempting to apply this feedback. "
    #     "End the conversation after exactly 10 clinician–patient exchanges. \n\n"
    # )
    df.loc[student_num, "Prompt"] += f"\n\n# Feedback to incorporate:\n{improvement_intro}{combined_feedback}"
    df.to_csv(save_as, index=False)
    return save_as


def plot_scores(scores_a, scores_b, label_a, label_b, title):
    """Plot bar chart comparing two sets of CBT scores."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 11

    categories = list(scores_a.keys())
    avg_a = [round(np.mean([v if v is not None else 0 for v in scores_a[cat]]), 2) for cat in categories]
    avg_b = [round(np.mean([v if v is not None else 0 for v in scores_b[cat]]), 2) for cat in categories]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width / 2, avg_a, width, label=label_a, color='skyblue')
    bars2 = ax.bar(x + width / 2, avg_b, width, label=label_b, color='orange')

    ax.set_ylabel('Average Score')
    ax.set_xlabel('Score Category')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 5.5)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()
    return fig


# =============================
# === Experiment Pipeline ====
# =============================

def run_experiment(student_num, num_patients, student_csv_path, label):
    """Run CBT experiment for one student."""
    start = time.time()
    random.seed(42)

    # Get random patient files
    all_files = [
        os.path.join(PATIENT_DIR, f)
        for f in os.listdir(PATIENT_DIR)
        if os.path.isfile(os.path.join(PATIENT_DIR, f))
    ]
    random_files = random.sample(all_files, num_patients)

    mode = MI_MODE
    df = pd.read_csv(student_csv_path, quotechar='"', skipinitialspace=True)
    all_results = []
    saved_data_directory = []

    # Run tests for each patient
    for patient in random_files:
        results = run_test(mode, patient, student_num)
        feedback_path = results["folder_name"] + "/detailed_feedback.txt"
        saved_data_directory.append(feedback_path)

    # Extract scores from all feedback files
    student_scores = extract_cbt_scores(saved_data_directory)

    # Build results table
    for idx, patient_file in enumerate(random_files):
        row = {
            "Student": df.iloc[student_num]["Name"],
            "Patient": os.path.basename(patient_file)
        }
        for key, values in student_scores.items():
            row[key] = values[idx] if idx < len(values) else None
        all_results.append(row)

    # Save results
    base_folder = "saved_tests"
    os.makedirs(base_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(base_folder, f"{label}_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    csv_path = os.path.join(save_folder, "MI_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Student", "Patient"] + list(student_scores.keys())
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Save elapsed time
    elapsed_seconds = time.time() - start
    with open(os.path.join(save_folder, "elapsed_time.txt"), "w") as f:
        f.write(f"Elapsed time: {elapsed_seconds:.2f} seconds\n")

    print(f"[{label.upper()} RUN COMPLETE] Results saved to {save_folder}")
    return student_scores, saved_data_directory


# =============================
# === Main Experiment Flow ====
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Automated CBT Testing Pipeline")
    print("=" * 60)

    # 1️⃣ Good vs Bad student comparison
    print("\n[STEP 1] Running Good Student Experiments...")
    scores_good, _ = run_experiment(
        GOOD_STUDENT_NUM,
        NUM_PATIENTS,
        STUDENT_CSV_PATH,
        label="good_student"
    )

    print("\n[STEP 2] Running Bad Student Experiments...")
    scores_bad, feedback_files = run_experiment(
        BAD_STUDENT_NUM,
        NUM_PATIENTS,
        STUDENT_CSV_PATH,
        label="bad_student"
    )

    print("\n[STEP 3] Generating comparison plot (Good vs Bad)...")
    fig1 = plot_scores(
        scores_good,
        scores_bad,
        "Good Student",
        "Bad Student",
        "Evaluation Scores for MI"
    )
    path1 = os.path.join(
        "saved_tests",
        f"comparison_good_vs_bad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig1.savefig(path1, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Plot 1 saved: {path1}")

    # 2️⃣ Bad student before vs after feedback (individualized feedback per session)
    print("\n[STEP 4] Running Bad Student with Individual Feedback...")
    scores_after_all = {cat: [] for cat in scores_bad.keys()}

    for i, feedback_path in enumerate(feedback_files):
        print(f"  Processing feedback {i + 1}/{len(feedback_files)}...")

        # Extract feedback only from this session
        feedback_text = extract_recommendations([feedback_path])

        # Create a temporary CSV with only that feedback
        updated_csv = augment_student_prompt(
            STUDENT_CSV_PATH,
            BAD_STUDENT_NUM,
            feedback_text,
            save_as=f"Student_Profiles/temp_student_feedback_{i}.csv"
        )

        # Run experiment for that single feedback improvement
        scores_after, _ = run_experiment(
            BAD_STUDENT_NUM,
            1,
            updated_csv,
            label=f"bad_student_after_feedback_{i}"
        )

        # Accumulate results (combine all single runs)
        for cat in scores_after_all.keys():
            scores_after_all[cat].extend(scores_after[cat])

    print("\n[STEP 5] Generating comparison plot (Before vs After Feedback)...")
    fig2 = plot_scores(
        scores_after_all,
        scores_bad,
        "After Feedback",
        "Before Feedback",
        "Evaluation Scores - Before vs After Individual Feedback"
    )
    path2 = os.path.join(
        "saved_tests",
        f"comparison_before_after_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig2.savefig(path2, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Plot 2 saved: {path2}")

    print("\n" + "=" * 60)
    print("✅ All experiments completed successfully!")
    print("=" * 60)