"""
Extract and analyze feedback scores from detailed_feedback.txt files.
Similar structure to sentiment_analysis.py but for MI feedback criteria scores.
"""

import os
import sys
import re
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the 9 MI criteria based on feedback_criteria_MI_5point.txt
MI_CRITERIA = [
    "Acceptance",
    "Empathy",
    "Direction",
    "Collaboration",
    "Autonomy Support",
    "Evocation",
    "Partnership",
    "Cultivating Change Talk",
    "Softening Sustain Talk"
]

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")


def ensure_csv_dir():
    """Create csv output directory if it does not exist."""
    os.makedirs(CSV_DIR, exist_ok=True)


def build_session_mode_map(root_dir):
    """
    Build a mapping from session folder name to (patient_id, mode) by reading
    all results_*.csv files in Session_Metadata.

    This is accurate because the results CSVs record the exact session folders
    used in each batch run. The previous timestamp-based approach was unreliable:
    multiple batches started within seconds of each other, causing sessions to be
    attributed to the wrong mode.

    Returns:
        dict: {session_folder_name: (patient_id, mode)}
    """
    import glob as _glob
    session_map = {}
    session_metadata_dir = os.path.join(root_dir, 'Session_Metadata')

    if not os.path.exists(session_metadata_dir):
        return session_map

    pattern = os.path.join(session_metadata_dir, '*', 'results_*.csv')
    for csv_path in _glob.glob(pattern):
        try:
            df = pd.read_csv(csv_path)
            if 'Session' not in df.columns or 'Mode' not in df.columns:
                continue
            for _, row in df.iterrows():
                session_name = str(row['Session'])
                mode = str(row['Mode'])
                # Extract patient_id from session folder name
                m = re.match(r'session_export_patient_(.+?)_\d{4}-\d{2}-\d{2}', session_name)
                patient_id = m.group(1) if m else 'unknown'
                session_map[session_name] = (patient_id, mode)
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")

    return session_map


def extract_scores_from_feedback(feedback_path):
    """
    Extract scores for all MI criteria from a detailed_feedback.txt file.
    Uses the same pattern as the reference extract_scores function.
    
    Returns:
        dict: {criterion_name: score (float)} or None if extraction fails
    """
    try:
        with open(feedback_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {feedback_path}: {e}")
        return None
    
    scores = {}
    
    # Extract scores for each criterion using the proven pattern from reference
    for cat in MI_CRITERIA:
        pattern = re.compile(
            rf"{re.escape(cat)}.*?Score:\s*\**\s*(\d+(\.\d+)?)\**",
            re.IGNORECASE | re.DOTALL
        )
        matches = pattern.findall(text)
        if matches:
            # Take the last match if multiple matches found
            score_value = float(matches[-1][0])
            scores[cat] = score_value
        else:
            scores[cat] = None
    
    # Check if we got at least some scores
    valid_scores = [s for s in scores.values() if s is not None]
    if len(valid_scores) == 0:
        return None
    
    return scores


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ensure_csv_dir()

    saved_sessions_dir = os.path.join(root_dir, 'Saved_Sessions')

    # 1. Build session -> (patient_id, mode) map from results CSVs.
    # This replaces the old timestamp-based approach which was unreliable when
    # multiple batch runs started within seconds of each other.
    print("Building session-to-mode map from Session_Metadata results CSVs...")
    session_map = build_session_mode_map(root_dir)
    print(f"Loaded {len(session_map)} session mappings")

    matched_data = {}

    # 2. Process Sessions
    if os.path.exists(saved_sessions_dir):
        print(f"Scanning sessions in {saved_sessions_dir}...")
        for entry in os.scandir(saved_sessions_dir):
            if not entry.is_dir():
                continue

            session_name = entry.name
            if session_name not in session_map:
                print(f"  Skipping {session_name}: not found in any results CSV")
                continue

            patient_id, mode = session_map[session_name]

            feedback_path = os.path.join(entry.path, 'detailed_feedback.txt')
            if not os.path.exists(feedback_path):
                continue

            scores = extract_scores_from_feedback(feedback_path)
            if not scores:
                print(f"Warning: Could not extract scores from {entry.name}")
                continue

            # Create key for grouping: (patient_id, therapist_mode)
            key = (patient_id, mode)

            print(f"Processing {entry.name} (patient_id={patient_id}) -> Mode: {mode}")

            if key not in matched_data:
                matched_data[key] = {
                    'patient_id': patient_id,
                    'therapist_mode': mode,
                    'all_scores': []  # List to store scores from all folds
                }

            matched_data[key]['all_scores'].append(scores)
    
    # 3. Aggregate Results
    # Average scores across all folds for each patient
    final_rows = []
    
    for key, data in matched_data.items():
        patient_id = data['patient_id']
        therapist_mode = data['therapist_mode']
        all_scores = data['all_scores']
        
        # Calculate average scores across all folds
        averaged_scores = {}
        for criterion in MI_CRITERIA:
            criterion_values = [scores.get(criterion) for scores in all_scores if scores.get(criterion) is not None]
            if criterion_values:
                averaged_scores[criterion] = sum(criterion_values) / len(criterion_values)
        
        # Add individual criterion scores
        for criterion in MI_CRITERIA:
            score = averaged_scores.get(criterion)
            if score is not None:
                final_rows.append({
                    'patient_id': patient_id,
                    'therapist_mode': therapist_mode,
                    'criterion': criterion,
                    'score': score,
                    'num_folds': len(all_scores)
                })
        
        # Calculate overall average for this patient (average of all valid criteria scores)
        if averaged_scores:
            overall_avg = sum(averaged_scores.values()) / len(averaged_scores)
            final_rows.append({
                'patient_id': patient_id,
                'therapist_mode': therapist_mode,
                'criterion': 'Overall Average',
                'score': overall_avg,
                'num_folds': len(all_scores)
            })
    
    # 4. Save Output
    if final_rows:
        df = pd.DataFrame(final_rows)
        output_path = os.path.join(CSV_DIR, 'feedback_scores_by_criterion.csv')
        df.to_csv(output_path, index=False)
        print(f"\nSaved feedback scores dataset to {output_path}")
        print(f"Total records: {len(df)}")
        print(df.head(20))
        
        # 5. Aggregate by Therapist Mode and Criterion (Average across sessions)
        print("\nCalculated average scores by Therapist Mode and Criterion:")
        
        unique_modes = df['therapist_mode'].unique()
        
        for mode in unique_modes:
            mode_df = df[df['therapist_mode'] == mode]
            
            # Count unique patients for this mode
            session_count = mode_df['patient_id'].nunique()
            
            # Group by criterion and calculate mean and std of scores
            grouped = mode_df.groupby('criterion')['score'].agg(['mean', 'std', 'count']).reset_index()
            
            # Round for cleaner output
            grouped['mean'] = grouped['mean'].round(2)
            grouped['std'] = grouped['std'].round(2)
            
            print(f"\n### Therapist Mode: {mode} ({session_count} sessions) ###")
            print(f"{'Criterion':<30} | {'Mean Score':<12} | {'Std Dev':<10} | {'Count':<10}")
            print("-" * 70)
            
            for _, row in grouped.iterrows():
                print(f"{row['criterion']:<30} | {row['mean']:<12.2f} | {row['std']:<10.2f} | {int(row['count']):<10}")
                
            # Save to CSV similar to other metrics files
            output_csv_name = f'feedback_metrics_{mode}.csv'
            output_csv_path = os.path.join(CSV_DIR, output_csv_name)
            grouped.to_csv(output_csv_path, index=False)
            print(f"Saved aggregated metrics to {output_csv_path}")
    
    else:
        print("No matched session data found to aggregate.")


if __name__ == "__main__":
    main()

