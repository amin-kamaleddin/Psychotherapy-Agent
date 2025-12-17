"""
Extract and analyze feedback scores from detailed_feedback.txt files.
Similar structure to sentiment_analysis.py but for MI feedback criteria scores.
"""

import os
import sys
import re
import json
from datetime import datetime
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


def get_session_timestamp(session_dir):
    """
    Extract timestamp and patient_id from session folder name.
    
    Returns:
        tuple: (datetime, patient_id) where patient_id can be None for old format
    """
    basename = os.path.basename(session_dir)
    
    # Try new format: session_export_patient_{patient_id}_{timestamp}
    new_format_pattern = r'session_export_patient_(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})$'
    match = re.match(new_format_pattern, basename)
    if match:
        patient_id = match.group(1)
        date_str = match.group(2)
        try:
            timestamp = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")
            return timestamp, patient_id
        except ValueError:
            pass
    
    # Try old format: session_export_{timestamp}
    try:
        date_str = basename.replace('session_export_', '')
        timestamp = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")
        return timestamp, None  # No patient_id in old format
    except ValueError:
        return None, None


def get_test_run_info(test_dir):
    """Extract timestamp and mode from test run metadata."""
    metadata_path = os.path.join(test_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                ts_str = data.get('timestamp', '')
                mode = data.get('therapist_mode', 'unknown')
                try:
                    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S"), mode
                except ValueError:
                    pass 
        except Exception:
            pass
    return None, None


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
    saved_tests_dir = os.path.join(root_dir, 'Session_Metadata')
    
    # 1. Map Tests
    test_runs = []
    if os.path.exists(saved_tests_dir):
        for entry in os.scandir(saved_tests_dir):
            if entry.is_dir():
                ts, mode = get_test_run_info(entry.path)
                if ts:
                    test_runs.append({'timestamp': ts, 'mode': mode, 'path': entry.path, 'ts_str': ts.strftime("%Y%m%d_%H%M%S")})
    test_runs.sort(key=lambda x: x['timestamp'])
    
    # Data structure: matched_data[(ts_str, patient_id, mode)] = { 'scores': {criterion: score} }
    matched_data = {}
    
    # 2. Process Sessions
    if os.path.exists(saved_sessions_dir):
        print(f"Scanning sessions in {saved_sessions_dir}...")
        for entry in os.scandir(saved_sessions_dir):
            if entry.is_dir():
                session_result = get_session_timestamp(entry.path)
                if not session_result or session_result[0] is None:
                    continue
                
                session_ts, patient_id = session_result
                # Use 'unknown' as patient_id for old format sessions
                if patient_id is None:
                    patient_id = 'unknown'
                
                # Match to test run
                matching_run = None
                for i in range(len(test_runs) - 1, -1, -1):
                    run = test_runs[i]
                    if run['timestamp'] <= session_ts:
                        matching_run = run
                        break
                
                if matching_run:
                    feedback_path = os.path.join(entry.path, 'detailed_feedback.txt')
                    if not os.path.exists(feedback_path):
                        continue
                    
                    scores = extract_scores_from_feedback(feedback_path)
                    if not scores:
                        print(f"Warning: Could not extract scores from {entry.name}")
                        continue
                    
                    # Create key for grouping: (timestamp, patient_id, therapist_mode)
                    key = (matching_run['ts_str'], patient_id, matching_run['mode'])
                    
                    print(f"Processing {entry.name} (patient_id={patient_id}) -> Test Run {matching_run['ts_str']}, Mode: {matching_run['mode']}")
                    
                    # Store scores
                    matched_data[key] = {
                        'timestamp': matching_run['ts_str'],
                        'patient_id': patient_id,
                        'therapist_mode': matching_run['mode'],
                        'scores': scores
                    }
    
    # 3. Aggregate Results
    # Group by therapist_mode and criterion, calculate averages
    final_rows = []
    
    for key, data in matched_data.items():
        timestamp = data['timestamp']
        patient_id = data['patient_id']
        therapist_mode = data['therapist_mode']
        scores = data['scores']
        
        # Add individual criterion scores
        for criterion in MI_CRITERIA:
            score = scores.get(criterion)
            if score is not None:
                final_rows.append({
                    'timestamp': timestamp,
                    'patient_id': patient_id,
                    'therapist_mode': therapist_mode,
                    'criterion': criterion,
                    'score': score
                })
        
        # Calculate overall average for this patient (average of all valid criteria scores)
        valid_scores = [scores.get(criterion) for criterion in MI_CRITERIA if scores.get(criterion) is not None]
        if valid_scores:
            overall_avg = sum(valid_scores) / len(valid_scores)
            final_rows.append({
                'timestamp': timestamp,
                'patient_id': patient_id,
                'therapist_mode': therapist_mode,
                'criterion': 'Overall Average',
                'score': overall_avg
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
            
            # Count sessions for this mode
            session_count = len(mode_df[['timestamp', 'patient_id']].drop_duplicates())
            
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

