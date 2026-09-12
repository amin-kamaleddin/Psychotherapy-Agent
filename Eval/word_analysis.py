
import re
import os
import glob
import statistics
import pandas as pd


CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")


def ensure_csv_dir():
    """Create csv output directory if it does not exist."""
    os.makedirs(CSV_DIR, exist_ok=True)

def parse_transcript_turns(file_path):
    """
    Parses the transcript and returns lists of metrics PER TURN PAIR.
    Returns:
        counselor_accum_avg (list of floats)
        client_accum_avg (list of floats)
        ratios (list of floats)
    or None if parsing fails.
    """
    counselor_label = "PSYCHIATRIST"
    client_label = "PATIENT"
    
    turns = []
    current_speaker = None
    current_text = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    # Parse file into turns
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(f"{counselor_label}:"):
            if current_speaker:
                turns.append({'speaker': current_speaker, 'text': " ".join(current_text)})
            current_speaker = 'counselor'
            content = line[len(counselor_label)+1:].strip()
            current_text = [content] if content else []
        elif line.startswith(f"{client_label}:"):
            if current_speaker:
                turns.append({'speaker': current_speaker, 'text': " ".join(current_text)})
            current_speaker = 'client'
            content = line[len(client_label)+1:].strip()
            current_text = [content] if content else []
        else:
            if current_speaker:
                current_text.append(line)
    
    if current_speaker:
        turns.append({'speaker': current_speaker, 'text': " ".join(current_text)})

    cleaned_turns = []
    for turn in turns:
        text = re.sub(r'\(.*?\)', '', turn['text'])
        word_count = len(text.split())
        if word_count > 0:
            cleaned_turns.append({'speaker': turn['speaker'], 'count': word_count})

    if not cleaned_turns:
        return None

    # Calculate accumulated averages
    counselor_accum_avg = []
    client_accum_avg = []
    counselor_accum_total = []
    client_accum_total = []
    
    c_total = 0
    c_turns = 0
    cl_total = 0
    cl_turns = 0
    
    for turn in cleaned_turns:
        if turn['speaker'] == 'counselor':
            c_total += turn['count']
            c_turns += 1
            avg = c_total / c_turns
            counselor_accum_avg.append(avg)
            counselor_accum_total.append(c_total)
        elif turn['speaker'] == 'client':
            cl_total += turn['count']
            cl_turns += 1
            avg = cl_total / cl_turns
            client_accum_avg.append(avg)
            client_accum_total.append(cl_total)
            
    # Align to pairs
    max_len = min(len(counselor_accum_avg), len(client_accum_avg))
    
    # Truncate to pairs
    counselor_accum_avg = counselor_accum_avg[:max_len]
    client_accum_avg = client_accum_avg[:max_len]
    c_accum_totals_trunc = counselor_accum_total[:max_len]
    cl_accum_totals_trunc = client_accum_total[:max_len]
    
    ratios = []
    for i in range(max_len):
        r = 0.0
        if cl_accum_totals_trunc[i] > 0:
            r = c_accum_totals_trunc[i] / cl_accum_totals_trunc[i]
        ratios.append(r)
        
    return {
        'c_acc_avg': counselor_accum_avg,
        'cl_acc_avg': client_accum_avg,
        'ratios': ratios
    }

def build_session_mode_map(root_dir):
    """
    Build a mapping from session folder name to (patient_id, mode) by reading
    all results_*.csv files in Session_Metadata.

    This replaces the previous timestamp-based approach, which was unreliable when
    multiple batch runs started within seconds of each other, causing sessions to be
    attributed to the wrong therapist mode.

    Returns:
        dict: {session_folder_name: (patient_id, mode)}
    """
    session_map = {}
    session_metadata_dir = os.getenv("PSYCH_META_DIR") or os.path.join(root_dir, 'Session_Metadata')
    if not os.path.exists(session_metadata_dir):
        return session_map
    for csv_path in glob.glob(os.path.join(session_metadata_dir, '*', 'results_*.csv')):
        try:
            df = pd.read_csv(csv_path)
            if 'Session' not in df.columns or 'Mode' not in df.columns:
                continue
            for _, row in df.iterrows():
                session_name = str(row['Session'])
                mode = str(row['Mode'])
                m = re.match(r'session_export_patient_(.+?)_\d{4}-\d{2}-\d{2}', session_name)
                patient_id = m.group(1) if m else 'unknown'
                session_map[session_name] = (patient_id, mode)
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
    return session_map

def aggregate_metrics(root_dir):
    saved_sessions_dir = os.getenv("PSYCH_SAVE_DIR") or os.path.join(root_dir, 'Saved_Sessions')
    ensure_csv_dir()

    # 1. Build session -> (patient_id, mode) map from results CSVs.
    # This replaces the old timestamp-based approach which was unreliable when
    # multiple batches started within seconds of each other.
    print("Building session-to-mode map from Session_Metadata results CSVs...")
    session_map = build_session_mode_map(root_dir)
    print(f"Loaded {len(session_map)} session mappings")

    if not session_map:
        print("No session mappings found.")
        return

    # 2. Collect Data per Agent - grouped by (patient_id, mode)
    matched_data = {}  # key: (patient_id, mode) -> value: list of session data

    if os.path.exists(saved_sessions_dir):
        for entry in os.scandir(saved_sessions_dir):
            if not entry.is_dir():
                continue

            session_name = entry.name
            if session_name not in session_map:
                continue

            patient_id, mode = session_map[session_name]

            transcript_path = os.path.join(entry.path, 'interview_transcript.txt')
            if not os.path.exists(transcript_path):
                continue

            data = parse_transcript_turns(transcript_path)
            if data:
                # Create key for grouping: (patient_id, therapist_mode)
                key = (patient_id, mode)

                # Initialize entry if it doesn't exist
                if key not in matched_data:
                    matched_data[key] = []

                # Append this session's data to the list (for averaging across folds)
                matched_data[key].append(data)

    # Check if we found any data
    if not matched_data:
        print("No matching sessions found for any test runs.")
        return

    # 3. First aggregate by patient (average across folds), then by mode
    mode_patient_data = {}  # mode -> list of patient-averaged data
    
    for key, session_list in matched_data.items():
        patient_id, mode = key
        
        if mode not in mode_patient_data:
            mode_patient_data[mode] = []
        
        # Determine max turns for this patient across all their folds
        max_turns = max(len(s['c_acc_avg']) for s in session_list)
        
        # Average across folds for this patient
        patient_avg_data = {
            'c_acc_avg': [],
            'cl_acc_avg': [],
            'ratios': []
        }
        
        for turn_idx in range(max_turns):
            # Collect values for this turn from all folds
            c_vals = [s['c_acc_avg'][turn_idx] for s in session_list if turn_idx < len(s['c_acc_avg'])]
            cl_vals = [s['cl_acc_avg'][turn_idx] for s in session_list if turn_idx < len(s['cl_acc_avg'])]
            r_vals = [s['ratios'][turn_idx] for s in session_list if turn_idx < len(s['ratios'])]
            
            if c_vals:
                patient_avg_data['c_acc_avg'].append(sum(c_vals) / len(c_vals))
            if cl_vals:
                patient_avg_data['cl_acc_avg'].append(sum(cl_vals) / len(cl_vals))
            if r_vals:
                patient_avg_data['ratios'].append(sum(r_vals) / len(r_vals))
        
        # Add this patient's averaged data to the mode's list
        mode_patient_data[mode].append(patient_avg_data)

    # 4. Aggregate across patients for each mode
    for mode, patient_sessions in mode_patient_data.items():
        print(f"\n### Agent Type: {mode} ({len(patient_sessions)} patients) ###")
        print(f"{'Turn_Pair':<10} | {'Avg Counselor Acc':<20} | {'Avg Client Acc':<20} | {'Avg Ratio':<15}")
        print("-" * 75)
        
        # Determine max turns for this mode
        max_turns = 0
        for s in patient_sessions:
            max_turns = max(max_turns, len(s['c_acc_avg']))
            
        output_rows = []
        output_rows.append("Turn_Pair,Counselor_Acc_Avg,Counselor_Acc_Std,Client_Acc_Avg,Client_Acc_Std,Acc_Ratio,Acc_Ratio_Std,Count")

        for i in range(max_turns):
            # Collect values for this turn index from all patients
            c_vals = []
            cl_vals = []
            r_vals = []
            
            for s in patient_sessions:
                if i < len(s['c_acc_avg']):
                    c_vals.append(s['c_acc_avg'][i])
                    cl_vals.append(s['cl_acc_avg'][i])
                    r_vals.append(s['ratios'][i])
            
            if not c_vals:
                continue
                
            avg_c = sum(c_vals) / len(c_vals)
            avg_cl = sum(cl_vals) / len(cl_vals)
            avg_r = sum(r_vals) / len(r_vals)
            
            # Calculate standard deviations across patients
            import numpy as np
            std_c = np.std(c_vals, ddof=1) if len(c_vals) > 1 else 0
            std_cl = np.std(cl_vals, ddof=1) if len(cl_vals) > 1 else 0
            std_r = np.std(r_vals, ddof=1) if len(r_vals) > 1 else 0
            count = len(c_vals)
            
            print(f"{i+1:<10} | {avg_c:<20.2f} | {avg_cl:<20.2f} | {avg_r:<15.2f}")
            output_rows.append(f"{i+1},{avg_c:.2f},{std_c:.2f},{avg_cl:.2f},{std_cl:.2f},{avg_r:.2f},{std_r:.2f},{count}")

        # Save to CSV
        out_file = os.path.join(CSV_DIR, f'metrics_output_{mode}.csv')
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_rows))
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    aggregate_metrics(project_root)
