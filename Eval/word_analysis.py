
import re
import os
import glob
import json
from datetime import datetime
import statistics


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

def get_session_info(session_dir):
    """
    Extract timestamp from session folder name.
    Supports both old format (session_export_{timestamp}) and new format (session_export_patient_{id}_{timestamp}_{uuid}).
    """
    basename = os.path.basename(session_dir)
    
    # Try new format with UUID: session_export_patient_{patient_id}_{timestamp}_{uuid}
    new_format_pattern = r'session_export_patient_(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_[a-f0-9-]+)?$'
    match = re.match(new_format_pattern, basename)
    if match:
        date_str = match.group(2)
        try:
            timestamp = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
            return timestamp
        except ValueError:
            pass
    
    # Try old format: session_export_{timestamp}
    try:
        date_str = basename.replace('session_export_', '')
        # Try with UUID first, then without
        date_str = re.sub(r'_[a-f0-9-]+$', '', date_str)
        timestamp = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
        return timestamp
    except ValueError:
        return None

def get_test_run_info(test_dir):
    metadata_path = os.path.join(test_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                ts_str = data.get('timestamp', '')
                mode = data.get('therapist_mode', 'unknown')
                try:
                    timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    return timestamp, mode
                except ValueError:
                    pass 
        except Exception:
            pass
    return None, None

def aggregate_metrics(root_dir):
    saved_sessions_dir = os.path.join(root_dir, 'Saved_Sessions')
    saved_tests_dir = os.path.join(root_dir, 'Session_Metadata')
    ensure_csv_dir()

    # 1. Collect Test Runs
    test_runs = []
    if os.path.exists(saved_tests_dir):
        for entry in os.scandir(saved_tests_dir):
            if entry.is_dir():
                ts, mode = get_test_run_info(entry.path)
                if ts:
                    test_runs.append({'timestamp': ts, 'mode': mode, 'path': entry.path})
    test_runs.sort(key=lambda x: x['timestamp'])

    if not test_runs:
        print("No test runs found.")
        return

    # 2. Collect Data per Agent
    agent_data = {} # mode -> list of session_data dicts

    if os.path.exists(saved_sessions_dir):
        for entry in os.scandir(saved_sessions_dir):
            if entry.is_dir():
                ts = get_session_info(entry.path)
                if not ts:
                    continue
                
                # Link to test run
                matching_mode = None
                for i in range(len(test_runs) - 1, -1, -1):
                    run = test_runs[i]
                    if run['timestamp'] <= ts:
                        matching_mode = run['mode']
                        break
                
                if matching_mode:
                    transcript_path = os.path.join(entry.path, 'interview_transcript.txt')
                    if not os.path.exists(transcript_path):
                        continue
                    
                    data = parse_transcript_turns(transcript_path)
                    if data:
                        if matching_mode not in agent_data:
                            agent_data[matching_mode] = []
                        agent_data[matching_mode].append(data)

    # Check if we found any data
    if not agent_data:
        print("No matching sessions found for any test runs.")
        return

    # 3. Aggregate and Print
    for mode, sessions in agent_data.items():
        print(f"\n### Agent Type: {mode} ({len(sessions)} sessions) ###")
        print(f"{'Turn_Pair':<10} | {'Avg Counselor Acc':<20} | {'Avg Client Acc':<20} | {'Avg Ratio':<15}")
        print("-" * 75)
        
        # Determine max turns for this agent
        max_turns = 0
        for s in sessions:
            max_turns = max(max_turns, len(s['c_acc_avg']))
            
        output_rows = []
        output_rows.append("Turn_Pair,Counselor_Acc_Avg,Counselor_Acc_Std,Client_Acc_Avg,Client_Acc_Std,Acc_Ratio,Acc_Ratio_Std,Count")

        for i in range(max_turns):
            # Collect values for this turn index from all valid sessions
            c_vals = []
            cl_vals = []
            r_vals = []
            
            for s in sessions:
                if i < len(s['c_acc_avg']):
                    c_vals.append(s['c_acc_avg'][i])
                    cl_vals.append(s['cl_acc_avg'][i])
                    r_vals.append(s['ratios'][i])
            
            if not c_vals:
                continue
                
            avg_c = sum(c_vals) / len(c_vals)
            avg_cl = sum(cl_vals) / len(cl_vals)
            avg_r = sum(r_vals) / len(r_vals)
            
            # Calculate standard deviations
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
