
import os
import sys
import glob
import json
import re
from datetime import datetime
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_key_loader import load_openai_api_key

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

def extract_patient_turns(transcript_path, max_turns=10):
    """
    Extracts the text of the first `max_turns` patient turns from the transcript.
    """
    client_label = "PATIENT"
    turns = []
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(f"{client_label}:"):
             current_speaker = 'client'
             content = line[len(client_label)+1:].strip()
             current_text = [content] if content else []
        elif line.startswith("PSYCHIATRIST:"):
            # End of client turn
            if current_speaker == 'client' and current_text:
                full_text = " ".join(current_text)
                clean_text = re.sub(r'\(.*?\)', '', full_text).strip()
                if clean_text:
                    turns.append(clean_text)
            current_speaker = 'counselor'
            current_text = [] 
        else:
            if current_speaker == 'client':
                current_text.append(line)
        
        if len(turns) >= max_turns:
            break
            
    if current_speaker == 'client' and current_text and len(turns) < max_turns:
         full_text = " ".join(current_text)
         clean_text = re.sub(r'\(.*?\)', '', full_text).strip()
         if clean_text:
             turns.append(clean_text)

    return turns[:max_turns]

def classify_sentiment_single(text, llm):
    """
    Classifies a single text turn. Returns 'positive', 'neutral', or 'negative'.
    """
    try:
        msg = HumanMessage(content=f"Classify the sentiment of the following text as 'positive', 'neutral', or 'negative'. Return ONLY the word.\n\nText: {text}")
        response = llm.invoke([msg])
        sentiment = response.content.strip().lower()
        
        if 'positive' in sentiment:
            return 'positive'
        elif 'negative' in sentiment:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error classifying: {e}")
        return 'neutral' # Default

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_key = load_openai_api_key()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key)
    
    saved_sessions_dir = os.path.join(root_dir, 'Saved_Sessions')
    saved_tests_dir = os.path.join(root_dir, 'saved_tests')

    # 1. Map Tests
    test_runs = []
    if os.path.exists(saved_tests_dir):
        for entry in os.scandir(saved_tests_dir):
            if entry.is_dir():
                ts, mode = get_test_run_info(entry.path)
                if ts:
                    test_runs.append({'timestamp': ts, 'mode': mode, 'path': entry.path, 'ts_str': ts.strftime("%Y%m%d_%H%M%S")})
    test_runs.sort(key=lambda x: x['timestamp'])

    # Data structure: matched_data[(ts_str, patient_id, mode)] = { 'turn_sentiments': { 0: [], 1: [], ... } }
    # Use tuple as key to group by timestamp, patient_id, and therapist_mode
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
                    transcript_path = os.path.join(entry.path, 'interview_transcript.txt')
                    turns = extract_patient_turns(transcript_path, max_turns=10)
                    
                    if not turns:
                        continue
                    
                    # Create key for grouping: (timestamp, patient_id, therapist_mode)
                    key = (matching_run['ts_str'], patient_id, matching_run['mode'])
                    
                    # Initialize entry if it doesn't exist
                    if key not in matched_data:
                        matched_data[key] = {
                            'timestamp': matching_run['ts_str'],
                            'patient_id': patient_id,
                            'therapist_mode': matching_run['mode'],
                            'turn_sentiments': {i: [] for i in range(10)}
                        }
                        
                    print(f"Processing {entry.name} (patient_id={patient_id}, {len(turns)} turns) -> Test Run {matching_run['ts_str']}")
                    
                    # Classify each turn
                    for i, text in enumerate(turns):
                        if i < 10:
                            sentiment = classify_sentiment_single(text, llm)
                            matched_data[key]['turn_sentiments'][i].append(sentiment)

    # 3. Aggregate Results
    # Group by (timestamp, patient_id, therapist_mode, turn_number)
    # Calculate CUMULATIVE proportions: for turn N, include all sentiments from turns 0 through N
    final_rows = []
    
    for key, data in matched_data.items():
        timestamp = data['timestamp']
        patient_id = data['patient_id']
        therapist_mode = data['therapist_mode']
        turn_map = data['turn_sentiments']
        
        # Track cumulative sentiments across turns
        cumulative_pos = 0
        cumulative_neu = 0
        cumulative_neg = 0
        cumulative_total = 0
        
        for i in range(10):
            # Get sentiments for this turn
            turn_sentiments = turn_map[i]
            if turn_sentiments:
                # Add this turn's sentiments to cumulative counts
                cumulative_pos += turn_sentiments.count('positive')
                cumulative_neu += turn_sentiments.count('neutral')
                cumulative_neg += turn_sentiments.count('negative')
                cumulative_total += len(turn_sentiments)
            
            # Only output row if we have at least one sentiment so far
            if cumulative_total > 0:
                prop_pos = cumulative_pos / cumulative_total
                prop_neu = cumulative_neu / cumulative_total
                prop_neg = cumulative_neg / cumulative_total
                
                final_rows.append({
                    'timestamp': timestamp,
                    'patient_id': patient_id,
                    'therapist_mode': therapist_mode,
                    'turn_number': i + 1,
                    'prop_pos': round(prop_pos, 2),
                    'prop_neu': round(prop_neu, 2),
                    'prop_neg': round(prop_neg, 2)
                })
    
    # 4. Save Output
    if final_rows:
        df = pd.DataFrame(final_rows)
        # Ensure csv directory exists
        csv_dir = os.path.join(root_dir, 'Eval', 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        output_path = os.path.join(csv_dir, 'sentiment_dataset_by_turn.csv')
        df.to_csv(output_path, index=False)
        print(f"\nSaved per-turn sentiment dataset to {output_path}")
        print(df.head(20))
    else:
        print("No matched session data found to aggregate.")

if __name__ == "__main__":
    main()
