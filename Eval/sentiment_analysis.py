
import os
import sys
import glob
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_key_loader import load_openai_api_key

# Thread-safe print lock
print_lock = threading.Lock()

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
    session_metadata_dir = os.path.join(root_dir, 'Session_Metadata')
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

def classify_sentiment_single(text, api_key):
    """
    Classifies a single text turn. Returns 'positive', 'neutral', or 'negative'.
    Each call creates its own LLM instance for thread safety.
    """
    try:
        llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key, timeout=60)
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
        with print_lock:
            print(f"Error classifying: {e}")
        return 'neutral' # Default

def process_single_session(session_data, api_key):
    """
    Process a single session - designed for parallel execution.
    
    Args:
        session_data: Tuple of (entry_path, session_ts, patient_id, matching_run)
        api_key: OpenAI API key
        
    Returns:
        Tuple of (key, processed_data) or (None, None) if processing fails
    """
    entry_path, session_ts, patient_id, matching_run = session_data
    
    try:
        transcript_path = os.path.join(entry_path, 'interview_transcript.txt')
        turns = extract_patient_turns(transcript_path, max_turns=10)
        
        if not turns:
            return None, None
        
        # Create key for grouping: (patient_id, therapist_mode)
        # Note: We don't include timestamp to avoid counting same patient multiple times
        key = (patient_id, matching_run['mode'])
        
        # Initialize data structure
        turn_sentiments = {i: [] for i in range(10)}
        
        with print_lock:
            print(f"Processing {os.path.basename(entry_path)} (patient_id={patient_id}, {len(turns)} turns) -> Test Run {matching_run['ts_str']}")
        
        # Classify each turn
        for i, text in enumerate(turns):
            if i < 10:
                sentiment = classify_sentiment_single(text, api_key)
                turn_sentiments[i].append(sentiment)
        
        return key, {
            'patient_id': patient_id,
            'therapist_mode': matching_run['mode'],
            'turn_sentiments': turn_sentiments
        }
        
    except Exception as e:
        with print_lock:
            print(f"Error processing {os.path.basename(entry_path)}: {e}")
        return None, None


def main(max_workers=20):
    """
    Main function to perform sentiment analysis with parallel processing.
    
    Args:
        max_workers: Number of parallel workers (default: 20)
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_key = load_openai_api_key()
    
    saved_sessions_dir = os.path.join(root_dir, 'Saved_Sessions')

    # 1. Build session -> (patient_id, mode) map from results CSVs.
    # This replaces the old timestamp-based approach which was unreliable when
    # multiple batches started within seconds of each other.
    print("Building session-to-mode map from Session_Metadata results CSVs...")
    session_map = build_session_mode_map(root_dir)
    print(f"Loaded {len(session_map)} session mappings")

    # Data structure to store results
    matched_data = {}

    # 2. Prepare Sessions for Processing
    sessions_to_process = []

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
            # Pack as a dict so process_single_session can use it without changes
            matching_run = {'mode': mode, 'ts_str': ''}
            sessions_to_process.append((entry.path, None, patient_id, matching_run))
    
    if not sessions_to_process:
        print("No sessions found to process.")
        return
    
    print(f"\nStarting parallel processing with {max_workers} workers...")
    print(f"Processing {len(sessions_to_process)} sessions\n")
    
    # 3. Process Sessions in Parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_session = {
            executor.submit(process_single_session, session_data, api_key): session_data 
            for session_data in sessions_to_process
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_session):
            completed += 1
            key, data = future.result()
            
            if key is not None and data is not None:
                # Merge results with same key
                if key in matched_data:
                    # Merge turn sentiments
                    for turn_idx in range(10):
                        matched_data[key]['turn_sentiments'][turn_idx].extend(
                            data['turn_sentiments'][turn_idx]
                        )
                else:
                    matched_data[key] = data
            
            with print_lock:
                print(f"Progress: {completed}/{len(sessions_to_process)} sessions processed")
    
    print(f"\nAll sessions processed!")
    print(f"Total unique session groups: {len(matched_data)}\n")

    # 4. Aggregate Results
    # Group by (timestamp, patient_id, therapist_mode, turn_number)
    # Calculate CUMULATIVE proportions: for turn N, include all sentiments from turns 0 through N
    final_rows = []
    
    for key, data in matched_data.items():
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
                    'patient_id': patient_id,
                    'therapist_mode': therapist_mode,
                    'turn_number': i + 1,
                    'prop_pos': round(prop_pos, 2),
                    'prop_neu': round(prop_neu, 2),
                    'prop_neg': round(prop_neg, 2)
                })
    
    # 5. Save Output
    if final_rows:
        df = pd.DataFrame(final_rows)
        # Ensure csv directory exists
        csv_dir = os.path.join(root_dir, 'Eval', 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        output_path = os.path.join(csv_dir, 'sentiment_dataset_by_turn.csv')
        df.to_csv(output_path, index=False)
        print(f"\nSaved per-turn sentiment dataset to {output_path}")
        print(f"Total records: {len(df)}")
        print(df.head(20))
    else:
        print("No matched session data found to aggregate.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sentiment Analysis with Parallel Processing")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Number of parallel workers (default: 20)")
    args = parser.parse_args()
    
    main(max_workers=args.max_workers)
