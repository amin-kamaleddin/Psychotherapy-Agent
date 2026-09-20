"""
Calculate Language Style Matching (LSM) scores for therapy sessions.

Follows the same structure as sentiment_analysis.py to calculate LSM
between counselor and client turns for each conversation turn.
"""

import os
import sys
import re
import glob
from collections import defaultdict
import pandas as pd

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")


def ensure_csv_dir():
    """Create csv output directory if missing."""
    os.makedirs(CSV_DIR, exist_ok=True)


class LSMCalculator:
    """
    Calculates the Linguistic Style Matching (LSM) score between two texts 
    based on the average of 8 functional word categories, using the 
    Sorensen-Dice-like similarity metric.
    """
    
    def __init__(self):
        # The 8 categories used in the final LSM average calculation 
        self.LSM_CATEGORIES_TO_AVERAGE = [
            'prepositions',
            'articles',
            'auxiliary_verbs',
            'adverbs',
            'conjunctions',
            'personal_pronouns',
            'impersonal_pronouns',
            'negations'
        ]
        
        # Refined dictionary of function words for the 9 standard LIWC-style categories.
        # Note: 'quantifiers' are included but skipped in the final average calculation 
        # to match the user-provided formula.
        self.function_words = {
            # 1. Personal Pronouns (ppron)
            'personal_pronouns': {
                'i', 'me', 'my', 'mine', 'myself', 
                'we', 'us', 'our', 'ours', 'ourselves',
                'you', 'your', 'yours', 'yourself', 'yourselves', 'youre', 
                'he', 'him', 'his', 'himself', 'hes',
                'she', 'her', 'hers', 'herself', 'shes',
                'they', 'them', 'their', 'theirs', 'themselves', 'theyre'
            },
            
            # 2. Impersonal Pronouns (ipron)
            'impersonal_pronouns': {
                'it', 'its', 'itself', 'this', 'that', 'these', 'those', 
                'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 
                'any', 'anyone', 'anything', 'some', 'someone', 'something', 'every', 
                'everyone', 'everything', 'none', 'nothing'
            },
            
            # 3. Articles (article)
            'articles': {'a', 'an', 'the'},
            
            # 4. Prepositions (prep)
            'prepositions': {
                'in', 'on', 'at', 'of', 'with', 'to', 'from', 'by', 'for', 'about', 
                'under', 'over', 'between', 'among', 'through', 'during', 'before', 
                'after', 'above', 'below', 'except', 'until', 'since', 'across', 
                'behind', 'beside', 'down', 'up', 'off', 'out', 'around', 'into', 
                'onto', 'upon', 'like', 'as', 'than'
            },
            
            # 5. Auxiliary Verbs (auxverb)
            'auxiliary_verbs': {
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'isnt', 
                'have', 'has', 'had', 'havent', 'hasnt', 'hadnt', 
                'do', 'does', 'did', 'dont', 'doesnt', 'didnt', 
                'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 
                'wont', 'wouldnt', 'shouldnt', 'cant', 'couldnt', 'maynt'
            },
            
            # 6. Adverbs (adverb)
            'adverbs': {
                'very', 'really', 'quite', 'just', 'too', 'so', 'much', 'more', 
                'most', 'less', 'least', 'only', 'almost', 'never', 'always', 
                'often', 'sometimes', 'maybe', 'perhaps', 'here', 'there', 
                'now', 'then', 'ever', 'further', 'away', 'back', 'well'
            },
            
            # 7. Conjunctions (conj)
            'conjunctions': {
                'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'although', 
                'because', 'if', 'unless', 'since', 'while', 'where', 'when', 
                'as', 'though', 'whereas'
            },
            
            # 8. Negations (negate)
            'negations': {
                'no', 'not', 'none', 'never', 'nobody', 'nothing', 'neither', 
                'nowhere', 'cannot', 'cant', 'dont', 'wont', 'didnt', 'isnt', 
                'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'shouldnt', 
                'couldnt', 'wouldnt', 'mustnt'
            },
            
            # 9. Quantifiers (Not used in the final average calculation per the formula image)
            'quantifiers': {
                'all', 'some', 'many', 'much', 'few', 'several', 'plenty', 'lot', 
                'lots', 'majority', 'minority', 'whole', 'entire', 'every', 'each'
            }
        }
    
    def _tokenize(self, text):
        """Cleans and splits text into words."""
        # Lowercase and use regex to find word tokens, ignoring apostrophes for contractions
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _get_category_counts(self, words):
        """Counts frequencies of function words in the provided word list."""
        counts = defaultdict(int)
        total_words = len(words)
        
        if total_words == 0:
            return counts, 0
        
        for category, vocab in self.function_words.items():
            # Count how many words in the text exist in this category
            count = sum(1 for word in words if word in vocab)
            counts[category] = count
            
        return counts, total_words
    
    def compare_texts(self, text1, text2):
        """
        Calculates the LSM score between two strings using the specified 8-category average formula.
        Returns a score between 0 (no match) and 1 (perfect match).
        """
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)
        
        counts1, total1 = self._get_category_counts(words1)
        counts2, total2 = self._get_category_counts(words2)
        
        if total1 == 0 or total2 == 0:
            # If one or both texts are empty, LSM is defined as 0.0 (no style to match)
            return 0.0
        
        cat_scores = []
        
        # Calculate LSM for ONLY the 8 categories specified in self.LSM_CATEGORIES_TO_AVERAGE
        for category in self.LSM_CATEGORIES_TO_AVERAGE:
            # 1. Calculate normalized frequencies (proportion of the total word count)
            p1 = counts1[category] / total1 
            p2 = counts2[category] / total2 
            
            # 2. Apply the individual LSM formula: 1 - (|p1 - p2| / (p1 + p2 + 0.0001))
            denominator = p1 + p2 + 0.0001  # Add a small epsilon to prevent division by zero
            score = 1 - (abs(p1 - p2) / denominator)
            
            cat_scores.append(score)
            
        # 3. Final score is the arithmetic average of the 8 category scores
        return sum(cat_scores) / len(cat_scores)


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


def extract_turn_pairs(transcript_path, max_turns=10):
    """
    Extract counselor and client turn pairs from transcript.
    
    Returns:
        list of tuples: [(counselor_text, client_text), ...] for each turn pair
    """
    counselor_label = "PSYCHIATRIST"
    client_label = "PATIENT"
    turn_pairs = []
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    
    current_speaker = None
    current_text = []
    counselor_turns = []
    client_turns = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(f"{counselor_label}:"):
            # Save previous turn if it was client
            if current_speaker == 'client' and current_text:
                full_text = " ".join(current_text)
                clean_text = re.sub(r'\(.*?\)', '', full_text).strip()
                if clean_text:
                    client_turns.append(clean_text)
            
            current_speaker = 'counselor'
            content = line[len(counselor_label)+1:].strip()
            current_text = [content] if content else []
            
        elif line.startswith(f"{client_label}:"):
            # Save previous turn if it was counselor
            if current_speaker == 'counselor' and current_text:
                full_text = " ".join(current_text)
                clean_text = re.sub(r'\(.*?\)', '', full_text).strip()
                if clean_text:
                    counselor_turns.append(clean_text)
            
            current_speaker = 'client'
            content = line[len(client_label)+1:].strip()
            current_text = [content] if content else []
            
        else:
            # Continuation of current turn
            if current_speaker:
                current_text.append(line)
        
        if len(counselor_turns) >= max_turns and len(client_turns) >= max_turns:
            break
    
    # Handle last turn
    if current_speaker == 'counselor' and current_text:
        full_text = " ".join(current_text)
        clean_text = re.sub(r'\(.*?\)', '', full_text).strip()
        if clean_text:
            counselor_turns.append(clean_text)
    elif current_speaker == 'client' and current_text:
        full_text = " ".join(current_text)
        clean_text = re.sub(r'\(.*?\)', '', full_text).strip()
        if clean_text:
            client_turns.append(clean_text)
    
    # Pair up counselor and client turns
    # Assumption: conversation starts with counselor, then alternates
    min_len = min(len(counselor_turns), len(client_turns))
    for i in range(min_len):
        turn_pairs.append((counselor_turns[i], client_turns[i]))
    
    return turn_pairs[:max_turns]


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ensure_csv_dir()
    
    saved_sessions_dir = os.getenv("PSYCH_SAVE_DIR") or os.path.join(root_dir, 'Saved_Sessions')

    # Initialize LSM calculator
    lsm_calc = LSMCalculator()

    # 1. Build session -> (patient_id, mode) map from results CSVs.
    # This replaces the old timestamp-based approach which was unreliable when
    # multiple batches started within seconds of each other.
    print("Building session-to-mode map from Session_Metadata results CSVs...")
    session_map = build_session_mode_map(root_dir)
    print(f"Loaded {len(session_map)} session mappings")

    # Data structure: matched_data[(patient_id, mode)] = { 'turn_lsm_scores': { 0: [], 1: [], ... } }
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

            transcript_path = os.path.join(entry.path, 'interview_transcript.txt')
            turn_pairs = extract_turn_pairs(transcript_path, max_turns=10)

            if not turn_pairs:
                continue

            # Create key for grouping: (patient_id, therapist_mode)
            key = (patient_id, mode)

            # Initialize entry if it doesn't exist
            if key not in matched_data:
                matched_data[key] = {
                    'patient_id': patient_id,
                    'therapist_mode': mode,
                    'turn_lsm_scores': {i: [] for i in range(10)}
                }

            print(f"Processing {entry.name} (patient_id={patient_id}, {len(turn_pairs)} turn pairs) -> Mode: {mode}")

            # Calculate LSM for each turn pair
            for i, (counselor_text, client_text) in enumerate(turn_pairs):
                if i < 10:
                    lsm_score = lsm_calc.compare_texts(counselor_text, client_text)
                    matched_data[key]['turn_lsm_scores'][i].append(lsm_score)
    
    # 3. Aggregate Results
    # Calculate per-turn LSM scores (LSM between counselor turn N and client turn N)
    # Following sentiment analysis pattern, we calculate cumulative average LSM up to each turn
    final_rows = []
    
    for key, data in matched_data.items():
        patient_id = data['patient_id']
        therapist_mode = data['therapist_mode']
        turn_map = data['turn_lsm_scores']
        
        # Track cumulative LSM scores across turns (similar to sentiment analysis)
        cumulative_scores = []
        
        for i in range(10):
            # Get LSM scores for this turn
            turn_scores = turn_map[i]
            if turn_scores:
                # Add this turn's scores to cumulative list
                cumulative_scores.extend(turn_scores)
            
            # Only output row if we have at least one score so far
            if cumulative_scores:
                avg_lsm = sum(cumulative_scores) / len(cumulative_scores)
                
                final_rows.append({
                    'patient_id': patient_id,
                    'therapist_mode': therapist_mode,
                    'turn_number': i + 1,
                    'avg_lsm': round(avg_lsm, 3)
                })
    
    # 4. Save the per-patient LSM (needed for per-patient distribution plots).
    #    Keeps patient_id so plot_metrics can draw patient-level distributions pooled
    #    across models, instead of only the aggregated mean/std.
    if final_rows:
        pdf = pd.DataFrame(final_rows)
        model_tag = os.getenv("PSYCH_MODEL", "")
        prefix = f"{model_tag}_" if model_tag else ""
        pat_csv = os.path.join(CSV_DIR, f'{prefix}lsm_per_patient.csv')
        pdf.to_csv(pat_csv, index=False)
        print(f"Saved per-patient LSM to {pat_csv} ({len(pdf)} rows)")

    # 5. Aggregate by Therapist Mode and Turn (Average across sessions)
    if final_rows:
        df = pd.DataFrame(final_rows)
        print("\nCalculated average LSM by Therapist Mode and Turn:")
        
        unique_modes = df['therapist_mode'].unique()
        
        for mode in unique_modes:
            mode_df = df[df['therapist_mode'] == mode]
            
            # Count unique patients for this mode
            session_count = mode_df['patient_id'].nunique()
            
            # Group by turn and calculate mean and std of avg_lsm
            grouped = mode_df.groupby('turn_number')['avg_lsm'].agg(['mean', 'std', 'count']).reset_index()
            
            # Rename columns for clarity
            grouped.columns = ['turn_number', 'avg_lsm', 'std_lsm', 'count']
            
            # Round for cleaner output
            grouped['avg_lsm'] = grouped['avg_lsm'].round(3)
            grouped['std_lsm'] = grouped['std_lsm'].round(3)
            
            print(f"\n### Therapist Mode: {mode} ({session_count} sessions) ###")
            print(f"{'Turn':<10} | {'Avg LSM':<20} | {'Std LSM':<20} | {'Count':<10}")
            print("-" * 65)
            
            for _, row in grouped.iterrows():
                print(f"{int(row['turn_number']):<10} | {row['avg_lsm']:<20.3f} | {row['std_lsm']:<20.3f} | {int(row['count']):<10}")
                
            # Save to CSV similar to metrics_output_{mode}.csv
            # Model-tagged when PSYCH_MODEL is set so per-model results don't overwrite.
            model_tag = os.getenv("PSYCH_MODEL", "")
            prefix = f"{model_tag}_" if model_tag else ""
            output_csv_name = f'{prefix}lsm_metrics_{mode}.csv'
            output_csv_path = os.path.join(CSV_DIR, output_csv_name)
            grouped.to_csv(output_csv_path, index=False)
            print(f"Saved aggregated metrics to {output_csv_path}")

    else:
        print("No matched session data found to aggregate.")


if __name__ == "__main__":
    main()

