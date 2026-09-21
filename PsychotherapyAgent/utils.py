"""
Utility classes for PsychotherapyAgent.

Provides file management and prompt loading functionality.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging


def sanitize_for_logging(text: str, max_length: int = 50) -> str:
    """
    Sanitize text for logging by replacing non-ASCII characters.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length to return
        
    Returns:
        Sanitized ASCII string
    """
    if not text:
        return ""
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '\u2192': '->',  # Right arrow
        '\u2190': '<-',  # Left arrow
        '\u2191': '^',   # Up arrow
        '\u2193': 'v',   # Down arrow
        '\u2014': '--',  # Em dash
        '\u2013': '-',   # En dash
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
    }
    sanitized = text[:max_length]
    for unicode_char, ascii_replacement in replacements.items():
        sanitized = sanitized.replace(unicode_char, ascii_replacement)
    # Replace any remaining non-ASCII characters
    sanitized = sanitized.encode('ascii', 'replace').decode('ascii')
    return sanitized


###############################################################
###                    FileManager Class                    ###
###############################################################

class FileManager:
    """Handles all file I/O operations for PsychotherapyAgent."""

    def __init__(self):
        """Initialize FileManager"""
        # Base session-save directory. Respects PSYCH_SAVE_DIR so model-specific run folders
        # (e.g. Model_Runs/<model>/Saved_Sessions) can be selected without code changes.
        self.base_dir = os.getenv("PSYCH_SAVE_DIR", "Saved_Sessions")

    def read_file(self, filepath: str) -> str:
        """
        Read and return contents of a file.

        Args:
            filepath: Path to the file to read

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If reading fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file {filepath}: {e}")

    def load_patient_profile(self, filepath: str) -> str:
        """
        Load and return a specific patient profile.

        Args:
            filepath: Path to the patient profile file

        Returns:
            Patient profile content as string
        """
        return self.read_file(filepath)

    def get_patient_id(self, profile_path: Optional[str] = None) -> Optional[str]:
        """
        Extract patient ID from profile path.

        Args:
            profile_path: Path to patient profile file (e.g., "Patient_Profiles/MI/IsabelleFournier.txt")

        Returns:
            Patient ID extracted from filename (without extension), or None if path is invalid
        """
        if not profile_path:
            return None
        try:
            # Extract filename without extension
            filename = os.path.basename(profile_path)
            patient_id = os.path.splitext(filename)[0]
            return patient_id
        except Exception:
            return None

    def _format_judge_proposer_conversation(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format judge-proposer conversation history into readable text.
        
        Args:
            conversation_history: List of turn records from judge-proposer interactions
            
        Returns:
            Formatted text string
        """
        if not conversation_history:
            return "No judge-proposer conversation recorded (single-agent mode or no interactions)."
        
        lines = []
        lines.append("=" * 80)
        lines.append("JUDGE-PROPOSER CONVERSATION HISTORY")
        lines.append("=" * 80)
        lines.append("")
        
        for turn_record in conversation_history:
            turn_num = turn_record.get("turn_number", "?")
            lines.append(f"{'=' * 80}")
            lines.append(f"TURN {turn_num}")
            lines.append(f"{'=' * 80}")
            lines.append("")
            
            # Candidate responses from proposer
            lines.append("CANDIDATE RESPONSES FROM PROPOSER:")
            lines.append("-" * 80)
            candidate_responses = turn_record.get("candidate_responses", [])
            for idx, candidate in enumerate(candidate_responses, 1):
                lines.append(f"\nResponse {idx}:")
                lines.append(candidate)
                lines.append("")
            
            # Judge request
            lines.append("-" * 80)
            lines.append("JUDGE REQUEST:")
            lines.append("-" * 80)
            judge_request = turn_record.get("judge_request", "")
            lines.append(judge_request)
            lines.append("")
            
            # Judge response
            lines.append("-" * 80)
            lines.append("JUDGE RESPONSE:")
            lines.append("-" * 80)
            judge_response = turn_record.get("judge_response", "")
            lines.append(judge_response)
            lines.append("")
            
            # Selected response
            selected_idx = turn_record.get("selected_response_index", "?")
            selected_response = turn_record.get("selected_response", "")
            lines.append("-" * 80)
            lines.append(f"SELECTED RESPONSE #{selected_idx}:")
            lines.append("-" * 80)
            lines.append(selected_response)
            lines.append("")
            lines.append("")
        
        return "\n".join(lines)

    def save_session_data(
            self,
            case_study: str,
            mi: str,
            detailed_feedback: str,
            transcript: str,
            presentation: str,
            base_dir: str = None,
            patient_id: Optional[str] = None,
            judge_proposer_conversation: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Save session data to timestamped folder with UUID.

        Args:
            case_study: Patient case study content
            mi: Formatted MI evaluation
            detailed_feedback: Detailed feedback text
            transcript: Interview transcript
            presentation: Case presentation
            base_dir: Base directory for saving (defaults to self.base_dir, from PSYCH_SAVE_DIR)
            patient_id: Optional patient ID/number to include in folder name
            judge_proposer_conversation: Optional list of judge-proposer conversation turn records

        Returns:
            Path to the created folder
        """
        import uuid

        if base_dir is None:
            base_dir = self.base_dir

        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped subfolder with UUID and optional patient_id
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
        if patient_id is not None:
            folder_name = os.path.join(base_dir, f"session_export_patient_{patient_id}_{timestamp}_{session_uuid}")
        else:
            folder_name = os.path.join(base_dir, f"session_export_{timestamp}_{session_uuid}")
        os.makedirs(folder_name, exist_ok=True)

        files = {
            "case_study.txt": case_study,
            "MI_EVAL.txt": mi,
            "detailed_feedback.txt": detailed_feedback,
            "interview_transcript.txt": transcript
        }
        
        # Add judge-proposer conversation if available
        if judge_proposer_conversation:
            conversation_text = self._format_judge_proposer_conversation(judge_proposer_conversation)
            files["judge_proposer_conversation.txt"] = conversation_text

        # Save each content to a text file
        for filename, content in files.items():
            try:
                file_path = os.path.join(folder_name, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Saved: {filename}")
            except Exception as e:
                logging.info(f"Error saving {filename}: {e}")
                continue

        return folder_name


