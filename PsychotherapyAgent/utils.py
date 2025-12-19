"""
Utility classes for PsychotherapyAgent.

Provides file management and prompt loading functionality.
"""

import os
import random
from datetime import datetime
from typing import Optional
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
        pass

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

    def load_random_case_profile(self, directory: str = 'Patient_Profiles') -> str:
        """
        Load a random patient profile from the appropriate directory.

        Args:
            directory: Base directory containing patient profiles

        Returns:
            Random patient profile content as string

        Raises:
            FileNotFoundError: If no profiles found in directory
        """
        directory += "/MI"
        files = [f for f in os.listdir(directory)]
        if not files:
            raise FileNotFoundError("No patient profiles found in the directory.")
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        return self.load_patient_profile(full_path)

    def calculate_session_duration(self, start_time_iso: str) -> float:
        """
        Calculate session duration in minutes.

        Args:
            start_time_iso: ISO format start time string

        Returns:
            Duration in minutes as float
        """
        try:
            start_time = datetime.fromisoformat(start_time_iso)
            duration = datetime.now() - start_time
            return duration.total_seconds() / 60
        except:
            return 0.0

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

    def save_session_data(
            self,
            case_study: str,
            mi: str,
            detailed_feedback: str,
            transcript: str,
            presentation: str,
            base_dir: str = "Saved_Sessions",
            patient_id: Optional[str] = None
    ) -> str:
        """
        Save session data to timestamped folder.

        Args:
            case_study: Patient case study content
            mi: Formatted MI evaluation
            detailed_feedback: Detailed feedback text
            transcript: Interview transcript
            presentation: Case presentation
            base_dir: Base directory for saving
            patient_id: Optional patient ID/number to include in folder name

        Returns:
            Path to the created folder
        """
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped subfolder with optional patient_id
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        if patient_id:
            folder_name = os.path.join(base_dir, f"session_export_patient_{patient_id}_{timestamp}")
        else:
            folder_name = os.path.join(base_dir, f"session_export_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        files = {
            "case_study.txt": case_study,
            "MI_EVAL.txt": mi,
            "detailed_feedback.txt": detailed_feedback,
            "interview_transcript.txt": transcript
        }

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


###############################################################
###                   PromptLoader Class                    ###
###############################################################

class PromptLoader:
    """Handles loading of prompts and context files."""

    def __init__(self):
        """Initialize PromptLoader."""
        self.file_manager = FileManager()

        # Define prompt paths
        self.prompt_paths = {
            'Clinical Summary': 'Prompts/MI/clinical_summary_prompt.txt',
            'Virtual Patient': 'Prompts/MI/virtual_patient_prompt.txt',
            'Verification': 'Prompts/MI/patient_profile_verification_prompt.txt',
            'Feedback': 'Prompts/MI/feedback_agent_prompt.txt',
            'MI Prompt': 'Prompts/MI/MI_form_prompt.txt',
            'Virtual Evaluator': 'Prompts/MI/virtual_evaluator_prompt.txt',
            'Virtual Therapist': 'Prompts/MI/virtual_therapist_prompt.txt',
        }

        # Define context paths
        self.context_paths = {
            'Feedback Criteria': 'Context/MI/feedback_criteria_MI_5point.txt',
            'Evaluation Form': 'Context/MI/MI_EVAL.txt',
        }

    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt file.

        Args:
            prompt_name: Name of the prompt to load (e.g., 'Clinical Summary')

        Returns:
            Prompt content as string

        Raises:
            FileNotFoundError: If prompt name is invalid
        """
        if prompt_name not in self.prompt_paths:
            raise FileNotFoundError(f"Invalid prompt name: {prompt_name}")

        filepath = self.prompt_paths[prompt_name]
        return self.file_manager.read_file(filepath)

    def load_context(self, context_name: str) -> str:
        """
        Load a context file.

        Args:
            context_name: Name of the context to load (e.g., 'Feedback Criteria')

        Returns:
            Context content as string

        Raises:
            FileNotFoundError: If context name is invalid
        """
        if context_name not in self.context_paths:
            raise FileNotFoundError(f"Invalid context name: {context_name}")

        filepath = self.context_paths[context_name]
        return self.file_manager.read_file(filepath)

