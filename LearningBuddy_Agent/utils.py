import os
import random
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import logging

from LearningBuddy import LearningBuddy


###############################################################
###                    FileManager Class                    ###
###############################################################

class FileManager:
    """Handles all file I/O operations for LearningBuddy."""

    def __init__(self):
        """
        Initialize FileManager
        """
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
            raise FileNotFoundError(f"file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

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
        # Select Patients available
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

    def save_session_data(
            self,
            case_study: str,
            mi: str,
            detailed_feedback: str,
            transcript: str,
            presentation: str,
            base_dir: str = "Saved_Sessions"
    ) -> str:
        """
        Save all session data to timestamped folder.

        Args:
            case_study: Patient case study content
            mi: Completed MI evaluation form
            detailed_feedback: Detailed feedback text
            transcript: Formatted interview transcript
            presentation: Formatted presentation transcript
            base_dir: Base directory for saving sessions

        Returns:
            Path to the created folder
        """
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = os.path.join(base_dir, f"session_export_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        # Define files to save 
        files = {
            "case_study.txt": case_study,
            "MI_EVAL.txt": mi,
            "detailed_feedback.txt": detailed_feedback,
            "interview_transcript.txt": transcript
        }

        # Save each file
        for filename, content in files.items():
            try:
                file_path = os.path.join(folder_name, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Saved: {filename}")
            except Exception as e:
                logging.error(f"Error saving {filename}: {e}")
                continue

        logging.info(f"Session data saved to: {folder_name}")
        return folder_name


###############################################################
###                   PromptLoader Class                    ###
###############################################################

class PromptLoader:
    """Handles loading of prompts and context files."""

    def __init__(self):
        """
        Initialize PromptLoader.
        """
        self.file_manager = FileManager()

        # Define prompt paths
        self.prompt_paths = {
                'Clinical Summary': 'Prompts/MI/clinical_summary_prompt.txt',
                'Virtual Patient': 'Prompts/MI/virtual_patient_prompt.txt',
                'Verification': 'Prompts/MI/patient_profile_verification_prompt.txt',
                'Feedback': 'Prompts/MI/feedback_agent_prompt.txt',
                'MI Prompt': 'Prompts/MI/MI_form_prompt.txt',
                'Virtual Evaluator': 'Prompts/MI/virtual_evaluator_prompt.txt',
        }

        # Define context paths
        self.context_paths = {
            'Feedback Criteria': 'Context/MI/feedback_criteria_MI_5point.txt',
            'Evaluation Form': 'Context/MI/MI_EVAL.txt',
        }

    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt file

        Args:
            prompt_name: Name of the prompt to load (e.g., 'Clinical Summary')

        Returns:
            Prompt content as string
        """
        if prompt_name not in self.prompt_paths:
            raise FileNotFoundError(
                f"Invalid prompt name for: {prompt_name}"
            )

        filepath = self.prompt_paths[prompt_name]
        return self.file_manager.read_file(filepath)

    def load_context(self, context_name: str) -> str:
        """
        Load a context file

        Args:
            context_name: Name of the context to load (e.g., 'Feedback Criteria')

        Returns:
            Context content as string
        """

        if context_name not in self.context_paths:
            raise FileNotFoundError(
                f"Invalid context name for: {context_name}"
            )

        filepath = self.context_paths[context_name]
        return self.file_manager.read_file(filepath)


###############################################################
###                  Formatting Functions                   ###
###############################################################

def format_interview_messages(messages: List) -> Tuple[List, str]:
    """
    Format interview messages for feedback generation.

    Filters out system messages and formats conversation into:
    1. List of LangChain message objects for LLM
    2. Plain text transcript string

    Args:
        messages: List of LangChain message objects (SystemMessage, HumanMessage, AIMessage)

    Returns:
        Tuple of (formatted_langchain_messages, formatted_transcript_string)
    """
    # Filter conversation messages (exclude system messages)
    conversation_messages = [
        m for m in messages
        if (isinstance(m, HumanMessage) or isinstance(m, AIMessage)) and m.content
    ]

    # Format for LangChain (with role prefixes)
    formatted_messages = []
    for m in conversation_messages:
        prefix = "PSYCHIATRIST: " if isinstance(m, AIMessage) else "PATIENT: "
        content = prefix + m.content
        msg_cls = AIMessage if isinstance(m, AIMessage) else HumanMessage
        formatted_messages.append(msg_cls(content=content))

    # Format as plain text transcript
    formatted_lines = []
    for m in conversation_messages:
        prefix = "PSYCHIATRIST: " if isinstance(m, AIMessage) else "PATIENT: "
        formatted_lines.append(prefix + m.content)

    formatted_transcript = "\n".join(formatted_lines)

    return formatted_messages, formatted_transcript


def format_presentation(presentation: List[str]) -> str:
    """
    Format presentation list into a single formatted string.

    Args:
        presentation: List of presentation text entries

    Returns:
        Formatted presentation string
    """
    if not presentation:
        return ""

    formatted_parts = [
        f"Psychiatrist Presentation: {p.strip()}"
        for p in presentation
        if p.strip()
    ]

    return "\n".join(formatted_parts)


def build_langchain_messages(chatlog: List[Dict]) -> List:
    """
    Convert chatlog dictionary format to LangChain message objects.

    Args:
        chatlog: List of message dictionaries with 'type' and 'content' keys

    Returns:
        List of LangChain message objects (SystemMessage, HumanMessage, AIMessage)
    """
    langchain_messages = []

    for msg in chatlog:
        msg_type = msg.get('type')
        content = msg.get('content', '')

        if msg_type == 'system':
            langchain_messages.append(SystemMessage(content=content))
        elif msg_type == 'human':
            langchain_messages.append(HumanMessage(content=content))
        elif msg_type == 'ai':
            langchain_messages.append(AIMessage(content=content))

    return langchain_messages
