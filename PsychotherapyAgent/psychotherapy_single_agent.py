"""
PsychotherapySingleAgent module for simulating single-agent therapist interactions in therapy training scenarios.
Supports MI (Motivational Interviewing) with configurable therapist profiles.

This class combines:
- Therapist response generation (single-agent modes: simple/complex)
- Patient simulation and session management (via PsychotherapyCore)
"""

import warnings
import logging
from typing import List, Optional

import pandas as pd
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Suppress langchain verbose deprecation warning
warnings.filterwarnings("ignore", message=".*Importing verbose from langchain root module.*", category=UserWarning)

from .psychotherapy_core import PsychotherapyCore
from .prompt_manager import PromptManager

# Type aliases for clarity
MessageList = List[BaseMessage]


class MessageCleaner:
    """Utility for cleaning and filtering message histories."""

    @staticmethod
    def clean_conversation(messages: MessageList, filter_patient_profile: bool = False) -> MessageList:
        """
        Clean and filter conversation messages based on patient profile access requirements.

        Args:
            messages: List of LangChain message objects
            filter_patient_profile: If True, filters out ALL system messages (used for simple agent index 0)
                                   If False, extracts patient profile from SystemMessage (used for complex agent index 1)
                                   System messages contain patient profile content from Patient_Profiles/MI/*.txt files

        Returns:
            Filtered list of messages:
            - If filter_patient_profile=True: Only HumanMessage/AIMessage (no system messages)
            - If filter_patient_profile=False: Patient profile as SystemMessage + HumanMessage/AIMessage
        """
        cleaned = []

        for i, msg in enumerate(messages):
            # Handle system messages
            if isinstance(msg, SystemMessage):
                if filter_patient_profile:
                    # Simple agent (index 0): Filter out ALL system messages
                    continue  # Skip all system messages for simple agent
                else:
                    # Complex agent (index 1): Extract patient profile only (not Virtual Patient role instruction)
                    if i == 0:  # First SystemMessage contains Virtual Patient prompt + patient profile
                        content = msg.content
                        # Extract patient profile part (usually starts with "# **Prompt:")
                        if "# **Prompt:" in content:
                            profile_start = content.find("# **Prompt:")
                            patient_profile = content[profile_start:]
                            # Add as background context, not as role instruction
                            cleaned.append(SystemMessage(content=f"PATIENT PROFILE (for reference):\n\n{patient_profile}"))
                        # Skip the original SystemMessage with Virtual Patient role
                        continue
                    else:
                        # Keep other system messages if any
                        cleaned.append(msg)
                continue

            # Skip messages containing large prompts/case files
            if isinstance(msg, HumanMessage) and "# **Prompt:" in msg.content:
                continue

            cleaned.append(msg)

        return cleaned


class PsychotherapySingleAgent:
    """
    Complete single-agent therapist system for therapy training scenarios.
    
    This class combines:
    - Therapist response generation (simple/complex modes)
    - Patient simulation and session management (via PsychotherapyCore)
    
    Used for simple and complex single-agent modes.
    
    Usage:
        therapist = PsychotherapySingleAgent(
            api_key="your-key",
            therapist_index=0,  # 0=simple, 1=complex
            use_base_prompt=True
        )
        therapist.start_session(case_study=case_study)
        result = therapist.send_message()  # Therapist generates response, patient responds
        therapist.end_session()
    """

    def __init__(
            self,
            api_key: str,
            therapist_index: int = 0,
            prompt_addon: str = "",
            model: str = "gpt-5-mini",
            temperature: float = 0.8,
            use_base_prompt: bool = True
    ):
        """
        Initialize a single-agent therapist with session management.

        Args:
            api_key: OpenAI API key
            therapist_index: Index of therapist profile to use (default: 0)
            prompt_addon: Additional text to append to therapist prompt
            model: OpenAI model to use (default: "gpt-5-mini" to match proposer)
            temperature: LLM temperature for response generation (default: 0.8 to match proposer, not used for gpt-5-mini)
            use_base_prompt: If True, includes base Virtual Therapist prompt when loading CSV prompts.
                           If False, uses only CSV prompt without base prompt.
        """
        self.api_key = SecretStr(api_key)
        self.therapist_index = therapist_index
        self.model = model
        self.temperature = temperature
        self.use_base_prompt = use_base_prompt

        # Initialize patient session management
        self.core = PsychotherapyCore(api_key, use_base_prompt=use_base_prompt)

        # Initialize prompt manager (same as multi-agent uses)
        self.prompt_manager = PromptManager(use_base_prompt=use_base_prompt)

        # Initialize LLM
        llm_kwargs = {
            "model": model,
            "api_key": self.api_key
        }
        # Only include temperature for non-gpt-5-mini models
        if model != "gpt-5-mini":
            llm_kwargs["temperature"] = temperature
        self.llm = ChatOpenAI(**llm_kwargs)

        # Load therapist prompt using PromptManager (same as multi-agent)
        prompt_content = self.prompt_manager.load_therapist_prompt_from_csv(therapist_index)
        self.system_prompt = prompt_content + prompt_addon
        
        # Get therapist name from CSV for logging
        try:
            df = pd.read_csv("Therapist_Prompts/therapist_prompts.csv", quotechar='"', skipinitialspace=True)
            if therapist_index < len(df):
                therapist_name = df.iloc[therapist_index]["Name"]
                logging.info(f"Therapist Profile Loaded: {therapist_name}")
        except Exception as e:
            logging.warning(f"Could not load therapist name: {e}")
        
        logging.info(f"PsychotherapySingleAgent initialized with therapist index {therapist_index}")

    def get_response(self, conversation: MessageList) -> str:
        """
        Generate therapist response based on conversation history.

        Args:
            conversation: List of LangChain message objects representing the conversation.
                         This includes SystemMessages that may contain patient profile content
                         (the .txt file content from Patient_Profiles/MI/ folder).

        Returns:
            Therapist's response as a string

        Raises:
            Exception: If LLM invocation fails
        """
        # For simple agent (index 0), filter out patient profile (case study from .txt files)
        # Complex agent (index 1) should have explicit access to patient profile like proposer
        if self.therapist_index == 0:
            # Simple agent: Filter out patient profile system messages
            filter_patient_profile = True
            cleaned_conversation = MessageCleaner.clean_conversation(
                conversation, 
                filter_patient_profile=filter_patient_profile
            )
            # Build full message list with system prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                *cleaned_conversation
            ]
        else:
            # Complex agent (index 1): Use conversation directly like proposer
            # This explicitly includes patient profile SystemMessage from interview_messages
            messages = [
                SystemMessage(content=self.system_prompt),
                *conversation  # Directly includes patient profile SystemMessage
            ]

        # Generate response
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"Failed to generate therapist response: {str(e)}")

    def get_profile_info(self) -> dict:
        """
        Get information about the loaded therapist profile.

        Returns:
            Dictionary with profile details
        """
        try:
            df = pd.read_csv("Therapist_Prompts/therapist_prompts.csv", quotechar='"', skipinitialspace=True)
            if self.therapist_index < len(df):
                therapist_name = df.iloc[self.therapist_index]["Name"]
                return {
                    "loaded": True,
                    "name": therapist_name,
                    "therapist_index": self.therapist_index,
                    "model": self.model,
                    "temperature": self.temperature
                }
        except Exception as e:
            logging.warning(f"Could not load therapist profile info: {e}")
        
        return {
            "loaded": False,
            "therapist_index": self.therapist_index,
            "model": self.model,
            "temperature": self.temperature
        }

    def update_prompt_addon(self, addon: str):
        """
        Update the prompt addon dynamically.

        Args:
            addon: Additional text to append to base prompt
        """
        # Reload base prompt and add new addon
        prompt_content = self.prompt_manager.load_therapist_prompt_from_csv(self.therapist_index)
        self.system_prompt = prompt_content + addon

    # Session management methods (delegate to PsychotherapyCore)
    def start_session(
        self,
        case_study: Optional[str] = None,
        selected_profile: Optional[str] = None,
        session_metadata: Optional[dict] = None
    ):
        """Start a new session."""
        return self.core.start_session(case_study, selected_profile, session_metadata)
    
    def send_message(self, user_input: Optional[str] = None) -> dict:
        """
        Send a message and get response.
        
        If user_input is provided, it's used as the therapist's message.
        If user_input is None, the therapist generates a response automatically.
        
        Args:
            user_input: Optional therapist message. If None, therapist generates response.
            
        Returns:
            Dictionary with therapist response, patient response, and session state
        """
        # Get current interview messages
        current_state = self.core.get_session_state()
        interview_messages = current_state.get("interview_messages", [])
        
        # Generate therapist response if not provided
        if user_input is None:
            therapist_response = self.get_response(interview_messages)
        else:
            therapist_response = user_input
        
        # Send therapist message and get patient response
        result = self.core.send_therapist_message(therapist_response)
        
        return {
            "therapist_response": therapist_response,
            "patient_response": result.get("patient_response"),
            "session_active": result.get("session_active", True),
            "current_phase": result.get("current_phase"),
            "chatlog": result.get("chatlog", [])
        }
    
    def end_session(self):
        """End session and generate feedback."""
        return self.core.end_session()
    
    def get_session_state(self):
        """Get current session state."""
        return self.core.get_session_state()
    
    def reset_session(self):
        """Reset current session state."""
        return self.core.reset_session()
    
    def get_session_statistics(self):
        """Get current session statistics."""
        return self.core.get_session_statistics()
    
    def export_session_data(self):
        """Export session data for analysis."""
        return self.core.export_session_data()
    
    def load_patient_profile(self, filepath: str) -> str:
        """Load a specific patient profile."""
        return self.core.load_patient_profile(filepath)
    
    def send_evaluator_message(self, user_input: str):
        """Send message to evaluator."""
        return self.core.send_evaluator_message(user_input)
    
    def add_presentation_message(self, user_input: str):
        """Add presentation message."""
        return self.core.add_presentation_message(user_input)
    
    @property
    def current_state(self):
        """Access to current state for backward compatibility."""
        return self.core.current_state
    
    @current_state.setter
    def current_state(self, value):
        """Setter for current state for backward compatibility."""
        self.core.current_state = value

    def __repr__(self) -> str:
        """String representation of the single-agent therapist."""
        profile_info = self.get_profile_info()
        if profile_info.get("loaded"):
            return f"PsychotherapySingleAgent(name={profile_info['name']})"
        return f"PsychotherapySingleAgent(index={self.therapist_index})"


