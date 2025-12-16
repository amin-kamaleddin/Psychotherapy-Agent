"""
Virtual Therapist module for simulating therapist interactions in therapy training scenarios.
Supports multiple therapy types (MI, CBT, STACER) with configurable therapist profiles.
"""

from typing import List, Literal, Optional
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from utils.api_key_loader import load_openai_api_key

# Type aliases for clarity
MessageList = List[BaseMessage]


@dataclass(frozen=True)
class TherapistProfile:
    """Represents a therapist's profile configuration."""
    name: str
    prompt_content: str


class PromptLoader:
    """Handles loading and caching of prompts from files."""

    _cache: dict[Path, str] = {}

    @classmethod
    def load(cls, filepath: str) -> str:
        """Load prompt from file with caching."""
        path = Path(filepath)

        if path not in cls._cache:
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {filepath}")

            with open(path, 'r', encoding='utf-8') as f:
                cls._cache[path] = f.read()

        return cls._cache[path]

    @classmethod
    def clear_cache(cls):
        """Clear the prompt cache (useful for testing/reloading)."""
        cls._cache.clear()


class TherapistProfileLoader:
    """Handles loading therapist profiles from CSV files."""

    PROFILE_PATHS = "Therapist_Prompts/therapist_prompts.csv"

    PROMPT_PATHS = "Prompts/MI/virtual_therapist_prompt.txt"

    @classmethod
    def load_profile(cls, therapist_index: int = 0) -> TherapistProfile:
        """
        Load a therapist profile for the given therapy type and index.

        Args:
            therapy_type: Type of therapy (MI, CBT, STACER)
            therapist_index: Index of therapist profile to load (default: 0)

        Returns:
            TherapistProfile with loaded data

        Raises:
            FileNotFoundError: If profile files don't exist
        """

        # Load CSV
        profile_path = Path(cls.PROFILE_PATHS)
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile CSV not found: {profile_path}")

        df = pd.read_csv(profile_path, quotechar='"', skipinitialspace=True)

        # Validate index
        if therapist_index >= len(df) or therapist_index < 0:
            raise ValueError(
                f"Therapist index {therapist_index} out of range. "
                f"Available: 0-{len(df) - 1}"
            )

        # Load base prompt and profile
        base_prompt = PromptLoader.load(cls.PROMPT_PATHS)
        therapist_row = df.iloc[therapist_index]

        return TherapistProfile(
            name=therapist_row["Name"],
            prompt_content=base_prompt + therapist_row["Prompt"],
        )


class MessageCleaner:
    """Utility for cleaning and filtering message histories."""

    @staticmethod
    def clean_conversation(messages: MessageList, filter_patient_profile: bool = False) -> MessageList:
        """
        Clean and filter conversation messages based on patient profile access requirements.

        Args:
            messages: List of LangChain message objects
            filter_patient_profile: If True, filters out ALL system messages (used for simple agent index 0)
                                   If False, KEEPS system messages (used for complex agent index 1)
                                   System messages contain patient profile content from Patient_Profiles/MI/*.txt files

        Returns:
            Filtered list of messages:
            - If filter_patient_profile=True: Only HumanMessage/AIMessage (no system messages)
            - If filter_patient_profile=False: All messages including system messages with patient profile
        """
        cleaned = []

        for msg in messages:
            # Handle system messages
            if isinstance(msg, SystemMessage):
                if filter_patient_profile:
                    # Simple agent (index 0): Filter out ALL system messages
                    # System messages in interview_messages contain patient profile content
                    # (from Patient_Profiles/MI/*.txt files appended to virtual patient prompt)
                    # All files from Patient_Profiles/MI/ are patient profiles
                    continue  # Skip all system messages for simple agent
                else:
                    # Complex agent (index 1): Keep system messages so they can see patient profile info
                    cleaned.append(msg)
                continue

            # Skip messages containing large prompts/case files
            if isinstance(msg, HumanMessage) and "# **Prompt:" in msg.content:
                continue

            cleaned.append(msg)

        return cleaned


class VirtualTherapist:
    """
    Simulates a therapist in therapy training scenarios.

    The virtual therapist responds to supervisor feedback and questions based on
    their profile and the conversation history.
    """

    def __init__(
            self,
            api_key: str,
            therapist_index: int = 0,
            prompt_addon: str = "",
            model: str = "gpt-5-mini",
            temperature: float = 0.8
    ):
        """
        Initialize a virtual therapist.

        Args:
            api_key: OpenAI API key
            therapist_index: Index of therapist profile to use (default: 0)
            prompt_addon: Additional text to append to therapist prompt
            model: OpenAI model to use (default: "gpt-5-mini" to match proposer)
            temperature: LLM temperature for response generation (default: 0.8 to match proposer, not used for gpt-5-mini)
        """
        self.api_key = SecretStr(api_key)
        self.therapist_index = therapist_index
        self.model = model
        self.temperature = temperature

        # Initialize LLM
        llm_kwargs = {
            "model": model,
            "api_key": self.api_key
        }
        # Only include temperature for non-gpt-5-mini models
        if model != "gpt-5-mini":
            llm_kwargs["temperature"] = temperature
        self.llm = ChatOpenAI(**llm_kwargs)

        # Load therapist profile
        self.profile: Optional[TherapistProfile] = None
        self.system_prompt: str = ""

        self.profile = TherapistProfileLoader.load_profile(therapist_index)
        self.system_prompt = self.profile.prompt_content + prompt_addon
        print(f"✓ Therapist Profile Loaded: {self.profile.name}")

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
        if not self.profile:
            return {
                "loaded": False,
            }

        return {
            "loaded": True,
            "name": self.profile.name,
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
        if self.profile:
            self.system_prompt = self.profile.prompt_content + addon
        else:
            self.system_prompt = addon

    def __repr__(self) -> str:
        """String representation of the virtual therapist."""
        if self.profile:
            return f"VirtualTherapist(name={self.profile.name})"
        return f"VirtualTherapist(no_profile)"


# Example usage
if __name__ == "__main__":
    # Initialize therapist
    key = load_openai_api_key()
    therapist = VirtualTherapist(
        api_key=key,
        therapist_index=1
    )

    # Get profile info
    print(therapist.get_profile_info())

    # Simulate conversation
    conversation = [
        HumanMessage(content="I think I am depressed")
    ]

    response = therapist.get_response(conversation)
    print(f"Therapist: {response}")