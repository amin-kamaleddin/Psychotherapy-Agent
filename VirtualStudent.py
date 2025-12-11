"""
Virtual Student module for simulating student interactions in therapy training scenarios.
Supports multiple therapy types (MI, CBT, STACER) with configurable student profiles.
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
class StudentProfile:
    """Represents a student's profile configuration."""
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


class StudentProfileLoader:
    """Handles loading student profiles from CSV files."""

    PROFILE_PATHS = "Student_Profiles/AK_StudentAgent_10Profiles - Prompts for MI Students.csv"

    PROMPT_PATHS = "Prompts/MI/virtual_student_prompt.txt"

    @classmethod
    def load_profile(cls, student_index: int = 0) -> StudentProfile:
        """
        Load a student profile for the given therapy type and index.

        Args:
            therapy_type: Type of therapy (MI, CBT, STACER)
            student_index: Index of student profile to load (default: 0)

        Returns:
            StudentProfile with loaded data

        Raises:
            FileNotFoundError: If profile files don't exist
        """

        # Load CSV
        profile_path = Path(cls.PROFILE_PATHS)
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile CSV not found: {profile_path}")

        df = pd.read_csv(profile_path, quotechar='"', skipinitialspace=True)

        # Validate index
        if student_index >= len(df) or student_index < 0:
            raise ValueError(
                f"Student index {student_index} out of range. "
                f"Available: 0-{len(df) - 1}"
            )

        # Load base prompt and profile
        base_prompt = PromptLoader.load(cls.PROMPT_PATHS)
        student_row = df.iloc[student_index]

        return StudentProfile(
            name=student_row["Name"],
            prompt_content=base_prompt + student_row["Prompt"],
        )


class MessageCleaner:
    """Utility for cleaning and filtering message histories."""

    @staticmethod
    def clean_conversation(messages: MessageList) -> MessageList:
        """
        Remove system messages and large prompt content from conversation history.

        Args:
            messages: List of LangChain message objects

        Returns:
            Filtered list of messages
        """
        cleaned = []

        for msg in messages:
            # Skip system messages
            if isinstance(msg, SystemMessage):
                continue

            # Skip messages containing large prompts/case files
            if isinstance(msg, HumanMessage) and "# **Prompt:" in msg.content:
                continue

            cleaned.append(msg)

        return cleaned


class VirtualStudent:
    """
    Simulates a student in therapy training scenarios.

    The virtual student responds to supervisor feedback and questions based on
    their profile and the conversation history.
    """

    def __init__(
            self,
            api_key: str,
            student_index: int = 0,
            prompt_addon: str = "",
            model: str = "gpt-4o-mini",
            temperature: float = 0.5
    ):
        """
        Initialize a virtual student.

        Args:
            api_key: OpenAI API key
            student_index: Index of student profile to use (default: 0)
            prompt_addon: Additional text to append to student prompt
            model: OpenAI model to use
            temperature: LLM temperature for response generation
        """
        self.api_key = SecretStr(api_key)
        self.student_index = student_index
        self.model = model
        self.temperature = temperature

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self.api_key
        )

        # Load student profile
        self.profile: Optional[StudentProfile] = None
        self.system_prompt: str = ""

        self.profile = StudentProfileLoader.load_profile('MI', student_index)
        self.system_prompt = self.profile.prompt_content + prompt_addon
        print(f"✓ Student Profile Loaded: {self.profile.name}")

    def get_response(self, conversation: MessageList) -> str:
        """
        Generate student response based on conversation history.

        Args:
            conversation: List of LangChain message objects representing the conversation

        Returns:
            Student's response as a string

        Raises:
            Exception: If LLM invocation fails
        """
        # Clean the conversation (remove system messages, large prompts)
        cleaned_conversation = MessageCleaner.clean_conversation(conversation)

        # Build full message list with system prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            *cleaned_conversation
        ]

        # Generate response
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"Failed to generate student response: {str(e)}")

    def get_profile_info(self) -> dict:
        """
        Get information about the loaded student profile.

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
            "student_index": self.student_index,
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
        """String representation of the virtual student."""
        if self.profile:
            return f"VirtualStudent(name={self.profile.name})"
        return f"VirtualStudent(no_profile)"


# Example usage
if __name__ == "__main__":
    # Initialize student
    key = load_openai_api_key()
    student = VirtualStudent(
        api_key=key,
        student_index=1
    )

    # Get profile info
    print(student.get_profile_info())

    # Simulate conversation
    conversation = [
        HumanMessage(content="I think I am depressed")
    ]

    response = student.get_response(conversation)
    print(f"Student: {response}")