"""
Graph State Definitions for LearningBuddy Modular.

Defines the TypedDict structures for LangGraph state management.
"""

from typing import TypedDict, Optional, List, Dict, Any
from langchain.schema import BaseMessage


class PatientContext(TypedDict, total=False):
    """Patient context information."""
    case_study: str
    diagnosis_summary: str
    clinical_summary: str
    profile_path: Optional[str]


class LBState(TypedDict, total=False):
    """
    LangGraph state schema for LearningBuddy.
    
    All fields are optional to allow incremental state updates.
    """
    # Conversational state
    interview_messages: List[BaseMessage]
    evaluator_messages: List[BaseMessage]
    
    # Session control
    current_phase: str
    session_active: bool
    evaluator_active: bool
    
    # Nested data structures
    patient_context: Optional[PatientContext]
    presentation: Optional[List[str]]
    evaluation: Optional[Dict[str, Any]]
    
    # Session metadata
    metadata: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    error_message: Optional[str]
    
    # Export path
    export_folder: Optional[str]
    folder_name: Optional[str]
    
    # Multi-agent specific
    therapist_candidates: Optional[List[str]]
    selected_therapist_response: Optional[str]
    
    # Feedback specific
    feedback_messages: Optional[List[BaseMessage]]
    mi: Optional[str]
    formatted_transcript: Optional[str]
    formatted_presentation: Optional[str]
    
    # Evaluator specific
    evaluator_chatlog: Optional[List[Dict[str, Any]]]
    user_input: Optional[str]
    ai_response: Optional[str]
    chatlog: Optional[List[Dict[str, Any]]]

