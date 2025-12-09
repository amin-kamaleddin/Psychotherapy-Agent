from enum import Enum
from typing import TypedDict, Annotated, Literal, Optional, Any

from langgraph.graph import add_messages



# Standard message format throughout the app
class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


# Session metadata - separate from conversational state
class SessionMetadata(TypedDict):
    session_id: str
    start_time: str
    end_time: Optional[str]
    selected_profile: Optional[str]


# Patient context - all patient-related data together
class PatientContext(TypedDict):
    case_study: str
    diagnosis_summary: Optional[str]
    clinical_summary: Optional[str]
    profile_path: Optional[str]


# Presentation data - keeping track of user's case presentation
class PresentationData(TypedDict):
    sections: list[str]  # List of presentation segments
    completed: bool


# Evaluation results - all feedback-related data together
class EvaluationResults(TypedDict):
    detailed_feedback: str
    formatted_stacer: str
    feedback_generated: bool


# Main state - much cleaner!
class LBState(TypedDict):
    # Conversational state - uses add_messages reducer
    interview_messages: Annotated[list[Message], add_messages]
    feedback_messages: Annotated[list[Message], add_messages]
    evaluator_messages: Annotated[list[Message], add_messages]

    cod_stacer: Optional[str]
    formatted_transcript: Optional[str]
    formatted_presentation: Optional[str]

    # Session control
    current_phase: str
    session_active: bool
    evaluator_active: bool

    # Nested data structures
    patient_context: Optional[PatientContext]
    presentation: Optional[PresentationData]
    evaluation: Optional[EvaluationResults]
    metadata: SessionMetadata

    # Error handling
    error: Optional[str]

    # Export path (only set at completion)
    folder_name: Optional[str]
    __extra_items__: Any
