"""
PsychotherapyCore - Shared patient simulation and session management.

This module contains the core functionality for patient simulation,
session lifecycle, and feedback generation that is shared between
single-agent and multi-agent therapist systems.
"""

import warnings
import logging
from datetime import datetime
from typing import Optional
from pydantic import SecretStr

# Suppress langchain verbose deprecation warning
warnings.filterwarnings("ignore", message=".*Importing verbose from langchain root module.*", category=UserWarning)

from .graph_state import LBState

from .llm_manager import LLMManager
from .prompt_manager import PromptManager
from .utils import FileManager, sanitize_for_logging
from langchain.schema import HumanMessage, AIMessage
from .workflows import (
    InitializationWorkflow,
    MessageWorkflow,
    FeedbackWorkflow,
    EvaluatorWorkflow
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("psychotherapy_agent.log"),
        logging.StreamHandler()
    ]
)


class PsychotherapyCore:
    """
    Core class for patient simulation and session management.
    
    Handles:
    - Patient profile loading and initialization
    - Session lifecycle (start, end)
    - Feedback generation
    - Evaluator workflow
    
    This class does NOT handle therapist response generation.
    Therapist responses should be provided by PsychotherapySingleAgent
    or PsychotherapyMultiAgent.
    """
    
    def __init__(self, api_key: str, use_base_prompt: bool = True):
        """
        Initialize PsychotherapyCore.
        
        Args:
            api_key: OpenAI API key
            use_base_prompt: If True, includes base Virtual Therapist prompt when loading CSV prompts.
        """
        self.api_key = SecretStr(api_key)
        
        # Initialize managers
        self.llm_manager = LLMManager(self.api_key)
        self.prompt_manager = PromptManager(use_base_prompt=use_base_prompt)
        self.file_manager = FileManager()
        
        # Initialize workflows (patient and session management only)
        self.init_workflow = InitializationWorkflow(
            self.llm_manager, 
            self.prompt_manager, 
            self.file_manager
        )
        # Message workflow for patient responses only (no therapist generation)
        self.patient_message_workflow = MessageWorkflow(
            self.llm_manager,
            self.prompt_manager,
            use_multi_agent=False,  # No therapist generation, just patient responses
            proposer_index=2,  # Not used when use_multi_agent=False
            judge_index=3  # Not used when use_multi_agent=False
        )
        self.feedback_workflow = FeedbackWorkflow(
            self.llm_manager,
            self.prompt_manager,
            self.file_manager
        )
        self.evaluator_workflow = EvaluatorWorkflow(self.llm_manager)
        
        # Build compiled workflows
        self.init_graph = self.init_workflow.build()
        self.patient_message_graph = self.patient_message_workflow.build()
        self.feedback_graph = self.feedback_workflow.build()
        self.evaluator_graph = self.evaluator_workflow.build()
        
        self.current_state = None
        logging.info("PsychotherapyCore initialized.")
    
    def start_session(
        self,
        case_study: Optional[str] = None,
        selected_profile: Optional[str] = None,
        session_metadata: Optional[dict] = None
    ) -> LBState:
        """
        Start a new session using initialization workflow.

        Args:
            case_study: Pre-loaded case study content (optional)
            selected_profile: Path to specific patient profile (optional)
            session_metadata: Additional metadata for folder naming (optional)
                             Expected keys: 'patient_number', 'therapist_type'

        Returns:
            Initial state after running initialization workflow

        Raises:
            Exception: If initialization workflow fails
        """
        logging.info("Starting new PsychotherapyCore session...")
        
        # Generate session metadata
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        # Build session_id with metadata if provided
        session_id_parts = [f"session_{timestamp}"]
        if session_metadata:
            if 'patient_number' in session_metadata:
                session_id_parts.append(f"patient_{session_metadata['patient_number']}")
            if 'therapist_type' in session_metadata:
                session_id_parts.append(session_metadata['therapist_type'])
        
        session_id = "_".join(session_id_parts)
        
        # Build base metadata
        base_metadata = {
            "session_id": session_id,
            "start_time": now.isoformat(),
            "end_time": None,
            "selected_profile": selected_profile
        }
        
        # Merge with provided session_metadata
        if session_metadata:
            base_metadata.update(session_metadata)
        
        # Build initial state
        initial_state: LBState = {
            "interview_messages": [],
            "evaluator_messages": [],
            "current_phase": "initializing",
            "session_active": False,
            "evaluator_active": False,
            "patient_context": None,
            "presentation": None,
            "evaluation": None,
            "metadata": base_metadata,
            "error": None,
            "export_folder": None
        }
        
        # If case_study provided, initialize patient_context early
        if case_study:
            initial_state["patient_context"] = {
                "case_study": case_study,
                "diagnosis_summary": "",
                "clinical_summary": "",
                "profile_path": selected_profile
            }
        
        try:
            # Run initialization workflow
            result = self.init_graph.invoke(initial_state)
            self.current_state = result
            
            # Log success
            if result.get("error"):
                logging.error(f"Session initialized with errors: {result['error']}")
            else:
                logging.info(f"Session initialized successfully. ID: {session_id}")
            
            return result
        
        except Exception as e:
            logging.error(f"Error starting session: {str(e)}", exc_info=True)
            error_state = initial_state
            error_state["current_phase"] = "error"
            error_state["error"] = f"Initialization failed: {str(e)}"
            return error_state
    
    def send_therapist_message(self, therapist_response: str) -> dict:
        """
        Send a therapist message and get patient response.
        
        This method handles the patient response generation after a therapist message.
        The therapist response should be provided by the caller (from PsychotherapySingleAgent
        or PsychotherapyMultiAgent).
        
        Args:
            therapist_response: The therapist's message/response as a string
            
        Returns:
            Dictionary with patient response and session state
        """
        if not self.current_state or not self.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}
        
        try:
            # Add therapist message to interview_messages
            self.current_state["interview_messages"] = [
                *self.current_state.get("interview_messages"), 
                HumanMessage(content=therapist_response)
            ]
            
            # Use patient message workflow to get patient response
            result = self.patient_message_graph.invoke(self.current_state)
            self.current_state = result
            
            return {
                "patient_response": result.get("ai_response"),
                "session_active": result.get("session_active", True),
                "current_phase": result.get("current_phase", "active_roleplay"),
                "chatlog": result.get("chatlog", [])
            }
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return {"error": str(e)}
    
    def end_session(self):
        """
        End session using feedback workflow.
        
        Public function for use by frontend to use this workflow.
        """
        if not self.current_state:
            return {"error": "No active session"}
        
        try:
            result = self.feedback_graph.invoke(self.current_state)
            self.current_state = result
            
            return result
        except Exception as e:
            logging.error(f"Error ending session: {str(e)}")
            return {"error": str(e)}
    
    def send_evaluator_message(self, user_input: str):
        """Public entrypoint for evaluator chat."""
        if not self.current_state or not self.current_state.get("evaluator_active"):
            logging.error("No active evaluation session.")
            return {"error": "No active session"}
        
        try:
            message_state = {**self.current_state, "user_input": user_input}
            result = self.evaluator_graph.invoke(message_state)
            self.current_state = result
            
            return {
                "ai_response": result.get("ai_response"),
                "evaluator_active": result.get("evaluator_active", True),
                "current_phase": "evaluator_chat",
                "chatlog": result.get("chatlog", [])
            }
        except Exception as e:
            logging.error(f"Error in evaluator chat: {str(e)}")
            return {"error": str(e)}
    
    def add_presentation_message(self, user_input: str):
        """
        Append and save user messages during presentation phase manually.
        
        Public function for frontend use.
        """
        if not self.current_state:
            logging.error("No active session found.")
            return {"error": "No active session"}
        
        presentation = self.current_state.get("presentation", [])
        presentation.append(user_input)
        
        self.current_state["presentation"] = presentation
        logging.info(f"Presentation input added: {sanitize_for_logging(user_input)}...")
        
        return {"presentation": presentation, "status": "updated"}
    
    # Utility methods
    def get_session_state(self):
        """Get current session state"""
        return self.current_state
    
    def reset_session(self):
        """Reset current session state"""
        self.current_state = None
        logging.info("Session state reset.")
    
    def get_session_statistics(self):
        """Get current session statistics"""
        if not self.current_state:
            return {}
        
        chatlog = self.current_state.get("chatlog", [])
        conversation_messages = [m for m in chatlog if m.get("role") in ["user", "assistant"]]
        
        return {
            "total_messages": len(conversation_messages),
            "total_exchanges": len(conversation_messages) // 2,
            "current_phase": self.current_state.get("current_phase"),
            "session_active": self.current_state.get("session_active", False),
            "session_metadata": self.current_state.get("metadata", {})
        }
    
    def export_session_data(self):
        """Export session data for analysis"""
        if not self.current_state:
            return None
        
        return {
            "session_metadata": self.current_state.get("metadata", {}),
            "case_study": self.current_state.get("patient_context", {}).get("case_study"),
            "clinical_summary": self.current_state.get("patient_context", {}).get("clinical_summary"),
            "diagnosis_summary": self.current_state.get("patient_context", {}).get("diagnosis_summary"),
            "chatlog": self.current_state.get("chatlog", []),
            "feedback": self.current_state.get("feedback"),
            "session_statistics": self.get_session_statistics(),
            "mi": self.current_state.get("mi"),
        }
    
    def load_patient_profile(self, filepath: str) -> str:
        """Load a specific patient profile."""
        return self.file_manager.load_patient_profile(filepath)

