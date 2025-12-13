"""
Main PsychotherapyAgent Orchestrator.

Coordinates all workflows and provides the public API.
"""

import logging
from datetime import datetime
from typing import Optional
from pydantic import SecretStr
from langchain.schema import HumanMessage

from .graph_state import LBState, PatientContext

from .llm_manager import LLMManager
from .prompt_manager import PromptManager
from .utils import FileManager
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


class PsychotherapyAgent:
    """
    Main orchestrator class for PsychotherapyAgent virtual patient simulation.
    
    Coordinates four main workflows:
    1. Initialization: Setup session and patient profile
    2. Interview: Virtual patient conversation (with optional multi-agent therapist)
    3. Feedback: Generate evaluation and MI form
    4. Evaluator: Q&A about evaluation
    """
    
    # Mode constants
    STACER_MODE = "STACER"
    MI_MODE = "MI"
    CBT_MODE = "CBT"
    
    def __init__(self, api_key: str, use_multi_agent: bool = True):
        """
        Initialize PsychotherapyAgent.
        
        Args:
            api_key: OpenAI API key
            use_multi_agent: If True, uses proposer+judge multi-agent system for therapist responses.
                           If False, uses original single-agent mode (direct user input).
        """
        self.api_key = SecretStr(api_key)
        self.use_multi_agent = use_multi_agent
        
        # Initialize managers
        self.llm_manager = LLMManager(self.api_key)
        self.prompt_manager = PromptManager()
        self.file_manager = FileManager()
        
        # Initialize workflows
        self.init_workflow = InitializationWorkflow(
            self.llm_manager, 
            self.prompt_manager, 
            self.file_manager
        )
        self.message_workflow = MessageWorkflow(
            self.llm_manager,
            self.prompt_manager,
            use_multi_agent=self.use_multi_agent,
            proposer_index=2,
            judge_index=3
        )
        self.feedback_workflow = FeedbackWorkflow(
            self.llm_manager,
            self.prompt_manager,
            self.file_manager
        )
        self.evaluator_workflow = EvaluatorWorkflow(self.llm_manager)
        
        # Build compiled workflows
        self.init_graph = self.init_workflow.build()
        self.message_graph = self.message_workflow.build()
        self.feedback_graph = self.feedback_workflow.build()
        self.evaluator_graph = self.evaluator_workflow.build()
        
        self.current_state = None
        mode_str = "multi-agent" if use_multi_agent else "single-agent (original)"
        logging.info(f"PsychotherapyAgent initialized with {mode_str} mode.")
    
    def set_multi_agent_therapist_profiles(self, proposer_index: int, judge_index: int):
        """
        Set therapist profile indices for multi-agent mode.
        
        Args:
            proposer_index: Index in CSV for proposer agent (default: 2)
            judge_index: Index in CSV for judge agent (default: 3)
        """
        self.message_workflow.proposer_index = proposer_index
        self.message_workflow.judge_index = judge_index
        # Rebuild workflow with new indices
        self.message_graph = self.message_workflow.build()
        logging.info(f"Multi-agent therapist profiles set: proposer={proposer_index}, judge={judge_index}")
    
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
        logging.info("Starting new PsychotherapyAgent session...")
        
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
    
    def send_message(self, user_input: str) -> dict:
        """
        Send a user message using message processing workflow.
        
        Public function for use by frontend to use this workflow.
        """
        if not self.current_state or not self.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}
        
        try:
            # Create state for message processing
            self.current_state["interview_messages"] = [
                *self.current_state.get("interview_messages"), 
                HumanMessage(content=user_input)
            ]
            result = self.message_graph.invoke(self.current_state)
            self.current_state = result
            
            return {
                "ai_response": result.get("ai_response"),
                "session_active": result.get("session_active", True),
                "current_phase": "active_roleplay",
                "chatlog": result.get("chatlog", [])
            }
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
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
        logging.info(f"Presentation input added: {user_input[:50]}...")
        
        return {"presentation": presentation, "status": "updated"}
    
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
