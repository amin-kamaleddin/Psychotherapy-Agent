"""
PsychotherapyMultiAgent - Multi-agent therapist system (proposer + judge).

This module handles the multi-agent therapist logic (proposer generates 3 candidates,
judge selects best one). Patient simulation and session management are handled by
PsychotherapyCore.
"""

import warnings
import logging
from typing import Optional
from pydantic import SecretStr
from langchain.schema import HumanMessage, AIMessage

# Suppress langchain verbose deprecation warning
warnings.filterwarnings("ignore", message=".*Importing verbose from langchain root module.*", category=UserWarning)

from .graph_state import LBState
from .psychotherapy_core import PsychotherapyCore
from .llm_manager import LLMManager
from .prompt_manager import PromptManager
from .workflows import MessageWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("psychotherapy_agent.log"),
        logging.StreamHandler()
    ]
)


class PsychotherapyMultiAgent:
    """
    Multi-agent therapist system using proposer + judge architecture.
    
    Uses PsychotherapyCore for patient simulation and session management.
    Adds multi-agent therapist logic: proposer generates 3 candidate responses,
    judge selects the best one.
    """
    
    def __init__(self, api_key: str, use_base_prompt: bool = True):
        """
        Initialize PsychotherapyMultiAgent.
        
        Args:
            api_key: OpenAI API key
            use_base_prompt: If True, includes base Virtual Therapist prompt when loading CSV prompts.
                           If False, uses only CSV prompt without base prompt.
        """
        # Initialize core for patient simulation and session management
        self.core = PsychotherapyCore(api_key, use_base_prompt=use_base_prompt)
        
        # Initialize multi-agent therapist workflow
        self.llm_manager = LLMManager(self.core.api_key)
        self.prompt_manager = PromptManager(use_base_prompt=use_base_prompt)
        
        self.message_workflow = MessageWorkflow(
            self.llm_manager,
            self.prompt_manager,
            use_multi_agent=True,
            proposer_index=2,
            judge_index=3
        )
        
        # Build compiled multi-agent message workflow
        self.message_graph = self.message_workflow.build()
        
        logging.info("PsychotherapyMultiAgent initialized with multi-agent therapist system.")
    
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
        """Start a new session using core initialization."""
        return self.core.start_session(case_study, selected_profile, session_metadata)
    
    def send_message(self, user_input: Optional[str] = None) -> dict:
        """
        Send a message using multi-agent therapist system (proposer + judge) and get patient response.
        
        The multi-agent workflow:
        1. Proposer generates 3 candidate therapist responses from conversation history
        2. Judge selects the best response
        3. Patient responds to the selected therapist message
        
        Args:
            user_input: Optional. In multi-agent mode, this should be None as the workflow
                       generates therapist responses automatically from conversation history.
        
        Returns:
            Dictionary with patient response and session state
        """
        if not self.core.current_state or not self.core.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}
        
        try:
            # The multi-agent workflow expects interview_messages to contain the conversation history
            # The last message should be the patient's previous response (or initial system message for first exchange)
            # The workflow will: propose_responses → judge_responses → generate_response (patient)
            
            # Log state before invocation
            interview_messages_before = self.core.current_state.get("interview_messages", [])
            logging.info(f"Before workflow: interview_messages count = {len(interview_messages_before)}")
            
            result = self.message_graph.invoke(self.core.current_state)
            
            # Verify that patient response was generated
            interview_messages_after = result.get("interview_messages", [])
            logging.info(f"After workflow: interview_messages count = {len(interview_messages_after)}")
            
            if len(interview_messages_after) <= len(interview_messages_before):
                logging.error(f"ERROR: Patient response not added! Before: {len(interview_messages_before)}, After: {len(interview_messages_after)}")
                return {"error": "Patient response was not generated"}
            
            # Check that last message is from patient (AIMessage)
            if interview_messages_after:
                last_msg = interview_messages_after[-1]
                if not isinstance(last_msg, AIMessage):
                    logging.warning(f"Last message is not AIMessage (patient), it's {type(last_msg).__name__}")
            
            self.core.current_state = result
            
            return {
                "ai_response": result.get("ai_response"),
                "session_active": result.get("session_active", True),
                "current_phase": "active_roleplay",
                "chatlog": result.get("chatlog", [])
            }
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def add_presentation_message(self, user_input: str):
        """Add presentation message using core."""
        return self.core.add_presentation_message(user_input)
    
    def end_session(self):
        """End session using core."""
        return self.core.end_session()
    
    def send_evaluator_message(self, user_input: str):
        """Send evaluator message using core."""
        return self.core.send_evaluator_message(user_input)
    
    # Utility methods - delegate to core
    def get_session_state(self):
        """Get current session state"""
        return self.core.get_session_state()
    
    def reset_session(self):
        """Reset current session state"""
        return self.core.reset_session()
    
    def get_session_statistics(self):
        """Get current session statistics"""
        return self.core.get_session_statistics()
    
    def export_session_data(self):
        """Export session data for analysis"""
        return self.core.export_session_data()
    
    def load_patient_profile(self, filepath: str) -> str:
        """Load a specific patient profile."""
        return self.core.load_patient_profile(filepath)
    
    @property
    def current_state(self):
        """Access to current state for backward compatibility."""
        return self.core.current_state
    
    @current_state.setter
    def current_state(self, value):
        """Setter for current state for backward compatibility."""
        self.core.current_state = value

