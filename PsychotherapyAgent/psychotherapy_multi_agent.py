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
from langchain_core.messages import HumanMessage, AIMessage

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
    
    def __init__(self, api_key: str):
        """
        Initialize PsychotherapyMultiAgent.
        
        Args:
            api_key: OpenAI API key
        """
        # Initialize core for patient simulation and session management
        self.core = PsychotherapyCore(api_key)
        
        # Initialize multi-agent therapist workflow
        self.llm_manager = LLMManager(self.core.api_key)
        self.prompt_manager = PromptManager()
        
        self.message_workflow = MessageWorkflow(
            self.llm_manager,
            self.prompt_manager,
            use_multi_agent=True,
            proposer_index=3,  # Row 5 in CSV: 3_MultiAgent_Proposer
            judge_index=4      # Row 6 in CSV: 4_MultiAgent_Judge
        )
        
        # Build compiled multi-agent message workflow
        self.message_graph = self.message_workflow.build()
        
        logging.info("PsychotherapyMultiAgent initialized with multi-agent therapist system.")
    
    def set_selection_mode(self, selection_mode: str):
        """Set the candidate-selection mechanism (judge or self).

        Ablation control: with selection_mode='self', the proposer generates the 3
        candidates AND selects the best one within the SAME single LLM call (no separate
        judge node). This is a matched-cost single-agent control (1 call vs the full
        system's 2), isolating the effect of the separate/learned judge module.

        Args:
            selection_mode: 'judge' (default) or 'self'
        """
        self.message_workflow.selection_mode = selection_mode
        # Rebuild workflow to apply the new selection mode
        self.message_graph = self.message_workflow.build()
        logging.info(f"Multi-agent candidate selection mode set to: {selection_mode}")
    
    def set_multi_agent_therapist_profiles(self, proposer_index: int, judge_index: int):
        """
        Set therapist profile indices for multi-agent mode.
        
        Args:
            proposer_index: Index in CSV for proposer agent (default: 3, row 5 = 3_MultiAgent_Proposer)
            judge_index: Index in CSV for judge agent (default: 4, row 6 = 4_MultiAgent_Judge)
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
        
        The exchange is RETRIED (up to a cap) until a full therapist+patient turn is actually
        appended to interview_messages. This guarantees each call produces one turn, so sessions
        never end up with fewer exchanges than the number of send_message calls (i.e. 10 turns).

        Args:
            user_input: Optional. In multi-agent mode, this should be None as the workflow
                       generates therapist responses automatically from conversation history.

        Returns:
            Dictionary with patient response and session state
        """
        if not self.core.current_state or not self.core.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}

        max_attempts = 5  # hard cap so a persistently failing exchange eventually surfaces
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                # Log state before invocation
                interview_messages_before = self.core.current_state.get("interview_messages", [])
                logging.info(f"Before workflow (attempt {attempt}): interview_messages count = {len(interview_messages_before)}")

                result = self.message_graph.invoke(self.core.current_state)

                # Verify that patient response was generated
                interview_messages_after = result.get("interview_messages", []) or []
                logging.info(f"After workflow (attempt {attempt}): interview_messages count = {len(interview_messages_after)}")

                if len(interview_messages_after) <= len(interview_messages_before):
                    last_error = "Patient response was not generated (no turn appended)."
                    logging.error(f"ERROR: Patient response not added! Before: {len(interview_messages_before)}, After: {len(interview_messages_after)} (attempt {attempt}/{max_attempts})")
                    # Do NOT commit the failed result; retry the whole exchange from the same
                    # (pre-iteration) state so the turn is not lost.
                    continue

                # Check that last message is from patient (AIMessage)
                if interview_messages_after:
                    last_msg = interview_messages_after[-1]
                    if not isinstance(last_msg, AIMessage):
                        logging.warning(f"Last message is not AIMessage (patient), it's {type(last_msg).__name__}")

                # Commit the successful exchange
                self.core.current_state = result

                return {
                    "ai_response": result.get("ai_response"),
                    "session_active": result.get("session_active", True),
                    "current_phase": "active_roleplay",
                    "chatlog": result.get("chatlog", [])
                }
            except Exception as e:
                last_error = str(e)
                logging.error(f"Error processing message (attempt {attempt}/{max_attempts}): {str(e)}")
                # Retry the exchange. current_state is unchanged because a failed invocation did not commit.

        # All attempts failed to produce a turn.
        logging.error(f"send_message failed to produce a turn after {max_attempts} attempts: {last_error}")
        return {"error": last_error, "message_attempts": max_attempts}
    
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

