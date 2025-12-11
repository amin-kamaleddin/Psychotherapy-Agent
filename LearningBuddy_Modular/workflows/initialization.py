"""
Initialization Workflow for LearningBuddy Modular.

Handles session initialization: patient profile selection, verification,
clinical summary generation, and roleplay setup.
"""

import logging
import os
import random
from langchain.schema import SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from ..graph_state import LBState


class InitializationWorkflow:
    """Workflow for initializing a new therapy session."""
    
    def __init__(self, llm_manager, prompt_manager, file_manager):
        """
        Initialize the workflow.
        
        Args:
            llm_manager: LLMManager instance
            prompt_manager: PromptManager instance
            file_manager: FileManager instance (from utils)
        """
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.file_manager = file_manager
    
    def build(self):
        """Build and return the compiled initialization workflow."""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("initialize", RunnableLambda(self._select_patient_profile))
        workflow.add_node("verify_profile", RunnableLambda(self._profile_verification))
        workflow.add_node("generate_summary", RunnableLambda(self._clinical_summary))
        workflow.add_node("setup_roleplay", RunnableLambda(self._roleplay_setup))
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "verify_profile")
        workflow.add_edge("verify_profile", "generate_summary")
        workflow.add_edge("generate_summary", "setup_roleplay")
        workflow.add_edge("setup_roleplay", END)
        
        logging.info("Initialization workflow built successfully.")
        return workflow.compile()
    
    def _select_patient_profile(self, state: LBState) -> dict:
        """Select or load patient profile."""
        logging.info("Selecting patient profile...")
        
        # Check if patient context already exists
        patient_ctx = state.get("patient_context")
        
        if patient_ctx and patient_ctx.get("case_study"):
            # Case study already provided
            logging.info("Using pre-loaded case study")
            return {}  # No changes needed
        
        # Load case study
        try:
            selected_profile = state["metadata"].get("selected_profile")
            
            if selected_profile:
                case_study = self.file_manager.load_patient_profile(selected_profile)
                logging.info(f"Loaded selected profile: {selected_profile}")
            else:
                case_study = self._load_random_case_profile()
                selected_profile = None  # Track that it was random
                logging.info("Loaded random profile")
            
            # Return new patient context
            return {
                "patient_context": {
                    "case_study": case_study,
                    "diagnosis_summary": "",
                    "clinical_summary": "",
                    "profile_path": selected_profile
                },
                "current_phase": "initializing"
            }
        
        except Exception as e:
            logging.error(f"Error loading case study: {str(e)}")
            return {
                "current_phase": "error",
                "error": f"Failed to load case study: {str(e)}"
            }
    
    def _profile_verification(self, state: LBState) -> dict:
        """Verifies the loaded patient profile."""
        logging.info("Starting profile verification phase.")
        
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Verification failed: Missing case_study in patient_context."
            }
        
        case_study = patient_ctx["case_study"]
        max_attempts = 3
        attempt = 0
        response_content = ""
        
        while attempt < max_attempts:
            attempt += 1
            try:
                instruction_prompt = self.prompt_manager.load_prompt('Verification')
                messages = [
                    SystemMessage(content=instruction_prompt + case_study),
                ]
                response = self.llm_manager.llm_mini_verification.invoke(messages)
                response_content = response.content
                break
            
            except Exception as e:
                logging.error(f"Error in profile verification: {str(e)}", exc_info=True)
                return {
                    "current_phase": "error",
                    "error": f"Profile verification failed: {str(e)}"
                }
        
        return {
            "patient_context": {
                **patient_ctx,
                "diagnosis_summary": response_content,
            },
            "current_phase": "generate_summary"
        }
    
    def _clinical_summary(self, state: LBState) -> dict:
        """Generates a clinical summary."""
        logging.info("Starting clinical summary phase.")
        
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Clinical Summary failed: Missing case_study in patient_context."
            }
        
        case_study = patient_ctx["case_study"]
        
        try:
            instruction_prompt = self.prompt_manager.load_prompt('Clinical Summary')
            messages = [
                SystemMessage(content=instruction_prompt + case_study),
            ]
            response = self.llm_manager.llm_mini_summary.invoke(messages)
            clinical_summary = response.content
            logging.info("Clinical summary generated.")
            
            return {
                "patient_context": {
                    **patient_ctx,
                    "clinical_summary": clinical_summary,
                },
                "current_phase": "setup_roleplay"
            }
        
        except Exception as e:
            logging.error(f"Error in clinical summary: {str(e)}", exc_info=True)
            return {
                "current_phase": "error",
                "error": f"Clinical summary generation failed: {str(e)}"
            }
    
    def _roleplay_setup(self, state: LBState) -> dict:
        """Sets up the roleplay session."""
        logging.info("Setting up virtual patient roleplay.")
        
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Roleplay setup failed: Missing case_study in patient_context."
            }
        
        case_study = patient_ctx["case_study"]
        
        try:
            instruction_prompt = self.prompt_manager.load_prompt('Virtual Patient')
            initial_messages = [
                SystemMessage(content=instruction_prompt + case_study),
            ]
            
            logging.info("Roleplay setup completed. Session ready for interaction.")
            
            return {
                "interview_messages": initial_messages,
                "current_phase": "active_roleplay",
                "session_active": True,
            }
        
        except Exception as e:
            logging.error(f"Error in roleplay setup: {str(e)}", exc_info=True)
            return {
                "current_phase": "error",
                "error": f"Roleplay setup failed: {str(e)}"
            }
    
    def _load_random_case_profile(self, directory='Patient_Profiles'):
        """Returns a randomly selected patient profile"""
        directory += "/MI"
        files = [f for f in os.listdir(directory)]
        if not files:
            raise FileNotFoundError("No patient profiles found in the directory.")
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        return self.file_manager.load_patient_profile(full_path)

