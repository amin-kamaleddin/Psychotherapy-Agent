"""
Feedback Workflow for LearningBuddy Modular.

Handles feedback generation, MI form filling, and session completion.
"""

import logging
from datetime import datetime
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from ..graph_state import LBState


class FeedbackWorkflow:
    """Workflow for generating feedback and completing sessions."""
    
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
        """Build and return the compiled feedback workflow."""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("evaluate_session", RunnableLambda(self._generate_feedback))
        workflow.add_node("complete_session", RunnableLambda(self._complete_session))
        workflow.set_entry_point("evaluate_session")
        workflow.add_edge("evaluate_session", "complete_session")
        workflow.add_edge("complete_session", END)
        
        logging.info("Feedback workflow built successfully.")
        return workflow.compile()
    
    def _generate_feedback(self, state: LBState):
        """Generate comprehensive feedback."""
        logging.info("Starting feedback evaluation phase.")
        
        # Validate and format required data
        case_study = state.get("patient_context").get("case_study")
        clinical_summary = state.get("patient_context").get("clinical_summary")
        interview_messages = state.get("interview_messages", [])
        presentation = state.get("presentation", [])
        
        # Handle empty lists
        if interview_messages == []:
            interview_messages = [" "]
        if presentation == [] or presentation is None:
            presentation = [" "]
        
        if not case_study:
            return {**state, "error_message": "Case study content is missing from state."}
        if not clinical_summary:
            return {**state, "error_message": "Clinical summary is missing from state."}
        if not interview_messages:
            return {**state, "error_message": "Interview messages is empty or missing from state."}
        if not presentation:
            return {**state, "error_message": "Presentation is empty or missing from state."}
        
        try:
            evaluation_context = self.prompt_manager.load_context('Feedback Criteria') or ""
            base_instruction_prompt = self.prompt_manager.load_prompt('Feedback') or ""
        except Exception as e:
            logging.exception("Failed to load feedback prompt/context")
            return {**state, "error_message": f"Error loading prompts or context: {str(e)}"}
        
        strict_feedback_prompt = f"""
                                {base_instruction_prompt}

                                {evaluation_context}

                                Patient Profile for Reference:
                                {case_study}

                                Clinical Summary:
                                {clinical_summary}
                                """
        
        # Filter conversation messages
        conversation_messages = [
            m for m in interview_messages
            if (isinstance(m, HumanMessage) or isinstance(m, AIMessage)) and m.content
        ]
        
        messages = [
            SystemMessage(content=strict_feedback_prompt),
            HumanMessage(content="Here is the therapy session transcript to evaluate:")
        ]
        
        formatted_lines = []
        for m in conversation_messages:
            prefix = "PSYCHIATRIST: " if isinstance(m, AIMessage) else "PATIENT: "
            content = prefix + m.content
            msg_cls = HumanMessage if isinstance(m, HumanMessage) else AIMessage
            messages.append(msg_cls(content=content))
            formatted_lines.append(content)
        
        formatted_transcript = "\n".join(formatted_lines)
        
        # Format the diagnostic presentation
        formatted_presentation = []
        if presentation:
            formatted_presentation = "\n".join(
                f"Psychiatrist Presentation: {p.strip()}" for p in presentation if p.strip()
            )
            messages.append(HumanMessage(
                content="Here is the Psychiatrist's diagnostic presentation:"
            ))
            messages.append(HumanMessage(content=formatted_presentation))
        else:
            logging.warning("Presentation was expected but empty or missing.")
        
        # Generate Full Feedback with scores
        messages.append(HumanMessage(content="""
                                            Now provide your comprehensive, strict evaluation of this diagnostic session. 
                                            Be thorough, critical, and specific in identifying areas for improvement.
                                            Structure your feedback using all the sections mentioned.
                                            """))
        
        try:
            response = self.llm_manager.llm_feedback.invoke(messages)
            messages.append(response)
            updates = {"feedback_messages": messages}
            feedback_text = response.content.strip()
        except Exception as e:
            logging.exception("LLM feedback generation failed")
            return {**state, "error_message": f"Failed to generate feedback: {str(e)}"}
        
        session_feedback = f"""
                            ## DETAILED CLINICAL EVALUATION

                            {feedback_text}

                          """
        logging.info("Finished detailed feedback generation.")
        logging.info("Starting MI generation")
        
        # Generate Formatted MI Form
        instruction_prompt = self.prompt_manager.load_prompt('MI Prompt')
        blank_mi_file = self.prompt_manager.load_context("Evaluation Form")
        
        mi_prompt = f"""
        {instruction_prompt}

        Here is the Full Transcript:
        {formatted_transcript}
        
        Here is the full Case Presentation:
        {formatted_presentation}

        Here is the Evaluation Criteria to base your answer off of:
        {feedback_text}

        And here is the blank MI form that you must use to format your answer:
        {blank_mi_file}
        """
        
        messages = [
            SystemMessage(content=mi_prompt),
            HumanMessage(content="Please provide the formatted answer below:")
        ]
        
        response = self.llm_manager.llm_feedback.invoke(messages)
        mi_response = response.content
        
        logging.info("Finished MI generation")
        logging.info("Starting Agent Setup")
        
        # Eval Agent Set Up
        evaluator_prompt = self.prompt_manager.load_prompt('Virtual Evaluator')
        
        # Build full system message with context
        full_system_prompt = f"""
              {evaluator_prompt}

              You are presenting the evaluation of the following psychiatric interview and case presentation:
              === INTERVIEW TRANSCRIPT ===
              {formatted_transcript}

              === PRESENTATION TRANSCRIPT ===
              {formatted_presentation}

              This is the completed evaluation and filled out MI form which you will use to create your responses:
              === COMPLETE EVALUATION ===
              {session_feedback}

              === COMPLETE MI ===
              {mi_response}

              Here is the full patient profile, which can be used to answer questions about the patient:
              {case_study}

              Begin by giving a 'feedback sandwich': positive, then constructive critique, then end with positive support. 
              After that, respond to any questions the user may ask about the evaluation or the scoring.
              """
        
        initial_messages = [
            SystemMessage(content=full_system_prompt.strip())
        ]
        
        eval_messages = initial_messages
        
        logging.info("Finished Agent Setup")
        logging.info("Feedback evaluation complete.")
        
        updates["mi"] = mi_response
        updates["formatted_transcript"] = formatted_transcript
        updates["formatted_presentation"] = formatted_presentation
        updates["current_phase"] = "evaluator_active"
        updates["session_active"] = False
        updates["evaluator_messages"] = eval_messages
        updates["evaluator_active"] = True
        
        return updates
    
    def _complete_session(self, state: LBState):
        """Complete the session by saving all session data"""
        logging.info("Saving session data...")
        
        case_study = state.get("patient_context").get("case_study")
        mi = state.get("mi")
        detailed_feedback = state.get("feedback_messages")
        detailed_feedback = ' \n '.join(item.content for item in detailed_feedback)
        transcript = state.get("formatted_transcript")
        presentation = state.get("formatted_presentation")
        
        # Use file_manager to save session data
        folder_name = self.file_manager.save_session_data(
            case_study=case_study,
            mi=mi,
            detailed_feedback=detailed_feedback,
            transcript=transcript,
            presentation=presentation
        )
        
        logging.info("Session data saved successfully.")
        
        # Update session metadata
        session_metadata = state.get("metadata", {})
        session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_exchanges": len(
                [m for m in state.get("interview_messages", []) 
                 if type(m) in [AIMessage, HumanMessage]]
            ) // 2,
            "session_duration_minutes": self._calculate_session_duration(
                session_metadata.get("start_time")
            ),
            "completion_status": "completed"
        })
        
        logging.info("Session completed successfully.")
        updates = {
            "metadata": session_metadata,
            "current_phase": "completed",
            "session_active": False,
            "export_folder": folder_name,
            "folder_name": folder_name  # Also set folder_name for compatibility
        }
        return updates
    
    def _calculate_session_duration(self, start_time_iso: str) -> float:
        """Calculate session duration in minutes"""
        try:
            start_time = datetime.fromisoformat(start_time_iso)
            duration = datetime.now() - start_time
            return duration.total_seconds() / 60
        except:
            return 0.0

