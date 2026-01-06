"""
Feedback Workflow for PsychotherapyAgent.

Handles feedback generation, MI form filling, and session completion.
"""

import logging
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from ..graph_state import LBState


class MIScore(BaseModel):
    """Individual MI criteria score with explanation."""
    score: int = Field(ge=1, le=5, description="Score from 1 to 5")
    explanation: str = Field(description="Detailed explanation referencing specific examples from the session")


class MIEvaluation(BaseModel):
    """Complete MI evaluation with structured scores for all 9 criteria."""
    acceptance: MIScore = Field(description="Extent to which counselor communicates unconditional positive regard")
    empathy: MIScore = Field(description="Extent to which counselor understands client's perspective")
    cultivating_change_talk: MIScore = Field(description="Extent to which counselor encourages client's language in favor of change")
    softening_sustain_talk: MIScore = Field(description="Extent that counselor avoids focus on reasons against changing")
    partnership: MIScore = Field(description="Extent to which counselor conveys that expertise resides within the client")
    direction: MIScore = Field(description="Degree to which counselor maintains focus on target behavior")
    autonomy_support: MIScore = Field(description="Extent to which clinician supports client perception of choice")
    collaboration: MIScore = Field(description="Extent to which interview occurs between equal partners")
    evocation: MIScore = Field(description="Extent to which counselor elicits motivation from within the client")
    overall_score: float = Field(ge=1.0, le=5.0, description="Average of all 9 scores")
    summary: str = Field(description="Overall summary with strengths and improvement areas")


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
        
        # Create structured output LLM for feedback
        self.structured_feedback_llm = llm_manager.llm_feedback.with_structured_output(MIEvaluation)
    
    def _format_structured_evaluation(self, evaluation: MIEvaluation) -> str:
        """Convert structured evaluation to formatted text."""
        return f"""# Motivational Interviewing Feedback: Psychiatrist Evaluation

## Global Ratings (Counselor Behaviors)

### 1. Acceptance
Score: {evaluation.acceptance.score}/5

Explanation:
{evaluation.acceptance.explanation}

---

### 2. Empathy
Score: {evaluation.empathy.score}/5

Explanation:
{evaluation.empathy.explanation}

---

### 3. Cultivating Change Talk
Score: {evaluation.cultivating_change_talk.score}/5

Explanation:
{evaluation.cultivating_change_talk.explanation}

---

### 4. Softening Sustain Talk
Score: {evaluation.softening_sustain_talk.score}/5

Explanation:
{evaluation.softening_sustain_talk.explanation}

---

### 5. Partnership
Score: {evaluation.partnership.score}/5

Explanation:
{evaluation.partnership.explanation}

---

### 6. Direction
Score: {evaluation.direction.score}/5

Explanation:
{evaluation.direction.explanation}

---

### 7. Autonomy Support
Score: {evaluation.autonomy_support.score}/5

Explanation:
{evaluation.autonomy_support.explanation}

---

### 8. Collaboration
Score: {evaluation.collaboration.score}/5

Explanation:
{evaluation.collaboration.explanation}

---

### 9. Evocation
Score: {evaluation.evocation.score}/5

Explanation:
{evaluation.evocation.explanation}

---

**Overall Global Ratings Score:** **{evaluation.overall_score:.1f}/5**

{evaluation.summary}
"""
    
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
            prefix = "PSYCHIATRIST: " if isinstance(m, HumanMessage) else "PATIENT: "
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
        
        try:
            # Use structured output for guaranteed 9-score format
            logging.info("Invoking structured feedback LLM with schema enforcement...")
            structured_evaluation = self.structured_feedback_llm.invoke(messages)
            
            # Convert structured evaluation to formatted text
            feedback_text = self._format_structured_evaluation(structured_evaluation)
            
            # Also store as regular message for compatibility
            response_message = AIMessage(content=feedback_text)
            messages.append(response_message)
            updates = {"feedback_messages": messages}
            
        except Exception as e:
            logging.exception("Structured LLM feedback generation failed, falling back to unstructured")
            try:
                # Fallback to unstructured if structured fails
                response = self.llm_manager.llm_feedback.invoke(messages)
                messages.append(response)
                updates = {"feedback_messages": messages}
                feedback_text = response.content.strip()
                
                # Remove example template if LLM included it in output
                feedback_text = self._remove_example_from_feedback(feedback_text)
            except Exception as e2:
                logging.exception("Fallback LLM feedback generation also failed")
                return {**state, "error_message": f"Failed to generate feedback: {str(e2)}"}
        
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
        
        # Get patient_id from metadata first (used by run_eval.py), fallback to extracting from profile_path
        metadata = state.get("metadata", {})
        patient_id = metadata.get("patient_number")
        
        if patient_id is None:
            # Fallback: try to extract from profile path
            profile_path = state.get("patient_context", {}).get("profile_path")
            patient_id = self.file_manager.get_patient_id(profile_path)

        # Get judge-proposer conversation if available (only for multi-agent mode)
        judge_proposer_conversation = state.get("judge_proposer_conversation")
        
        # Use file_manager to save session data
        folder_name = self.file_manager.save_session_data(
            case_study=case_study,
            mi=mi,
            detailed_feedback=detailed_feedback,
            transcript=transcript,
            presentation=presentation,
            patient_id=patient_id,
            judge_proposer_conversation=judge_proposer_conversation
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
    
    def _remove_example_from_feedback(self, feedback_text: str) -> str:
        """Remove example template from LLM output if present.
        
        The prompt includes an example (Linda case) to guide formatting.
        Some LLMs may include this example in their output. This function
        detects and removes it, keeping only the actual evaluation.
        """
        # Check for end-of-example marker
        marker = "[END OF EXAMPLE - YOUR ACTUAL EVALUATION SHOULD START BELOW]"
        if marker in feedback_text:
            parts = feedback_text.split(marker)
            if len(parts) > 1:
                cleaned = parts[-1].strip()
                logging.info("Removed example template from feedback (found marker)")
                return cleaned
        
        # Fallback: detect if Linda example is present by checking for specific text
        # The example mentions "Linda" and specific phrases
        if "Linda's ambivalence" in feedback_text or "Linda's attachment to smoking" in feedback_text:
            # Split on a pattern that indicates start of actual evaluation
            # Look for the second occurrence of "## Global Ratings" or "### 1. Acceptance"
            lines = feedback_text.split('\n')
            acceptance_indices = [i for i, line in enumerate(lines) if '### 1. Acceptance' in line or '1. Acceptance' in line]
            
            if len(acceptance_indices) > 1:
                # Keep everything from the second Acceptance section onward
                actual_start = acceptance_indices[1]
                cleaned = '\n'.join(lines[actual_start:]).strip()
                logging.info("Removed Linda example template from feedback (detected duplicate sections)")
                return cleaned
        
        return feedback_text

