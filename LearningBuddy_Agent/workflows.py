import logging
from datetime import datetime
from typing import Optional, Dict, List
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from .graph_state import LBState, PatientContext
from .utils import (
    FileManager,
    PromptLoader,
    format_interview_messages,
    format_presentation,
    build_langchain_messages
)


###############################################################
###          InitializationWorkflow Class                   ###
###############################################################

class InitializationWorkflow:
    """
    Handles session initialization workflow:
    1. Select patient profile
    2. Verify profile
    3. Generate clinical summary
    4. Setup roleplay
    """

    def __init__(self, llm_verification, llm_summary, file_manager: FileManager,
                 prompt_loader: PromptLoader):
        """
        Initialize the workflow with required dependencies.

        Args:
            llm_verification: LLM client for profile verification
            llm_summary: LLM client for clinical summary
            file_manager: FileManager instance
            prompt_loader: PromptLoader instance
        """
        self.llm_verification = llm_verification
        self.llm_summary = llm_summary
        self.file_manager = file_manager
        self.prompt_loader = prompt_loader
        self.graph = self._build_graph()
        logging.info("InitializationWorkflow initialized.")

    def _build_graph(self):
        """Build initialization workflow graph - linear, no loops."""
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

        logging.info("Initialization workflow graph built successfully.")
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
                case_study = self.file_manager.load_random_case_profile()
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
        """
        Verify the loaded patient profile.
        Updates patient_context with diagnosis summary.
        """
        logging.info("Starting profile verification phase.")

        # Access nested data and check for required content
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
                instruction_prompt = self.prompt_loader.load_prompt('Verification')
                messages = [
                    SystemMessage(content=instruction_prompt + case_study),
                ]
                response = self.llm_verification.invoke(messages)
                response_content = response.content
                # Assuming verification passes
                break

            except Exception as e:
                logging.error(f"Error in profile verification: {str(e)}", exc_info=True)
                return {
                    "current_phase": "error",
                    "error": f"Profile verification failed: {str(e)}"
                }

        # Return updates using nested structure
        return {
            "patient_context": {
                **patient_ctx,
                "diagnosis_summary": response_content,
            },
            "current_phase": "generate_summary"
        }

    def _clinical_summary(self, state: LBState) -> dict:
        """
        Generate clinical summary.
        Brief intake information before interview starts.
        """
        logging.info("Starting clinical summary phase.")

        # Safely access nested data
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Clinical Summary failed: Missing case_study in patient_context."
            }

        case_study = patient_ctx["case_study"]

        try:
            instruction_prompt = self.prompt_loader.load_prompt('Clinical Summary')

            messages = [
                SystemMessage(content=instruction_prompt + case_study),
            ]

            response = self.llm_summary.invoke(messages)
            clinical_summary = response.content
            logging.info("Clinical summary generated.")

            # Return updates using nested structure
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
        """
        Set up roleplay session by loading patient prompt and
        initializing conversation history.
        """
        logging.info("Setting up virtual patient roleplay.")

        # Safely access nested data
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Roleplay setup failed: Missing case_study in patient_context."
            }

        case_study = patient_ctx["case_study"]

        try:
            instruction_prompt = self.prompt_loader.load_prompt('Virtual Patient')

            # Initialize conversation history with system message
            initial_messages = [
                SystemMessage(content=instruction_prompt + case_study),
            ]

            logging.info("Roleplay setup completed. Session ready for interaction.")

            # Return updates
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

    def run(self, initial_state: LBState) -> LBState:
        """
        Run the initialization workflow.

        Args:
            initial_state: Initial state with metadata

        Returns:
            Updated state after initialization
        """
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            logging.error(f"Error running initialization workflow: {str(e)}")
            initial_state["current_phase"] = "error"
            initial_state["error"] = f"Initialization failed: {str(e)}"
            return initial_state


###############################################################
###              InterviewWorkflow Class                    ###
###############################################################

class InterviewWorkflow:
    """
    Handles virtual patient interview workflow:
    1. Process user input
    2. Generate AI patient response
    """

    def __init__(self, llm_virtual_patient):
        """
        Initialize the workflow with required dependencies.

        Args:
            llm_virtual_patient: LLM client for virtual patient responses
        """
        self.llm_virtual_patient = llm_virtual_patient
        self.graph = self._build_graph()
        logging.info("InterviewWorkflow initialized.")

    def _build_graph(self):
        """Build message processing workflow - linear, no loops."""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("process_input", RunnableLambda(self._process_user_input))
        workflow.add_node("generate_response", RunnableLambda(self._generate_ai_response))

        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_response")
        workflow.add_edge("generate_response", END)

        logging.info("Message processing workflow graph built successfully.")
        return workflow.compile()

    def _process_user_input(self, state: LBState) -> dict:
        """
        Process user input during interview.
        Validates that input exists.
        """
        user_input = state.get("interview_messages")[-1].content if state.get("interview_messages") else None
        logging.info(f"Processing user input...")

        if not user_input:
            logging.error("User input is Empty...")
            return state

        return {
            "current_phase": "generate_response",
        }

    def _generate_ai_response(self, state: LBState) -> dict:
        """
        Generate AI response during roleplay.
        Appends response to interview messages.
        """
        logging.info("Generating AI patient response...")

        try:
            interview_messages = state.get("interview_messages", [])

            # Generate response
            response = self.llm_virtual_patient.invoke(interview_messages)

            logging.info("AI response generated successfully.")

            return {
                "interview_messages": [AIMessage(content=response.content)],
                "current_phase": "message_complete"
            }
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}")
            return {
                "current_phase": "error",
                "error": f"AI response generation failed: {str(e)}"
            }

    def run(self, state: LBState, user_input: str) -> LBState:
        """
        Run the interview workflow with user input.

        Args:
            state: Current state
            user_input: User's message

        Returns:
            Updated state with AI response
        """
        try:
            # Add user message to state
            state["interview_messages"] = [
                *state.get("interview_messages", []),
                HumanMessage(content=user_input)
            ]

            # Run workflow
            result = self.graph.invoke(state)
            return result

        except Exception as e:
            logging.error(f"Error running interview workflow: {str(e)}")
            state["current_phase"] = "error"
            state["error"] = f"Interview failed: {str(e)}"
            return state


###############################################################
###              FeedbackWorkflow Class                     ###
###############################################################

class FeedbackWorkflow:
    """
    Handles feedback generation workflow:
    1. Generate comprehensive feedback and STACER
    2. Complete session and save data
    """

    def __init__(self, llm_feedback, llm_virtual_evaluator, prompt_loader: PromptLoader,
                 file_manager: FileManager, mode: int):
        """
        Initialize the workflow with required dependencies.

        Args:
            llm_feedback: LLM client for feedback generation
            llm_virtual_evaluator: LLM client for virtual evaluator
            prompt_loader: PromptLoader instance
            file_manager: FileManager instance
            mode: Current mode (STACER_MODE, MI_MODE, or CBT_MODE)
        """
        self.llm_feedback = llm_feedback
        self.llm_virtual_evaluator = llm_virtual_evaluator
        self.prompt_loader = prompt_loader
        self.file_manager = file_manager
        self.mode = mode
        self.graph = self._build_graph()
        logging.info("FeedbackWorkflow initialized.")

    def _build_graph(self):
        """Build feedback workflow - linear, no loops."""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("evaluate_session", RunnableLambda(self._generate_feedback))
        workflow.add_node("complete_session", RunnableLambda(self._complete_session))

        workflow.set_entry_point("evaluate_session")
        workflow.add_edge("evaluate_session", "complete_session")
        workflow.add_edge("complete_session", END)

        logging.info("Feedback workflow graph built successfully.")
        return workflow.compile()

    def _generate_feedback(self, state: LBState) -> dict:
        """
        Generate comprehensive feedback.
        Scores criteria, writes feedback, fills out STACER/evaluation form,
        and sets up evaluator agent.
        """
        logging.info("Starting feedback evaluation phase.")

        # Validate and format required data
        patient_ctx = state.get("patient_context")
        if not patient_ctx:
            return {
                "current_phase": "error",
                "error": "Patient context is missing from state."
            }

        case_study = patient_ctx.get("case_study")
        clinical_summary = patient_ctx.get("clinical_summary")
        interview_messages = state.get("interview_messages", [])
        presentation = state.get("presentation", [])

        # Add empty string if lists are empty (user skipped a step)
        if not interview_messages:
            interview_messages = [HumanMessage(content=" ")]
        if not presentation:
            presentation = [" "]

        # Validate required data
        if not case_study:
            return {"current_phase": "error", "error": "Case study content is missing."}
        if not clinical_summary:
            return {"current_phase": "error", "error": "Clinical summary is missing."}

        try:
            evaluation_context = self.prompt_loader.load_context('Feedback Criteria') or ""
            base_instruction_prompt = self.prompt_loader.load_prompt('Feedback') or ""
        except Exception as e:
            logging.exception("Failed to load feedback prompt/context")
            return {"current_phase": "error", "error": f"Error loading prompts: {str(e)}"}

        # Build feedback prompt
        strict_feedback_prompt = f"""
{base_instruction_prompt}

{evaluation_context}

Patient Profile for Reference:
{case_study}

Clinical Summary:
{clinical_summary}
"""

        # Format interview messages
        conversation_messages, formatted_transcript = format_interview_messages(interview_messages)

        # Build messages for LLM
        messages = [
            SystemMessage(content=strict_feedback_prompt),
            HumanMessage(content="Here is the therapy session transcript to evaluate:")
        ]
        messages.extend(conversation_messages)

        # Format and add presentation
        formatted_presentation = format_presentation(presentation)
        if formatted_presentation:
            messages.append(HumanMessage(
                content="Here is the Psychiatrist's diagnostic presentation:"
            ))
            messages.append(HumanMessage(content=formatted_presentation))
        else:
            logging.warning("Presentation was expected but empty or missing.")

        # Request comprehensive evaluation
        messages.append(HumanMessage(content="""
Now provide your comprehensive, strict evaluation of this diagnostic session. 
Be thorough, critical, and specific in identifying areas for improvement.
Structure your feedback using all the sections mentioned.
"""))

        # Generate detailed feedback
        try:
            response = self.llm_feedback.invoke(messages)
            messages.append(response)
            feedback_text = response.content.strip()
        except Exception as e:
            logging.exception("LLM feedback generation failed")
            return {"current_phase": "error", "error": f"Failed to generate feedback: {str(e)}"}

        session_feedback = f"""
## DETAILED CLINICAL EVALUATION

{feedback_text}
"""

        logging.info("Finished detailed feedback generation.")
        logging.info("Starting STACER generation.")

        # Generate formatted STACER/Evaluation form
        instruction_prompt = self.prompt_loader.load_prompt('STACER Prompt')
        blank_stacer_file = self.prompt_loader.load_context("Evaluation Form")

        stacer_prompt = f"""
{instruction_prompt}

Here is the Full Transcript:
{formatted_transcript}

Here is the full Case Presentation:
{formatted_presentation}

Here is the Evaluation Criteria to base your answer off of:
{feedback_text}

And here is the blank STACER form that you must use to format your answer:
{blank_stacer_file}
"""

        messages_stacer = [
            SystemMessage(content=stacer_prompt),
            HumanMessage(content="Please provide the formatted answer below:")
        ]

        response = self.llm_feedback.invoke(messages_stacer)
        cod_stacer_response = response.content

        logging.info("Finished STACER generation.")
        logging.info("Starting Evaluator Agent setup.")

        # Set up Virtual Evaluator Agent
        evaluator_prompt = self.prompt_loader.load_prompt('Virtual Evaluator')

        full_system_prompt = f"""
{evaluator_prompt}

You are presenting the evaluation of the following psychiatric interview and case presentation:
=== INTERVIEW TRANSCRIPT ===
{formatted_transcript}

=== PRESENTATION TRANSCRIPT ===
{formatted_presentation}

This is the completed evaluation and filled out STACER form which you will use to create your responses:
=== COMPLETE EVALUATION ===
{session_feedback}

=== COMPLETE STACER ===
{cod_stacer_response}

Here is the full patient profile, which can be used to answer questions about the patient:
{case_study}

Begin by giving a 'feedback sandwich': positive, then constructive critique, then end with positive support. 
After that, respond to any questions the user may ask about the evaluation or the scoring.
"""

        initial_messages = [SystemMessage(content=full_system_prompt.strip())]

        logging.info("Finished Evaluator Agent setup.")
        logging.info("Feedback evaluation complete.")

        # Return updates
        return {
            "feedback_messages": messages,
            "cod_stacer": cod_stacer_response,
            "formatted_transcript": formatted_transcript,
            "formatted_presentation": formatted_presentation,
            "current_phase": "evaluator_active",
            "session_active": False,
            "evaluator_messages": initial_messages,
            "evaluator_active": True
        }

    def _complete_session(self, state: LBState) -> dict:
        """
        Complete the session by saving all session data to files.
        """
        logging.info("Saving session data...")

        # Extract data from state
        patient_ctx = state.get("patient_context", {})
        case_study = patient_ctx.get("case_study", "")
        stacer = state.get("cod_stacer", "")

        # Extract detailed feedback from messages
        feedback_messages = state.get("feedback_messages", [])
        detailed_feedback = ' \n '.join(
            item.content for item in feedback_messages if hasattr(item, 'content')
        )

        transcript = state.get("formatted_transcript", "")
        presentation = state.get("formatted_presentation", "")

        # Save session data using FileManager
        try:
            folder_name = self.file_manager.save_session_data(
                case_study=case_study,
                stacer=stacer,
                detailed_feedback=detailed_feedback,
                transcript=transcript,
                presentation=presentation,
                mode=self.mode
            )
        except Exception as e:
            logging.error(f"Error saving session data: {str(e)}")
            return {
                "current_phase": "error",
                "error": f"Failed to save session data: {str(e)}"
            }

        # Update session metadata
        session_metadata = state.get("metadata", {})
        start_time = session_metadata.get("start_time")

        # Count exchanges
        interview_messages = state.get("interview_messages", [])
        total_exchanges = len([
            m for m in interview_messages
            if isinstance(m, (AIMessage, HumanMessage))
        ]) // 2

        session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_exchanges": total_exchanges,
            "session_duration_minutes": self.file_manager.calculate_session_duration(start_time) if start_time else 0,
            "completion_status": "completed"
        })

        logging.info("Session completed successfully.")

        return {
            "metadata": session_metadata,
            "current_phase": "completed",
            "session_active": False,
            "folder_name": folder_name
        }

    def run(self, state: LBState) -> LBState:
        """
        Run the feedback workflow.

        Args:
            state: Current state

        Returns:
            Updated state with feedback and saved data
        """
        try:
            result = self.graph.invoke(state)
            return result
        except Exception as e:
            logging.error(f"Error running feedback workflow: {str(e)}")
            state["current_phase"] = "error"
            state["error"] = f"Feedback generation failed: {str(e)}"
            return state


###############################################################
###              EvaluatorWorkflow Class                    ###
###############################################################

class EvaluatorWorkflow:
    """
    Handles virtual evaluator interaction workflow:
    1. Process user input
    2. Generate evaluator response
    """

    def __init__(self, llm_virtual_evaluator):
        """
        Initialize the workflow with required dependencies.

        Args:
            llm_virtual_evaluator: LLM client for virtual evaluator
        """
        self.llm_virtual_evaluator = llm_virtual_evaluator
        self.graph = self._build_graph()
        logging.info("EvaluatorWorkflow initialized.")

    def _build_graph(self):
        """Build evaluator workflow - single-turn message processing."""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("process_input", RunnableLambda(self._process_user_input_evaluator))
        workflow.add_node("generate_feedback", RunnableLambda(self._generate_evaluator_response))

        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_feedback")
        workflow.add_edge("generate_feedback", END)

        logging.info("Evaluator workflow graph built successfully.")
        return workflow.compile()

    def _process_user_input_evaluator(self, state: LBState) -> dict:
        """
        Process user input during evaluator interaction.
        Note: User input comes from the 'user_input' field for evaluator.
        """
        user_input = state.get("user_input")
        if not user_input:
            return state

        logging.info(f"Processing evaluator user input: {user_input[:50]}...")

        # Add user message to evaluator messages
        evaluator_messages = state.get("evaluator_messages", [])
        evaluator_messages_updated = [
            *evaluator_messages,
            HumanMessage(content=user_input)
        ]

        return {
            "evaluator_messages": evaluator_messages_updated,
            "current_phase": "generate_eval_feedback",
            "user_input": None
        }

    def _generate_evaluator_response(self, state: LBState) -> dict:
        """
        Generate response from evaluator agent.
        """
        logging.info("Generating evaluator feedback...")

        try:
            evaluator_messages = state.get("evaluator_messages", [])

            # Generate response
            response = self.llm_virtual_evaluator.invoke(evaluator_messages)
            ai_response = response.content

            # Add AI response to messages
            evaluator_messages_updated = [
                *evaluator_messages,
                AIMessage(content=ai_response)
            ]

            logging.info("Evaluator response generated.")

            return {
                "evaluator_messages": evaluator_messages_updated,
                "current_phase": "eval_feedback_complete",
                "evaluator_active": True
            }

        except Exception as e:
            logging.error(f"Evaluator response failed: {str(e)}")
            return {
                "current_phase": "error",
                "error": f"Evaluator error: {str(e)}"
            }

    def run(self, state: LBState, user_input: str) -> LBState:
        """
        Run the evaluator workflow with user input.

        Args:
            state: Current state
            user_input: User's question/message

        Returns:
            Updated state with evaluator response
        """
        try:
            # Add user input to state
            state["user_input"] = user_input

            # Run workflow
            result = self.graph.invoke(state)
            return result

        except Exception as e:
            logging.error(f"Error running evaluator workflow: {str(e)}")
            state["current_phase"] = "error"
            state["error"] = f"Evaluator interaction failed: {str(e)}"
            return state