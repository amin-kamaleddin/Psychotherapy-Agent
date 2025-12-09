import logging
import os
import random
from datetime import datetime

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import SecretStr

from LearningBuddy_Agent.graph_state import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("learningbuddy.log"),
        logging.StreamHandler()
    ]
)


# Defining LangGraph State Variables
class LearningBuddy:
    # Modes
    STACER_MODE = 1
    MI_MODE = 2
    CBT_MODE = 3

    def __init__(self, api_key: str, mode=STACER_MODE):
        self.api_key = SecretStr(api_key)
        self.llm_mini_verification = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=self.api_key)
        self.llm_mini_summary = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, api_key=self.api_key)
        self.llm_mini_virtualPatient = ChatOpenAI(model="gpt-4.1-mini", temperature=1, api_key=self.api_key)
        self.llm_feedback = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, api_key=self.api_key)
        self.llm_mini_virtualEvaluator = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, api_key=self.api_key)
        self.init_graph = self._build_init_workflow()
        self.message_graph = self._build_message_workflow()
        self.feedback_graph = self._build_feedback_workflow()
        self.evaluator_graph = self._build_evaluator_workflow()
        self.current_state = None
        self.mode = mode
        logging.info("LearningBuddy initialized with separate LangGraph workflows.")

    ###############################################################
    ###              Workflow 1: Initialization                 ###
    ###############################################################
    #   This workflow initializes the virtual patient interview:
    #            1. Selects a patient profile
    #            2. Verifies the selected profile
    #            3. Generates an introductory 'clinical summary'
    #            4. Creates the prompt to be used for the chatbot
    #
    #   Public function to use this workflow: start_session()

    def _build_init_workflow(self):
        """Build initialization workflow - linear, no loops"""
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
                case_study = self.load_patient_profile(selected_profile)
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
        """
        Verifies the loaded patient profile. Updates patient_context with the
        diagnosis summary (or handles the profile rejection/retry logic if enabled).
        """
        logging.info("Starting profile verification phase.")

        # 1. Access nested data and safely check for required content
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

        # NOTE: The original loop structure is kept for context,
        # but the internal verification logic is still commented out.
        while attempt < max_attempts:
            attempt += 1
            try:
                instruction_prompt = self._load_prompt('Verification')
                # Use standard LangChain Message objects for LLM invocation
                messages = [
                    SystemMessage(content=instruction_prompt + case_study),
                ]
                response = self.llm_mini_verification.invoke(messages)
                response_content = response.content
                # Assuming verification passes for now (since original logic was commented out)
                break

            except Exception as e:
                logging.error(f"Error in profile verification: {str(e)}", exc_info=True)
                # 2. Use new error format for conditional routing
                return {
                    "current_phase": "error",
                    "error": f"Profile verification failed: {str(e)}"
                }

        # 3. Return updates using the nested structure
        return {
            "patient_context": {
                **patient_ctx,  # Keep existing fields
                "diagnosis_summary": response_content,
            },
            "current_phase": "generate_summary"
        }

    def _clinical_summary(self, state: LBState) -> dict:
        """
        Generates a clinical summary. This summary includes brief intake information
        on the patient before the interview starts.
        """
        logging.info("Starting clinical summary phase.")

        # 1. Safely access nested data
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Clinical Summary failed: Missing case_study in patient_context."
            }

        case_study = patient_ctx["case_study"]

        try:
            instruction_prompt = self._load_prompt('Clinical Summary')

            messages = [
                SystemMessage(content=instruction_prompt + case_study),
            ]

            response = self.llm_mini_summary.invoke(messages)
            clinical_summary = response.content
            logging.info("Clinical summary generated.")

            # 2. Return updates using the nested structure
            return {
                "patient_context": {
                    **patient_ctx,  # Keep existing fields
                    "clinical_summary": clinical_summary,
                },
                "current_phase": "setup_roleplay"
            }

        except Exception as e:
            logging.error(f"Error in clinical summary: {str(e)}", exc_info=True)
            # 3. Use new error format
            return {
                "current_phase": "error",
                "error": f"Clinical summary generation failed: {str(e)}"
            }

    def _roleplay_setup(self, state: LBState) -> dict:
        """
        Sets up the roleplay session by loading the patient prompt and initializing
        the conversation history with required system and context messages.
        """
        logging.info("Setting up virtual patient roleplay.")

        # 1. Safely access nested data
        patient_ctx = state.get("patient_context")
        if not patient_ctx or not patient_ctx.get("case_study"):
            return {
                "current_phase": "error",
                "error": "Roleplay setup failed: Missing case_study in patient_context."
            }

        case_study = patient_ctx["case_study"]

        try:
            instruction_prompt = self._load_prompt('Virtual Patient')

            # 2. Initialize conversation history with LangChain Message objects
            initial_messages = [
                SystemMessage(content=instruction_prompt + case_study),
            ]

            logging.info("Roleplay setup completed. Session ready for interaction.")

            # 3. Return updates using the new state fields
            return {
                "interview_messages": initial_messages,  # New field for conversation history
                "current_phase": "active_roleplay",
                "session_active": True,
                # Removed redundant 'user_input' and 'ai_response' fields
            }

        except Exception as e:
            logging.error(f"Error in roleplay setup: {str(e)}", exc_info=True)
            # 4. Use new error format
            return {
                "current_phase": "error",
                "error": f"Roleplay setup failed: {str(e)}"
            }

    def start_session(
            self,
            case_study: Optional[str] = None,
            selected_profile: Optional[str] = None
    ) -> LBState:
        """
        Start a new session using initialization workflow.

        Args:
            case_study: Pre-loaded case study content (optional)
            selected_profile: Path to specific patient profile (optional)

        Returns:
            Initial state after running initialization workflow

        Raises:
            Exception: If initialization workflow fails
        """
        logging.info("Starting new LearningBuddy session...")

        # Generate session metadata
        now = datetime.now()
        session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}"

        # Build initial state with new structure
        initial_state: LBState = {
            # Conversational state - empty at start
            "interview_messages": [],
            "evaluator_messages": [],

            # Session control
            "current_phase": "initializing",
            "session_active": False,
            "evaluator_active": False,

            # Nested data structures - all None at start
            "patient_context": None,
            "presentation": None,
            "evaluation": None,

            # Session metadata
            "metadata": {
                "session_id": session_id,
                "start_time": now.isoformat(),
                "end_time": None,
                "selected_profile": selected_profile
            },

            # Error handling
            "error": None,

            # Export path
            "export_folder": None
        }

        # If case_study provided, initialize patient_context early
        if case_study:
            initial_state["patient_context"] = PatientContext(
                case_study=case_study,
                diagnosis_summary="",
                clinical_summary="",
                profile_path=selected_profile)

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

            # Return error state
            error_state = initial_state
            error_state["current_phase"] = "error"
            error_state["error"] = f"Initialization failed: {str(e)}"
            return error_state

    ###############################################################
    ###           Workflow 2: Virtual Patient Chatbot           ###
    ###############################################################
    #   This workflow handles one turn of the virtual patient interview:
    #            1. Read user input
    #            2. Generate patient response
    #
    #   Public function to use this workflow: send_message()

    def _build_message_workflow(self):
        """ Build message processing workflow - linear, no loops"""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("process_input", RunnableLambda(self._process_user_input))
        workflow.add_node("generate_response", RunnableLambda(self._generate_ai_response))
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_response")
        workflow.add_edge("generate_response", END)

        logging.info("Message processing workflow built successfully.")
        return workflow.compile()

    def _process_user_input(self, state: LBState):
        """ Process user input during interview.
            Appends all entries to chatlog for future reference.
        """
        user_input = state.get("interview_messages")[-1].content if state.get("interview_messages") else None
        logging.info(f"Processing user input: ...")

        if not user_input:
            logging.error("User input is Empty...")
            return state

        return {
            "current_phase": "generate_response",
        }

    def _generate_ai_response(self, state: LBState):
        """ Generate AI response during roleplay.
            Appends all entries to chatlog for future reference.
        """
        logging.info("Generating AI patient response...")

        try:
            interview_messages = state.get("interview_messages", [])

            # Generate response
            response = self.llm_mini_virtualPatient.invoke(interview_messages)

            # Add AI response to chatlog

            logging.info("AI response generated successfully.")

            return {
                "interview_messages": [AIMessage(content=response.content)],
                "current_phase": "message_complete"
            }
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}")
            return {**state, "error_message": f"AI response generation failed: {str(e)}"}

    def send_message(self, user_input: str) -> dict:
        """ Send a user message using message processing workflow.
            Public function for use by frontend to use this workflow.
        """
        if not self.current_state or not self.current_state.get("session_active"):
            logging.error("No active session found.")
            return {"error": "No active session"}

        try:
            # Create state for message processing
            self.current_state["interview_messages"] = [*self.current_state.get("interview_messages"), HumanMessage(
                content=user_input)]
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

    ###############################################################
    ###             Workflow 2b : Case Presentation             ###
    ###############################################################
    #   This step handles the case presentation.
    #   The UI prompts the user for input, and uses this function to append and save the user's whole presentation:
    #            1. Read and save user input(s)
    #
    #   Public function to use this step: add_presentation_message()

    def add_presentation_message(self, user_input: str):
        """ Append and save user messages during presentation phase manually.
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

    ###############################################################
    ###                Workflow 3: STACER Feedback              ###
    ###############################################################
    #   This workflow generates all static feedback after an interview ends:
    #            1. Generates detailed feedback and fills out STACER
    #            2. Initializes Feedback Agent
    #            2. Ends current session and saves all session data
    #
    #   Public function to use this workflow: end_session()

    def _build_feedback_workflow(self):
        """Build feedback workflow - linear, no loops"""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("evaluate_session", RunnableLambda(self._generate_feedback))
        workflow.add_node("complete_session", RunnableLambda(self._complete_session))
        workflow.set_entry_point("evaluate_session")
        workflow.add_edge("evaluate_session", "complete_session")
        workflow.add_edge("complete_session", END)

        logging.info("Feedback workflow built successfully.")
        return workflow.compile()

    def _generate_feedback(self, state: LBState):
        """ Generate comprehensive feedback.
            First scores all defined criteria on a 1-5 scale and writes
            thorough feedback based on provided rubric.
            Then uses full feedback to fill out a formatted COD STACER.
        """
        logging.info("Starting feedback evaluation phase.")

        # Validate and format required data
        case_study = state.get("patient_context").get("case_study")
        clinical_summary = state.get("patient_context").get("clinical_summary")
        interview_messages = state.get("interview_messages", [])
        presentation = state.get("presentation", [])

        # Feedback fails if chatlog or presentaion lists are empty. Adds empty string if user skips a step and the lists are empty
        if interview_messages == []:
            chatlog = [" "]
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
            evaluation_context = self._load_context('Feedback Criteria') or ""
            base_instruction_prompt = self._load_prompt('Feedback') or ""
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
        # Filter conversation messages (exclude system messages and case study repeats)
        conversation_messages = [
            m for m in interview_messages
            if (isinstance(m, HumanMessage) or isinstance(m, AIMessage)) and
               m.content
            # and m.content.strip() != case_study.strip()
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
            response = self.llm_feedback.invoke(messages)
            messages.append(response)
            updates = {"feedback_messages": messages}
            feedback_text = response.content.strip()
        except Exception as e:
            logging.exception("LLM feedback generation failed")
            return {**state, "error_message": f"Failed to generate feedback: {str(e)}"}

        num_exchanges = len(conversation_messages) // 2
        session_feedback = f"""
                            ## DETAILED CLINICAL EVALUATION

                            {feedback_text}

                          """
        logging.info("Finished detailed feedback generation. ")
        logging.info("Starting STACER generation ")
        # Generate Formatted COD STACER Form
        instruction_prompt = self._load_prompt('STACER Prompt')
        blank_stacer_file = self._load_context("Evaluation Form")

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

        messages = [
            SystemMessage(content=stacer_prompt),
            HumanMessage(content="Please provide the formatted answer below:")
        ]

        response = self.llm_feedback.invoke(messages)
        cod_stacer_response = response.content

        logging.info("Finished STACER generation ")
        logging.info("Starting Agent Setup ")

        ####   Eval Agent Set Up
        evaluator_prompt = self._load_prompt('Virtual Evaluator')

        # Build full system message with context
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

        initial_messages = [
                SystemMessage(content=full_system_prompt.strip())
        ]

        eval_messages = initial_messages

        logging.info("Finished Agent Setup ")

        logging.info("Feedback evaluation complete.")
        updates["cod_stacer"] = cod_stacer_response
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
        ###Save Data to text Files###

        case_study = state.get("patient_context").get("case_study")
        stacer = state.get("cod_stacer")
        detailed_feedback = state.get("feedback_messages")
        detailed_feedback = ' \n '.join(item.content for item in detailed_feedback)
        transcript = state.get("formatted_transcript")
        presentation = state.get("formatted_presentation")

        # Create base directory if it doesn't exist
        base_dir = "Saved_Sessions"
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped subfolder inside Saved_Sessions
        base_dir = "Saved_Sessions"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = os.path.join(base_dir, f"session_export_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        if self.mode == LearningBuddy.STACER_MODE:
            files = {
                "case_study.txt": case_study,
                "cod_stacer.txt": stacer,
                "detailed_feedback.txt": detailed_feedback,
                "interview_transcript.txt": transcript,
                "presentation_transcript.txt": presentation,
            }
        if self.mode == LearningBuddy.MI_MODE:
            files = {
                "case_study.txt": case_study,
                "MI_EVAL.txt": stacer,
                "detailed_feedback.txt": detailed_feedback,
                "interview_transcript.txt": transcript
            }
        if self.mode == LearningBuddy.CBT_MODE:
            files = {
                "case_study.txt": case_study,
                "CBT_EVAL.txt": stacer,
                "detailed_feedback.txt": detailed_feedback,
                "interview_transcript.txt": transcript,
                "presentation_transcript.txt": presentation,
            }

        # Save each content to a text file
        for filename, content in files.items():
            try:
                file_path = os.path.join(folder_name, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Saved: {filename}")
            except Exception as e:
                logging.info(f"Error saving {filename}: {e}")
                continue

        logging.info("Session data saved successfully.")

        # Update session metadata
        session_metadata = state.get("session_metadata", {})
        session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_exchanges": len(
                [m for m in state.get("interview_messages", []) if type(m) in [AIMessage, HumanMessage]]) // 2,
            "session_duration_minutes": self._calculate_session_duration(session_metadata.get("start_time")),
            "completion_status": "completed"
        })

        logging.info("Session completed successfully.")
        updates = {
            "session_metadata": session_metadata,
            "current_phase": "completed",
            "session_active": False,
            "folder_name": folder_name
        }
        return updates

    def end_session(self):
        """ End session using feedback workflow
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

    ###############################################################
    ###          Workflow 4: Virtual Evaluator Chatbot          ###
    ###############################################################
    #   This workflow handles one turn of the virtual evaluator feedback and Q&A:
    #            1. Read user input
    #            2. Generate evaluator response
    #
    #   Public function to use this workflow: send_message()

    def _build_evaluator_workflow(self):
        """Build workflow for evaluator bot – single-turn message processing."""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("process_input", RunnableLambda(self._process_user_input_evaluator))
        workflow.add_node("generate_feedback", RunnableLambda(self._generate_evaluator_response))
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_feedback")
        workflow.add_edge("generate_feedback", END)

        logging.info("Evaluator workflow built successfully.")
        return workflow.compile()

    def _process_user_input_evaluator(self, state: LBState):
        """Process user input during evaluator interaction.
           Appends entries to evaluator_chatlog.
        """
        user_input = state.get("user_input")
        if not user_input:
            return state

        logging.info(f"Processing evaluator user input: {user_input[:50]}...")

        # Append to separate evaluator-specific chatlog
        chatlog = state.get("evaluator_chatlog", [])
        chatlog.append({
            "type": "human",
            "role": "user",
            "content": user_input
        })

        return {
            **state,
            "evaluator_chatlog": chatlog,
            "current_phase": "generate_eval_feedback",
            "user_input": None
        }

    def _generate_evaluator_response(self, state: LBState):
        """Generate response from evaluator agent using transcript and scoring."""
        logging.info("Generating evaluator feedback...")

        try:
            eval_chatlog = state.get("evaluator_chatlog", [])

            # Build full system message with context
            langchain_messages = []
            for msg in eval_chatlog:
                if msg.get('type') == 'system':
                    langchain_messages.append(SystemMessage(content=msg['content']))
                elif msg.get('type') == 'human':
                    langchain_messages.append(HumanMessage(content=msg['content']))
                elif msg.get('type') == 'ai':
                    langchain_messages.append(AIMessage(content=msg['content']))

            response = self.llm_mini_virtualEvaluator.invoke(langchain_messages)
            ai_response = response.content

            eval_chatlog.append({
                "type": "ai",
                "role": "evaluator",
                "content": ai_response
            })

            logging.info("Evaluator response generated.")

            return {
                **state,
                "evaluator_chatlog": eval_chatlog,
                "ai_response": ai_response,
                "evaluator_session_active": True,
                "current_phase": "eval_feedback_complete"
            }

        except Exception as e:
            logging.error(f"Evaluator response failed: {str(e)}")
            return {**state, "error_message": f"Evaluator error: {str(e)}"}

    def send_evaluator_message(self, user_input: str):
        """Public entrypoint for evaluator chat."""
        if not self.current_state or not self.current_state.get("evaluator_session_active"):
            logging.error("No active evaluation session.")
            return {"error": "No active session"}

        try:
            message_state = {**self.current_state, "user_input": user_input}
            result = self.evaluator_graph.invoke(message_state)
            self.current_state = result

            return {
                "ai_response": result.get("ai_response"),
                "evaluator_session_active": result.get("evaluator_session_active", True),
                "current_phase": "evaluator_chat",
                "chatlog": result.get("chatlog", [])
            }
        except Exception as e:
            logging.error(f"Error in evaluator chat: {str(e)}")
            return {"error": str(e)}

    ###############################################################
    ###                Additional Utility Methods               ###
    ###############################################################

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
            "session_metadata": self.current_state.get("session_metadata", {})
        }

    def export_session_data(self):
        """Export session data for analysis"""
        if not self.current_state:
            return None

        return {
            "session_metadata": self.current_state.get("session_metadata", {}),
            "case_study": self.current_state.get("case_study"),
            "clinical_summary": self.current_state.get("clinical_summary"),
            "diagnosis_summary": self.current_state.get("diagnosis_summary"),
            "chatlog": self.current_state.get("chatlog", []),
            "feedback": self.current_state.get("feedback"),
            "session_statistics": self.get_session_statistics(),
            "cod_stacer": self.current_state.get("cod_stacer"),
        }

    def _calculate_session_duration(self, start_time_iso: str) -> float:
        """Calculate session duration in minutes"""
        try:
            start_time = datetime.fromisoformat(start_time_iso)
            duration = datetime.now() - start_time
            return duration.total_seconds() / 60
        except:
            return 0.0


    def _load_random_case_profile(self, directory='Patient_Profiles'):
        """ Returns a randomly selected patient profile"""

        # Select Patients available in the current mode
        if self.mode == LearningBuddy.STACER_MODE or self.mode == LearningBuddy.CBT_MODE:
            directory += "/STACER"
        elif self.mode == LearningBuddy.MI_MODE:
            directory += "/MI"
        # Add more modes ...

        files = [f for f in os.listdir(directory)]
        if not files:
            raise FileNotFoundError("No patient profiles found in the directory.")
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        return self.load_patient_profile(full_path)

    def _load_prompt(self, prompt_name: str) -> str:
        """ Returns the selected prompt"""
        # Standard LB STACER Evaluation
        if self.mode == LearningBuddy.STACER_MODE:
            if prompt_name == 'Clinical Summary':
                filepath = 'Prompts/STACER/clinical_summary_prompt.txt'
            elif prompt_name == 'Virtual Patient':
                filepath = 'Prompts/STACER/virtual_patient_prompt.txt'
            elif prompt_name == 'Verification':
                filepath = 'Prompts/STACER/patient_profile_verification_prompt.txt'
            elif prompt_name == 'Feedback':
                filepath = 'Prompts/STACER/feedback_agent_prompt.txt'
            elif prompt_name == 'STACER Prompt':
                filepath = 'Prompts/STACER/STACER_form_prompt.txt'
            elif prompt_name == 'Virtual Evaluator':
                filepath = 'Prompts/STACER/virtual_evaluator_prompt.txt'
            else:
                raise FileNotFoundError(f"File not found. Invalid Prompt Name: {prompt_name}")

        # Motivational Interviewing Evaluation
        elif self.mode == LearningBuddy.MI_MODE:
            if prompt_name == 'Clinical Summary':
                filepath = 'Prompts/MI/clinical_summary_prompt.txt'
            elif prompt_name == 'Virtual Patient':
                filepath = 'Prompts/MI/virtual_patient_prompt.txt'
            elif prompt_name == 'Verification':
                filepath = 'Prompts/MI/patient_profile_verification_prompt.txt'
            elif prompt_name == 'Feedback':
                filepath = 'Prompts/MI/feedback_agent_prompt.txt'
            elif prompt_name == 'STACER Prompt':
                filepath = 'Prompts/MI/MI_form_prompt.txt'
            elif prompt_name == 'Virtual Evaluator':
                filepath = 'Prompts/MI/virtual_evaluator_prompt.txt'
            else:
                raise FileNotFoundError(f"File not found. Invalid Prompt Name: {prompt_name}")

        # Motivational Interviewing Evaluation
        elif self.mode == LearningBuddy.CBT_MODE:
            if prompt_name == 'Clinical Summary':
                filepath = 'Prompts/CBT/clinical_summary_prompt.txt'
            elif prompt_name == 'Virtual Patient':
                filepath = 'Prompts/CBT/virtual_patient_prompt.txt'
            elif prompt_name == 'Verification':
                filepath = 'Prompts/CBT/patient_profile_verification_prompt.txt'
            elif prompt_name == 'Feedback':
                filepath = 'Prompts/CBT/feedback_agent_prompt.txt'
            elif prompt_name == 'STACER Prompt':
                filepath = 'Prompts/CBT/CBT_form_prompt.txt'
            elif prompt_name == 'Virtual Evaluator':
                filepath = 'Prompts/CBT/virtual_evaluator_prompt.txt'
            else:
                raise FileNotFoundError(f"File not found. Invalid Prompt Name: {prompt_name}")

        # Add more modes here...
        else:
            raise FileNotFoundError(f"File not found. Invalid Mode: {self.mode}")

        return self._read_file(filepath)

    def _load_context(self, context_name: str) -> str:
        """ Returns the selected LLM context"""

        # STACER Mode
        if self.mode == LearningBuddy.STACER_MODE:
            if context_name == 'Feedback Criteria':
                filepath = 'Context/STACER/feedback_criteria.txt'
            elif context_name == 'Evaluation Form':
                filepath = 'Context/STACER/COD_STACER.txt'
            else:
                raise FileNotFoundError(f"File not found. Invalid Prompt Name: {context_name}")

        # Motivational Interviewing Mode
        elif self.mode == LearningBuddy.MI_MODE:
            if context_name == 'Feedback Criteria':
                filepath = 'Context/MI/feedback_criteria_MI_5point.txt'
            elif context_name == 'Evaluation Form':
                filepath = 'Context/MI/MI_EVAL.txt'
            else:
                raise FileNotFoundError(f"File not found. Invalid Prompt Name: {context_name}")


        elif self.mode == LearningBuddy.CBT_MODE:
            if context_name == 'Feedback Criteria':
                filepath = 'Context/CBT/feedback_criteria_CBT.txt'
            elif context_name == 'Evaluation Form':
                filepath = 'Context/CBT/CBT_EVAL.txt'
            else:
                raise FileNotFoundError(f"File not found. Invalid Prompt Name: {context_name}")

        # Add more modes here...
        else:
            raise FileNotFoundError(f"File not found. Invalid Mode: {self.mode}")

        return self._read_file(filepath)
