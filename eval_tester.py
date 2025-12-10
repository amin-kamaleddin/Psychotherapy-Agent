import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import SecretStr
import random
from datetime import datetime
import os
from pathlib import Path
import pandas as pd


# Defining LangGraph State Variables
class LBState(TypedDict):
    case_study: Optional[str]
    diagnosis_summary: str
    clinical_summary: str
    chatlog: List[Dict[str, Any]]
    feedback: str
    current_phase: str
    session_active: bool
    user_input: Optional[str]
    ai_response: Optional[str]
    session_metadata: Dict[str, Any]
    error_message: Optional[str]
    next_action: Optional[str]
    cod_stacer: str
    presentation: List[str]
    evaluator_chatlog: List[Dict[str, Any]]
    evaluator_session_active: bool
    formatted_transcript: str
    formatted_presentation: str
    folder_name: str

class LearningBuddy:

    def __init__(self, api_key: str):
        self.api_key = SecretStr(api_key)
        self.llm_mini_verification = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=self.api_key)
        self.llm_mini_summary = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, api_key=self.api_key)
        self.llm_mini_virtualPatient = ChatOpenAI(model="gpt-4.1-mini", temperature=1, api_key=self.api_key)
        self.llm_feedback = ChatOpenAI(model="gpt-4.1", temperature=0.1, api_key=self.api_key)
        self.llm_mini_virtualEvaluator = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, api_key=self.api_key)

        def _try_build(name):
            builder = getattr(self, name, None)
            if callable(builder):
                return builder()
            logging.debug("Skipping %s; builder not available in evaluation-only configuration.", name)
            return None

        self.init_graph = _try_build("_build_init_workflow")
        self.message_graph = _try_build("_build_message_workflow")
        self.feedback_graph = _try_build("_build_feedback_workflow")
        self.evaluator_graph = _try_build("_build_evaluator_workflow")
        if self.feedback_graph is None or self.evaluator_graph is None:
            raise RuntimeError("Feedback workflows are required for evaluation mode.")
        self.voice = ""
        self.current_state = None
        logging.info("LearningBuddy initialized with separate LangGraph workflows.")

    def _build_state_from_transcript(self, transcript_rows: pd.DataFrame) -> LBState:
        """Build an LBState instance from a motivational interviewing transcript."""
        required_columns = {
            "mi_quality",
            "transcript_id",
            "video_title",
            "video_url",
            "topic",
            "utterance_id",
            "interlocutor",
            "timestamp",
            "utterance_text",
        }
        missing_columns = required_columns.difference(transcript_rows.columns)
        if missing_columns:
            raise ValueError(
                f"Transcript is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        transcript_rows = transcript_rows.sort_values("timestamp")
        first_row = transcript_rows.iloc[0]

        def _safe_str(value, fallback):
            if pd.isna(value) or value is None:
                return fallback
            value_str = str(value).strip()
            return value_str if value_str else fallback

        topic = _safe_str(first_row.get("topic"), "Unknown topic")
        video_title = _safe_str(first_row.get("video_title"), "Unknown video title")
        transcript_id = _safe_str(first_row.get("transcript_id"), "unknown_transcript")
        mi_quality = _safe_str(first_row.get("mi_quality"), "unspecified quality")

        case_study = (
            f"Topic: {topic}"
            f"Video Title: {video_title}"
            f"Transcript ID: {transcript_id}"
            f"Quality Label: {mi_quality}"
        )
        clinical_summary = (
            f"Motivational interviewing session on '{topic}' labelled '{mi_quality}'. "
            f"Source video: {video_title}."
        )

        chatlog: List[Dict[str, Any]] = []
        for _, row in transcript_rows.iterrows():
            content = _safe_str(row.get("utterance_text"), "")
            if not content:
                continue
            interlocutor = _safe_str(row.get("interlocutor"), "client").lower()
            role = "user" if interlocutor == "therapist" else "assistant"
            chatlog.append({"role": role, "content": content})

        if not chatlog:
            raise ValueError("Transcript contains no utterances.")

        timestamps = transcript_rows["timestamp"].dropna().astype(str).str.strip()
        start_timestamp = timestamps.iloc[0] if not timestamps.empty else None
        end_timestamp = timestamps.iloc[-1] if not timestamps.empty else None
        start_seconds = self._timestamp_to_seconds(start_timestamp)
        end_seconds = self._timestamp_to_seconds(end_timestamp)
        transcript_duration_minutes = self._duration_minutes_from_seconds(start_seconds, end_seconds)

        session_metadata = {
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "transcript_start_seconds": start_seconds,
            "transcript_end_seconds": end_seconds,
            "transcript_duration_minutes": transcript_duration_minutes,
        }

        state: LBState = {
            "case_study": case_study,
            "diagnosis_summary": "",
            "clinical_summary": clinical_summary,
            "chatlog": chatlog,
            "feedback": "",
            "current_phase": "evaluate_session",
            "session_active": False,
            "user_input": None,
            "ai_response": None,
            "session_metadata": session_metadata,
            "error_message": None,
            "next_action": None,
            "cod_stacer": "",
            "presentation": ["Presentation not provided in this transcript."],
            "evaluator_chatlog": [],
            "evaluator_session_active": False,
            "formatted_transcript": "",
            "formatted_presentation": "",
            "folder_name": "",
        }
        return state

    @staticmethod
    def _timestamp_to_seconds(timestamp: Optional[str]) -> Optional[float]:
        if timestamp is None:
            return None
        value = str(timestamp).strip()
        if not value:
            return None
        parts = value.split(':')
        if len(parts) != 3:
            return None
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        except ValueError:
            return None
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return float(total_seconds)

    @staticmethod
    def _duration_minutes_from_seconds(start_seconds: Optional[float], end_seconds: Optional[float]) -> Optional[float]:
        if start_seconds is None or end_seconds is None:
            return None
        if end_seconds < start_seconds:
            return None
        return round((end_seconds - start_seconds) / 60.0, 2)

    def run_directory_evaluations(self, csv_dir: str) -> Dict[str, Dict[str, str]]:
        """Run the feedback workflow for every transcript CSV in the given directory."""
        csv_path = Path(csv_dir).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"Directory not found: {csv_path}")
        if not csv_path.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: {csv_path}")

        results: Dict[str, Dict[str, str]] = {}
        for csv_file in sorted(csv_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                logging.exception("Failed to read transcript file %s: %s", csv_file, exc)
                continue

            if df.empty:
                logging.warning("Skipping empty transcript file %s", csv_file)
                continue

            if "transcript_id" not in df.columns:
                logging.error("Skipping %s because it lacks a transcript_id column", csv_file)
                continue

            for transcript_id, transcript_rows in df.groupby("transcript_id"):
                try:
                    self.current_state = self._build_state_from_transcript(transcript_rows)
                except ValueError as exc:
                    logging.error(
                        "Skipping transcript %s in %s: %s",
                        transcript_id,
                        csv_file,
                        exc,
                    )
                    continue

                result_state = self.feedback_graph.invoke(self.current_state)
                if result_state.get("error_message"):
                    logging.error(
                        "Evaluation failed for transcript %s in %s: %s",
                        transcript_id,
                        csv_file,
                        result_state["error_message"],
                    )
                    continue

                self.current_state = result_state
                results[str(transcript_id)] = {
                    "detailed_feedback": result_state.get("feedback", ""),
                    "cod_stacer": result_state.get("cod_stacer", ""),
                    "saved_session_folder": result_state.get("folder_name", ""),
                    "source_file": str(csv_file),
                }

        return results

###############################################################
    ###                Workflow 3: CBT Feedback              ###
    ###############################################################
    #   This workflow generates all static feedback after an interview ends:
    #            1. Generates detailed feedback and fills out CBT
    #            2. Initializes Feedback Agent
    #            2. Ends current session and saves all session data
    #
    #   Public function to use this workflow: end_session()

    def _build_feedback_workflow(self):
        """Build feedback workflow - linear, no loops"""
        workflow = StateGraph(state_schema=LBState)

        workflow.add_node("evaluate_session",   RunnableLambda(self._generate_feedback))
        workflow.add_node("complete_session",   RunnableLambda(self._complete_session))
        workflow.set_entry_point("evaluate_session")
        workflow.add_edge("evaluate_session", "complete_session")
        workflow.add_edge("complete_session", END)

        logging.info("Feedback workflow built successfully.")
        return workflow.compile()

    def _generate_feedback(self, state: LBState):
        """ Generate comprehensive feedback.
            First scores all defined criteria on a 1-5 scale and writes
            thorough feedback based on provided rubric.
            Then uses full feedback to fill out a formatted COD CBT.
        """
        logging.info("Starting feedback evaluation phase.")

        # Validate and format required data
        case_study = state.get("case_study")
        clinical_summary = state.get("clinical_summary")
        chatlog = state.get("chatlog", [])
        presentation = state.get("presentation", [])

        if not case_study:
            case_study = "Case study details were not supplied with this transcript."
        if not clinical_summary:
            clinical_summary = "Clinical summary missing; rely on transcript content."
        if not chatlog:
            return {**state, "error_message": "Chatlog is empty or missing from state."}
        if not presentation:
            presentation = ["Presentation not provided in this transcript."]
        elif isinstance(presentation, str):
            presentation = [presentation]

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
            m for m in chatlog
            if m.get("role") in ["user", "assistant"] and
               m.get("content") and
               m["content"].strip() != case_study.strip()
        ]

        messages = [
            SystemMessage(content=strict_feedback_prompt),
            HumanMessage(content="Here is the therapy session transcript to evaluate:")
        ]

        formatted_lines = []
        for m in conversation_messages:
            speaker = "THERAPIST" if m["role"] == "user" else "CLIENT"
            utterance = m["content"].strip()
            content = f"{speaker}: {utterance}"
            msg_cls = HumanMessage if m["role"] == "user" else AIMessage
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
        logging.info("Starting CBT generation ")
        # Generate Formatted COD CBT Form
        instruction_prompt = self._load_prompt('CBT Prompt')
        blank_stacer_file = self._load_context("Evaluation Form")

        stacer_prompt = f"""
        {instruction_prompt}

        Here is the Full Transcript:
        {formatted_transcript}
        
        Here is the full Case Presentation:
        {formatted_presentation}

        Here is the Evaluation Criteria to base your answer off of:
        {feedback_text}

        And here is the blank CBT form that you must use to format your answer:
        {blank_stacer_file}
        """

        messages = [
            SystemMessage(content=stacer_prompt),
            HumanMessage(content="Please provide the formatted answer below:")
        ]

        response = self.llm_feedback.invoke(messages)
        cod_stacer_response = response.content

        logging.info("Finished CBT generation ")
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

              This is the completed evaluation and filled out CBT form which you will use to create your responses:
              === COMPLETE EVALUATION ===
              {session_feedback}

              === COMPLETE CBT ===
              {cod_stacer_response}

              Here is the full patient profile, which can be used to answer questions about the patient:
              {case_study}

              Begin by giving a 'feedback sandwich': positive, then constructive critique, then end with positive support. 
              After that, respond to any questions the user may ask about the evaluation or the scoring.
              """

        initial_messages = [
            {
                "type": "system",
                "role": "system",
                "content": full_system_prompt.strip()
            }
        ]

        eval_chatlog = initial_messages

        logging.info("Finished Agent Setup ")

        logging.info("Feedback evaluation complete.")
        return {
            **state,
            "feedback": session_feedback,
            "cod_stacer": cod_stacer_response,
            "formatted_transcript": formatted_transcript,
            "formatted_presentation": formatted_presentation,
            "current_phase": "evaluator_active",
            "session_active": False,
            "evaluator_chatlog": eval_chatlog,
            "evaluator_session_active": True,
            "user_input": None,
            "ai_response": None
        }

    def _complete_session(self, state: LBState):
        """Complete the session by saving all session data"""

        logging.info("Saving session data...")
        ###Save Data to text Files###

        case_study = state.get("case_study")
        stacer = state.get("cod_stacer")
        detailed_feedback = state.get("feedback")
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

        files = {
            "case_study.txt": case_study,
            "MI_EVAL.txt": stacer,
            "detailed_feedback.txt": detailed_feedback,
            "interview_transcript.txt": transcript
        }
        # Save each content to a text file
        for filename, content in files.items():
            with open(os.path.join(folder_name, filename), "w", encoding="utf-8") as f:
                f.write(content)

        logging.info("Session data saved successfully.")

        # Update session metadata
        session_metadata = dict(state.get("session_metadata", {}) or {})
        duration_minutes = session_metadata.get("transcript_duration_minutes")
        if duration_minutes is None:
            duration_minutes = self._duration_minutes_from_seconds(
                session_metadata.get("transcript_start_seconds"),
                session_metadata.get("transcript_end_seconds"),
            )
        if duration_minutes is None:
            duration_minutes = 0.0

        session_metadata["transcript_duration_minutes"] = duration_minutes

        # Ensure end timestamp is captured if available from transcript
        if session_metadata.get("end_timestamp") is None:
            session_metadata["end_timestamp"] = session_metadata.get("start_timestamp")

        session_metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_exchanges": len(
                [m for m in state.get("chatlog", []) if m.get("role") in ["user", "assistant"]]) // 2,
            "session_duration_minutes": duration_minutes,
            "completion_status": "completed"
        })

        logging.info("Session completed successfully.")
        return {
            **state,
            "session_metadata": session_metadata,
            "current_phase": "completed",
            "session_active": False,
            "folder_name": folder_name
        }

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

    def _read_file(self, filepath):
        """ Reads and returns the selected file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    def load_patient_profile(self, filepath: str) -> str:
        """ Returns the selected patient profile"""
        return self._read_file(filepath)

    def _load_random_case_profile(self, directory='Patient_Profiles'):
        """ Returns a randomly selected patient profile"""

        directory += "/MI"

        files = [f for f in os.listdir(directory)]
        if not files:
            raise FileNotFoundError("No patient profiles found in the directory.")
        random_file = random.choice(files)
        full_path = os.path.join(directory, random_file)
        return self.load_patient_profile(full_path)

    def _load_prompt(self, prompt_name: str) -> str:
        """ Returns the selected prompt"""

        # Motivational Interviewing Evaluation
        if prompt_name == 'Clinical Summary':
            filepath = 'Prompts/MI/clinical_summary_prompt.txt'
        elif prompt_name == 'Virtual Patient':
            filepath = 'Prompts/MI/virtual_patient_prompt.txt'
        elif prompt_name == 'Verification':
            filepath = 'Prompts/MI/patient_profile_verification_prompt.txt'
        elif prompt_name == 'Feedback':
            filepath = 'Prompts/MI/feedback_agent_prompt.txt'
        elif prompt_name == 'CBT Prompt':
            filepath = 'Prompts/MI/MI_form_prompt.txt'
        elif prompt_name == 'Virtual Evaluator':
            filepath = 'Prompts/MI/virtual_evaluator_prompt.txt'
        else:
            raise FileNotFoundError(f"File not found. Invalid Prompt Name: {prompt_name}")

        return self._read_file(filepath)

    def _load_context(self, context_name: str) -> str:
        """ Returns the selected LLM context"""

        if context_name == 'Feedback Criteria':
            filepath = 'Context/MI/feedback_criteria_MI_5point.txt'
        elif context_name == 'Evaluation Form':
            filepath = 'Context/MI/MI_EVAL.txt'
        else:
            raise FileNotFoundError(f"File not found. Invalid Prompt Name: {context_name}")

        return self._read_file(filepath)


def extract_case_feedback_data(root_path: str) -> pd.DataFrame:
    """
    Traverse all folders in `root_path`, extract Transcript ID, Quality Label, 
    and Final Overall Score from the two text files (case_study.txt and detailed_feedback.txt),
    and return a DataFrame with columns:
        ['Transcript ID', 'Quality Label', 'Predicted Score'].
    """
    data = []

    for folder, _, files in os.walk(root_path):
        # Expect both files to be in the same folder
        case_study_path = os.path.join(folder, "case_study.txt")
        feedback_path = os.path.join(folder, "MI_EVAL.txt")

        transcript_id = None
        quality_label = None
        predicted_score = None

        # --- Extract from case_study.txt ---
        if os.path.exists(case_study_path):
            with open(case_study_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Extract "Transcript ID"
            match_id = re.search(r"Transcript ID:\s*(\d{0,3})", text)
            transcript_id = match_id.group(1) if match_id else None

            # Extract "Quality Label"
            match_quality = re.search(r"Quality Label:\s*(high|low)", text, re.IGNORECASE)
            quality_label = match_quality.group(1).lower() if match_quality else None

        # --- Extract from cod_stacer.txt ---
        if os.path.exists(feedback_path):
            with open(feedback_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Try case 1: Overall MI Adherence: 3
            match_direct = re.search(r"Overall MI Adherence:\s*(\d+)", text)

            # Try range form followed by "Score:"
            match_score = re.search(r"Score:\s*(\d+)", text)

            # (3) Try checkbox form (☑ 3 or ✓ 3)
            match_checkbox = re.search(r"[☑✓]\s*(\d+)", text)

            if match_checkbox:
                predicted_score = int(match_checkbox.group(1))
            elif match_score:
                predicted_score = int(match_score.group(1))
            elif match_direct:
                predicted_score = int(match_direct.group(1))
            else:
                predicted_score = None

        # Only add if at least one item was found
        if transcript_id or quality_label or predicted_score:
            data.append({
                "Transcript ID": transcript_id,
                "Quality Label": quality_label,
                "Predicted Score": predicted_score
            })

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Transcript ID", "Quality Label", "Predicted Score"])
    return df

def evaluate_accuracy(csv_path: str) -> float:
    """
    Evaluate prediction accuracy based on a CSV containing:
        - 'Quality Label' (values: 'high' or 'low')
        - 'Predicted Score' (integer values)

    Rules:
        - Correct if (Quality Label == 'high' and Predicted Score > 2)
        - Correct if (Quality Label == 'low' and Predicted Score == 1)
        - Otherwise incorrect

    Returns:
        accuracy (float): Percentage of correct predictions
    """

    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {"Quality Label", "Predicted Score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain the columns: {required_cols}")

    # Calculate correctness per row
    df["Correct"] = (
        ((df["Quality Label"].str.lower() == "high") & (df["Predicted Score"] > 2))
        | ((df["Quality Label"].str.lower() == "low") & (df["Predicted Score"] <= 2))
    )

    # Tally results
    correct = df["Correct"].sum()
    total = len(df)
    incorrect = total - correct
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Print summary
    print(f"✅ Total: {total}")
    print(f"✔️  Correct: {correct}")
    print(f"❌ Incorrect: {incorrect}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")

    return accuracy

def _prompt_for_value(env_key: str, prompt_text: str) -> str:
    value = os.getenv(env_key, '').strip() if os.getenv(env_key) else ''
    if value:
        return value
    try:
        return input(prompt_text).strip()
    except EOFError:
        return ''

def _format_result(transcript_id: str, payload: Dict[str, str]) -> str:
    divider = '=' * 80
    parts = [
        divider,
        f"Transcript ID: {transcript_id} (source: {payload.get('source_file', 'unknown')})",
        '-' * 80,
        'Filled CBT Form:',
        payload.get('cod_stacer', '(no CBT output)'),
        '',
        'Detailed Feedback:',
        payload.get('detailed_feedback', '(no feedback generated)'),
        '',
        f"Saved session folder: {payload.get('saved_session_folder', 'N/A')}"
    ]
    return ''.join(parts)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    transcripts_dir = _prompt_for_value('MI_TRANSCRIPTS_DIR', 'Enter the path to the MI transcript CSV directory: ')
    if not transcripts_dir:
        print('No transcript directory provided; exiting.')
        return

    api_key = _prompt_for_value('OPENAI_API_KEY', 'Enter your OpenAI / Azure OpenAI API key: ')
    if not api_key:
        print('An OpenAI-compatible API key is required to run the evaluation workflow.')
        return

    buddy = LearningBuddy(api_key=api_key)
    try:
        results = buddy.run_directory_evaluations(transcripts_dir)
    except Exception as exc:
        logging.exception('Evaluation run failed')
        print(f'Evaluation failed: {exc}')
        return

    if not results:
        print('No transcripts were processed. Verify the directory path and CSV schema.')
        return

    for transcript_id, payload in results.items():
        print(_format_result(transcript_id, payload))
    print('=' * 80)
    print('Evaluation complete.')
    
    df = extract_case_feedback_data("./Saved_Sessions")
    df.to_csv("Evaluation_Metric_Evaluator_Agent.csv")    


if __name__ == '__main__':
    main()