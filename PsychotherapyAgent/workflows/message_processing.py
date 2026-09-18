"""
Message Processing Workflow for PsychotherapyAgent.

Handles message processing with optional multi-agent therapist system
(proposer + judge agents).
"""

import logging
import re
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from ..graph_state import LBState
from ..utils import sanitize_for_logging


class SelfSelectResult(BaseModel):
    """Structured proposal: 3 candidate therapist responses + which the model chooses.

    Using structured output removes the fragile text parsing of 'SELECTED RESULT:'
    that some models (GLM / DeepSeek) failed to emit consistently, which caused
    fallbacks to candidate #1.
    """
    response_1: str = Field(description="Candidate response 1 (empathic reflection)")
    response_2: str = Field(description="Candidate response 2 (collaborative reflection plus question)")
    response_3: str = Field(description="Candidate response 3 (warm reflection of change talk)")
    selected_index: int = Field(ge=1, le=3, description="Index (1, 2, or 3) of the best candidate")


class SelectionResult(BaseModel):
    """Structured selection: the model picks an index from already-generated candidates.

    Used as a retry/fallback so the LLM (not a hard-coded rule) makes the choice even when
    the full self-select structured call failed to return JSON.
    """
    selected_index: int = Field(ge=1, le=3, description="Index (1, 2, or 3) of the best candidate")


class MessageWorkflow:
    """Workflow for processing messages during therapy session."""
    
    def __init__(self, llm_manager, prompt_manager, use_multi_agent: bool = True,
                 proposer_index: int = 3, judge_index: int = 4,
                 self_select_index: int = 5,
                 selection_mode: str = "judge"):
        """
        Initialize the workflow.
        
        Args:
            llm_manager: LLMManager instance
            prompt_manager: PromptManager instance
            use_multi_agent: If True, uses proposer+judge system
            proposer_index: CSV index for proposer agent (default: 3, row 5 = 3_MultiAgent_Proposer)
            judge_index: CSV index for judge agent (default: 4, row 6 = 4_MultiAgent_Judge)
            self_select_index: CSV index for the single-call self-selection prompt
                               (default: 5, row 7 = 5_Proposer_SelfSelect)
            selection_mode: How to pick among the candidate responses.
                            'judge' -> use the learned judge agent (full multi-agent).
                            'self'  -> the proposer both generates 3 candidates and selects the
                                       best one in a SINGLE LLM call (true single-agent control
                                       with matched 1-call cost).
        """
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.use_multi_agent = use_multi_agent
        self.proposer_index = proposer_index
        self.judge_index = judge_index
        self.self_select_index = self_select_index
        self.selection_mode = selection_mode
    
    def build(self):
        """Build and return the compiled message workflow."""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("process_input", RunnableLambda(self._process_user_input))
        
        if self.use_multi_agent:
            # Multi-agent mode: proposer → judge → patient response
            workflow.add_node("propose_responses", RunnableLambda(self._propose_therapist_responses))
            workflow.add_node("judge_responses", RunnableLambda(self._judge_therapist_responses))
            workflow.add_node("generate_response", RunnableLambda(self._generate_ai_response))
            workflow.set_entry_point("process_input")
            if self.selection_mode == "self":
                # Single-call self-selection control: proposer generates AND selects, judge is skipped.
                workflow.add_edge("process_input", "propose_responses")
                workflow.add_edge("propose_responses", "generate_response")
                logging.info("Message processing workflow built with single-call self-selection (no judge).")
            else:
                workflow.add_edge("process_input", "propose_responses")
                workflow.add_edge("propose_responses", "judge_responses")
                workflow.add_edge("judge_responses", "generate_response")
            workflow.add_edge("generate_response", END)
            logging.info("Message processing workflow built with multi-agent therapist system.")
        else:
            # Original mode: direct user input → patient response
            workflow.add_node("generate_response", RunnableLambda(self._generate_ai_response_original))
            workflow.set_entry_point("process_input")
            workflow.add_edge("process_input", "generate_response")
            workflow.add_edge("generate_response", END)
            logging.info("Message processing workflow built with original single-agent mode.")
        
        return workflow.compile()
    
    def _process_user_input(self, state: LBState):
        """Process user input during interview."""
        interview_messages = state.get("interview_messages", [])
        
        if self.use_multi_agent:
            # Multi-agent mode: Generate therapist response from conversation history
            # The last message in interview_messages is typically the patient's previous response
            # (or initial system message for first exchange)
            if not interview_messages:
                logging.error("No interview messages found for multi-agent mode.")
                return state
            logging.info(f"Multi-agent mode: Generating therapist response from conversation history ({len(interview_messages)} messages)...")
            next_phase = "propose_responses"
        else:
            # Single-agent mode: Expect user input (therapist message)
            user_input = interview_messages[-1].content if interview_messages else None
            logging.info(f"Processing user input: ...")
            
            if not user_input:
                logging.error("User input is Empty...")
                return state
            next_phase = "generate_response"
        
        return {
            "current_phase": next_phase,
        }
    
    def _propose_therapist_responses(self, state: LBState):
        """Generate 3 candidate therapist responses using the proposer agent."""
        logging.info("Therapist proposer agent generating 3 candidate responses...")
        
        try:
            interview_messages = state.get("interview_messages", [])
            
            # Load proposer prompt from CSV
            proposer_prompt = self.prompt_manager.load_therapist_prompt_from_csv(self.proposer_index)
            logging.info(f"Loaded proposer prompt from CSV (index {self.proposer_index})")
            
            # Extract patient profile from first SystemMessage if it exists
            patient_profile = None
            if interview_messages and isinstance(interview_messages[0], SystemMessage):
                # The first SystemMessage contains Virtual Patient prompt + patient profile
                # Extract ONLY factual background info, excluding:
                # - Role instructions ("You are playing the role...")
                # - Presenting Concern (verbatim patient dialogue examples)
                # - Behavioral Presentation, Simulation Instructions, etc.
                content = interview_messages[0].content
                
                # Find Background Information section
                if "### **Background Information**" in content:
                    profile_start = content.find("### **Background Information**")
                    # Find the end of Background Information (before next section)
                    remaining_content = content[profile_start:]
                    lines = remaining_content.split('\n')
                    factual_lines = []
                    in_background = False
                    for line in lines:
                        if "### **Background Information**" in line:
                            in_background = True
                            factual_lines.append(line)
                        elif line.strip() == "---" and in_background:
                            # End of Background Information section
                            break
                        elif in_background and not line.strip().startswith("* **Presenting Concern"):
                            factual_lines.append(line)
                        elif "Presenting Concern" in line or "Behavioral Presentation" in line:
                            # Stop before these sections
                            break
                    
                    if factual_lines:
                        patient_profile = '\n'.join(factual_lines)
            
            # Filter to only HumanMessage and AIMessage (conversation history)
            conversation_history = [
                msg for msg in interview_messages 
                if isinstance(msg, (HumanMessage, AIMessage))
            ]
            logging.info(f"Filtered conversation history: {len(conversation_history)} messages")
            
            # Build messages: therapist prompt + patient profile as context + conversation
            messages = [SystemMessage(content=proposer_prompt)]
            
            if patient_profile:
                # Add patient profile as background information (not as role instruction)
                messages.append(SystemMessage(content=f"PATIENT BACKGROUND INFORMATION:\n\n{patient_profile}"))
                logging.info("Added patient profile as background context")
            
            messages.extend(conversation_history)

            # Single-call self-selection control: the ONE proposer call both generates the 3
            # candidates AND picks the best one. The prompt lives in the CSV (5_Proposer_SelfSelect)
            # so no selection text is hardcoded here. This is the true single-agent ablation
            # (1 LLM call vs the full system's 2), so any performance gap cannot be attributed
            # to model count or a separate judge role.
            if self.selection_mode == "self":
                # Build the same message structure as the normal proposer path, but use the
                # dedicated single-call self-selection prompt from the CSV.
                self_select_prompt = self.prompt_manager.load_therapist_prompt_from_csv(self.self_select_index)
                logging.info(f"Loaded self-select prompt from CSV (index {self.self_select_index})")

                messages = [
                    SystemMessage(content=self_select_prompt),
                    *([SystemMessage(content=f"PATIENT BACKGROUND INFORMATION:\n\n{patient_profile}")]
                      if patient_profile else []),
                    *conversation_history
                ]
                # Single-call self-selection control: the ONE proposer call both generates the 3
            # candidates AND picks the best one. The prompt lives in the CSV (5_Proposer_SelfSelect)
            # so no selection text is hardcoded here. This is the true single-agent ablation
            # (1 LLM call vs the full system's 2), so any performance gap cannot be attributed
            # to model count or a separate judge role.
            if self.selection_mode == "self":
                # Build the same message structure as the normal proposer path, but use the
                # dedicated single-call self-selection prompt from the CSV.
                self_select_prompt = self.prompt_manager.load_therapist_prompt_from_csv(self.self_select_index)
                logging.info(f"Loaded self-select prompt from CSV (index {self.self_select_index})")

                messages = [
                    SystemMessage(content=self_select_prompt),
                    *([SystemMessage(content=f"PATIENT BACKGROUND INFORMATION:\n\n{patient_profile}")]
                      if patient_profile else []),
                    *conversation_history
                ]

                # First attempt: STRUCTURED OUTPUT so the model must return the 3 candidates plus a
                # chosen index as typed fields. This removes the fragile 'SELECTED RESULT:' text
                # parsing that some models (GLM / DeepSeek) did not reliably emit (causing fallbacks
                # to candidate #1).
                selected_response = None
                candidates = []
                try:
                    structured_llm = self.llm_manager.llm_therapist_proposer.with_structured_output(SelfSelectResult)
                    result = structured_llm.invoke(messages)
                    candidates = [result.response_1, result.response_2, result.response_3]
                    idx = max(1, min(3, getattr(result, 'selected_index', 1))) - 1
                    selected_response = candidates[idx]
                    logging.info("Self-selection (structured output) produced the chosen therapist response.")
                except Exception as e:
                    # Some models (e.g. GLM) ignore the structured-output contract and return plain
                    # text ('RESPONSE 1: ...', 'Selected Response: ...') instead of JSON. Retry:
                    # first re-invoke WITHOUT schema enforcement to recover the generated candidates,
                    # then have the LLM explicitly choose among them. The selection is always made by
                    # the model, never by a hard-coded rule.
                    logging.warning(f"Self-selection structured output failed ({e}); retrying unstructured.")
                    try:
                        plain_result = self.llm_manager.llm_therapist_proposer.invoke(messages)
                        raw_text = plain_result.content
                        recovered = self._parse_candidate_responses(raw_text, interview_messages)
                        # Prefer an explicit selection tag the model already emitted...
                        selected = self._extract_selected_response(raw_text, interview_messages)
                        candidates = recovered
                        if selected:
                            selected_response = selected
                            logging.info("Self-selection (unstructured retry) recovered an explicit choice.")
                        else:
                            # ...otherwise ask the LLM to pick among the recovered candidates.
                            selected_response, _ = self._llm_select_candidate(recovered, interview_messages)
                            if selected_response:
                                logging.info("Self-selection (unstructured retry + LLM pick) chosen.")
                    except Exception as e2:
                        logging.warning(f"Self-selection unstructured retry also failed ({e2}).")

                if not selected_response:
                    logging.warning("No LLM-selected therapist response produced this turn.")

                _parsed_for_log = candidates or []

                conversation_history = state.get("judge_proposer_conversation", [])
                turn_number = sum(1 for msg in interview_messages if isinstance(msg, HumanMessage)) + 1
                turn_record = {
                    "turn_number": turn_number,
                    "candidate_responses": _parsed_for_log,
                    "judge_request": "SELF-SELECTION (single call): proposer also selected",
                    "judge_response": "Self-selected within the proposer call",
                    "selected_response_index": "self",
                    "selected_response": selected_response
                }
                conversation_history.append(turn_record)

                return {
                    "selected_therapist_response": selected_response,
                    "judge_proposer_conversation": conversation_history,
                    "current_phase": "generate_response"
                }

            # Generate 3 candidate responses
            response = self.llm_manager.llm_therapist_proposer.invoke(messages)
            response_text = response.content
            
            # Parse the 3 responses
            candidate_responses = self._parse_candidate_responses(response_text, interview_messages)
            
            logging.info(f"Generated {len(candidate_responses)} candidate therapist responses.")
            
            return {
                "therapist_candidates": candidate_responses,
                "current_phase": "judge_responses"
            }
        
        except Exception as e:
            logging.error(f"Error in therapist proposer: {str(e)}")
            fallback_response = "I understand. Can you tell me more about that?"
            return {
                "therapist_candidates": [fallback_response, fallback_response, fallback_response],
                "current_phase": "judge_responses",
                "error": f"Therapist proposer error: {str(e)}"
            }
    
    def _parse_candidate_responses(self, response_text: str, interview_messages: list) -> list:
        """Parse candidate responses from proposer output."""
        candidate_responses = []
        lines = response_text.split('\n')
        current_response = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('RESPONSE 1:') or line.startswith('Response 1:') or line.startswith('1:'):
                if current_response:
                    candidate_responses.append(' '.join(current_response).strip())
                current_response = [line.split(':', 1)[1].strip() if ':' in line else line]
            elif line.startswith('RESPONSE 2:') or line.startswith('Response 2:') or line.startswith('2:'):
                if current_response:
                    candidate_responses.append(' '.join(current_response).strip())
                current_response = [line.split(':', 1)[1].strip() if ':' in line else line]
            elif line.startswith('RESPONSE 3:') or line.startswith('Response 3:') or line.startswith('3:'):
                if current_response:
                    candidate_responses.append(' '.join(current_response).strip())
                current_response = [line.split(':', 1)[1].strip() if ':' in line else line]
            elif line and current_response:
                current_response.append(line)
        
        if current_response:
            candidate_responses.append(' '.join(current_response).strip())
        
        # Ensure we have exactly 3 responses
        if len(candidate_responses) < 3:
            if len(candidate_responses) == 0:
                # Fallback: try to regenerate with clearer instructions.
                # Use the proposer prompt from CSV (index 3) instead of the removed
                # 'Virtual Therapist' key, so this path never throws on a valid config.
                base_prompt = self.prompt_manager.load_therapist_prompt_from_csv(self.proposer_index)
                if not base_prompt:
                    base_prompt = ("You are an expert mental health counselor conducting a therapy session. "
                                   "Your task: Generate exactly 3 different response options.")
                fallback_prompt = f"""{base_prompt}

Generate exactly 3 different therapist responses. Format strictly as:
1. [first response]
2. [second response]
3. [third response]"""
                fallback_messages = [
                    SystemMessage(content=fallback_prompt),
                    *interview_messages
                ]
                fallback_response = self.llm_manager.llm_therapist_proposer.invoke(fallback_messages)
                fallback_text = fallback_response.content
                # Try to extract numbered items
                matches = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', fallback_text, re.DOTALL)
                if len(matches) >= 3:
                    candidate_responses = [m.strip() for m in matches[:3]]
                else:
                    # Last resort: split the response into 3 parts
                    parts = fallback_text.split('\n\n')
                    if len(parts) >= 3:
                        candidate_responses = [p.strip() for p in parts[:3]]
                    else:
                        candidate_responses = [fallback_text.strip()] * 3
            else:
                # Pad with variations if we have fewer than 3
                while len(candidate_responses) < 3:
                    candidate_responses.append(
                        candidate_responses[-1] if candidate_responses 
                        else "I understand. Can you tell me more?"
                    )
        
        # Keep only first 3
        return candidate_responses[:3]

    def _llm_select_candidate(self, candidates: list, interview_messages: list):
        """Ask the LLM to explicitly choose one of the already-generated candidates.

        Used when the full self-select structured call failed to return JSON but we could
        recover the 3 candidates from the raw text. The choice is made by the model (with a
        structured-output schema), not by a hard-coded rule. Returns (selected_response, tried).

        tried is True if a fresh LLM selection call was actually attempted/completed.
        """
        if len(candidates) < 3:
            return None, False
        try:
            select_prompt = (
                "You are selecting the best therapist response for the current turn.\n\n"
                "Here are 3 candidate responses from the proposer:\n"
                f"1. {candidates[0]}\n"
                f"2. {candidates[1]}\n"
                f"3. {candidates[2]}\n\n"
                "Pick the one (index 1, 2, or 3) that is the most empathic, most aligned with "
                "Motivational Interviewing, and most helpful for the patient right now."
            )
            select_messages = [
                SystemMessage(content=select_prompt),
                *interview_messages,
            ]
            select_llm = self.llm_manager.llm_therapist_proposer.with_structured_output(SelectionResult)
            sel = select_llm.invoke(select_messages)
            idx = max(1, min(3, getattr(sel, 'selected_index', 1))) - 1
            return candidates[idx], True
        except Exception as e:
            logging.warning(f"LLM candidate selection failed ({e}).")
            return None, True

    def _extract_selected_response(self, response_text: str, interview_messages: list):
        """Robustly extract the self-selected therapist response from a single-call proposal.

        Handles the formats observed across gpt-5-mini / Qwen / GLM:
          1. Explicit '<tag>: <response>' result tags (SELECTED RESULT, Selected, Best response, etc.)
          2. Reference to a candidate number ('Response 2 is best', 'Best response is 1', 'I choose 3')
          3. None -> returns None so the caller falls back to candidate #1.

        Returns:
            The chosen response string, or None if no reliable selection is found.
        """
        candidates = self._parse_candidate_responses(response_text, interview_messages)

        # 1) Explicit result tags carrying the verbatim chosen text.
        for pattern in [
            r'SELECTED RESULT\s*:\s*(.+)$',
            r'Selected Response\s*:\s*(.+)$',
            r'Best response\s*[:\-]\s*(.+)$',
            r'Final Response\s*[:\-]\s*(.+)$',
        ]:
            m = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()
                if val and len(val) > 5:
                    return val

        # 2) Reference to a candidate number (e.g. "Response 2 is best", "I choose 1").
        if len(candidates) >= 1:
            for pattern in [
                r'Best response (?:is|:)\s*([123])\b',
                r'Response ([123])\s+is\s+best',
                r'(?:Selected|I select|I choose|choose)\s+(?:response\s*)?([123])\b',
                r'^([123])\s*\)\s*$',
            ]:
                m = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(candidates):
                        return candidates[idx]

        # 3) If exactly one candidate parsed cleanly and looks like a full sentence, prefer it.
        if len(candidates) == 1 and len(candidates[0]) > 20:
            return candidates[0]

        return None

    def _judge_therapist_responses(self, state: LBState):
        """Judge agent selects the best response from 3 candidates."""
        logging.info("Therapist judge agent selecting best response...")
        
        try:
            candidate_responses = state.get("therapist_candidates", [])
            interview_messages = state.get("interview_messages", [])
            
            if not candidate_responses or len(candidate_responses) < 3:
                logging.warning("Less than 3 candidate responses available, using first one.")
                selected_response = candidate_responses[0] if candidate_responses else "I understand. Can you tell me more?"
                
                # Track conversation even in error case
                conversation_history = state.get("judge_proposer_conversation", [])
                turn_number = sum(1 for msg in interview_messages if isinstance(msg, HumanMessage)) + 1
                turn_record = {
                    "turn_number": turn_number,
                    "candidate_responses": candidate_responses.copy() if candidate_responses else [],
                    "judge_request": "ERROR: Less than 3 candidate responses available",
                    "judge_response": "Using first available response",
                    "selected_response_index": 1,
                    "selected_response": selected_response
                }
                conversation_history.append(turn_record)
                
                return {
                    "selected_therapist_response": selected_response,
                    "judge_proposer_conversation": conversation_history,
                    "current_phase": "generate_response"
                }

            # Load judge prompt from CSV
            judge_prompt = self.prompt_manager.load_therapist_prompt_from_csv(self.judge_index)
            logging.info(f"Loaded judge prompt from CSV (index {self.judge_index})")
            
            # Format candidates for judge
            candidates_text = "\n\n".join([
                f"CANDIDATE RESPONSE {i+1}:\n{candidate}" 
                for i, candidate in enumerate(candidate_responses)
            ])
            
            # Extract patient profile from first SystemMessage if it exists
            patient_profile = None
            if interview_messages and isinstance(interview_messages[0], SystemMessage):
                # The first SystemMessage contains Virtual Patient prompt + patient profile
                # Extract ONLY factual background info, excluding:
                # - Role instructions ("You are playing the role...")
                # - Presenting Concern (verbatim patient dialogue examples)
                # - Behavioral Presentation, Simulation Instructions, etc.
                content = interview_messages[0].content
                
                # Find Background Information section
                if "### **Background Information**" in content:
                    profile_start = content.find("### **Background Information**")
                    # Find the end of Background Information (before next section)
                    remaining_content = content[profile_start:]
                    lines = remaining_content.split('\n')
                    factual_lines = []
                    in_background = False
                    for line in lines:
                        if "### **Background Information**" in line:
                            in_background = True
                            factual_lines.append(line)
                        elif line.strip() == "---" and in_background:
                            # End of Background Information section
                            break
                        elif in_background and not line.strip().startswith("* **Presenting Concern"):
                            factual_lines.append(line)
                        elif "Presenting Concern" in line or "Behavioral Presentation" in line:
                            # Stop before these sections
                            break
                    
                    if factual_lines:
                        patient_profile = '\n'.join(factual_lines)
            
            # Filter to only HumanMessage and AIMessage (conversation history)
            context_messages = [
                msg for msg in interview_messages 
                if isinstance(msg, (HumanMessage, AIMessage))
            ]
            logging.info(f"Filtered conversation history for judge: {len(context_messages)} messages")
            
            # Build messages: judge prompt + patient profile as context + conversation + evaluation request
            messages = [SystemMessage(content=judge_prompt)]
            
            if patient_profile:
                # Add patient profile as background information (not as role instruction)
                messages.append(SystemMessage(content=f"PATIENT BACKGROUND INFORMATION:\n\n{patient_profile}"))
                logging.info("Added patient profile as background context for judge")
            
            messages.extend(context_messages)
            messages.append(
                HumanMessage(content=f"Here are the 3 candidate therapist responses to evaluate:\n\n{candidates_text}\n\nWhich response (1, 2, or 3) is the best? Respond with the number and a brief explanation.")
            )
            
            # Get judge's selection
            judge_response = self.llm_manager.llm_therapist_judge.invoke(messages)
            judge_text = judge_response.content.strip()
            
            # Extract the selected number
            match = re.search(r'\b([123])\b', judge_text)
            if match:
                selected_index = int(match.group(1)) - 1  # Convert to 0-based index
            else:
                logging.warning("Could not parse judge selection, using first response.")
                selected_index = 0
            
            # Ensure index is valid
            if selected_index < 0 or selected_index >= len(candidate_responses):
                selected_index = 0
            
            selected_response = candidate_responses[selected_index]
            
            logging.info(f"Judge selected response {selected_index + 1} out of {len(candidate_responses)} candidates.")
            
            # Track judge-proposer conversation for this turn
            # Get the judge request (the HumanMessage content sent to judge)
            judge_request_text = messages[-1].content if messages else ""
            
            # Get or initialize judge-proposer conversation history
            conversation_history = state.get("judge_proposer_conversation", [])
            
            # Calculate turn number (number of therapist responses so far)
            # Count HumanMessages in interview_messages (these are therapist responses)
            turn_number = sum(1 for msg in interview_messages if isinstance(msg, HumanMessage)) + 1
            
            # Store this turn's conversation
            turn_record = {
                "turn_number": turn_number,
                "candidate_responses": candidate_responses.copy(),
                "judge_request": judge_request_text,
                "judge_response": judge_text,
                "selected_response_index": selected_index + 1,  # Store as 1-based for readability
                "selected_response": selected_response
            }
            conversation_history.append(turn_record)
            
            return {
                "selected_therapist_response": selected_response,
                "judge_proposer_conversation": conversation_history,
                "current_phase": "generate_response"
            }
        
        except Exception as e:
            logging.error(f"Error in therapist judge: {str(e)}")
            candidate_responses = state.get("therapist_candidates", [])
            selected_response = candidate_responses[0] if candidate_responses else "I understand. Can you tell me more?"
            
            # Track conversation even in error case
            conversation_history = state.get("judge_proposer_conversation", [])
            interview_messages = state.get("interview_messages", [])
            turn_number = sum(1 for msg in interview_messages if isinstance(msg, HumanMessage)) + 1
            turn_record = {
                "turn_number": turn_number,
                "candidate_responses": candidate_responses.copy() if candidate_responses else [],
                "judge_request": f"ERROR: {str(e)}",
                "judge_response": "Error occurred, using fallback response",
                "selected_response_index": 1,
                "selected_response": selected_response
            }
            conversation_history.append(turn_record)
            
            return {
                "selected_therapist_response": selected_response,
                "judge_proposer_conversation": conversation_history,
                "current_phase": "generate_response",
                "error": f"Therapist judge error: {str(e)}"
            }
    
    def _generate_ai_response_original(self, state: LBState):
        """Generate AI patient response in original mode (direct user input)."""
        logging.info("Generating AI patient response (original mode)...")
        
        try:
            interview_messages = state.get("interview_messages", [])
            response = self.llm_manager.llm_mini_virtualPatient.invoke(interview_messages)
            
            updated_messages = [
                *interview_messages,
                AIMessage(content=response.content)
            ]
            
            logging.info("AI patient response generated successfully (original mode).")
            
            return {
                "interview_messages": updated_messages,
                "current_phase": "message_complete"
            }
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}")
            return {**state, "error_message": f"AI response generation failed: {str(e)}"}
    
    def _generate_ai_response(self, state: LBState):
        """Generate AI patient response after therapist's selected response (multi-agent mode)."""
        logging.info("Generating AI patient response (multi-agent mode)...")
        
        try:
            interview_messages = state.get("interview_messages", [])
            selected_therapist_response = state.get("selected_therapist_response")
            
            if not selected_therapist_response:
                logging.error("No selected therapist response found in state!")
                return {**state, "error_message": "No selected therapist response found"}
            
            # Log current state of interview_messages before adding therapist response
            logging.info(f"Current interview_messages count: {len(interview_messages)}")
            if interview_messages:
                last_msg_type = type(interview_messages[-1]).__name__
                logging.info(f"Last message type: {last_msg_type}")
            
            # Add therapist response to conversation history (keep all previous messages)
            interview_messages_with_therapist = [
                *interview_messages,
                HumanMessage(content=selected_therapist_response)
            ]
            logging.info(f"Therapist selected response added (length: {len(selected_therapist_response)} chars)")
            
            response = self.llm_manager.llm_mini_virtualPatient.invoke(interview_messages_with_therapist)
            
            # Add patient response to complete the exchange
            updated_messages = [
                *interview_messages_with_therapist,
                AIMessage(content=response.content)
            ]
            
            logging.info("AI patient response generated successfully (multi-agent mode).")
            
            return {
                "interview_messages": updated_messages,
                "ai_response": response.content,  # Also return for API compatibility
                "current_phase": "message_complete"
            }
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}", exc_info=True)
            return {**state, "error_message": f"AI response generation failed: {str(e)}"}

