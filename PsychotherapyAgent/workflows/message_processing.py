"""
Message Processing Workflow for PsychotherapyAgent.

Handles message processing with optional multi-agent therapist system
(proposer + judge agents).
"""

import logging
import re
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from ..graph_state import LBState
from ..utils import sanitize_for_logging


class MessageWorkflow:
    """Workflow for processing messages during therapy session."""
    
    def __init__(self, llm_manager, prompt_manager, use_multi_agent: bool = True,
                 proposer_index: int = 2, judge_index: int = 3):
        """
        Initialize the workflow.
        
        Args:
            llm_manager: LLMManager instance
            prompt_manager: PromptManager instance
            use_multi_agent: If True, uses proposer+judge system
            proposer_index: CSV index for proposer agent (default: 2)
            judge_index: CSV index for judge agent (default: 3)
        """
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.use_multi_agent = use_multi_agent
        self.proposer_index = proposer_index
        self.judge_index = judge_index
    
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
            
            messages = [
                SystemMessage(content=proposer_prompt),
                *interview_messages
            ]
            
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
                # Fallback: try to regenerate with clearer instructions
                base_prompt = self.prompt_manager.load_prompt('Virtual Therapist')
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
    
    def _judge_therapist_responses(self, state: LBState):
        """Judge agent selects the best response from 3 candidates."""
        logging.info("Therapist judge agent selecting best response...")
        
        try:
            candidate_responses = state.get("therapist_candidates", [])
            interview_messages = state.get("interview_messages", [])
            
            if not candidate_responses or len(candidate_responses) < 3:
                logging.warning("Less than 3 candidate responses available, using first one.")
                selected_response = candidate_responses[0] if candidate_responses else "I understand. Can you tell me more?"
                return {
                    "selected_therapist_response": selected_response,
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
            
            # Build messages list
            context_messages = interview_messages[:-1] if interview_messages else []
            messages = [
                SystemMessage(content=judge_prompt),
                *context_messages,
                HumanMessage(content=f"Here are the 3 candidate therapist responses to evaluate:\n\n{candidates_text}\n\nWhich response (1, 2, or 3) is the best? Respond with the number and a brief explanation.")
            ]
            
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
            
            return {
                "selected_therapist_response": selected_response,
                "current_phase": "generate_response"
            }
        
        except Exception as e:
            logging.error(f"Error in therapist judge: {str(e)}")
            candidate_responses = state.get("therapist_candidates", [])
            selected_response = candidate_responses[0] if candidate_responses else "I understand. Can you tell me more?"
            return {
                "selected_therapist_response": selected_response,
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

