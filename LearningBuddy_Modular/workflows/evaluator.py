"""
Evaluator Workflow for LearningBuddy Modular.

Handles evaluator chatbot interactions for Q&A about feedback.
"""

import logging
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from ..graph_state import LBState


class EvaluatorWorkflow:
    """Workflow for evaluator chatbot interactions."""
    
    def __init__(self, llm_manager):
        """
        Initialize the workflow.
        
        Args:
            llm_manager: LLMManager instance
        """
        self.llm_manager = llm_manager
    
    def build(self):
        """Build and return the compiled evaluator workflow."""
        workflow = StateGraph(state_schema=LBState)
        
        workflow.add_node("process_input", RunnableLambda(self._process_user_input_evaluator))
        workflow.add_node("generate_feedback", RunnableLambda(self._generate_evaluator_response))
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_feedback")
        workflow.add_edge("generate_feedback", END)
        
        logging.info("Evaluator workflow built successfully.")
        return workflow.compile()
    
    def _process_user_input_evaluator(self, state: LBState):
        """Process user input during evaluator interaction."""
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
            eval_messages = state.get("evaluator_messages", [])
            
            # Build full message list from evaluator messages and chatlog
            langchain_messages = list(eval_messages)  # Start with system message
            for msg in eval_chatlog:
                if msg.get('type') == 'system':
                    langchain_messages.append(SystemMessage(content=msg['content']))
                elif msg.get('type') == 'human':
                    langchain_messages.append(HumanMessage(content=msg['content']))
                elif msg.get('type') == 'ai':
                    langchain_messages.append(AIMessage(content=msg['content']))
            
            response = self.llm_manager.llm_mini_virtualEvaluator.invoke(langchain_messages)
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
                "evaluator_active": True,
                "current_phase": "eval_feedback_complete"
            }
        
        except Exception as e:
            logging.error(f"Evaluator response failed: {str(e)}")
            return {**state, "error_message": f"Evaluator error: {str(e)}"}

