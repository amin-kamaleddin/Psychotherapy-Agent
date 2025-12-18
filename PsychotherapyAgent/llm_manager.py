"""
LLM Manager for PsychotherapyAgent.

Handles initialization and management of all ChatOpenAI instances.
"""

import logging
import os
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class LLMManager:
    """Manages all LLM instances for PsychotherapyAgent."""
    
    def __init__(self, api_key: SecretStr):
        """
        Initialize all LLM instances.
        
        Args:
            api_key: OpenAI API key (as SecretStr)
        """
        self.api_key = api_key
        
        # Load environment variables for OpenRouter
        load_dotenv()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Core LLMs (OpenAI gpt-5-mini)
        self.llm_mini_verification = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key
        )
        self.llm_mini_summary = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key
        )
        self.llm_mini_virtualPatient = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key
        )
        self.llm_feedback = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key
        )
        
        # Evaluator should use OpenRouter Gemini when available
        if openrouter_api_key:
            self.llm_mini_virtualEvaluator = ChatOpenAI(
                model="google/gemini-2.5-flash",
                temperature=0.1,
                api_key=SecretStr(openrouter_api_key),
                base_url="https://openrouter.ai/api/v1"
            )
            logging.info("Virtual evaluator initialized with OpenRouter Gemini 2.5 Flash.")
        else:
            logging.warning("OPENROUTER_API_KEY not found in environment. Using OpenAI for virtual evaluator.")
            self.llm_mini_virtualEvaluator = ChatOpenAI(
                model="gpt-5-mini",
                api_key=self.api_key
            )
        
        # Therapist agents (for multi-agent mode) - all gpt-5-mini on OpenAI
        self.llm_therapist_proposer = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key
        )
        self.llm_therapist_judge = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key
        )
        
        logging.info("LLM Manager initialized with all ChatOpenAI instances.")

