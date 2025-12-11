"""
LLM Manager for LearningBuddy Modular.

Handles initialization and management of all ChatOpenAI instances.
"""

import logging
import os
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class LLMManager:
    """Manages all LLM instances for LearningBuddy."""
    
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
        
        # Core LLMs
        self.llm_mini_verification = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=0, 
            api_key=self.api_key
        )
        self.llm_mini_summary = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=0.2, 
            api_key=self.api_key
        )
        self.llm_mini_virtualPatient = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=1, 
            api_key=self.api_key
        )
        self.llm_feedback = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=0.1, 
            api_key=self.api_key
        )
        self.llm_mini_virtualEvaluator = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=0.7, 
            api_key=self.api_key
        )
        
        # Therapist agents (for multi-agent mode)
        self.llm_therapist_proposer = ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=0.8, 
            api_key=self.api_key
        )
        
        # Therapist judge uses OpenRouter Gemini Mini Flash 2.5
        if openrouter_api_key:
            self.llm_therapist_judge = ChatOpenAI(
                model="google/gemini-2.5-flash",  # Gemini Mini Flash 2.5 on OpenRouter
                temperature=0.1,
                api_key=SecretStr(openrouter_api_key),
                base_url="https://openrouter.ai/api/v1"
            )
            logging.info("Therapist judge initialized with OpenRouter Gemini Mini Flash 2.5.")
        else:
            # Fallback to OpenAI if OpenRouter key not found
            logging.warning("OPENROUTER_API_KEY not found in environment. Using OpenAI for judge agent.")
            self.llm_therapist_judge = ChatOpenAI(
                model="gpt-4.1-mini", 
                temperature=0.1, 
                api_key=self.api_key
            )
        
        logging.info("LLM Manager initialized with all ChatOpenAI instances.")

