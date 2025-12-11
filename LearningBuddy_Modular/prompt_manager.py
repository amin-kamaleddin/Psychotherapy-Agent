"""
Prompt Manager for LearningBuddy Modular.

Handles loading of prompts from files and CSV.
"""

import logging
import pandas as pd
from typing import Optional


class PromptManager:
    """Manages prompt loading from files and CSV."""
    
    def __init__(self):
        """Initialize PromptManager."""
        # Define prompt file paths
        self.prompt_paths = {
            'Clinical Summary': 'Prompts/MI/clinical_summary_prompt.txt',
            'Virtual Patient': 'Prompts/MI/virtual_patient_prompt.txt',
            'Verification': 'Prompts/MI/patient_profile_verification_prompt.txt',
            'Feedback': 'Prompts/MI/feedback_agent_prompt.txt',
            'MI Prompt': 'Prompts/MI/MI_form_prompt.txt',
            'Virtual Evaluator': 'Prompts/MI/virtual_evaluator_prompt.txt',
            'Virtual Therapist': 'Prompts/MI/virtual_therapist_prompt.txt',
        }
        
        # Define context file paths
        self.context_paths = {
            'Feedback Criteria': 'Context/MI/feedback_criteria_MI_5point.txt',
            'Evaluation Form': 'Context/MI/MI_EVAL.txt',
        }
    
    def _read_file(self, filepath: str) -> str:
        """Read and return contents of a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file {filepath}: {e}")
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from file.
        
        Args:
            prompt_name: Name of the prompt to load
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt name is invalid
        """
        if prompt_name not in self.prompt_paths:
            raise FileNotFoundError(f"File not found. Invalid Prompt Name: {prompt_name}")
        
        filepath = self.prompt_paths[prompt_name]
        return self._read_file(filepath)
    
    def load_context(self, context_name: str) -> str:
        """
        Load a context file.
        
        Args:
            context_name: Name of the context to load
            
        Returns:
            Context content as string
            
        Raises:
            FileNotFoundError: If context name is invalid
        """
        if context_name not in self.context_paths:
            raise FileNotFoundError(f"File not found. Invalid Prompt Name: {context_name}")
        
        filepath = self.context_paths[context_name]
        return self._read_file(filepath)
    
    def load_therapist_prompt_from_csv(self, therapist_index: int) -> str:
        """
        Load therapist prompt from CSV file.
        
        Args:
            therapist_index: Index of therapist profile in CSV 
                            (0=simple, 1=complex, 2=proposer, 3=judge)
            
        Returns:
            Prompt content as string
        """
        csv_path = "Therapist_Prompts/therapist_prompts.csv"
        try:
            df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
            if therapist_index >= len(df) or therapist_index < 0:
                raise ValueError(
                    f"Therapist index {therapist_index} out of range. "
                    f"Available: 0-{len(df) - 1}"
                )
            
            therapist_row = df.iloc[therapist_index]
            csv_prompt = therapist_row["Prompt"]
            
            # For simple (0) and complex (1) agents, combine with base prompt
            # For proposer (2) and judge (3), use CSV prompt as-is (they are standalone)
            if therapist_index in [0, 1]:
                base_prompt = self.load_prompt('Virtual Therapist')
                return base_prompt + "\n\n" + csv_prompt
            else:
                # Proposer (2) and Judge (3) have standalone prompts
                return csv_prompt
        except Exception as e:
            logging.error(f"Error loading therapist prompt from CSV: {str(e)}")
            # Fallback prompts based on agent type
            if therapist_index == 2:  # Proposer
                return """You are a therapist response proposer agent. Your role is to generate exactly 3 different candidate therapist responses based on the conversation history. Each response should be appropriate for the therapeutic context and follow motivational interviewing principles. Make sure each response is distinct and offers a different therapeutic approach or phrasing. Format your response as: RESPONSE 1: [first response] RESPONSE 2: [second response] RESPONSE 3: [third response]"""
            elif therapist_index == 3:  # Judge
                return """You are an expert judge evaluating therapist responses in a motivational interviewing context. Your task is to select the BEST response from 3 candidate therapist responses based on: 1. Appropriateness for the therapeutic context, 2. Adherence to motivational interviewing principles, 3. Empathy and rapport building, 4. Clinical effectiveness, 5. Natural flow in the conversation. Review the conversation history and the 3 candidate responses. Select the response number (1, 2, or 3) that is most effective and appropriate. Respond with ONLY the number (1, 2, or 3) followed by a brief one-sentence explanation."""
            else:  # Simple or Complex (0 or 1)
                try:
                    base_prompt = self.load_prompt('Virtual Therapist')
                    return base_prompt
                except:
                    return """You are a medical clinician conducting a motivational interview on a patient.
You will play the role of the clinician. Only respond as if you are the clinician treating your patient.
You must only speak from the perspective of the clinician.
Your responses should be relevant to the context of the interview and should flow logically."""

