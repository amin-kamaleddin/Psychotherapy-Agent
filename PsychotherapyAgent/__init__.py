"""
PsychotherapyAgent - Virtual Patient Simulation System with Multi-Agent Support

A modular therapy training system supporting MI (Motivational Interviewing)
with multi-agent therapist system (proposer + judge) and single-agent therapist.

Main Components:
- PsychotherapyMultiAgent: Multi-agent orchestrator class (proposer + judge)
- PsychotherapySingleAgent: Single-agent therapist class (simple/complex modes)
- Workflows: Initialization, Interview (with multi-agent support), Feedback, and Evaluator workflows
- Utils: File management, prompt loading, and formatting utilities
- State management: Type-safe state definitions

Usage:
    from PsychotherapyAgent import PsychotherapyMultiAgent, PsychotherapySingleAgent

    # Multi-agent system
    agent = PsychotherapyMultiAgent(api_key="your-api-key")
    state = agent.start_session(case_study=case_study)
    response = agent.send_message("Hello, how are you feeling?")

    # Single-agent system
    therapist = PsychotherapySingleAgent(api_key="your-api-key", therapist_index=0)
"""

# Import classes from new module names
from .psychotherapy_multi_agent import PsychotherapyMultiAgent
from .psychotherapy_single_agent import PsychotherapySingleAgent

# Import state types
from .graph_state import LBState, PatientContext

# Version information
__version__ = "2.1.0"
__author__ = "PsychotherapyAgent Team"

# Public API - what users can import
__all__ = [
    # Main classes
    'PsychotherapyMultiAgent',
    'PsychotherapySingleAgent',

    # State types (for type hints and advanced usage)
    'LBState',
    'PatientContext',

    # Version info
    '__version__',
]

