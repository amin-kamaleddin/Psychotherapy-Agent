"""
PsychotherapyAgent - Virtual Patient Simulation System with Multi-Agent Support

A modular therapy training system supporting MI (Motivational Interviewing)
with optional multi-agent therapist system (proposer + judge).

Main Components:
- PsychotherapyAgent: Main orchestrator class
- Workflows: Initialization, Interview (with multi-agent support), Feedback, and Evaluator workflows
- Utils: File management, prompt loading, and formatting utilities
- State management: Type-safe state definitions

Usage:
    from PsychotherapyAgent import PsychotherapyAgent

    agent = PsychotherapyAgent(
        api_key="your-api-key",
        use_multi_agent=True  # Enable multi-agent system
    )

    # Start session
    state = agent.start_session(case_study=case_study)

    # Send message
    response = agent.send_message("Hello, how are you feeling?")

    # End session and get feedback
    final_state = agent.end_session()
"""

# Import main class from psychotherapy_agent module
from .psychotherapy_agent import PsychotherapyAgent

# Import state types
from .graph_state import LBState, PatientContext

# Version information
__version__ = "2.1.0"
__author__ = "PsychotherapyAgent Team"

# Public API - what users can import
__all__ = [
    # Main class
    'PsychotherapyAgent',

    # State types (for type hints and advanced usage)
    'LBState',
    'PatientContext',

    # Version info
    '__version__',
]

