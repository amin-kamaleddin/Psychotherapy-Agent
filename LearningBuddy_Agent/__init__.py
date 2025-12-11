"""
LearningBuddy Agent - Virtual Patient Simulation System

A multi-modal therapy training system supporting MI (Motivational Interviewing)

Main Components:
- LearningBuddy: Main orchestrator class
- Workflows: Initialization, Interview, Feedback, and Evaluator workflows
- Utils: File management, prompt loading, and formatting utilities
- State management: Type-safe state definitions

Usage:
    from LearningBuddy_Agent import LearningBuddy

    buddy = LearningBuddy(
        api_key="your-api-key",
    )

    # Start session
    state = buddy.start_session()

    # Send message
    response = buddy.send_message("Hello, how are you feeling?")

    # End session and get feedback
    final_state = buddy.end_session()
"""

# Import main class and constants from learning_buddy module
from .learning_buddy import LearningBuddy

# Import state types
from .graph_state import LBState, PatientContext

# Version information
__version__ = "2.0.0"
__author__ = "LearningBuddy Team"

# Public API - what users can import
__all__ = [
    # Main class
    'LearningBuddy',

    # State types (for type hints and advanced usage)
    'LBState',
    'PatientContext',

    # Version info
    '__version__',
]