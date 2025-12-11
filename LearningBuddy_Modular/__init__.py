"""
LearningBuddy Modular - Virtual Patient Simulation System with Multi-Agent Support

A modular therapy training system supporting MI (Motivational Interviewing)
with optional multi-agent therapist system (proposer + judge).

Main Components:
- LearningBuddy: Main orchestrator class
- Workflows: Initialization, Interview (with multi-agent support), Feedback, and Evaluator workflows
- Utils: File management, prompt loading, and formatting utilities
- State management: Type-safe state definitions (shared with LearningBuddy_Agent)

Usage:
    from LearningBuddy_Modular import LearningBuddy

    buddy = LearningBuddy(
        api_key="your-api-key",
        use_multi_agent=True  # Enable multi-agent system
    )

    # Start session
    state = buddy.start_session(case_study=case_study)

    # Send message
    response = buddy.send_message("Hello, how are you feeling?")

    # End session and get feedback
    final_state = buddy.end_session()
"""

# Import main class from learning_buddy module
from .learning_buddy import LearningBuddy

# Import state types (from shared module)
from .graph_state import LBState, PatientContext

# Version information
__version__ = "2.1.0"
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

