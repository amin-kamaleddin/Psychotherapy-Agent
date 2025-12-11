import streamlit as st
from datetime import datetime
import time
from LearningBuddy_Modular import LearningBuddy
import base64
import markdown2
import uuid
import os
import io
import streamlit.components.v1 as components

# Streamlit App Configuration
st.set_page_config(
    page_title="Clinical Learning Buddy - LangGraph Powered",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

selected_mode = None 
# CSS for enhanced chat interface 
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat container */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Therapist messages */
    [data-testid="chat-message-user"] {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    
    /* Patient messages */
    [data-testid="chat-message-assistant"] {
        background-color: #f3e5f5;
        margin-right: 20%;
    }
    
    /* LangGraph status indicators */
    .langgraph-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .phase-indicator {
        background: rgba(255,255,255,0.2);
        padding: 5px 15px;
        border-radius: 15px;
        display: inline-block;
        margin: 5px;
        font-size: 0.9em;
    }
    
    /* Evaluation card */
    .evaluation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Critical feedback card */
    .critical-feedback {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%) !important;
    }
    
    /* Feedback sections */
    .feedback-section {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid rgba(255,255,255,0.5);
    }
    
    /* Session info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* LangGraph workflow indicator */
    .workflow-indicator {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #2e7d32;
    }
    
    /* Processing overlay */
    .processing-overlay {
        background: rgba(102, 126, 234, 0.1);
        padding: 30px;
        border-radius: 15px;
        border: 2px solid #667eea;
        text-align: center;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Session header */
    .session-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #4caf50;
        animation: pulse 2s infinite;
    }
    
    .status-processing {
        background-color: #ff9800;
        animation: pulse 2s infinite;
    }
    
    .status-complete {
        background-color: #2196f3;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
    
    /* Patient selection styling */
    .patient-selection {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 4px solid #4caf50;
    }
    
    /* LangGraph branding */
    .langgraph-branding {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.9em;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def autoplay_audio(audio_bytes: bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    unique_id = f"audio_{uuid.uuid4().hex}"

    html = f"""
    <html>
        <body>
            <audio id="{unique_id}" autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
            <script>
                const audio = document.getElementById("{unique_id}");
                if (audio) {{
                    audio.volume = 1.0;
                    audio.play().catch((e) => {{
                        console.warn("Autoplay blocked:", e);
                    }});
                }}
            </script>
        </body>
    </html>
    """

    # This embeds it in an iframe so Streamlit doesn't re-render it
    components.html(html, height=0)





def get_available_patient_profiles(directory='Patient_Profiles'):
    """Get list of available patient profile files"""

    # Select Patients available in the current mode
    directory += "/MI"

    try:
        if not os.path.exists(directory):
            return []
        files = [f for f in os.listdir(directory) if f.endswith('.txt') or not '.' in f]
        return sorted(files)
    except Exception:
        return []

def get_patient_profile_preview(filepath):
    """Get a preview of the patient profile content"""
    try:
        content = st.session_state.learning_buddy.load_patient_profile(filepath)
        return content[:200] + "..." if len(content) > 200 else content
    except Exception:
        return "Unable to load preview"

def format_session_duration(start_time_iso):
    """Format session duration from ISO string"""
    try:
        start_time = datetime.fromisoformat(start_time_iso)
        duration = datetime.now() - start_time
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes}m {seconds}s"
    except:
        return "0m 0s"

def reset_session_state():
    """Reset all session-related state variables"""
    st.session_state.session_data = None
    st.session_state.chat_messages = []
    st.session_state.session_active = False
    st.session_state.selected_patient_profile = None
    st.session_state.processing_phase = None
    st.session_state.show_patient_info = False
    st.session_state.show_end_confirmation = False
    st.session_state.presentation_messages = []
    st.session_state.presentation_started = False
    st.session_state.presentation_phase = False
    st.session_state.evaluator_chat_messages = []
    st.presentation_start_time = None
    st.session_start_time = None

def get_phase_display_name(phase):
    """Get user-friendly phase names"""
    phase_names = {
        "initialize_node": "🔄 Initializing",
        "verify_profile": "✅ Profile Verification", 
        "generate_summary": "📋 Clinical Summary",
        "setup_roleplay": "🎭 Roleplay Setup",
        "active_roleplay": "💬 Active Session",
        "evaluate_session": "🎯 Generating Feedback",
        "complete_session": "✅ Session Complete",
        "completed": "✅ Completed",
        "error": "❌ Error"
    }
    return phase_names.get(phase, phase)
def save_evaluator_transcript():
    filepath = st.session_state.session_data.get('folder_name')
    print(filepath)
    if not filepath:
        return  # Exit if no filepath is defined

    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, "evaluator_transcript.txt")

    with open(full_path, "w", encoding="utf-8") as f:
        for msg in st.session_state.evaluator_chat_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            f.write(f"{role}: {content}\n\n")


# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'learning_buddy' not in st.session_state:
    st.session_state.learning_buddy = None
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'completed_sessions' not in st.session_state:
    st.session_state.completed_sessions = []
if 'selected_patient_profile' not in st.session_state:
    st.session_state.selected_patient_profile = None
if 'available_profiles' not in st.session_state:
    st.session_state.available_profiles = get_available_patient_profiles()
if 'processing_phase' not in st.session_state:
    st.session_state.processing_phase = None
if 'show_patient_info' not in st.session_state:
    st.session_state.show_patient_info = False
if 'show_end_confirmation' not in st.session_state:
    st.session_state.show_end_confirmation = False
if "muted" not in st.session_state:
    st.session_state.muted = False
if 'presentation_phase' not in st.session_state:
    st.session_state.presentation_phase = False
if 'presentation_messages' not in st.session_state:
    st.session_state.presentation_messages = []
if 'presentation_started' not in st.session_state:
    st.session_state.presentation_started = False
if "evaluator_chat_messages" not in st.session_state:
    st.session_state.evaluator_chat_messages = []
if "evaluator_muted" not in st.session_state:
    st.session_state.evaluator_muted = False
if "mode" not in st.session_state:
    st.session_state.evaluator_muted = 0
if " mode_label" not in st.session_state:
    st.session_state.evaluator_muted = ""

# Sidebar Configuration
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 350px !important;  
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 350px !important;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.title("🏥 Clinical Learning Buddy")
st.sidebar.markdown("---")

# API Key Management
if not st.session_state.get("api_key"):
    st.sidebar.warning("⚠️ Please enter your API keys")

    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    # Mode selection dropdown
    mode_options = {
        "STACER Evaluation": LearningBuddy.STACER_MODE,
        "Motivational Interviewing (MI) Evaluation": LearningBuddy.MI_MODE,
        "Cognitive Behavioural Therapy (CBT) Evaluation": LearningBuddy.CBT_MODE
        # APPEND MORE OPTIONS HERE
    }
    selected_mode_label = st.sidebar.selectbox("Select Mode", list(mode_options.keys()))
    selected_mode = mode_options[selected_mode_label]

    if st.sidebar.button("🔑 Set API Keys"):
        if api_key_input:
            try:
                st.session_state.api_key = api_key_input
                st.session_state.mode = selected_mode
                st.session_state.mode_label = selected_mode_label
                st.session_state.available_profiles = get_available_patient_profiles(mode = st.session_state.mode)
                st.session_state.learning_buddy = LearningBuddy(api_key_input)
                st.session_state.learning_buddy.mode = selected_mode

                st.sidebar.success("✅ API keys set successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error initializing LearningBuddy: {str(e)}")
        else:
            st.sidebar.error("❌ Please enter both API keys.")
else:
    st.sidebar.success(f"✅ API Keys configured | Mode: { st.session_state.mode_label}")
    if st.sidebar.button("🔑 Change API Keys/Mode", disabled=st.session_state.processing_phase is not None):
        st.session_state.api_key = None
        st.session_state.learning_buddy = None
        st.session_state.mode = None
        st.session_state.mode_label = None
        reset_session_state()
        st.rerun()

# Patient Profile Selection
if st.session_state.api_key:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Patient Selection")
    
    # Refresh profiles button
    if st.sidebar.button("🔄 Refresh Profiles"):
        print(st.session_state.mode)
        st.session_state.available_profiles = get_available_patient_profiles(mode=st.session_state.mode)
        st.rerun()
    
    # Patient profile selection
    profile_options = ["Random"] + st.session_state.available_profiles
    
    selected_option = st.sidebar.selectbox(
        "Choose Patient Profile:",
        options=profile_options,
        index=0,
        disabled=st.session_state.session_active or st.session_state.processing_phase is not None
    )
    
    if selected_option != "Random":
        print("Selected option",selected_option)
        if st.session_state.learning_buddy.mode == LearningBuddy.STACER_MODE:
            st.session_state.selected_patient_profile = os.path.join('Patient_Profiles\STACER', selected_option)
        elif st.session_state.learning_buddy.mode == LearningBuddy.MI_MODE:
            st.session_state.selected_patient_profile = os.path.join('Patient_Profiles\MI', selected_option)
        elif st.session_state.learning_buddy.mode == LearningBuddy.CBT_MODE:
            st.session_state.selected_patient_profile = os.path.join('Patient_Profiles\STACER', selected_option)

        # Show preview
        with st.sidebar.expander("📋 Profile Preview"):
            preview = get_patient_profile_preview(st.session_state.selected_patient_profile)
            st.text_area("Profile Preview", preview, height=100, disabled=True, key="profile_preview", label_visibility="collapsed")
    else:
        st.session_state.selected_patient_profile = None
        st.sidebar.info("🎲 Random patient will be selected")

# Feature Navigation
mode = selected_mode
print(mode)
if mode == None:
    if st.session_state.learning_buddy.mode == LearningBuddy.STACER_MODE:
        feature = st.sidebar.radio(
            "Navigate",
            ["💬 STACER Practice Session",],      # "📊 Session Analytics🎯", "Performance Review"
            index=0,
            disabled=st.session_state.processing_phase is not None
        )
    elif st.session_state.learning_buddy.mode == LearningBuddy.MI_MODE:
        feature = st.sidebar.radio(
            "Navigate",
            ["💬 MI Practice Session",],      # "📊 Session Analytics🎯", "Performance Review"
            index=0,
            disabled=st.session_state.processing_phase is not None
        )
    elif st.session_state.learning_buddy.mode == LearningBuddy.CBT_MODE:
        feature = st.sidebar.radio(
            "Navigate",
            ["💬 CBT Practice Session",],      # "📊 Session Analytics🎯", "Performance Review"
            index=0,
            disabled=st.session_state.processing_phase is not None
        )
else:
    if mode == LearningBuddy.STACER_MODE:
        feature = st.sidebar.radio(
            "Navigate",
            ["💬 STACER Practice Session",],      # "📊 Session Analytics🎯", "Performance Review"
            index=0,
            disabled=st.session_state.processing_phase is not None
        )
    elif mode == LearningBuddy.MI_MODE:
        feature = st.sidebar.radio(
            "Navigate",
            ["💬 MI Practice Session",],      # "📊 Session Analytics🎯", "Performance Review"
            index=0,
            disabled=st.session_state.processing_phase is not None
        )
    elif mode == LearningBuddy.CBT_MODE:
        feature = st.sidebar.radio(
            "Navigate",
            ["💬 CBT Practice Session",],      # "📊 Session Analytics🎯", "Performance Review"
            index=0,
            disabled=st.session_state.processing_phase is not None
        )
# LangGraph Session Status
if st.session_state.session_data or st.session_state.processing_phase:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 System Status")
    
    if st.session_state.processing_phase:
        phase_name = get_phase_display_name(st.session_state.processing_phase)
        st.sidebar.markdown(f'<span class="status-indicator status-processing"></span> **{phase_name}**', unsafe_allow_html=True)
        st.sidebar.warning("⏳ Core workflows running...")
    elif st.session_state.session_data:
        current_phase = st.session_state.session_data.get('current_phase', 'unknown')
        phase_name = get_phase_display_name(current_phase)
        session_active = st.session_state.session_data.get('session_active', False)
        
        if session_active:
            st.sidebar.markdown(f'<span class="status-indicator status-active"></span> **{phase_name}**', unsafe_allow_html=True)
            # Show session stats
            if st.session_state.learning_buddy:
                session_stats = st.session_state.learning_buddy.get_session_statistics()
                st.sidebar.info(f"Exchanges: {session_stats.get('total_exchanges', 0)}")
                
                # Show duration if available
                session_metadata = st.session_state.session_data.get('session_metadata', {})
                if session_metadata.get('start_time'):
                    duration = format_session_duration(session_metadata['start_time'])
                    st.sidebar.info(f"Duration: {duration}")
        else:
            st.sidebar.markdown(f'<span class="status-indicator status-complete"></span> **{phase_name}**', unsafe_allow_html=True)

# Initialize LearningBuddy if needed
if not st.session_state.learning_buddy and st.session_state.api_key:
    try:
        st.session_state.learning_buddy = LearningBuddy(st.session_state.api_key)
    except Exception as e:
        st.error(f"❌ Error initializing LearningBuddy: {str(e)}")

# Main Content
if feature == "💬 STACER Practice Session" or feature == "💬 MI Practice Session" or feature == "💬 CBT Practice Session":
    # Session header
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    st.markdown("# 💬 Virtual Patient Interview")
    st.markdown("Practice clinical skills with AI-powered patient simulations")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.api_key:
        st.error("🔐 Please enter your OpenAI API key in the sidebar to continue.")
    
    elif not st.session_state.learning_buddy:
        st.error("❌ LearningBuddy not initialized. Please check your API key.")
    
    else:
        # Session Controls
        col1, col2, col3 = st.columns([2, 2, 2])
        
        controls_disabled = st.session_state.processing_phase is not None
        
        with col1:
            if st.button("🎭 Start Session", 
                       use_container_width=True, 
                       disabled=st.session_state.session_active or controls_disabled, 
                       type="primary"):
                
                # Set processing phase
                reset_session_state()
                st.session_state.processing_phase = "initialize_node"
                st.rerun()
        
        with col2:
            show_end_button = ((st.session_state.session_active or st.session_state.presentation_phase) and
                               not st.session_state.show_end_confirmation and
                               not controls_disabled)
            
            if st.button("🏁 End Session", 
                       use_container_width=True, 
                       disabled=not show_end_button, 
                       type="secondary"):
                st.session_state.presentation_phase = False
                st.session_state.show_end_confirmation = True
                st.rerun()

        with col3:
            if st.button("🔄 Reset All", 
                       use_container_width=True,
                       disabled=controls_disabled):
                if st.session_state.learning_buddy:
                    st.session_state.learning_buddy.reset_session()
                reset_session_state()
                st.rerun()
        
        # Process session initialization
        if st.session_state.processing_phase == "initialize_node":
            st.markdown('<div class="processing-overlay">', unsafe_allow_html=True)
            st.markdown("### 🔄 Initializing AI-centric Workflow...")
            st.markdown("Running through automated phases: Profile Verification → Clinical Summary → Roleplay Setup")
            
            # Show progress
            progress_placeholder = st.empty()
            with progress_placeholder:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Phase indicators
                status_text.text("🔄 Starting workflow...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("✅ Profile verification...")
                progress_bar.progress(30)
                time.sleep(0.3)
                
                status_text.text("📋 Generating clinical summary...")
                progress_bar.progress(60)
                time.sleep(0.3)
                
                status_text.text("🎭 Setting up roleplay...")
                progress_bar.progress(90)
                time.sleep(0.3)
                
                try:
                    session_result = st.session_state.learning_buddy.start_session(
                        case_study=None,
                        selected_profile=st.session_state.selected_patient_profile
                    )
                    
                    if session_result.get("error_message"):
                        st.error(f"❌ Error: {session_result['error_message']}")
                        st.session_state.processing_phase = None
                    else:
                        progress_bar.progress(100)
                        status_text.text("✅ Session ready!")
                        time.sleep(0.5)
                        
                        # Update session state
                        st.session_state.session_data = session_result
                        st.session_state.session_active = session_result.get('session_active', False)
                        st.session_state.chat_messages = []
                        st.session_state.processing_phase = None
                        
                        st.success("✅ Initial setup completed successfully!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ System error encountered: {str(e)}")
                    st.session_state.processing_phase = None
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(1)
            st.rerun()
        
        # End session confirmation
        if st.session_state.show_end_confirmation and not st.session_state.processing_phase:
            st.warning("⚠️ Are you sure you want to end this session?")
            st.info("🔍 This will trigger AI feedback evaluation workflow.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, End Session", use_container_width=True, type="primary"):
                    st.session_state.processing_phase = "evaluate_session"
                    st.session_state.show_end_confirmation = False
                    st.rerun()
            with col2:
                if st.button("❌ Continue Session", use_container_width=True):
                    st.session_state.show_end_confirmation = False
                    st.rerun()
        
        # Process session ending
        if st.session_state.processing_phase == "evaluate_session":
            st.markdown('<div class="processing-overlay">', unsafe_allow_html=True)
            st.markdown("### 🎯 Running AI Feedback Evaluation...")
            st.markdown("AI supervisor analyzing session using comprehensive clinical criteria. This may take a moment(Around 3-5 minutes).")
            
            progress_placeholder = st.empty()
            with progress_placeholder:
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                progress_bar.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            try:
                # End session using LangGraph
                result = st.session_state.learning_buddy.end_session()
                
                if result.get("error"):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Save completed session
                    session_export = st.session_state.learning_buddy.export_session_data()
                    if session_export:
                        st.session_state.completed_sessions.append({
                            "timestamp": datetime.now(),
                            "langgraph_data": session_export,
                            "profile_used": st.session_state.selected_patient_profile or "Random"
                        })
                    
                    # Update session state
                    st.session_state.session_data = result
                    st.session_state.session_active = False
                    st.session_state.processing_phase = None
                    
                    st.success("✅ AI feedback evaluation completed!")
                    st.info("📊 Check the feedback below and visit Session Analytics for detailed review.")
                
            except Exception as e:
                st.error(f"❌ AI feedback error: {str(e)}")
                st.session_state.processing_phase = None
            
            time.sleep(1)
            st.rerun()
        
        # Patient Information Display
        if (st.session_state.session_data and 
            st.session_state.show_patient_info and 
            not st.session_state.processing_phase):
            
            with st.container():
                st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
                st.markdown("### 📋 Session Information")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    case_study = st.session_state.session_data.get('case_study', '')
                    st.text_area("Patient Case Study", case_study, height=150, disabled=True, label_visibility="visible")
                
                with col2:
                    if st.session_state.learning_buddy:
                        session_stats = st.session_state.learning_buddy.get_session_statistics()
                        st.markdown("**System Stats:**")
                        st.markdown(f"- Phase: {get_phase_display_name(session_stats.get('current_phase', 'unknown'))}")
                        st.markdown(f"- Active: {'Yes' if session_stats.get('session_active') else 'No'}")
                        st.markdown(f"- Exchanges: {session_stats.get('total_exchanges', 0)}")
                        
                        profile_name = "Selected" if st.session_state.selected_patient_profile else "Random"
                        st.markdown(f"- Profile: {profile_name}")
                
                # Clinical Summary
                clinical_summary = st.session_state.session_data.get('clinical_summary')
                if clinical_summary:
                    st.markdown("### 📄 AI-Generated Clinical Summary")
                    st.info(clinical_summary)
                
                # Diagnosis Summary
                diagnosis_summary = st.session_state.session_data.get('diagnosis_summary')
                if diagnosis_summary:
                    st.markdown("### 🔍 Profile Verification Result")
                    st.info(diagnosis_summary)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Active Chat Interface
        if (st.session_state.session_active and 
            not st.session_state.processing_phase and 
            not st.session_state.show_end_confirmation):
            
            st.markdown("---")
            st.markdown("### 💭 AI-Powered Patient Interview")
            
            # Clinical Summary Reference
            if st.session_state.session_data.get('clinical_summary'):
                with st.expander("📋 **Clinical Summary** - Therapist Reference", expanded=False):
                    st.markdown("**LangGraph Generated Summary:**")
                    st.markdown(st.session_state.session_data['clinical_summary'])
                    
                    # Session stats
                    col1, col2, col3 = st.columns(3)
                    if st.session_state.learning_buddy:
                        session_stats = st.session_state.learning_buddy.get_session_statistics()
                        with col1:
                            profile_type = "Selected" if st.session_state.selected_patient_profile else "Random"
                            st.markdown(f"**Profile**: {profile_type}")
                        with col2:
                            st.markdown(f"**Exchanges**: {session_stats.get('total_exchanges', 0)}")
                        with col3:
                            session_metadata = st.session_state.session_data.get('session_metadata', {})
                            if session_metadata.get('start_time'):
                                duration = format_session_duration(session_metadata['start_time'])
                                st.markdown(f"**Duration**: {duration}")

            #Chat Set up
            # --- Initialize session start time ---
            if 'session_start_time' not in st.session_state:
                st.session_state.session_start_time = datetime.now()

            # --- Initialize chat messages ---
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []

            # --- Add Evaluator Welcome Message (only once) ---
            if 'evaluator_intro_shown' not in st.session_state:
                mode = selected_mode
                if mode == LearningBuddy.STACER_MODE:
                    st.session_state.chat_messages.append({
                        "role": "evaluator",
                        "content": "Hello and welcome to your simulated STACER evaluation. I will be evaluating your patient interviewing skills today. When you have finished speaking with the patient, press the 'End Interview' button. "
                    })
                elif mode == LearningBuddy.MI_MODE:
                       st.session_state.chat_messages.append({
                        "role": "evaluator",
                        "content": "Hello and welcome to your simulated MI evaluation. I will be evaluating your motivational interviewing skills today. When you have finished speaking with the patient, press the 'End Interview' button. "
                    })
                elif mode == LearningBuddy.CBT_MODE:
                       st.session_state.chat_messages.append({
                        "role": "evaluator",
                        "content": "Hello and welcome to your simulated CBT evaluation. I will be evaluating your cognitive behavioural therapy skills today. When you have finished speaking with the patient, press the 'End Interview' button. "
                    })
                st.session_state.chat_messages.append({
                    "role": "evaluator",
                    "content": "The virtual patient is in the chatroom with us. You may now begin."
                })
                st.session_state.evaluator_intro_shown = True

            # --- Timed Evaluator Messages ---
            now = datetime.now()
            elapsed = (now - st.session_state.session_start_time).total_seconds()

            # Add 50-minute evaluator message
            if elapsed >= 50 * 60 and not any(
                    m.get("tag") == "evaluator_50min" for m in st.session_state.chat_messages):
                st.session_state.chat_messages.append({
                    "role": "evaluator",
                    "content": "*50 minutes have passed. 5 minutes remaining.*",
                    "tag": "evaluator_50min"
                })

            # Add 55-minute evaluator message
            if elapsed >= 55 * 60 and not any(
                    m.get("tag") == "evaluator_55min" for m in st.session_state.chat_messages):
                st.session_state.chat_messages.append({
                    "role": "evaluator",
                    "content": "*55 minutes have elapsed. The session should now conclude.*",
                    "tag": "evaluator_55min"
                })

            # Chat Interface
            chat_container = st.container(height=400)
            
            with chat_container:

                # Display chat history
                for msg in st.session_state.chat_messages:
                    role = msg.get('role')
                    content = msg.get('content', '')

                    if role == 'user':  # Therapist (Right side)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background-color: #e3f2fd; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                <strong>You:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif role == 'assistant':  # Patient (Left side)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #f3e5f5; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                <strong>Patient:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif role == 'evaluator':
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #f1f8e9; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                <strong>Evaluator:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Mute Toggle
            st.session_state.muted = st.toggle("🔇 Mute audio", value=st.session_state.get("muted", False))

            # Chat Input
            if prompt := st.chat_input("Type your response as therapist..."):
                # Add user message to display
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                # Send message through LangGraph
                with st.spinner("🔄 System processing response..."):
                    try:
                        response = st.session_state.learning_buddy.send_message(prompt)
                        
                        if response.get("error"):
                            st.error(f"❌ Error: {response['error']}")
                        else:
                            # Add AI response to display
                            ai_response = response.get("ai_response", "")
                            if ai_response:
                                st.session_state.chat_messages.append({
                                    "role": "assistant", 
                                    "content": ai_response
                                })
                            
                            # Update session data
                            st.session_state.session_data = st.session_state.learning_buddy.get_session_state()
                            st.session_state.session_active = response.get("session_active", False)
                    
                    except Exception as e:
                        st.error(f"❌ LangGraph communication error: {str(e)}")
                
                st.rerun()


            # End Interview Button
            if st.button("🛑 End Interview"):
                mode = selected_mode
                st.session_state.presentation_phase = True
                st.session_state.session_active = False  # Stop current chat
                st.rerun()

            #Audio Playback
            if "latest_audio" in st.session_state:
                autoplay_audio(st.session_state.latest_audio)
                del st.session_state["latest_audio"]

        if st.session_state.presentation_phase and not st.session_state.show_end_confirmation:
            st.markdown("---")
            st.markdown("### 🧠 Clinical Presentation Entry")

            # Initialize evaluator's opening message
            if not st.session_state.presentation_started:
                evaluator_intro = "Please begin your diagnostic presentation based on the interview. You may type it as one single message, or a series of smaller messages. When you are done presenting, press the 'End Session' Button"
                st.session_state.presentation_messages.append({"role": "evaluator", "content": evaluator_intro})
                st.session_state.presentation_started = True

            # --- Initialize session start time ---
            if 'presentation_start_time' not in st.session_state:
                st.session_state.presentation_start_time = datetime.now()
            # --- Timed Evaluator Messages ---
            now = datetime.now()
            elapsed = (now - st.session_state.presentation_start_time).total_seconds()


            # Add 20-minute evaluator message
            if elapsed >= 20 * 60 and not any(
                    m.get("tag") == "evaluator_20min" for m in st.session_state.presentation_messages):
                st.session_state.presentation_messages.append({
                    "role": "evaluator",
                    "content": "*20 minutes have passed. 5 minutes remaining.*",
                    "tag": "evaluator_20min"
                })

            # Add 25-minute evaluator message
            if elapsed >= 25 * 60 and not any(
                    m.get("tag") == "evaluator_25min" for m in st.session_state.presentation_messages):
                st.session_state.presentation_messages.append({
                    "role": "evaluator",
                    "content": "*25 minutes have elapsed. The presentation should now conclude.*",
                    "tag": "evaluator_25min"
                })

            # Scrollable Presentation Container
            with st.container(height=400):
                for msg in st.session_state.presentation_messages:
                    role = msg.get("role")
                    content = msg.get("content", '')

                    if role == 'evaluator':
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #f1f8e9; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                <strong>Evaluator:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif role == 'user':
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background-color: #e3f2fd ; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                <strong>You:</strong><br>{content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Input box for presentation entries
            if user_input := st.chat_input("Type your case presentation here..."):
                st.session_state.presentation_messages.append({
                    "role": "user",
                    "content": user_input
                })

                # Call helper to store presentation
                try:
                    st.session_state.learning_buddy.add_presentation_message(user_input)
                except Exception as e:
                    st.error(f"Error storing presentation input: {str(e)}")

                st.rerun()

        # Display Feedback After Session
        if (st.session_state.session_data and
                st.session_state.session_data.get('feedback') and
                st.session_state.session_data.get('cod_stacer') and
                not st.session_state.session_active and
                not st.session_state.processing_phase):
            # Use tabs for Feedback and STACER form
            print(st.session_state.learning_buddy.mode)
            mode = st.session_state.learning_buddy.mode
            # Use tabs for Feedback and STACER/MI form
            if mode == LearningBuddy.STACER_MODE:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Evaluator Chat", "📝 COD STACER Form", " 📊 Detailed Feedback" , "🎤 Patient Interview Transcript", "🗣️ Case Presentation Transcript", "👤 Patient Profile"])
            elif mode == LearningBuddy.MI_MODE: 
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Evaluator Chat", "📝 MI Evaluation Form", " 📊 Detailed Feedback" , "🎤 Patient Interview Transcript", "🗣️ Case Presentation Transcript", "👤 Patient Profile"])
            elif mode == LearningBuddy.CBT_MODE: 
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Evaluator Chat", "📝 CBT Evaluation Form", " 📊 Detailed Feedback" , "🎤 Patient Interview Transcript", "🗣️ Case Presentation Transcript", "👤 Patient Profile"])
            
            # Tab 1: Evaluator Chat
            with tab1:
                st.markdown("### 💬 Interactive Evaluator")
                evaluator_chat_container = st.container(height=400)

                # Only allow access if evaluator session is active
                evaluator_active = (
                        st.session_state.learning_buddy and
                        st.session_state.learning_buddy.current_state.get("evaluator_session_active", False)
                )

                if not evaluator_active:
                    st.warning("⚠️ Evaluator session not available. Complete your interview and presentation first.")
                else:
                    # Initialize chat messages if missing
                    if "evaluator_chat_messages" not in st.session_state:
                        st.session_state.evaluator_chat_messages = []

                    # Initialize muted state if missing
                    if "evaluator_muted" not in st.session_state:
                        st.session_state.evaluator_muted = False

                    #Append the first message
                    if st.session_state.evaluator_chat_messages == []:
                        try:
                            response = st.session_state.learning_buddy.send_evaluator_message("Please Begin the Evaluation")

                            if response.get("error"):
                                st.error(f"❌ Error: {response['error']}")
                            else:
                                ai_response = response.get("ai_response", "")
                                if ai_response:
                                    st.session_state.evaluator_chat_messages.append({
                                        "role": "evaluator",
                                        "content": ai_response
                                    })
                                    save_evaluator_transcript()

                        except Exception as e:
                            st.error(f"❌ Evaluator processing error: {e}")

                    with evaluator_chat_container:
                        for msg in st.session_state.evaluator_chat_messages:
                            role = msg.get("role")
                            content = msg.get("content", "")

                            if role == "user":
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                        <div style="background-color: #e3f2fd; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                            <strong>You:</strong><br>{content}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                            elif role == "evaluator":
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                        <div style="background-color: #f1f8e9; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                            <strong>Evaluator:</strong><br>{content}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    st.session_state.evaluator_muted = st.toggle("🔇 Mute evaluator",
                                                                 value=st.session_state.get("evaluator_muted", False))

                    if prompt := st.chat_input("Ask a question about your evaluation..."):
                        st.session_state.evaluator_chat_messages.append({
                            "role": "user",
                            "content": prompt
                        })

                        with st.spinner("💬 Evaluator preparing response..."):
                            try:
                                response = st.session_state.learning_buddy.send_evaluator_message(prompt)

                                if response.get("error"):
                                    st.error(f"❌ Error: {response['error']}")
                                else:
                                    ai_response = response.get("ai_response", "")
                                    if ai_response:
                                        st.session_state.evaluator_chat_messages.append({
                                            "role": "evaluator",
                                            "content": ai_response
                                        })
                                        save_evaluator_transcript()

                            except Exception as e:
                                st.error(f"❌ Evaluator processing error: {e}")

                        st.rerun()

                    # Audio Playback
                    if 'latest_evaluator_audio' in st.session_state:
                        audio_base64 = base64.b64encode(st.session_state.latest_evaluator_audio).decode('utf-8')
                        audio_html = f'''
                                <audio autoplay>
                                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                                </audio>
                                '''
                        st.markdown(audio_html, unsafe_allow_html=True)

                        del st.session_state['latest_evaluator_audio']

            # Tab 2: COD STACER Form
            with tab2:
                # Extract STACER text
                STACER_text = st.session_state.session_data.get('cod_stacer')

                st.session_state.session_data['cod_stacer'] = STACER_text

                html = markdown2.markdown(STACER_text,
                                          extras=["tables", "fenced-code-blocks", "strike", "cuddled-lists"])

                # Display with Streamlit
                if mode == LearningBuddy.STACER_MODE:
                    st.markdown("### 📄 Completed COD STACER Evaluation Form")
                    st.info(
                        "This is a structured Core STACER form based on your most recent session. This form evaluates the expectations of PGY3 and PGY4 students.")
                elif mode == LearningBuddy.MI_MODE:
                    st.markdown("### 📄 Completed MI Evaluation Form")
                    st.info(
                        "This is a structured Motivational Interviewing form based on your most recent session. This form evaluates the expectations of PGY3 and PGY4 students.")
                elif mode == LearningBuddy.CBT_MODE:
                    st.markdown("### 📄 Completed CBT Evaluation Form")
                    st.info(
                        "This is a structured Cognitive Bahavioural Therapy form based on your most recent session. This form evaluates the expectations of PGY3 and PGY4 students.")
                     
                st.markdown("---")
                st.markdown(html, unsafe_allow_html=True)

            # Tab 3: Feedback
            with tab3:
                st.markdown("---")
                st.markdown("### 🎯 AI Clinical Supervision Feedback")
                st.warning(
                    "⚠️ **Note**: This feedback was generated using workflow with DSM-5 guidelines. It is intentionally strict and critical to accelerate learning.")

                feedback_text = st.session_state.session_data['feedback']

                if "CRITICAL ERROR" in feedback_text or "critical error" in feedback_text.lower():
                    st.error("🚨 **CRITICAL ERRORS DETECTED** - Review immediately for patient safety")

                feedback_container = st.container()
                with feedback_container:
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown('<div class="evaluation-card">', unsafe_allow_html=True)
                        st.markdown("#### 📋 AI Clinical Evaluation")
                        st.markdown(feedback_text)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
                        st.markdown("#### 🔬 AI Generated Summary")

                        import re

                        score_match = re.search(r'SCORE:\s*(\d+)/100', feedback_text, re.IGNORECASE)
                        if score_match:
                            score = int(score_match.group(1))
                            if score >= 80:
                                st.success(f"Score: {score}/100")
                            elif score >= 70:
                                st.warning(f"Score: {score}/100")
                            else:
                                st.error(f"Score: {score}/100")

                        if st.session_state.learning_buddy:
                            session_stats = st.session_state.learning_buddy.get_session_statistics()
                            st.markdown(f"**Exchanges**: {session_stats.get('total_exchanges', 0)}")

                            session_metadata = st.session_state.session_data.get('session_metadata', {})
                            if session_metadata.get('session_duration_minutes'):
                                st.markdown(f"**Duration**: {session_metadata['session_duration_minutes']:.1f} min")

                            profile_type = "Selected" if st.session_state.selected_patient_profile else "Random"
                            st.markdown(f"**Profile**: {profile_type}")

                        st.markdown('</div>', unsafe_allow_html=True)

            # Tab 4: Transcript
            with tab4:
                # Chat Interface
                chat_container = st.container(height=400)

                with chat_container:
                    # Welcome message
                    if not st.session_state.chat_messages:
                        st.chat_message("assistant").markdown(
                            "👋 *Patient simulation ready. The session is now active and ready for therapeutic interaction.*"
                        )

                    # Display chat history
                    for msg in st.session_state.chat_messages:
                        role = msg.get('role')
                        content = msg.get('content', '')

                        if role == 'user':  # Therapist (Right side)
                            st.markdown(f"""
                                                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                                    <div style="background-color: #e3f2fd; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                                        <strong>You:</strong><br>{content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)

                        elif role == 'assistant':  # Patient (Left side)
                            st.markdown(f"""
                                                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                                    <div style="background-color: #f3e5f5; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                                        <strong>Patient:</strong><br>{content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
            #Tab 5: Presentation
            with tab5:
                # Scrollable Presentation Container
                with st.container(height=400):
                    for msg in st.session_state.presentation_messages:
                        role = msg.get("role")
                        content = msg.get("content", '')

                        if role == 'evaluator':
                            st.markdown(f"""
                                                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                                    <div style="background-color: #f1f8e9; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: left;">
                                                        <strong>Evaluator:</strong><br>{content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)

                        elif role == 'user':
                            st.markdown(f"""
                                                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                                    <div style="background-color: #e3f2fd ; padding: 12px 18px; border-radius: 15px; max-width: 70%; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                                                        <strong>You:</strong><br>{content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)

            #Tab 6: Patient Profile
            with tab6:

                clinical_summary = st.session_state.session_data.get('clinical_summary', '')
                if clinical_summary:
                    st.markdown("**AI-Generated Clinical Summary:**")
                    st.info(clinical_summary)

                case_study = st.session_state.session_data.get('case_study', '')
                st.markdown("**Patient Profile Used During Simulation:**")
                st.info(case_study)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 Save Feedback", use_container_width=True):
                    st.success("✅ Feedback saved!")
            with col2:
                if st.button("🎭 New Session", use_container_width=True):
                    if st.session_state.learning_buddy:
                        st.session_state.learning_buddy.reset_session()
                    reset_session_state()
                    st.rerun()
            with col3:
                if st.button("📊 View Analytics", use_container_width=True):
                    st.session_state.selected_feature = "📊 Session Analytics"
                    st.rerun()




elif feature == "📊 Session Analytics":
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    st.markdown("# 📊 Session Analytics & History")
    st.markdown("Track your progress with AI-powered insights")
    st.markdown('<div class="phase-indicator">🔬 AI Workflow Analytics</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.completed_sessions:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Sessions", len(st.session_state.completed_sessions))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_exchanges = sum(
                session.get('langgraph_data', {}).get('session_statistics', {}).get('total_exchanges', 0)
                for session in st.session_state.completed_sessions
            )
            avg_exchanges = total_exchanges / len(st.session_state.completed_sessions) if st.session_state.completed_sessions else 0
            st.metric("Avg Exchanges", f"{avg_exchanges:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_duration = sum(
                session.get('langgraph_data', {}).get('session_metadata', {}).get('session_duration_minutes', 0)
                for session in st.session_state.completed_sessions
            )
            st.metric("Total Time", f"{total_duration:.0f} min")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if st.session_state.completed_sessions:
                last_session = st.session_state.completed_sessions[-1]['timestamp']
                st.metric("Last Session", last_session.strftime("%m/%d %H:%M"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # LangGraph Analytics
        st.markdown("### 🔬 Workflow Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Workflow Performance")
            
            successful_sessions = len([s for s in st.session_state.completed_sessions 
                                     if not s.get('langgraph_data', {}).get('error_message')])
            success_rate = (successful_sessions / len(st.session_state.completed_sessions)) * 100
            
            st.markdown(f"**Success Rate**: {success_rate:.1f}%")
            st.markdown(f"**Completed Workflows**: {successful_sessions}")
            st.markdown("**Phases**: Profile → Summary → Roleplay → Feedback")
            st.markdown("**AI Models**: GPT-4o-mini workflow")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Profile usage analytics
            selected_count = sum(1 for s in st.session_state.completed_sessions if s.get('profile_used') != 'Random')
            random_count = len(st.session_state.completed_sessions) - selected_count
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 👤 Profile Selection Analytics")
            st.markdown(f"**Selected Profiles**: {selected_count}")
            st.markdown(f"**Random Profiles**: {random_count}")
            
            if selected_count > random_count:
                st.markdown("📊 **Strategic Learner** - Prefers targeted practice")
            elif random_count > selected_count:
                st.markdown("🎲 **Adaptive Learner** - Embraces variety")
            else:
                st.markdown("⚖️ **Balanced Learner** - Good mix of approaches")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Session History
        st.markdown("### 📚 Session History")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First"])
        
        sessions_to_display = st.session_state.completed_sessions.copy()
        if sort_order == "Newest First":
            sessions_to_display.reverse()
        
        mode = st.session_state.learning_buddy.mode
        for i, session in enumerate(sessions_to_display):
            session_num = len(st.session_state.completed_sessions) - i if sort_order == "Newest First" else i + 1
            profile_info = session.get('profile_used', 'Unknown')
            profile_display = "Random" if profile_info == 'Random' else os.path.basename(profile_info)
            
            langgraph_data = session.get('langgraph_data', {})
            session_metadata = langgraph_data.get('session_metadata', {})
            duration = session_metadata.get('session_duration_minutes', 0)
            
            with st.expander(f"🔬 Chat Session {session_num} - {session['timestamp'].strftime('%Y-%m-%d %H:%M')} ({duration:.1f}m) - {profile_display}"):
                if mode == LearningBuddy.STACER_MODE:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Patient Data", "💬 Conversation", "📝 Feedback",  "📋 COD STACER", "📊 Metrics", "🔬 LangGraph Details"])
                
                elif mode ==LearningBuddy.MI_MODE:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Patient Data", "💬 Conversation", "📝 Feedback",  "📋 Motivational Interviewing", "📊 Metrics", "🔬 LangGraph Details"])
                elif mode ==LearningBuddy.CBT_MODE:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Patient Data", "💬 Conversation", "📝 Feedback",  "📋 Cognitive Behavioural Therapy", "📊 Metrics", "🔬 LangGraph Details"])
                with tab1:
                    st.markdown("**Patient Profile (Verified):**")
                    case_study = langgraph_data.get('case_study', '')
                    st.text_area("Case Study Content", case_study, height=200, disabled=True, key=f"case_{i}", label_visibility="collapsed")
                    
                    clinical_summary = langgraph_data.get('clinical_summary', '')
                    if clinical_summary:
                        st.markdown("**AI-Generated Clinical Summary:**")
                        st.info(clinical_summary)
                
                with tab2:
                    st.markdown("**AI Therapy Conversation:**")
                    chatlog = langgraph_data.get('chatlog', [])
                    conversation_messages = [m for m in chatlog if m.get('role') in ['user', 'assistant']]
                    
                    for msg in conversation_messages:
                        if msg.get('role') == 'user':
                            st.chat_message("user").markdown(f"**Therapist:** {msg.get('content', '')}")
                        elif msg.get('role') == 'assistant':
                            st.chat_message("assistant").markdown(f"**Patient:** {msg.get('content', '')}")
                
                with tab3:
                    st.markdown("**AI Supervisor Feedback:**")
                    feedback = langgraph_data.get('feedback', '')
                    if feedback:
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        st.markdown(feedback)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No feedback available for this session")

                with tab4:
                    cod_stacer = langgraph_data.get('cod_stacer', '')
                    if cod_stacer:
                        if mode == LearningBuddy.STACER_MODE:
                            st.markdown("**COD Clinical Observation Document (STACER):**")
                        elif mode == LearningBuddy.MI_MODE:
                            st.markdown("**MI Observation Document (MI Evaluation Form):**")
                        elif mode == LearningBuddy.CBT_MODE:
                            st.markdown("**CBT Observation Document (CBT Evaluation Form):**")
                        st.markdown(cod_stacer, unsafe_allow_html=True)
                    else:
                        if mode == LearningBuddy.STACER_MODE:
                            st.warning("No COD STACER data available for this session.")
                        elif mode == LearningBuddy.MI_MODE:
                            st.warning("No MI Evaluation data available for this session.")
                        elif mode == LearningBuddy.CBT_MODE:
                            st.warning("No CBT Evaluation data available for this session.")

                with tab5:
                    col1, col2 = st.columns(2)
                    with col1:
                        session_stats = langgraph_data.get('session_statistics', {})
                        st.markdown("**LangGraph Session Metrics:**")
                        st.markdown(f"- Duration: {duration:.1f} minutes")
                        st.markdown(f"- Total Exchanges: {session_stats.get('total_exchanges', 0)}")
                        st.markdown(f"- Total Messages: {session_stats.get('total_messages', 0)}")
                        st.markdown(f"- Profile Type: {profile_display}")
                    
                    with col2:
                        st.markdown("**Workflow Timing:**")
                        st.markdown(f"- Date: {session['timestamp'].strftime('%Y-%m-%d')}")
                        st.markdown(f"- Time: {session['timestamp'].strftime('%H:%M:%S')}")
                        st.markdown("**Technology Stack:**")
                        st.markdown("- Engine: LangGraph")
                        st.markdown("- Models: GPT-4o-mini")
                        st.markdown("- Workflow: 4-Phase")
                
                with tab6:
                    st.markdown("**Complete LangGraph Workflow Data:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Workflow Phases:**+")
                        st.markdown("✅ Phase 1: Profile Verification")
                        st.markdown("✅ Phase 2: Clinical Summary Generation")
                        st.markdown("✅ Phase 3: Interactive Roleplay")
                        st.markdown("✅ Phase 4: Comprehensive Feedback")
                        
                        diagnosis_summary = langgraph_data.get('diagnosis_summary', '')
                        if diagnosis_summary:
                            st.markdown("**Profile Verification Result:**")
                            st.text_area("Diagnosis Summary", diagnosis_summary, height=100, disabled=True, key=f"diag_{i}", label_visibility="collapsed")
                    
                    with col2:
                        st.markdown("**Session Metadata:**")
                        metadata = langgraph_data.get('session_metadata', {})
                        for key, value in metadata.items():
                            if isinstance(value, str) and len(value) > 50:
                                st.markdown(f"- **{key}**: {value[:50]}...")
                            else:
                                st.markdown(f"- **{key}**: {value}")
    else:
        st.info("📭 No completed LangGraph sessions yet. Start a therapy session to begin tracking your progress!")
        if st.button("🎭 Start First Session", use_container_width=True):
            st.session_state.selected_feature = "💬 Therapy Session"
            st.rerun()

elif feature == "🎯 Performance Review":
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    st.markdown("# 🎯 Performance Review")
    st.markdown("Analyze your clinical performance with LangGraph insights")
    st.markdown('<div class="phase-indicator">🔬 AI-Powered Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.completed_sessions:
        # Performance Overview
        st.markdown("### 📈 Performance Overview")
        
        total_sessions = len(st.session_state.completed_sessions)
        total_exchanges = sum(
            session.get('langgraph_data', {}).get('session_statistics', {}).get('total_exchanges', 0)
            for session in st.session_state.completed_sessions
        )
        avg_duration = sum(
            session.get('langgraph_data', {}).get('session_metadata', {}).get('session_duration_minutes', 0)
            for session in st.session_state.completed_sessions
        ) / total_sessions if total_sessions > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
            st.markdown("#### 🎓 Experience Level")
            if total_sessions < 5:
                st.markdown("**Beginner** - Getting familiar with workflow!")
            elif total_sessions < 15:
                st.markdown("**Intermediate** - Good LangGraph progress!")
            else:
                st.markdown("**Advanced** - LangGraph expert level!")
            st.progress(min(total_sessions / 20, 1.0))
            st.markdown(f"**Sessions Completed**: {total_sessions}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### ⏱️ Session Management")
            st.markdown(f"**Avg Duration:** {avg_duration:.1f} minutes")
            if avg_duration < 10:
                st.markdown("💡 Consider longer sessions for deeper therapy")
            elif avg_duration > 30:
                st.markdown("✅ Excellent session length for practice")
            else:
                st.markdown("👍 Good session management")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 💬 Engagement Quality")
            avg_exchanges = total_exchanges / total_sessions if total_sessions > 0 else 0
            st.markdown(f"**Avg Exchanges:** {avg_exchanges:.1f}")
            if avg_exchanges < 5:
                st.markdown("💡 Try more back-and-forth dialogue")
            else:
                st.markdown("✅ Great conversational engagement!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # LangGraph Workflow Insights
        st.markdown("### 🔬 Workflow Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Workflow Success Metrics")
            
            successful_sessions = len([s for s in st.session_state.completed_sessions 
                                     if not s.get('langgraph_data', {}).get('error_message')])
            success_rate = (successful_sessions / total_sessions) * 100 if total_sessions > 0 else 0
            
            st.markdown(f"✅ **Success Rate**: {success_rate:.1f}%")
            st.markdown(f"🎭 **Completed Workflows**: {successful_sessions}/{total_sessions}")
            st.markdown("📋 **Profile Verification**: Automated")
            st.markdown("🔍 **Clinical Summaries**: AI-Generated")
            st.markdown("💬 **Virtual Patients**: Dynamic AI")
            st.markdown("📝 **Feedback**: Comprehensive Analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Profile strategy analysis
            selected_profiles = sum(1 for s in st.session_state.completed_sessions if s.get('profile_used') != 'Random')
            random_profiles = total_sessions - selected_profiles
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 👤 Learning Strategy Analysis")
            st.markdown(f"**Selected Profiles**: {selected_profiles}/{total_sessions}")
            st.markdown(f"**Random Profiles**: {random_profiles}/{total_sessions}")
            
            if selected_profiles > random_profiles:
                st.markdown("📊 **Strategic Learner**")
                st.markdown("You prefer targeted practice with specific patient types")
            elif random_profiles > selected_profiles:
                st.markdown("🎲 **Adaptive Learner**")
                st.markdown("You embrace variety and unexpected scenarios")
            else:
                st.markdown("⚖️ **Balanced Learner**")
                st.markdown("Perfect mix of strategic and adaptive learning")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Learning Progress Visualization
        st.markdown("### 📊 Learning Progress")
        
        if len(st.session_state.completed_sessions) > 1:
            import pandas as pd
            
            # Prepare data for visualization
            dates = [s['timestamp'].date() for s in st.session_state.completed_sessions]
            durations = [s.get('langgraph_data', {}).get('session_metadata', {}).get('session_duration_minutes', 0) 
                        for s in st.session_state.completed_sessions]
            exchanges = [s.get('langgraph_data', {}).get('session_statistics', {}).get('total_exchanges', 0) 
                        for s in st.session_state.completed_sessions]
            
            df = pd.DataFrame({
                'Date': dates,
                'Duration (min)': durations,
                'Exchanges': exchanges
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Session Duration Trend**")
                st.line_chart(df.set_index('Date')['Duration (min)'])
            with col2:
                st.markdown("**Engagement Trend**")
                st.line_chart(df.set_index('Date')['Exchanges'])
        
        # Enhanced LangGraph Recommendations
        st.markdown("### 💡 Recommendations")
        st.markdown('<div class="workflow-indicator">', unsafe_allow_html=True)
        st.markdown("**Personalized insights based on your session data:**")
        
        recommendations = []
        
        if avg_duration < 15:
            recommendations.append("⏱️ **Session Duration**: Extend sessions to fully leverage application benefits")
        
        if avg_exchanges < 8:
            recommendations.append("💬 **Engagement Depth**: Increase dialogue exchanges to maximize AI feedback quality")
        
        if selected_profiles < total_sessions * 0.3:
            recommendations.append("🎯 **Profile Strategy**: Try more selected profiles to practice targeted clinical scenarios")
        
        if total_sessions < 10:
            recommendations.append("🔄 **Practice Frequency**: Regular sessions help us provide increasingly personalized feedback")
        
        # Success rate recommendations
        if success_rate < 90:
            recommendations.append("🔧 **Workflow Optimization**: Review session endings to ensure full workflow completion")
        
        # # Always include LangGraph-specific recommendations
        # recommendations.extend([
        #     "🔬 **LangGraph Advantage**: Each session's AI analysis improves with more diverse practice data",
        #     "📊 **Feedback Patterns**: Review AI-generated feedback trends to identify consistent growth areas",
        #     "🎭 **Virtual Patient Variety**: Mix profile types to train against diverse clinical presentations",
        #     "📈 **Progress Tracking**: Use session analytics to monitor improvement over time"
        # ])
        
        for recommendation in recommendations:
            st.markdown(recommendation)
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("📭 Complete some  therapy sessions to see your comprehensive performance review!")
        if st.button("🎭 Start First Session", use_container_width=True):
            st.session_state.selected_feature = "💬 Therapy Session"
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Clinical Learning Buddy v27m742nr")
