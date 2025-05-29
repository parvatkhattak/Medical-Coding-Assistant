import streamlit as st
import requests
import json
import uuid
from typing import Dict, List, Any
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Coding Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern dark theme with improved mobile responsiveness
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global dark theme */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        color: #e2e8f0;
    }
    
    .main .block-container {
        background: transparent;
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1200px;
    }
    
.main-header {
    font-size: clamp(2.8rem, 6vw, 4rem);  /* Bigger font */
    font-weight: 800;
    background: linear-gradient(135deg, #4fc3f7 0%, #03a9f4 50%, #0288d1 100%);
    text-align: center;
    margin-bottom: 2rem;
    letter-spacing: -0.03em;
    line-height: 1.2;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);  /* Add subtle shadow */
}

    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: clamp(0.9rem, 3vw, 1.1rem);
        margin-bottom: 2rem;
        font-weight: 400;
        padding: 0 1rem;
        line-height: 1.4;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
        border-left: 4px solid #3b82f6;
        margin-left: 10%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-left: 4px solid #10b981;
        margin-right: 10%;
    }
    
    /* Debug message styling */
    .debug-message {
        background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%);
        border-left: 4px solid #ea580c;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .debug-content {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .chat-message h4 {
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: clamp(0.9rem, 3vw, 1.1rem);
    }
    
    .chat-content {
        color: #e2e8f0;
        line-height: 1.6;
        font-size: clamp(0.85rem, 2.5vw, 0.95rem);
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
        font-size: clamp(0.8rem, 2.5vw, 0.9rem);
    }
    
    .status-offline {
        color: #ef4444;
        font-weight: 600;
        font-size: clamp(0.8rem, 2.5vw, 0.9rem);
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
        font-size: clamp(0.8rem, 2.5vw, 0.9rem);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .sidebar-section {
        background: rgba(30, 41, 59, 0.6);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Sidebar text sizing */
    .css-1d391kg .markdown-text-container {
        font-size: clamp(0.8rem, 2.2vw, 0.9rem);
    }
    
    /* Input styling */
    .stTextInput input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-size: clamp(0.85rem, 2.5vw, 0.95rem) !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Chat input specific styling */
    .stChatInput input {
        font-size: clamp(0.9rem, 2.5vw, 1rem) !important;
        padding: 1rem !important;
        min-height: 3rem !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        font-size: clamp(0.8rem, 2.2vw, 0.9rem) !important;
        padding: 0.75rem 1rem !important;
        width: 100% !important;
        min-height: 2.5rem !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Debug button styling */
    .debug-button button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
    }
    
    .debug-button button:hover {
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Quick action buttons */
    .quick-btn {
        background: rgba(71, 85, 105, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        color: #ffffff !important;
        text-align: left !important;
        font-size: clamp(0.75rem, 2vw, 0.9rem) !important;
        margin: 0.2rem 0 !important;
        padding: 0.6rem 0.8rem !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    .quick-btn:hover {
        background: rgba(71, 85, 105, 0.8) !important;
        border-color: #cbd5e1 !important;
        color: #ffffff !important;
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(30, 41, 59, 0.6);
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.2);
        font-size: clamp(0.8rem, 2.2vw, 0.9rem);
    }
    
    /* Info boxes */
    .stInfo {
        font-size: clamp(0.75rem, 2vw, 0.85rem) !important;
    }
    
    .stError {
        font-size: clamp(0.75rem, 2vw, 0.85rem) !important;
    }
    
    /* Loading spinner */
    .stSpinner {
        color: #3b82f6 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: clamp(0.7rem, 1.8vw, 0.85rem);
        margin-top: 2rem;
        padding: 1.5rem 0;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
        line-height: 1.4;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Toggle switch for debug mode */
    .debug-toggle {
        background: rgba(30, 41, 59, 0.6);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Responsive design for mobile */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.5rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        .user-message, .assistant-message, .debug-message {
            margin-left: 0;
            margin-right: 0;
            padding: 0.8rem;
        }
        
        .chat-message {
            margin: 0.8rem 0;
        }
        
        .main-header {
            margin-bottom: 1rem;
        }
        
        .subtitle {
            margin-bottom: 1.5rem;
            padding: 0 0.5rem;
        }
        
        /* Sidebar adjustments for mobile */
        .css-1d391kg {
            width: 100% !important;
        }
        
        /* Make buttons more touch-friendly */
        .stButton button {
            min-height: 3rem !important;
            padding: 1rem !important;
        }
        
        .quick-btn {
            min-height: 2.5rem !important;
            padding: 0.8rem !important;
        }
    }
    
    /* Extra small screens */
    @media (max-width: 480px) {
        .main .block-container {
            padding-left: 0.25rem;
            padding-right: 0.25rem;
        }
        
        .chat-message {
            padding: 0.6rem;
            margin: 0.6rem 0;
        }
        
        .sidebar-section {
            padding: 0.6rem;
            margin: 0.6rem 0;
        }
        
        .footer {
            padding: 1rem 0;
            margin-top: 1.5rem;
        }
    }
    
    /* Large screens */
    @media (min-width: 1200px) {
        .user-message {
            margin-left: 20%;
        }
        
        .assistant-message {
            margin-right: 20%;
        }
        
        .chat-message {
            padding: 1.5rem;
        }
    }
    
    /* Tablet portrait */
    @media (max-width: 1024px) and (min-width: 769px) {
        .user-message {
            margin-left: 15%;
        }
        
        .assistant-message {
            margin-right: 15%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Configuration
FASTAPI_URL = "https://chatbot-vl3b.onrender.com"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "user"
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "last_api_response" not in st.session_state:
    st.session_state.last_api_response = None

def call_chatbot_api(question: str) -> Dict[str, Any]:
    """Call the FastAPI chatbot endpoint"""
    try:
        payload = {
            "question": question,
            "chat_id": st.session_state.chat_id,
            "user_id": st.session_state.user_id
        }
        
        response = requests.post(
            f"{FASTAPI_URL}/api/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            api_response = response.json()
            # Store the full API response for debug purposes
            st.session_state.last_api_response = api_response
            return api_response
        elif response.status_code == 429:
            return {
                "answer": "⚠️ Rate limit exceeded. Please wait before trying again."
            }
        else:
            return {
                "answer": f"❌ Server error (Status: {response.status_code})"
            }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "❌ Cannot connect to server. Please ensure the FastAPI server is running."
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "⏱️ Request timed out. Please try again."
        }
    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}"
        }

def check_api_health() -> Dict[str, str]:
    """Check API health status"""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return {"status": "online", "message": "API is healthy"}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except:
        return {"status": "offline", "message": "Cannot connect to API"}

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with modern styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <h4>👤 You</h4>
            <div class="chat-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <h4>🤖 Medical Coding Assistant</h4>
            <div class="chat-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)

def display_debug_info(api_response: Dict[str, Any]):
    """Display debug information about the API response"""
    st.markdown("""
    <div class="chat-message debug-message">
        <h4>🔧 Debug Information</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different debug sections
    debug_tabs = st.tabs(["📊 Summary", "🔍 Sources", "🧠 Structured Query", "📈 Full Response"])
    
    with debug_tabs[0]:
        # Summary information
        st.subheader("Response Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sources_count = len(api_response.get("sources", []))
            st.metric("Sources Found", sources_count)
        
        with col2:
            has_structured_query = api_response.get("structured_query") is not None
            st.metric("Structured Query", "Yes" if has_structured_query else "No")
        
        with col3:
            response_length = len(api_response.get("answer", ""))
            st.metric("Response Length", f"{response_length} chars")
    
    with debug_tabs[1]:
        # Sources information
        st.subheader("Retrieved Sources")
        sources = api_response.get("sources", [])
        
        if sources:
            for i, source in enumerate(sources):
                with st.expander(f"Source {i+1}: {source.get('source_group', 'Unknown')} - Score: {source.get('score', 0):.4f}"):
                    st.write("**File Name:**", source.get('file_name', 'Unknown'))
                    st.write("**Source Description:**", source.get('source_description', 'N/A'))
                    st.write("**Priority:**", source.get('source_priority', 'N/A'))
                    st.write("**Score:**", source.get('score', 'N/A'))
                    
                    # Display metadata if available
                    metadata = source.get('metadata', {})
                    if metadata:
                        st.write("**Metadata:**")
                        st.json(metadata)
                    
                    # Display source text
                    st.write("**Content:**")
                    text_content = source.get('text', 'No content available')
                    if len(text_content) > 500:
                        st.text_area("Source Content", text_content, height=200, disabled=True, key=f"source_text_{i}")
                    else:
                        st.write(text_content)
        else:
            st.info("No sources retrieved for this query")
    
    with debug_tabs[2]:
        # Structured query information
        st.subheader("Structured Query Analysis")
        structured_query = api_response.get("structured_query")
        
        if structured_query:
            st.json(structured_query)
            
            # Additional analysis
            if isinstance(structured_query, dict):
                st.write("**Query Intent:**", structured_query.get("intent", "Unknown"))
                st.write("**Context Aware:**", structured_query.get("context_aware", False))
                
                filters = structured_query.get("filters", {})
                if filters:
                    st.write("**Applied Filters:**")
                    for key, value in filters.items():
                        if value:  # Only show non-empty filters
                            st.write(f"- **{key.capitalize()}:** {value}")
        else:
            st.info("No structured query information available")
    
    with debug_tabs[3]:
        # Full API response
        st.subheader("Complete API Response")
        st.json(api_response)

def safe_rerun():
    """Safely handle page rerun across different Streamlit versions"""
    try:
        # Try the new method first (Streamlit >= 1.27.0)
        st.rerun()
    except AttributeError:
        try:
            # Try the experimental method (older versions)
            st.experimental_rerun()
        except AttributeError:
            # If neither works, use a workaround with session state
            st.session_state.force_rerun = True

def process_sample_question(question: str):
    """Process a sample question and get AI response"""
    # Add user message
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    
    # Get AI response
    with st.spinner("🔍 Analyzing your question..."):
        response = call_chatbot_api(question)
        assistant_message = {
            "role": "assistant", 
            "content": response["answer"],
            "debug_data": response if st.session_state.debug_mode else None
        }
        st.session_state.messages.append(assistant_message)

def main():
    # Header
    st.markdown("""
    <h1 class="main-header">🏥 Medical Coding Assistant</h1>
    <p class="subtitle">AI-powered ICD-10 and CPT coding guidance for medical professionals</p>
    """, unsafe_allow_html=True)
    
    # Check API health
    health = check_api_health()
    api_online = health["status"] == "online"
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # API Status
        if api_online:
            st.markdown('<p class="status-online">🟢 API Online</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-offline">🔴 API Offline</p>', unsafe_allow_html=True)
            st.error(health["message"])
        
        # Debug Mode Toggle
        st.markdown("## 🔧 Debug Settings")
        st.session_state.debug_mode = st.toggle(
            "Enable Debug Mode", 
            value=st.session_state.debug_mode,
            help="Show detailed information about API responses and retrieved data"
        )
        
        if st.session_state.debug_mode:
            st.success("🔍 Debug mode enabled")
            
            # # Show debug button for last response if available
            # if st.session_state.last_api_response:
            #     if st.button("🔧 Show Last Debug Info", help="Display debug information for the last API response"):
            #         # We'll display this in the main area
            #         st.session_state.show_debug_popup = True
        else:
            st.info("Debug mode disabled")
        
        # Session info
        st.markdown("## 💬 Session Info")
        st.info(f"**Messages:** {len(st.session_state.messages)}")
        st.info(f"**Session ID:** `{st.session_state.chat_id[:8]}...`")
        
        # Quick actions
        st.markdown("## ⚡ Quick Start")
        
        sample_questions = [
            "What is the ICD-10 code for diabetes?",
            "How do I code pneumonia?",
            "Hypertension coding guidelines",
            "ICD-10 sequencing rules",
            "CPT evaluation codes"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}", disabled=not api_online, 
                        help="Click to ask this question"):
                process_sample_question(question)
                safe_rerun()

        # Chat management
        st.markdown("## 💬 Chat Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Chat", type="primary", help="Start a new conversation"):
                st.session_state.messages = []
                st.session_state.chat_id = str(uuid.uuid4())
                st.session_state.last_api_response = None
                safe_rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary", help="Clear current conversation"):
                st.session_state.messages = []
                st.session_state.last_api_response = None
                safe_rerun()
        
        # Instructions
        st.markdown("## 📋 How to Use")
        st.markdown("""
        1. **Ask Questions**: Type medical coding questions below
        2. **Get Answers**: Receive AI-powered coding guidance
        3. **Debug Mode**: Enable to see retrieved sources and query analysis
        4. **Follow Up**: Ask clarifying questions as needed
        
        **Example Questions:**
        - "What is the ICD-10 code for acute MI?"
        - "How do I code diabetes with complications?"
        - "What are the guidelines for sequencing codes?"
        """)

    # Main chat area
    # Show debug popup if requested
    if hasattr(st.session_state, 'show_debug_popup') and st.session_state.show_debug_popup:
        if st.session_state.last_api_response:
            st.markdown("---")
            display_debug_info(st.session_state.last_api_response)
            st.markdown("---")
        st.session_state.show_debug_popup = False
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            display_chat_message(message, is_user=True)
        else:
            display_chat_message(message, is_user=False)
            
            # Show debug info if debug mode is enabled and debug data is available
            if st.session_state.debug_mode and message.get("debug_data"):
                display_debug_info(message["debug_data"])
    
    # Chat input
    placeholder_text = "Ask me about medical coding..." if api_online else "API server offline"
    
    if prompt := st.chat_input(placeholder_text, disabled=not api_online):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_chat_message(user_message, is_user=True)
        
        # Get AI response
        with st.spinner("🔍 Searching knowledge base..."):
            response = call_chatbot_api(prompt)
            assistant_message = {
                "role": "assistant",
                "content": response["answer"],
                "debug_data": response if st.session_state.debug_mode else None
            }
            st.session_state.messages.append(assistant_message)
            display_chat_message(assistant_message, is_user=False)
            
            # Show debug info immediately if debug mode is enabled
            if st.session_state.debug_mode:
                display_debug_info(response)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🏥 Medical Coding Assistant | Powered by AI Technology</p>
        <p>⚠️ For educational purposes only. Always consult official coding guidelines for clinical use.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
