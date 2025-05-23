import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Source RAG Medical Coding Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    /* Global dark theme overrides */
    .main .block-container {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    .stApp {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    /* Sidebar dark styling */
    .css-1d391kg, .css-1cypcdb {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* All text elements */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4FC3F7 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4FC3F7;
        color: #ffffff !important;
    }
    .user-message {
        background-color: #2d3748 !important;
        border-left-color: #4FC3F7;
        color: #ffffff !important;
    }
    .assistant-message {
        background-color: #1a202c !important;
        border-left-color: #68D391;
        color: #ffffff !important;
    }
    .warning-message {
        background-color: #744210 !important;
        border-left-color: #F6E05E;
        color: #FBD38D !important;
        border: 1px solid #975A16;
    }
    .chat-message strong {
        color: #ffffff !important;
    }
    .chat-message * {
        color: #ffffff !important;
    }
    
    /* Source information */
    .source-info {
        background-color: #2d3748 !important;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border: 1px solid #4a5568;
        font-size: 0.9rem;
        color: #ffffff !important;
    }
    .source-info * {
        color: #ffffff !important;
    }
    
    /* Structured query */
    .structured-query {
        background-color: #1a365d !important;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #2c5282;
        font-size: 0.9rem;
        color: #ffffff !important;
    }
    
    /* Priority badges */
    .priority-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .priority-1 { background-color: #22543d; color: #68d391; }
    .priority-2 { background-color: #744210; color: #f6e05e; }
    .priority-3 { background-color: #742a2a; color: #feb2b2; }
    
    /* Group styling */
    .group-icd { border-left: 4px solid #68d391; }
    .group-cpt { border-left: 4px solid #f6e05e; }
    .group-terminology { border-left: 4px solid #63b3ed; }
    
    /* Metrics container */
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    /* API status indicators */
    .api-status-online {
        color: #68d391 !important;
        font-weight: bold;
    }
    .api-status-offline {
        color: #fc8181 !important;
        font-weight: bold;
    }
    .api-status-limited {
        color: #f6e05e !important;
        font-weight: bold;
    }
    
    /* Streamlit components dark styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    .stButton button {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    .stButton button:hover {
        background-color: #4a5568 !important;
        color: #ffffff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1a202c !important;
        color: #ffffff !important;
    }
    
    /* Code blocks */
    .stCode, code, pre {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Metrics */
    .metric-container {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar elements */
    .css-1cypcdb .stMarkdown, .css-1cypcdb .stText {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Footer styling */
    .footer-text {
        color: #a0aec0 !important;
    }
    
    /* Override any remaining light backgrounds */
    div[data-testid="stSidebar"] {
        background-color: #2d2d2d !important;
    }
    
    div[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .stAlert {
        background-color: #744210 !important;
        color: #ffffff !important;
        border: 1px solid #975A16 !important;
    }
    
    .stSuccess {
        background-color: #22543d !important;
        color: #ffffff !important;
        border: 1px solid #2f855a !important;
    }
    
    .stError {
        background-color: #742a2a !important;
        color: #ffffff !important;
        border: 1px solid #c53030 !important;
    }
    
    .stWarning {
        background-color: #744210 !important;
        color: #ffffff !important;
        border: 1px solid #975A16 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
FASTAPI_URL = "http://localhost:8000"  # Change this to your FastAPI server URL

# Document groups mapping (matching the FastAPI structure)
DOCUMENT_GROUPS = {
    "ICD_CODES": {
        "name": "ICD-10 Guidelines & References",
        "description": "ICD-10 Coding Guidelines and References",
        "priority": 1,
        "color": "#28a745"
    },
    "CPT_PROCEDURES": {
        "name": "CPT Procedures & Documentation",
        "description": "CPT Procedure Codes and Documentation",
        "priority": 2,
        "color": "#ffc107"
    },
    "MEDICAL_TERMINOLOGY": {
        "name": "Medical Terminology & Definitions",
        "description": "Medical Terminology and Definitions",
        "priority": 3,
        "color": "#17a2b8"
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "api_health" not in st.session_state:
    st.session_state.api_health = {}

def call_chatbot_api(question: str, show_retrieved_data: bool = False) -> Dict[str, Any]:
    """Call the FastAPI chatbot endpoint with enhanced error handling"""
    try:
        payload = {
            "question": question,
            "chat_id": st.session_state.chat_id,
            "user_id": st.session_state.user_id,
            "show_retrieved_data": show_retrieved_data
        }
        
        response = requests.post(
            f"{FASTAPI_URL}/api/chat",
            json=payload,
            timeout=60  # Increased timeout for Gemini API calls
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        elif response.status_code == 429:
            return {
                "answer": "⚠️ API rate limit exceeded. Please wait a moment before trying again.",
                "sources": [],
                "structured_query": None,
                "warning": "Rate limit exceeded"
            }
        else:
            return {
                "answer": f"❌ API Error: Server returned status code {response.status_code}",
                "sources": [],
                "structured_query": None,
                "warning": f"HTTP {response.status_code} error"
            }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "❌ Connection Error: Could not connect to the chatbot API. Please make sure the FastAPI server is running on the correct port.",
            "sources": [],
            "structured_query": None,
            "warning": "Connection failed"
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "⏱️ Request timed out. The API might be processing a complex query or experiencing high load. Please try again.",
            "sources": [],
            "structured_query": None,
            "warning": "Request timeout"
        }
    except Exception as e:
        return {
            "answer": f"❌ Unexpected Error: {str(e)}",
            "sources": [],
            "structured_query": None,
            "warning": "Unexpected error occurred"
        }

def get_api_health() -> Dict[str, Any]:
    """Get detailed API health information"""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/health", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_collection_stats() -> Dict[str, Any]:
    """Get collection statistics"""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/collection/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def display_structured_query(structured_query: Dict[str, Any]):
    """Display the structured query information with enhanced formatting"""
    if structured_query:
        with st.expander("🔍 Query Analysis & Processing", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Intent Detection:**")
                intent = structured_query.get("intent", "Unknown")
                st.code(intent)
                
                st.markdown("**🔄 Query Refinement:**")
                refined_query = structured_query.get("search_query", structured_query.get("query", ""))
                st.text_area("Processed Query", refined_query, height=68, disabled=True)
            
            with col2:
                filters = structured_query.get("filters", {})
                st.markdown("**🔍 Extracted Filters:**")
                
                if filters.get("keywords"):
                    st.write("**Keywords:**", ", ".join(filters["keywords"]))
                if filters.get("code"):
                    st.write("**Code:**", filters["code"])
                
                # Show the original query for comparison
                st.markdown("**📝 Original Query:**")
                st.text_area("User Input", structured_query.get("query", ""), height=68, disabled=True)

def display_sources(sources: List[Dict[str, Any]]):
    """Display source information with enhanced formatting"""
    if sources:
        with st.expander(f"📚 Knowledge Sources ({len(sources)} found)", expanded=False):
            # Group sources by priority
            sources_by_priority = {}
            for source in sources:
                priority = source.get('source_priority', 99)
                if priority not in sources_by_priority:
                    sources_by_priority[priority] = []
                sources_by_priority[priority].append(source)
            
            # Display sources grouped by priority
            for priority in sorted(sources_by_priority.keys()):
                priority_sources = sources_by_priority[priority]
                
                st.markdown(f"**Priority {priority} Sources ({len(priority_sources)} items):**")
                
                for i, source in enumerate(priority_sources):
                    group = source.get('source_group', 'UNKNOWN')
                    group_info = DOCUMENT_GROUPS.get(group, {})
                    
                    # Get CSS class for group styling
                    group_class = f"group-{group.lower().split('_')[0]}" if '_' in group else ""
                    
                    st.markdown(f"""
                    <div class="source-info {group_class}">
                        <strong>📄 {group_info.get('name', group)}</strong>
                        <span class="priority-badge priority-{priority}">Priority {priority}</span><br>
                        <small><strong>File:</strong> {source.get('file_name', 'Unknown')}</small><br>
                        <small><strong>Relevance Score:</strong> {source.get('score', 0):.4f}</small><br>
                        <small><strong>Chunk:</strong> #{source.get('chunk_index', 0)}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")

def display_retrieved_data(retrieved_data: List[Dict[str, Any]]):
    """Display retrieved data for debugging purposes"""
    if retrieved_data:
        with st.expander(f"🔍 Retrieved Data ({len(retrieved_data)} chunks)", expanded=False):
            for i, data in enumerate(retrieved_data):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Chunk {i+1}:** {data.get('source_group', 'Unknown')}")
                        st.text_area(
                            f"Content Preview",
                            data.get('text', ''),
                            height=100,
                            key=f"retrieved_data_{i}",
                            disabled=True
                        )
                    
                    with col2:
                        st.metric("Score", f"{data.get('score', 0):.4f}")
                        st.write(f"**File:** {data.get('file_name', 'Unknown')}")
                        st.write(f"**Chunk:** #{data.get('chunk_index', 0)}")

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling and enhanced features"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display warning if present
        if message.get('warning'):
            st.markdown(f"""
            <div class="chat-message warning-message">
                <strong>⚠️ Notice:</strong> {message['warning']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Medical Coding Assistant:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display additional information if available
        if 'structured_query' in message and message['structured_query']:
            display_structured_query(message['structured_query'])
        
        if 'sources' in message and message['sources']:
            display_sources(message['sources'])
        
        if 'retrieved_data' in message and message['retrieved_data'] and st.session_state.show_debug:
            display_retrieved_data(message['retrieved_data'])

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Multi-Source RAG Medical Coding Assistant</h1>', unsafe_allow_html=True)
    
    # Get API health status
    health_info = get_api_health()
    st.session_state.api_health = health_info
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ System Information")
        
        # API Status with detailed information
        status = health_info.get("status", "unknown")
        if status == "healthy":
            st.markdown('<p class="api-status-online">🟢 API: Online & Healthy</p>', unsafe_allow_html=True)
            
            # Show Gemini status
            gemini_status = health_info.get("gemini_status", "unknown")
            if "limited" in str(gemini_status).lower():
                st.markdown(f'<p class="api-status-limited">⚠️ Gemini: {gemini_status}</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="api-status-online">🟢 Gemini: Available</p>', unsafe_allow_html=True)
            
            # Show daily requests if available
            daily_requests = health_info.get("daily_requests_used")
            if daily_requests is not None:
                st.write(f"**Daily API Usage:** {daily_requests}/500 requests")
                if daily_requests > 450:
                    st.warning("⚠️ Approaching daily limit")
        else:
            st.markdown('<p class="api-status-offline">🔴 API: Offline</p>', unsafe_allow_html=True)
            st.error("❌ FastAPI server is not running. Please start the server before using the chatbot.")
        
        st.write(f"**Chat Session:** `{st.session_state.chat_id[:8]}...`")
        st.write(f"**Total Messages:** {len(st.session_state.messages)}")
        
        # Collection Statistics
        if status == "healthy":
            st.header("📊 Knowledge Base Stats")
            stats = get_collection_stats()
            
            if "error" not in stats:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
                
                groups = stats.get("groups", {})
                for group_key, count in groups.items():
                    group_info = DOCUMENT_GROUPS.get(group_key, {})
                    group_name = group_info.get("name", group_key)
                    st.write(f"**{group_name}:** {count} chunks")
        
        # Document Groups Info
        st.header("📁 Knowledge Sources")
        for group_key, group_info in DOCUMENT_GROUPS.items():
            priority = group_info.get("priority", 99)
            st.markdown(f"""
            **{group_info['name']}** (Priority {priority})  
            {group_info['description']}
            """)
        
        # Settings
        st.header("⚙️ Settings")
        st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.show_debug = st.checkbox("Show Debug Information", value=st.session_state.show_debug)
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_id = str(uuid.uuid4())
            st.rerun()
        
        # Instructions
        st.header("📋 Usage Instructions")
        st.markdown("""
        **How to use this assistant:**
        1. Type your medical coding question below
        2. Review the query analysis and processing
        3. Read the comprehensive answer with sources
        4. Check source priorities and relevance scores
        5. Ask follow-up questions for clarification
        
        **Example questions:**
        - "What is the ICD-10 code for Type 2 diabetes with complications?"
        - "How do I code acute myocardial infarction with STEMI?"
        - "What are the sequencing guidelines for multiple diagnoses?"
        - "Explain the difference between CPT codes 99213 and 99214"
        - "What documentation is required for diabetes coding?"
        """)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    display_chat_message(message, is_user=True)
                else:
                    display_chat_message(message, is_user=False)
        
        # Chat input
        api_available = health_info.get("status") == "healthy"
        placeholder_text = "Ask me about medical coding..." if api_available else "API server offline - please start the FastAPI server"
        
        if prompt := st.chat_input(placeholder_text, disabled=not api_available):
            # Add user message to chat history
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Display user message immediately
            display_chat_message(user_message, is_user=True)
            
            # Show loading spinner with context
            with st.spinner("🔍 Analyzing query and searching knowledge base..."):
                # Call the API with debug data if requested
                response = call_chatbot_api(prompt, show_retrieved_data=st.session_state.show_debug)
                
                # Add assistant response to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "structured_query": response.get("structured_query"),
                    "warning": response.get("warning"),
                    "retrieved_data": response.get("retrieved_data")
                }
                st.session_state.messages.append(assistant_message)
            
            # Rerun to display the new message
            st.rerun()
    
    with col2:
        # Statistics and metrics
        st.header("📊 Session Analytics")
        
        if st.session_state.messages:
            user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Questions", len(user_messages))
            with col_b:
                st.metric("Responses", len(assistant_messages))
            
            # Analyze recent warnings
            recent_warnings = []
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and msg.get("warning"):
                    recent_warnings.append(msg["warning"])
                    if len(recent_warnings) >= 3:
                        break
            
            if recent_warnings:
                st.header("⚠️ Recent Notices")
                for warning in recent_warnings:
                    st.warning(warning)
            
            # Recent sources analysis
            recent_sources = []
            source_groups = {}
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and msg.get("sources"):
                    for source in msg["sources"]:
                        group = source.get('source_group', 'Unknown')
                        source_groups[group] = source_groups.get(group, 0) + 1
                        recent_sources.append(source)
                    if len(recent_sources) >= 10:
                        break
            
            if source_groups:
                st.header("📚 Source Usage")
                for group, count in source_groups.items():
                    group_info = DOCUMENT_GROUPS.get(group, {})
                    group_name = group_info.get("name", group)
                    st.write(f"• **{group_name}:** {count} references")
        
        # Quick actions
        st.header("⚡ Quick Start Questions")
        
        sample_questions = [
            "What is the ICD-10 code for diabetes mellitus?",
            "How do I code pneumonia with complications?",
            "What are the hypertension coding guidelines?",
            "Explain ICD-10 sequencing rules",
            "What's the difference between I21.0 and I21.9?",
            "How do I document CPT evaluation codes?",
            "What are the requirements for modifier usage?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}", disabled=not api_available):
                # Add to session state to trigger processing
                st.session_state.temp_question = question
                st.rerun()
        
        # Handle sample question
        if hasattr(st.session_state, 'temp_question'):
            question = st.session_state.temp_question
            delattr(st.session_state, 'temp_question')
            
            # Add user message
            user_message = {"role": "user", "content": question}
            st.session_state.messages.append(user_message)
            
            # Call API
            with st.spinner("🔍 Processing your question..."):
                response = call_chatbot_api(question, show_retrieved_data=st.session_state.show_debug)
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "structured_query": response.get("structured_query"),
                    "warning": response.get("warning"),
                    "retrieved_data": response.get("retrieved_data")
                }
                st.session_state.messages.append(assistant_message)
            
            st.rerun()

    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a0aec0; font-size: 0.9rem;" class="footer-text">
        <p>🏥 Multi-Source RAG Medical Coding Assistant | Powered by Gemini AI & Qdrant Vector Database</p>
        <p>⚠️ For educational purposes only. Always consult official coding manuals and guidelines for clinical use.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()