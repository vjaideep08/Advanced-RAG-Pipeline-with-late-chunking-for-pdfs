# streamlit_chatbot.py
import streamlit as st
import requests
import json
from datetime import datetime
import time
from typing import List, Dict, Any
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #2b313e;
        border-left: 4px solid #1f77b4;
    }
    
    .chat-message.bot {
        background-color: #262730;
        border-left: 4px solid #ff6b6b;
    }
    
    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        margin-bottom: 0.5rem;
    }
    
    .chat-message .message {
        color: white;
        margin: 0;
        line-height: 1.6;
    }
    
    .source-tag {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem 0.2rem 0 0;
        display: inline-block;
    }
    
    .stats-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .fine-print-card {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .fine-print-high {
        border-left: 4px solid #f44336;
    }
    
    .fine-print-medium {
        border-left: 4px solid #ff9800;
    }
    
    .fine-print-low {
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8080"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
FINE_PRINTS_ENDPOINT = f"{API_BASE_URL}/fine-prints"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_status" not in st.session_state:
    st.session_state.api_status = None

# Helper functions
def check_api_status():
    """Check if the RAG API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to API at {API_BASE_URL}"
    except requests.exceptions.Timeout:
        return False, "API request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"

def send_chat_message(query: str, chat_history: List[Dict] = None):
    """Send message to RAG chat endpoint"""
    if chat_history is None:
        chat_history = []
    
    payload = {
        "query": query,
        "chat_history": chat_history
    }
    
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. The query might be too complex."
    except Exception as e:
        return False, f"Error sending message: {str(e)}"

def get_fine_prints():
    """Get fine-prints from the API"""
    try:
        response = requests.get(FINE_PRINTS_ENDPOINT, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Error fetching fine-prints: {str(e)}"

def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Error fetching stats: {str(e)}"

def format_chat_history_for_api(messages):
    """Convert Streamlit messages to API format"""
    chat_history = []
    for msg in messages[:-1]:  # Exclude the current message
        role = "user" if msg["role"] == "user" else "assistant"
        chat_history.append({
            "role": role,
            "content": msg["content"]
        })
    return chat_history

# Main App Layout
def main():
    # Header
    st.title("🤖 RAG Pipeline Chat Assistant")
    st.markdown("*Intelligent document analysis and proposal writing assistance*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # API Status Check
        if st.button("🔄 Check API Status", use_container_width=True):
            with st.spinner("Checking API status..."):
                is_healthy, status_info = check_api_status()
                st.session_state.api_status = (is_healthy, status_info)
        
        # Display API Status
        if st.session_state.api_status:
            is_healthy, status_info = st.session_state.api_status
            if is_healthy:
                st.success("✅ API is running")
                if isinstance(status_info, dict):
                    st.json(status_info)
            else:
                st.error(f"❌ API Error: {status_info}")
        
        st.divider()
        
        # System Stats
        st.header("📊 System Statistics")
        if st.button("📈 Get Stats", use_container_width=True):
            success, stats = get_stats()
            if success:
                st.json(stats)
            else:
                st.error(stats)
        
        st.divider()
        
        # Chat Controls
        st.header("💬 Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Export Chat
        if st.session_state.messages:
            if st.button("💾 Export Chat", use_container_width=True):
                chat_export = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="📥 Download Chat JSON",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Main Chat Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show sources for assistant messages
                    if message["role"] == "assistant" and "sources" in message:
                        if message["sources"]:
                            st.markdown("**📚 Sources:**")
                            sources_html = ""
                            for source in message["sources"]:
                                sources_html += f'<span class="source-tag">{source}</span>'
                            st.markdown(sources_html, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Check API status first
            is_healthy, _ = check_api_status()
            if not is_healthy:
                st.error("❌ API is not available. Please check the connection.")
                return
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Prepare chat history
                    chat_history = format_chat_history_for_api(st.session_state.messages)
                    
                    # Send request
                    success, response = send_chat_message(prompt, chat_history)
                    
                    if success:
                        assistant_response = response["response"]
                        sources = response.get("sources", [])
                        
                        # Display response
                        st.markdown(assistant_response)
                        
                        # Display sources
                        if sources:
                            st.markdown("**📚 Sources:**")
                            sources_html = ""
                            for source in sources:
                                sources_html += f'<span class="source-tag">{source}</span>'
                            st.markdown(sources_html, unsafe_allow_html=True)
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": assistant_response,
                            "sources": sources
                        })
                    else:
                        error_msg = f"❌ Error: {response}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    
    with col2:
        st.header("📋 Fine-prints & Key Details")
        
        if st.button("🔍 Extract Fine-prints", use_container_width=True):
            with st.spinner("Extracting key details..."):
                success, fine_prints = get_fine_prints()
                
                if success and fine_prints:
                    st.success(f"✅ Found {len(fine_prints)} fine-prints")
                    
                    # Group by category
                    categories = {}
                    for fp in fine_prints:
                        category = fp.get("category", "general")
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(fp)
                    
                    # Display by category
                    for category, items in categories.items():
                        with st.expander(f"📁 {category.title()} ({len(items)} items)"):
                            for fp in items:
                                importance = fp.get("importance", "low")
                                importance_color = {
                                    "high": "🔴",
                                    "medium": "🟡", 
                                    "low": "🟢"
                                }.get(importance, "⚪")
                                
                                st.markdown(f"""
                                <div class="fine-print-card fine-print-{importance}">
                                    <strong>{importance_color} {fp.get('title', 'No title')}</strong><br>
                                    <small>Source: {fp.get('source_document', 'Unknown')}</small><br>
                                    {fp.get('content', 'No content')}
                                </div>
                                """, unsafe_allow_html=True)
                elif success:
                    st.info("ℹ️ No fine-prints found")
                else:
                    st.error(f"❌ Error: {fine_prints}")
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What are the main requirements?",
            "What are the submission deadlines?",
            "What technical specifications are needed?",
            "What is the pricing structure?",
            "What are the evaluation criteria?",
            "What documents need to be submitted?",
            "Who are the key contacts?",
            "What are the project timelines?"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                # Simulate clicking the chat input
                st.session_state.sample_query = query
                st.rerun()
        
        # Handle sample query
        if hasattr(st.session_state, 'sample_query'):
            prompt = st.session_state.sample_query
            del st.session_state.sample_query
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response
            with st.spinner("Processing sample query..."):
                chat_history = format_chat_history_for_api(st.session_state.messages)
                success, response = send_chat_message(prompt, chat_history)
                
                if success:
                    assistant_response = response["response"]
                    sources = response.get("sources", [])
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response,
                        "sources": sources
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ Error: {response}"
                    })
                
                st.rerun()

# Footer
def show_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔗 **API Endpoint:** http://localhost:8080")
    with col2:
        st.markdown("🤖 **Model:** Claude 3.7")
    with col3:
        st.markdown("🗄️ **Vector DB:** Qdrant")

if __name__ == "__main__":
    main()
    show_footer()

# requirements_streamlit.txt (Additional packages needed)
"""
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.0.0
"""

# run_streamlit.py (Helper script to run the Streamlit app)
import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not already installed"""
    required_packages = [
        "streamlit>=1.28.0",
        "requests>=2.31.0", 
        "pandas>=2.0.0"
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

def main():
    print("🚀 Starting Streamlit RAG Chatbot...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Installing required packages...")
        install_requirements()
    
    # Run the streamlit app
    os.system("streamlit run streamlit_chatbot.py --server.port 8501")

if __name__ == "__main__":
    main()
