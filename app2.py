"""
Voice RAG Search Assistant
A Streamlit application that enables voice search using Pinecone vector database.
"""

import os
import time
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Import the AI search functionality
from pinecone_client import dense_index, namespace

# ----- CONFIGURATION -----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMPLE_RATE = 16000  # Audio sample rate

# Initialize API clients
client = OpenAI(api_key=OPENAI_API_KEY)


# ----- HELPER FUNCTIONS -----
def load_css():
    """Load custom CSS styling from file"""
    try:
        css_file = Path("custom_style.css")
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning("custom_style.css not found. UI styling may be affected.")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")


def init_session_state():
    """Initialize session state variables with defaults"""
    defaults = {
        "audio_data": None,
        "recording": False,
        "transcript": "",
        "processing": False,
        "search_results": None,
        "toast_shown": False,
        "record_duration": 15,
        "top_k": 10
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def save_audio():
    """Save recorded audio to a temporary file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio = st.session_state.audio_data
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    sf.write(temp_file.name, audio, SAMPLE_RATE)
    return temp_file.name


def transcribe_audio(file_path):
    """Transcribe audio file using OpenAI Whisper API"""
    try:
        with open(file_path, "rb") as file:
            response = client.audio.transcriptions.create(model="whisper-1", file=file)
        os.unlink(file_path)  # Delete temporary file
        return response.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None


def record_audio():
    """Start audio recording"""
    st.session_state.audio_data = sd.rec(
        int(SAMPLE_RATE * st.session_state.record_duration),
        samplerate=SAMPLE_RATE, channels=1, dtype=np.int16
    )
    st.session_state.recording = True
    st.session_state.transcript = ""
    st.session_state.search_results = None
    st.session_state.toast_shown = False


def stop_recording():
    """Stop audio recording and process the audio"""
    st.session_state.recording = False
    sd.stop()
    st.toast("Recording stopped successfully!", icon="✅")
    process_audio()


def process_audio():
    """Process recorded audio by transcribing and searching"""
    if st.session_state.audio_data is None:
        return

    st.session_state.processing = True
    status = st.empty()
    status.markdown("""
    <div class="status-processing">
        <h3>🔄 Processing your audio...</h3>
        <p>Converting speech to text using OpenAI Whisper. This may take a few moments.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        audio_path = save_audio()
        result = transcribe_audio(audio_path)

        if result:
            st.session_state.transcript = result
            status.empty()
            if not st.session_state.toast_shown:
                st.toast("Transcription complete!", icon="✅")
                st.session_state.toast_shown = True
            perform_search(result)
        else:
            status.empty()
            st.error("Failed to transcribe audio. Please try again.")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
    finally:
        st.session_state.processing = False


def ai_search(query, top_k=10):
    """Search Pinecone database with the query"""
    try:
        results = dense_index.search(
            namespace=namespace,
            query={"top_k": top_k, "inputs": {"text": query}}
        )
        return results["result"]["hits"]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def perform_search(query):
    """Execute search and update session state with results"""
    with st.spinner("Searching knowledge base..."):
        results = ai_search(query, top_k=st.session_state.top_k)
        st.session_state.search_results = results
        if results:
            st.toast(f"Found {len(results)} results", icon="🔍")
        else:
            st.toast("No results found. Try a different query.", icon="ℹ️")


def highlight_query_terms(text, query):
    """Highlight query terms in text for better visibility"""
    if not query or not text:
        return text

    query_terms = query.lower().split()
    for term in query_terms:
        if len(term) > 2:  # Only highlight terms longer than 2 characters
            idx = text.lower().find(term.lower())
            while idx != -1:
                original = text[idx:idx + len(term)]
                text = text[:idx] + f'<span class="highlight">{original}</span>' + text[idx + len(term):]
                # Adjust index to account for added HTML
                idx = text.lower().find(term.lower(), idx + len(f'<span class="highlight">{original}</span>'))
    return text


# ----- UI COMPONENTS -----
def render_header():
    """Render the application header"""
    st.markdown("""
    <div class="header-container">
        <div class="header-emoji">🎙️</div>
        <h1>Voice RAG Search Assistant</h1>
        <p>Speak naturally and search your knowledge base with AI-powered RAG technology</p>
    </div>
    """, unsafe_allow_html=True)


def render_recording_ui():
    """Render the recording interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.recording:
            if st.button("⏹️ Stop Recording", type="primary", use_container_width=True):
                stop_recording()
        else:
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                record_audio()

    # Display recording progress if recording
    if st.session_state.recording:
        rec_status = st.empty()
        start = time.time()
        while st.session_state.recording and (time.time() - start) < st.session_state.record_duration:
            elapsed = time.time() - start
            remaining = st.session_state.record_duration - elapsed
            progress_percentage = (elapsed / st.session_state.record_duration) * 100

            rec_status.markdown(f"""
            <div class="status-recording">
                <h3>🔴 Recording...</h3>
                <p>{remaining:.1f} seconds remaining.</p>
                <div style="height: 5px; background-color: #f1f1f1; border-radius: 5px;">
                    <div style="height: 5px; width: {progress_percentage}%;
                    background-color: #e74c3c; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.1)
            if not st.session_state.recording:
                break

        # If recording ended due to timeout
        if st.session_state.recording:
            st.session_state.recording = False
            sd.stop()
            rec_status.empty()
            st.toast("Recording complete!", icon="✅")
            process_audio()


def render_transcript_ui():
    """Render the transcript and search input area"""
    if not st.session_state.transcript:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        st.markdown("### 📝 Your Query")
        col1, col2 = st.columns([3, 1])
        with col1:
            transcript_text = st.text_area("Edit your query:", value=st.session_state.transcript, height=100)
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Search", type="primary", use_container_width=True):
                perform_search(transcript_text)
        st.markdown('</div>', unsafe_allow_html=True)


def render_search_results():
    """Render search results from Pinecone"""
    if not st.session_state.search_results:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.container():
        count = len(st.session_state.search_results)
        st.markdown(f"""
        <div class="results-container">
            <div class="results-header">
                <h2>🔎 Search Results ({count})</h2>
                <span class="badge">Top {min(count, st.session_state.top_k)} matches</span>
            </div>
        """, unsafe_allow_html=True)

        if count > 0:
            for i, hit in enumerate(st.session_state.search_results):
                score = hit.get("_score", 0)
                text = hit.get("fields", {}).get("chunk_text", "No text available")
                hit_id = hit.get("_id", "N/A")

                # Determine relevance class based on score
                relevance = "high-relevance" if score > 0.7 else "medium-relevance" if score > 0.4 else "low-relevance"
                score_pct = round(score * 100)

                # Highlight query terms in the text
                highlighted = highlight_query_terms(text, st.session_state.transcript)

                st.markdown(f"""
                <div class="card {relevance}">
                    <div class="result-number">{i + 1}</div>
                    <div class="card-header">
                        <h3>Match #{i + 1}</h3>
                        <span class="score-badge">Relevance: {score_pct}%</span>
                    </div>
                    <div class="card-content">{highlighted}</div>
                    <div class="result-id">ID: {hit_id}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Empty state for no results
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <h3>No results found</h3>
                <p>Try adjusting your search query or increasing the number of results to display.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_help_section():
    """Render the help expander section"""
    with st.expander("📚 How to Use This App"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Quick Start:
            1. Click **Start Recording**
            2. Speak your search query
            3. The app transcribes and searches automatically
            """)
        with col2:
            st.markdown("""
            ### Tips:
            - Speak clearly and be specific
            - Edit your query for more accurate results
            - Adjust duration or result count in sidebar
            """)


def render_sidebar():
    """Render the sidebar with settings"""
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("🎙️ Recording Duration")
        duration = st.slider("Seconds", 5, 60, st.session_state.record_duration)
        if duration != st.session_state.record_duration:
            st.session_state.record_duration = duration
            st.toast(f"Recording duration set to {duration} seconds", icon="⏱️")

        st.subheader("🔍 Result Settings")
        top_k = st.slider("Top Results", 1, 100, st.session_state.top_k)
        if top_k != st.session_state.top_k:
            st.session_state.top_k = top_k
            st.toast(f"Top results set to {top_k}", icon="🔢")
            if st.session_state.transcript:
                perform_search(st.session_state.transcript)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Search", use_container_width=True):
                st.session_state.transcript = ""
                st.session_state.search_results = None
                st.experimental_rerun()
        with col2:
            if st.button("⬇️ Export Results", use_container_width=True):
                export_results()


def render_footer():
    """Render the application footer"""
    st.markdown("""
    <div style="text-align:center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;">
        <p style="color: #666; font-size: 0.8rem;">Powered by OpenAI Whisper & Pinecone | Built with ❤️ in Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


def export_results():
    """Export search results to a file"""
    if not st.session_state.search_results:
        st.toast("No results to export!", icon="ℹ️")
        return

    try:
        # Placeholder for actual export implementation
        # You would generate CSV/JSON here based on st.session_state.search_results
        st.toast("Export feature coming soon!", icon="ℹ️")
    except Exception as e:
        st.error(f"Export error: {str(e)}")


# ----- MAIN APP FUNCTION -----
def main():
    """Main application function"""
    # Page setup
    st.set_page_config(
        page_title="Voice RAG Search Assistant",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Set overall page background gradient
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load CSS and initialize session state
    load_css()
    init_session_state()

    # Render app components
    render_header()
    render_recording_ui()
    render_transcript_ui()
    render_search_results()
    render_help_section()
    render_sidebar()
    render_footer()


# Run the application
if __name__ == "__main__":
    main()