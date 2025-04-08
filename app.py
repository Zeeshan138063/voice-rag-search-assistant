import os
import tempfile
import time

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
# ----- PAGE SETUP -----
st.set_page_config(page_title="Voice RAG Search Assistant", page_icon="🎙️", layout="wide",
    initial_sidebar_state="collapsed")


def load_css():
    with open("custom_style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ----- LOAD EXTERNAL CSS -----


# ----- SESSION STATE -----
defaults = {"audio_data": None, "recording": False, "transcript": "", "processing": False, "search_results": None,
    "toast_shown": False, "record_duration": 15, "top_k": 10}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ----- CONSTANTS -----
sample_rate = 16000

# ----- HEADER -----
st.markdown("""
<div class="header-container">
    <div class="header-emoji">🎙️</div>
    <h1>Voice RAG Search Assistant</h1>
    <p>Speak naturally and search your knowledge base with AI-powered RAG technology</p>
</div>
""", unsafe_allow_html=True)


# ----- FUNCTIONS -----
def save_audio():
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio = st.session_state.audio_data
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    sf.write(temp_file.name, audio, sample_rate)
    return temp_file.name


def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as file:
            response = client.audio.transcriptions.create(model="whisper-1", file=file)
        os.unlink(file_path)
        return response.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None


def record_audio():
    st.session_state.audio_data = sd.rec(int(sample_rate * st.session_state.record_duration), samplerate=sample_rate,
        channels=1, dtype=np.int16)
    st.session_state.recording = True
    st.session_state.transcript = ""
    st.session_state.search_results = None
    st.session_state.toast_shown = False


def stop_recording():
    st.session_state.recording = False
    sd.stop()
    st.toast("Recording stopped successfully!", icon="✅")
    process_audio()


def process_audio():
    if st.session_state.audio_data is not None:
        st.session_state.processing = True
        status = st.empty()
        status.markdown("""
        <div class="status-processing">
            <h3>🔄 Processing your audio...</h3>
            <p>Converting speech to text using OpenAI Whisper. This may take a few moments.</p>
        </div>
        """, unsafe_allow_html=True)

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
        st.session_state.processing = False


def ai_search(query, top_k=10):
    try:
        results = dense_index.search(namespace=namespace, query={"top_k": top_k, "inputs": {"text": query}})
        return results["result"]["hits"]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def perform_search(query):
    with st.spinner("Searching knowledge base..."):
        results = ai_search(query, top_k=st.session_state.top_k)
        st.session_state.search_results = results
        if results:
            st.toast(f"Found {len(results)} results", icon="🔍")
        else:
            st.toast("No results found. Try a different query.", icon="ℹ️")


def highlight_query_terms(text, query):
    if not query or not text:
        return text
    query_terms = query.lower().split()
    for term in query_terms:
        if len(term) > 2:
            idx = text.lower().find(term.lower())
            while idx != -1:
                original = text[idx:idx + len(term)]
                text = text[:idx] + f'<span class="highlight">{original}</span>' + text[idx + len(term):]
                idx = text.lower().find(term.lower(), idx + len(f'<span class="highlight">{original}</span>'))
    return text


# ----- RECORDING UI -----
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.session_state.recording:
        if st.button("⏹️ Stop Recording", type="primary", use_container_width=True):
            stop_recording()
    else:
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            record_audio()

if st.session_state.recording:
    rec_status = st.empty()
    start = time.time()
    while st.session_state.recording and (time.time() - start) < st.session_state.record_duration:
        elapsed = time.time() - start
        remaining = st.session_state.record_duration - elapsed
        rec_status.markdown(f"""
        <div class="status-recording">
            <h3>🔴 Recording...</h3>
            <p>{remaining:.1f} seconds remaining.</p>
            <div style="height: 5px; background-color: #f1f1f1; border-radius: 5px;">
                <div style="height: 5px; width: {(elapsed / st.session_state.record_duration) * 100}%;
                background-color: #e74c3c; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.1)
        if not st.session_state.recording:
            break
    if st.session_state.recording:
        st.session_state.recording = False
        sd.stop()
        rec_status.empty()
        st.toast("Recording complete!", icon="✅")
        process_audio()

# ----- TRANSCRIPT & SEARCH INPUT -----
if st.session_state.transcript:
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

# ----- SEARCH RESULTS -----
if st.session_state.search_results:
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

        for i, hit in enumerate(st.session_state.search_results):
            score = hit.get("_score", 0)
            text = hit.get("fields", {}).get("chunk_text", "No text")
            hit_id = hit.get("_id", "N/A")
            relevance = "high-relevance" if score > 0.7 else "medium-relevance" if score > 0.4 else "low-relevance"
            score_pct = round(score * 100)
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
        st.markdown("</div>", unsafe_allow_html=True)

# ----- HELP EXPANDER -----
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

# ----- SIDEBAR SETTINGS -----
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
        if st.button("🔄 New Search"):
            st.session_state.transcript = ""
            st.session_state.search_results = None
            st.experimental_rerun()
    with col2:
        if st.button("⬇️ Export Results"):
            st.toast("Export feature coming soon!", icon="ℹ️")

# ----- FOOTER -----
st.markdown("""
<div style="text-align:center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;">
    <p style="color: #666; font-size: 0.8rem;">Powered by OpenAI Whisper & Pinecone | Built with ❤️ in Streamlit</p>
</div>
""", unsafe_allow_html=True)
