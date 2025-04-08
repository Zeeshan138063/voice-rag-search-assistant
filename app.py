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
from ai_search.pinecone_client import dense_index, namespace

# ----- CONFIGURATION -----
load_dotenv()
# ----- PAGE SETUP -----
st.set_page_config(page_title="Voice AI Search Assistant", page_icon="🎙️", layout="wide",
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
    <h1>🎙️ Voice AI Search Assistant</h1>
    <p>Speak naturally and search your knowledge base with AI-powered technology</p>
</div>
""", unsafe_allow_html=True)

import json
import pandas as pd
import streamlit as st


# ----- FUNCTIONS -----
# Add this function to handle product catalog
def display_product_catalog():
    """
    Display all products from records.json in a searchable table
    """
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2>📋 Product Catalog</h2>', unsafe_allow_html=True)

    # Add description
    st.markdown("""
    <p style="margin-bottom: 20px;">
        Browse the available products below to help with formulating your search queries.
        You can filter and sort the table to find specific items.
    </p>
    """, unsafe_allow_html=True)

    try:
        # Load products directly from records.json
        with open('records.json', 'r', encoding='utf-8') as f:
            products = json.load(f)

        # Convert to DataFrame for display
        df = pd.DataFrame([{"ID": item["_id"], "Product Name": item["chunk_text"]} for item in products])

        # Add search functionality
        search_term = st.text_input("Search products:", "")
        if search_term:
            filtered_df = df[df["Product Name"].str.contains(search_term, case=False)]
        else:
            filtered_df = df

        # Display the DataFrame as an interactive table
        st.dataframe(filtered_df, column_config={"ID": st.column_config.TextColumn("ID", width="small"),
                                                 "Product Name": st.column_config.TextColumn("Product Name",
                                                                                             width="large"), },
                     use_container_width=True, hide_index=True, )

        # Add a note about the number of products
        st.markdown(f"<p><small>Showing {len(filtered_df)} of {len(products)} products</small></p>",
                    unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("records.json file not found. Make sure it's in the same directory as your app.")
    except Exception as e:
        st.error(f"Error loading product catalog: {str(e)}")


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

        # Create a styled table with consistent widths
        st.markdown("""
        <div class="results-table">
            <table style="width:100%; table-layout:fixed;">
                <thead>
                    <tr>
                        <th style="width:80px;">Number</th>
                        <th style="width:calc(100% - 230px);">Product Name</th>
                        <th style="width:150px;">Relevancy Score</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)

        # Create table rows for each result
        for i, hit in enumerate(st.session_state.search_results):
            score = hit.get("_score", 0)
            text = hit.get("fields", {}).get("chunk_text", "No text available")
            score_pct = round(score * 100)

            # Determine relevance class based on score
            relevance_class = "high-relevance" if score > 0.7 else "medium-relevance" if score > 0.4 else "low-relevance"

            # Create a table row with the exact same widths as header
            st.markdown(f"""
                <tr class="{relevance_class}">
                    <td style="width:80px; text-align:center;">{i + 1}</td>
                    <td style="width:calc(100% - 230px);">{text}</td>
                    <td style="width:150px; text-align:center;"><span class="score-badge">{score_pct}%</span></td>
                </tr>
            """, unsafe_allow_html=True)

        # Close the table
        st.markdown("""
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

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
    # Add product catalog toggle
    st.subheader("📋 Product Catalog")
    if st.checkbox("Show Product Catalog", value=False, help="Display all available products for reference"):
        display_product_catalog()

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
        # To this:
        if st.button("🔄 New Search"):
            st.session_state.transcript = ""
            st.session_state.search_results = None
            st.rerun()  # Fixed line
    with col2:
        if st.button("⬇️ Export Results"):
            st.toast("Export feature coming soon!", icon="ℹ️")

# ----- FOOTER -----
st.markdown("""
<div style="text-align:center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;">
    <p style="color: #666; font-size: 0.8rem;"></p>
</div>
""", unsafe_allow_html=True)
