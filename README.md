.# 🎙️ Voice RAG Search Assistant

An AI-powered voice-based search assistant built with [Streamlit](https://streamlit.io/), [OpenAI Whisper](https://openai.com/research/whisper), and [Pinecone](https://www.pinecone.io/) vector search. Just speak your query, and let the app transcribe and semantically search your knowledge base using Retrieval-Augmented Generation (RAG).

## 🔧 Features

- 🎤 Voice recording and transcription via Whisper API
- 🧠 Vector-based semantic search using Pinecone
- ✨ Modern, responsive UI with custom CSS
- 📄 Editable transcriptions and dynamic search results
- ⚙️ Configurable result limits and recording duration

## 🚀 Demo
Coming soon...

## 📦 Installation

```bash
git clone https://github.com/your-username/voice-rag-search-assistant.git
cd voice-rag-search-assistant
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
```

## 🧪 Run the App

```bash
streamlit run app.py
```

## 📁 Folder Structure

```
├── app.py                 # Streamlit app
├── custom_style.css       # UI styling
├── pinecone_client.py     # Pinecone vector search logic
├── .env                   # Your API keys
```

## 📌 To Do
- [ ] Add CSV/JSON export feature
- [ ] Deploy on Streamlit Cloud
- [ ] Add voice waveform visualizations

## 🙌 Credits

- [OpenAI Whisper](https://openai.com/research/whisper)
- [Streamlit](https://streamlit.io/)
- [Pinecone](https://www.pinecone.io/)
