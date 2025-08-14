# PDF Assistant Demo

A Streamlit-based AI assistant that:
- Reads and processes PDF documents.
- Stores document chunks in a **Chroma** vector database.
- Uses **NoSQL-style metadata** for document context.
- Embeds and retrieves relevant content to answer user questions.

This project demonstrates:
- **Vector embeddings** with `nomic-embed-text`
- **Chroma vector database** for semantic search
- **NoSQL-style metadata** handling
- PDF ingestion, text splitting, and contextual retrieval

---

## Features
- Upload one or more PDFs.
- Get AI-generated answers that include:
  - **Red Flags / Risks**
  - **Positive Aspects**
  - **Recommendations**
- View retrieved document chunks for transparency.
- Persistent local vector DB so embeddings aren't recalculated unnecessarily.

---

## Requirements
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally.
- The following Ollama models available:
  - `llama3.2`
  - `nomic-embed-text`

---

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Bredicus/pdf-assistant-demo.git
   cd pdf-assistant-demo
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```

5. **Pull required models**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

---

## Running the App
```bash
streamlit run app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## Project Structure
```
pdf-assistant-demo/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── chroma_db/          # Persistent Chroma vector database (auto-created)
```

---

## Example Workflow
1. Upload one or more PDF files.
2. The app extracts text, splits it into chunks, and stores embeddings in Chroma.
3. Ask a question about the content.
4. The assistant retrieves relevant chunks and provides a structured answer:
   - Risks
   - Positives
   - Recommendations

