import streamlit as st
import ollama
import os
from tempfile import NamedTemporaryFile
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- Constants ----------------
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "demo_vector_store"
PERSIST_DIRECTORY = "./chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------- Configure logging ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- Helper Functions ----------------
def ingest_pdfs(uploaded_files):
    """Load all uploaded PDF documents."""
    documents = []
    for file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = UnstructuredPDFLoader(file_path=tmp_path)
        docs = loader.load()

        # Add NoSQL style metadata
        for doc in docs:
            doc.metadata = {
                "filename": file.name,
                "source": tmp_path,
                "page": doc.metadata.get("page", None)
            }

        documents.extend(docs)
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def create_or_load_vector_db(_documents=None):
    """Create or load the vector database with embeddings."""
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        if not _documents:
            return None
        chunks = split_documents(_documents)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db

def build_context(retriever, question, top_k=5):
    """Retrieve relevant chunks and return a combined context string."""
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(
        [f"{doc.metadata['filename']} - Page {doc.metadata.get('page')}: {doc.page_content}" for doc in relevant_docs[:top_k]]
    )
    return context, relevant_docs[:top_k]

def generate_prompt(context, question):
    """Prompt for the LLM."""
    template = f"""
You are an AI expert assistant. Based ONLY on the following context, provide a structured answer.

Context:
{context}

Instructions:
1. Identify any **Red Flags / Risks** (e.g., legal, financial, structural, or general issues).
2. Identify **Positive Aspects / Strengths**.
3. Provide **Recommendations / Actionable Advice**.

Answer clearly, use bullet points, and ensure the answer is helpful to a user evaluating the content.

Question: {question}
"""
    return template

def print_colored_answer(answer_text):
    """Color-coded output for Streamlit."""
    lines = answer_text.split("\n")
    for line in lines:
        lower = line.lower()
        if "red flag" in lower or "risk" in lower:
            st.markdown(f"<span style='color:red'>{line}</span>", unsafe_allow_html=True)
        elif "positive" in lower or "strength" in lower:
            st.markdown(f"<span style='color:green'>{line}</span>", unsafe_allow_html=True)
        elif "recommendation" in lower or "advice" in lower:
            st.markdown(f"<span style='color:blue'>{line}</span>", unsafe_allow_html=True)
        else:
            st.write(line)

# ---------------- Streamlit Interface ----------------
st.set_page_config(page_title="PDF Assistant Demo", layout="wide")
st.title("PDF Assistant Demo")

st.markdown("""
Upload one or more PDF files. The AI will analyze the content and provide:
- Red Flags / Risks
- Positive Aspects
- Recommendations / Actionable Advice
""")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = ingest_pdfs(uploaded_files)
    st.success(f"Loaded {len(documents)} document(s).")

    vector_db = create_or_load_vector_db(documents)
    if vector_db is None:
        st.error("Vector database could not be created.")
    else:
        st.success("Vector database ready for querying.")

        user_question = st.text_input("Enter your question:", "")

        if user_question:
            with st.spinner("Generating response..."):
                llm = ChatOllama(model=MODEL_NAME)
                retriever = vector_db.as_retriever()
                context, retrieved_docs = build_context(retriever, user_question)
                prompt_text = generate_prompt(context, user_question)

                # Run LLM
                chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template(prompt_text)
                    | llm
                    | StrOutputParser()
                )
                response = chain.invoke(input={"context": context, "question": user_question})

                st.markdown("### Structured Answer")
                print_colored_answer(response)

                # Show retrieved chunks
                with st.expander("Show Retrieved Chunks"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}** - {doc.metadata['filename']} (Page {doc.metadata.get('page')})")
                        st.write(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
else:
    st.info("Upload one or more PDF files to get started.")

