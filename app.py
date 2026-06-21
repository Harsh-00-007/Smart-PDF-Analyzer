import streamlit as st
from dotenv import load_dotenv
from src.rag_engine import process_pdf, ask_document

# Load environment variables (Groq API Key)
load_dotenv()

st.set_page_config(page_title="Smart PDF Reader", layout="centered")
st.title("📚 Intelligent PDF Analyzer")
st.markdown("Upload any PDF document and ask questions about its contents. Powered by Hugging Face embeddings, ChromaDB, and LLaMA 3.3.")

# Initialize session state for the retriever and chat history
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: File Upload ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        with st.spinner("Chunking text and building vector database..."):
            try:
                st.session_state.retriever = process_pdf(uploaded_file)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = [] # Clear history on new upload
                st.success("Database built successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

# --- MAIN CHAT INTERFACE ---
if st.session_state.retriever:
    st.header("2. Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("What is this document about?"):
        
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    answer = ask_document(st.session_state.retriever, prompt)
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"[SYSTEM ERROR]: {e}")
else:
    st.info("👈 Please upload a PDF in the sidebar to begin.")