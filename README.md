# 📚 Smart PDF RAG System

An intelligent, stateful web application that allows users to upload any PDF document and instantly interact with its contents. This system leverages a **Retrieval-Augmented Generation (RAG)** architecture to completely eliminate LLM hallucination by grounding the AI's responses strictly in the provided document context.

## 🛠️ Technical Architecture

*   **Frontend UI:** Streamlit (with Stateful Chat Memory)
*   **LLM Orchestration:** LangChain
*   **Inference Engine:** Groq API (Ultra-low latency)
*   **Generative Model:** LLaMA 3.3 70B Versatile
*   **Embedding Model:** Hugging Face (`sentence-transformers/all-mpnet-base-v2`)
*   **Vector Database:** ChromaDB (Ephemeral, in-memory processing)
*   **Document Parsing:** PyPDFLoader

## ⚙️ Core System Features

### 1. Dynamic Document Ingestion & Processing
*   Accepts user-uploaded PDFs via Streamlit's in-memory file uploader.
*   Utilizes secure OS-level temporary file generation (`tempfile`) to bridge the gap between Streamlit's byte-stream and LangChain's file-path requirements.
*   Automatically cleans up temporary storage immediately after processing to prevent memory leaks.

### 2. Optimized Text Chunking
*   Implements `RecursiveCharacterTextSplitter` with a strict `chunk_size=1000` and `chunk_overlap=100`.
*   This specific overlap ensures that contextual meaning is not lost across chunk boundaries, resulting in highly accurate embeddings.

### 3. Localized Vector Search (MMR)
*   Uses a localized Hugging Face sentence-transformer model to generate dense vector embeddings, reducing API costs and latency.
*   Stores embeddings in a ChromaDB vector store.
*   Utilizes **Maximum Marginal Relevance (MMR)** search (`lambda_mult=0.5`) to retrieve the top 4 chunks that are not only highly relevant to the user query but also diverse from each other.

### 4. Stateful UI & Caching
*   Leverages `st.session_state` to cache the instantiated `retriever` engine. This ensures the computationally expensive embedding process only occurs once per document upload, rather than refreshing on every user interaction.
