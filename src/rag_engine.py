import os
import tempfile
import warnings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["USER_AGENT"] = "RAG_Portfolio_Project/1.0"

def process_pdf(uploaded_file):
    """Saves the uploaded PDF temporarily, processes it, and returns a retriever."""
    # 1. Save Streamlit upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. Load the PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 3. Split the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # 4. Embed and store in Chroma (Ephemeral memory for dynamic uploads)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"} 
        )
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        
        # 5. Return the MMR retriever
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )
    
    finally:
        # 6. Clean up the temporary file
        os.remove(tmp_path)

def ask_document(retriever, query: str):
    """Executes the RAG chain with the user's query."""
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's question based ONLY on the provided Context. If the answer is not in the context, say 'I cannot find the answer in the provided document'."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = template | model
    response = chain.invoke({"context": context, "question": query})
    
    return response.content