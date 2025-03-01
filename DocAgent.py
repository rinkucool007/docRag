import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Follow these rules:
    1. Provide all relevant details from the context
    2. If answer isn't in context, say "answer is not available in the context"
    3. Never invent answers or speculate beyond the context
    4. Keep responses clear and concise
    
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = Ollama(model="deepseek-r1", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True  # Security parameter
        )
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.write("**Reply:**", response["output_text"])
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        st.info("Please process PDFs first before asking questions")

def main():
    st.set_page_config("Chat PDF", page_icon="📄")
    st.header("PDF Chatbot with Local AI (deepseek-r1) 🖥️")

    user_question = st.text_input("Ask questions about your PDF content:")
    
    with st.sidebar:
        st.title("Workflow")
        st.markdown("""
        1. Upload PDF files
        2. Process documents
        3. Ask questions
        """)
        
        pdf_docs = st.file_uploader(
            "Upload PDF files", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
                return
                
            with st.status("Processing documents..."):
                st.write("Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)
                
                st.write("Splitting text into chunks...")
                text_chunks = get_text_chunks(raw_text)
                
                st.write("Creating vector store...")
                get_vector_store(text_chunks)
                
            st.success("Documents processed successfully! You can now ask questions.")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()