import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#embedding model -spacy
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

#LLaMA 3 via Ollama
llm = Ollama(model="llama3", temperature=0)


# ---------- PDF Processing ----------
def pdf_read(pdf_doc):
    """Extract text from uploaded PDFs"""
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Split text into chunks
def get_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

#Store chunks in FAISS DB
def vector_store(text_chunks):

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


#Ask LLaMA 3 using retrieved context
def ask_llama3(context, question):

    prompt = f"""
    You are a helpful assistant. Use ONLY the following context to answer the question.

    Context:
    {context}

    Question: {question}

    If the answer is not in the context, just say: "Answer is not available in the context."
    """

    response = llm.invoke(prompt)
    st.write("**Reply:**", response)

#Retrieve relevant chunks and pass them to LLaMA 3
def user_input(user_question):

    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    ask_llama3(context, user_question)


# ---------- Streamlit UI ----------
def main():
    st.set_page_config("Chat PDF with LLaMA 3")
    st.header("📄 RAG-based Chat with PDF (LLaMA 3 - Ollama)")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("✅ PDF processing done")


if __name__ == "__main__":
    main()
