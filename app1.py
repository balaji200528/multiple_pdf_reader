import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from duckduckgo_search import DDGS
import google.generativeai as genai


# =============================
# 1. ENV + API CONFIG
# =============================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# =============================
# 2. PDF PROCESSING
# =============================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# =============================
# 3. SEARCH HELPERS
# =============================
def web_search(query: str, num_results: int = 3):
    with DDGS() as ddgs:
        results = [r["body"] for r in ddgs.text(query, max_results=num_results)]
    return " ".join(results)


# =============================
# 4. USER INPUT HANDLER
# =============================
def user_input(question, mode):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if mode == "PDF":
        # Load FAISS DB
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question, k=3)
        context = " ".join([doc.page_content for doc in docs]) if docs else "No relevant info found in PDF."
    else:
        # Web Search
        context = web_search(question)

    # Prompt Template
    prompt_template = """
    You are a helpful assistant.
    Answer the following question based on the context below. 
    If the context does not contain the answer, say "The answer is not available in the provided context."

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Run Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"context": context, "question": question})

    # Output
    st.write("Reply:", response)


# =============================
# 5. STREAMLIT APP
# =============================
def main():
    st.set_page_config("Chat PDF + Web Search")
    st.header("Chat with PDF + Web Search using Gemini 💁")

    # Select mode
    mode = st.radio("Choose Mode:", ["PDF", "Web Search"])

    # User question
    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question, mode)

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✅ Index Saved!")


if __name__ == "__main__":
    main()
