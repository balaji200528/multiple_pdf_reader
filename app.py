from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from pypdf import PdfReader
# from langchain.text.splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS   # line ~13
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.chains import LLMChain
from duckduckgo_search import DDGS
# from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text




def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) 
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain(vector_store):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context just say,
    "answer is not available in the context".
    Do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)  # ← Updated model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain




def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # new_db=FAISS.load_local("faiss_index",embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # ✅ added

    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain(new_db)

    response =chain(
        {"input_documents":docs,"question":user_question},
        return_only_outputs=True)
    
    print(response)
    st.write("reply: ",response['output_text'])

def web_search(query: str, num_results: int = 3):
    with DDGS() as ddgs:
        results = [r["body"] for r in ddgs.text(query, max_results=num_results)]
    return " ".join(results)
    
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()