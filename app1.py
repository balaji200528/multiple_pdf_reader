# import streamlit as st
# import os
# from dotenv import load_dotenv
# from pypdf import PdfReader

# from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.vectorstores import FAISS
# from duckduckgo_search import DDGS
# import google.generativeai as genai


# # =============================
# # 1. ENV + API CONFIG
# # =============================
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# # =============================
# # 2. PDF PROCESSING
# # =============================
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# # =============================
# # 3. SEARCH HELPERS
# # =============================
# def web_search(query: str, num_results: int = 3):
#     with DDGS() as ddgs:
#         results = [r["body"] for r in ddgs.text(query, max_results=num_results)]
#     return " ".join(results)


# # =============================
# # 4. USER INPUT HANDLER
# # =============================
# def user_input(question, mode):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     if mode == "PDF":
#         # Load FAISS DB
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(question, k=3)
#         context = " ".join([doc.page_content for doc in docs]) if docs else "No relevant info found in PDF."
#     else:
#         # Web Search
#         context = web_search(question)

#     # Prompt Template
#     prompt_template = """
#     You are a helpful assistant.
#     Answer the following question based on the context below. 
#     If the context does not contain the answer, say "The answer is not available in the provided context."

#     Context: {context}
#     Question: {question}
#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     # Run Gemini
#     # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

#     chain = LLMChain(llm=llm, prompt=prompt)
#     response = chain.run({"context": context, "question": question})

#     # Output
#     st.write("Reply:", response)


# # =============================
# # 5. STREAMLIT APP
# # =============================
# def main():
#     st.set_page_config("Chat PDF + Web Search")
#     st.header("Chat with PDF + Web Search using Gemini 💁")

#     # Select mode
#     mode = st.radio("Choose Mode:", ["PDF", "Web Search"])

#     # User question
#     user_question = st.text_input("Ask a Question")

#     if user_question:
#         user_input(user_question, mode)

#     # Sidebar
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF Files and Click on the Submit & Process Button", 
#             accept_multiple_files=True
#         )
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done ✅ Index Saved!")


# if __name__ == "__main__":
#     main()


import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent, AgentType

# =============================
# 1. ENV + API CONFIG
# =============================
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# =============================
# 2. Initialize Gemini + Tools
# =============================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Tavily search tool
search = TavilySearchResults()
tools = [search]

# Fix parsing errors with handle_parsing_errors=True
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,   # 👈 important
)

# =============================
# 3. Embeddings + Prompt
# =============================
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

prompt_template = """
You are an AI assistant. Use the following context to answer the user question.
If the answer is not in the context, say "The answer is not available in the provided context."

Context: {context}
Question: {question}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = LLMChain(llm=llm, prompt=PROMPT)

# =============================
# 4. PDF Processing
# =============================
vector_store = None

def process_pdf(uploaded_file):
    global vector_store
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

# =============================
# 5. User Input Handler
# =============================
def user_input(question, mode):
    if mode == "PDF":
        if not vector_store:
            st.error("Please upload and process a PDF first.")
            return
        docs = vector_store.similarity_search(question, k=3)
        context = " ".join([d.page_content for d in docs])
        response = chain.run({"context": context, "question": question})
        st.write("Reply:", response)

    elif mode == "Web Search":
        response = agent.run(question)   # Tavily agent
        st.write("Reply:", response)

# =============================
# 6. Streamlit App
# =============================
def main():
    st.set_page_config(page_title="Chat with PDF + Web Search using Gemini", layout="wide")
    st.markdown("## 🤖 Chat with PDF + Web Search using Gemini")

    with st.sidebar:
        st.markdown("### Menu:")
        uploaded_file = st.file_uploader("Upload your PDF Files", type="pdf")
        if uploaded_file:
            st.write(uploaded_file.name)
        if st.button("Submit & Process") and uploaded_file:
            process_pdf(uploaded_file)
            st.success("PDF processed successfully!")

    mode = st.radio("Choose Mode:", ["PDF", "Web Search"])
    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question, mode)

if __name__ == "__main__":
    main()
