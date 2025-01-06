<<<<<<< HEAD
import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load secrets from secrets.toml
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("RAG Webpage Assistant Using Llama 3.3")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are Webpage Assistant that helps users to find information in a context.
    Please provide the most accurate response based on the context and inputs
    only give information that is in the context not in general
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to process the webpage link
def vector_embedding(webpage_link):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/textembedding-gecko-multilingual@001")
        st.session_state.loader = WebBaseLoader(webpage_link)  # Load webpage
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Input for webpage link
webpage_link = st.text_input("Enter a Webpage Link")

# Embedding button
if st.button("Load and Embed Webpage") and webpage_link:
    vector_embedding(webpage_link)
    st.write("Vector Store DB is ready")

# Input for the question
prompt1 = st.text_input("Enter Your Question From Webpage")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
=======
import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load secrets from secrets.toml
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("RAG Webpage Assistant Using Llama 3.3")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are Webpage Assistant that helps users to find information in a context.
    Please provide the most accurate response based on the context and inputs
    only give information that is in the context not in general
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to process the webpage link
def vector_embedding(webpage_link):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/textembedding-gecko-multilingual@001")
        st.session_state.loader = WebBaseLoader(webpage_link)  # Load webpage
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Input for webpage link
webpage_link = st.text_input("Enter a Webpage Link")

# Embedding button
if st.button("Load and Embed Webpage") and webpage_link:
    vector_embedding(webpage_link)
    st.write("Vector Store DB is ready")

# Input for the question
prompt1 = st.text_input("Enter Your Question From Webpage")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
>>>>>>> 4d35f98 (initial commit)
        st.write("Please embed the webpage first.") 