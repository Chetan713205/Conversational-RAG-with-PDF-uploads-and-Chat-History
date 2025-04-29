# Importing libraries
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain                # for RAG
from langchain.chains import create_retrieval_chain                                        # for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Credential and Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ChatBot with LLM"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3jjlk9rl6FBUASv21T1aBAFo_h_R6rTk"
os.environ["HF_TOKEN"] = "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"
os.environ["GROQ_API_KEY"] = "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"

# Calling the embedding model
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2") 

# Setting up Streamlit Frontend User Interface
st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDFs and chat with the content")

# Input the GROQ API key to the UI
api_key = st.text_input("Enter your GROQ-API key to begin: ", type = "password")

# Initialize the LLM model:
if api_key:
    llm = ChatGroq(groq_api_key = api_key, model_name = "gemma2-9b-it")

    # Chat Interface
    session_id = st.text_input("Session ID", value = "default_session")        # creates a text input field in the Streamlit labeled "Session ID"
    
    # Manage Chat History                                                      This dictionary could later be used to store chat history or other session-specific data
    if "store" not in st.session_state:                                        # checks if a key called "store" exists in st.session_state
        st.session_state.store = {}                                            # If not, it initializes it as an empty dictionary
        
    # Upload File
    uploaded_files = st.file_uploader("Upload a PDF", type = "pdf", accept_multiple_files = True)
    
    # Process the uploaded file
    if uploaded_files:
        documents = []
        for i in uploaded_files:
            temp_pdf = f"./temp.pdf"                                     # Defines a temporary file path named temp.pdf in the current directory
            with open(temp_pdf, "wb") as file:                           # Openstemp.pdf in write-binary mode
                file.write(i.getvalue())                                 # Gets the binary content of the uploaded file
                file_name = i.name                                       # Extracts the original file name of the uploaded file
            
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)        
            
        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
        retriever = vectorstore.as_retriever()
        
        # Contextualize 
        contextualize_q_system_prompt = (
            "Given a chat history and latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question that can be understood"
            "without the chat history, do NOT answer the question"
            "just reformulate it if needed and otherwhise return it as it is"
        )
        
        # In Prompt Template
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        # Answer question
        system_prompt = (
            "You are an assistant for question answering task"
            "use the following piece of retrieved context to answer"
            "the question. If you don't know the answer say that you"
            "don't know. Use three sentence to keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)                    # This wraps your LLM with the QA prompt to form a chain that can take retrieved documents and generate answers.
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)      # combines The context-aware retriever (rewrites queries using history) and question_answer_chain
        
        def get_session_history(session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:                                        # Checks if the session ID (from earlier st.text_input) exists in the st.session_state.store dictionary.
                st.session_state.store[session_id] = ChatMessageHistory()                       # If not, initializes a new ChatMessageHistory object (from LangChain).
            return st.session_state.store[session_id]                                           # Returns the session's chat history object.
         
        conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, 
            input_messages_key = "input", 
            history_messages_key = "chat_history",
            output_message_key = "answer")
        
        user_input = st.text_input("Your question: ")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input" : user_input},
                config = {"configurable" : {"session_id" : session_id}}          # constructs a key "abc123" in store
            )
            
            st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)
             
else:
    st.write("Please enter your GROQ-API key")
    
        
        
                