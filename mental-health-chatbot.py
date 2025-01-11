import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter

import gradio as gr

# Load environment variables
load_dotenv()

# Access API key from .env file
api_key = os.getenv("groq_api_key")

def initialize_llm():
    # Replace ChatGroq with the correct model initialization
    llm = ChatGroq(
        temperature=0,
        groq_api_key= api_key,
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Create Vector Database
def create_vector_db():
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Use HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and persist Chroma database
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()

    print("ChromaDB created and data saved")
    return vector_db

# Setup QA Chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a helpful and concise chatbot specializing in mental health topics. 
    Context: {context}
    User: {question}
    Chatbot: """

    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize Chatbot
print("Initializing Chatbot...")
llm = initialize_llm()
db_path = "./chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)
print("QA Chain initialized successfully")

# Gradio App
def chatbot_response(message, history):
    try:
        if not message.strip():
            return None
        response = qa_chain.run(message)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as app:
    chatbot = gr.ChatInterface(
        fn=chatbot_response,
        title="ThriveChat",
        type="messages"
    )
app.launch()
