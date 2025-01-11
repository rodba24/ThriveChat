# ThriveChat

[![Hugging Face Spaces](https://img.shields.io/badge/Running%20on-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/rodba24/Thrive-Chat)

ThriveChat is a Retrieval-Augmented Generation (RAG) chatbot designed to assist users by providing concise, helpful responses and mental health resources. It combines advanced retrieval mechanisms with powerful generative models to create meaningful and contextually accurate conversations.

---

## **Features**
- **Retrieval-Augmented Generation (RAG)**: Combines a retriever to find relevant content and a language model to generate responses.
- **Interactive Chat Interface**: Engage in conversations with the chatbot via an intuitive web interface powered by Gradio.
- **Document Retrieval**: Scans uploaded PDFs for relevant content and provides accurate answers based on the documents.
- **State-of-the-Art NLP**: Built with ChatGroq and HuggingFace Sentence Transformers for precise understanding and responses.
- **Deployed on Hugging Face Spaces**: Accessible online with no setup required.

### Try it here: [Thrive-Chat on Hugging Face](https://huggingface.co/spaces/rodba24/Thrive-Chat)

---

## **What It Uses**
- **LangChain**: Orchestrates conversational and retrieval-based workflows.
- **ChatGroq**: Serves as the language model to generate conversational responses.
- **HuggingFace Sentence Transformers**: Encodes document chunks into embeddings for semantic search.
- **Chroma**: Acts as the vector database for storing embeddings and retrieving relevant context.
- **Gradio**: Provides a user-friendly interface for chatbot interactions.
- **Environment Variables**: Securely manages API keys and configurations using Python's `dotenv` library.

---

## **How RAG Works in ThriveChat**
1. **Document Embedding**: PDFs in the `data/` directory are converted into vector embeddings using HuggingFace Sentence Transformers.
2. **Vector Database Retrieval**: Relevant document chunks are retrieved from the Chroma vector database based on user queries.
3. **Prompt Augmentation**: The retrieved content is passed as context to the ChatGroq model.
4. **Response Generation**: ChatGroq generates a response that combines the retrieved context and user query.

---
title: ThriveChat
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: mental-health-chatbot.py
pinned: true
---
