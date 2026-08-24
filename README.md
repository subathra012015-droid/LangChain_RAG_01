LangChain RAG Learning Project

A step-by-step learning project for building and understanding Retrieval-Augmented Generation (RAG) applications using Python, LangChain, OpenAI, Hugging Face, FAISS, and ChromaDB.

The project progresses from a basic RAG implementation to separate ingestion/query flows, LangChain pipelines, conversational RAG, and Hugging Face embeddings with ChromaDB.

🎯 Project Objective

The objective of this project is to understand how a RAG application works and build different implementations step by step.

The project covers:

Basic RAG using OpenAI and FAISS

Separate ingestion and query processes

LangChain RAG pipelines

Conversational RAG

Hugging Face embeddings

ChromaDB vector storage and retrieval

FAISS and ChromaDB inspection

🏗️ Project Architecture

Source Data
    │
    ▼
Text Splitter
    │
    ▼
Chunks
    │
    ▼
Embedding Model
    │
    ▼
Vector Store / Database
    │
    ▼
Retriever
    │
    ▼
Relevant Context
    │
    ▼
LangChain Pipeline
    │
    ▼
LLM / Chat Model
    │
    ▼
Answer

The project currently demonstrates two main vector-search implementations:

OpenAI Embeddings → FAISS

Hugging Face Embeddings → ChromaDB

🔄 RAG Pipeline

A RAG application has two main stages.

1. Ingestion

Source Document
    ↓
Load Data
    ↓
Split into Chunks
    ↓
Create Embeddings
    ↓
Store in Vector Store / Database

Run ingestion when the source data, chunk configuration, or embedding model changes.

2. Query

User Question
    ↓
Create Query Embedding
    ↓
Semantic Search
    ↓
Retrieve Relevant Chunks
    ↓
Provide Context to LLM
    ↓
Generate Answer

The vector database does not need to be rebuilt for every new user question.

🧠 Technologies Used

