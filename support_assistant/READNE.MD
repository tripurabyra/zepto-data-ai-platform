# Zepto Support Assistant

A LangGraph + FastAPI + ChromaDB based Retrieval-Augmented Generation (RAG) support assistant that answers Zepto policy questions using a curated document knowledge base.

 

## Overview

The Support Assistant module provides an API that retrieves relevant Zepto policy information and returns answers with source document references and confidence scores.

The system uses:

* FastAPI for API endpoints
* LangGraph for workflow orchestration
* Sentence Transformers for embeddings
* ChromaDB as the vector database
* Docker for container deployment

 

## Architecture


User Question
      |
      v
FastAPI Endpoint
      |
      v
LangGraph Workflow
      |
      +     -+
      | Intent Router  |
      +     -+
              |
              v
      Policy Question?
              |
              v
     Sentence Transformer
        Embeddings
              |
              v
       ChromaDB Vector Store
              |
              v
      Zepto Policy Documents
              |
              v
 Answer + Sources + Confidence


 

## Project Structure


support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   └── doc_08.txt
│
├── vectorstore/
│   └── chroma_db/
│
├── main.py
├── graph.py
├── ingest.py
├── schemas.py
├── prompts.py
├── Dockerfile
├── requirements.txt
└── README.md


 

## Setup (Local)

Create and activate virtual environment:

 
python -m venv .venv


Windows:

powershell
.venv\Scripts\activate


Install dependencies:

 
pip install -r requirements.txt


 

## Create Vector Database

Run ingestion:

 
python ingest.py


This loads Zepto policy documents and stores embeddings in ChromaDB.

 

## Run API Locally

Start FastAPI:

 
uvicorn main:app --reload


Open Swagger documentation:


http://localhost:8000/docs


 

# Docker Setup

## Build Docker Image

 
docker build -t support-assistant .


## Run Container

 
docker run -p 8000:8000 support-assistant


API documentation:


http://localhost:8000/docs


 

# Example API Request

Request:

json
{
  "query": "What is Zepto Pass?"
}


Response:

json
{
  "answer": "Based on the retrieved context: Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories)...",
  "sources": [
    "doc_03"
  ],
  "confidence": 1
}


 

# Example Test Conversations

## Test 1

User:


What is Zepto Pass?


Assistant:


Zepto Pass costs INR 49 per month and provides free standard delivery and 5% off select categories.


Source:


doc_03


 

## Test 2

User:


How can I track my order?


Assistant:


Orders show a live rider-tracking map from packing until delivery through the Track Order screen.


Source:


doc_04


 

## Test 3

User:


Can I cancel my order?


Assistant:


Orders can be cancelled free of cost before the order status changes to Packed.


Source:


doc_05


 

# Retrieval Pipeline

1. User submits a question.
2. Intent classifier determines whether it is a Zepto policy question.
3. Policy questions are converted into embeddings.
4. ChromaDB searches the closest matching documents.
5. The retrieved context is returned with:

   * Answer
   * Source document ID
   * Confidence score

 

# Docker Verification

Successful container startup:


Graph loaded successfully

Uvicorn running on:
http://0.0.0.0:8000


Successful RAG test:


Question:
What is Zepto Pass?

Source:
doc_03


 

# Future Improvements

* Add real LLM generation instead of context-only responses
* Add conversation memory
* Add authentication
* Add automated evaluation tests
* Add monitoring and logging

 

## Status

FastAPI API
LangGraph workflow
 ChromaDB vector search
 Sentence Transformer embeddings
 Docker deployment
 Policy document retrieval
