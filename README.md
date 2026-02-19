# 📄 Smart Contract Q&A Assistant

A RAG (Retrieval Augmented Generation) application that allows users to
upload contracts and legal documents, then ask questions and get
AI-powered answers with source citations.

## Features

- **Document Upload**: Support for PDF and DOCX files
- **Intelligent Chunking**: Recursive text splitting with overlap
- **Semantic Search**: FAISS-powered vector similarity search
- **Conversational QA**: Ask follow-up questions with context memory
- **Source Citations**: Every answer includes references to source chunks
- **Guard Rails**: Input validation, prompt injection detection, hallucination checks
- **Document Summarization**: Generate concise summaries of uploaded documents
- **Evaluation Pipeline**: Metrics for retrieval and answer quality

##  Architecture
Document Upload → Chunking → Embedding → FAISS Search → QA Chain → Guardrails → Output

## 🛠️ Technology Stack

| Component         | Technology                    |
| ----------------- | ----------------------------- |
| LLM Framework     | LangChain                     |
| LLM Provider      | grok                          |
| Vector Store      | FAISS                         |
| Embeddings        | SentenceTransformers (MiniLM) |
| Backend API       | FastAPI                       |
| Frontend UI       | Gradio                        |
| Document Parsing  | PyMuPDF, python-docx          |

## usage 
1. Upload a PDF or DOCX file.
2. Ask a question in the chat box.
3. summarize the contract
4. View the AI answer along with source citations.

## Quick Start

### Clone and Setup

```bash
git clone <https://github.com/TasneemSamir/Smart-Contract-Summary-and-Q-A-Assistant->
cd smart-contract-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
