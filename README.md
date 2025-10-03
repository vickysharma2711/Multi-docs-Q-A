# Multi-PDF Q&A Bot with LangChain

## Overview
This project is a **multi-document Q&A system** that allows users to ask questions over multiple PDF documents.  
It uses **LangChain**, **vector embeddings**, and the **Groq Chat model** for retrieval-augmented generation (RAG) capabilities.  

**Key Features:**
- Load and process multiple PDF documents
- Split documents into smaller chunks for efficient retrieval
- Convert text into vector embeddings using HuggingFace `all-MiniLM-L6-v2`
- Store vectors in **FAISS** for fast semantic search
- Use **ChatGroq LLM** with `refine` chain type for accurate answers
- Supports multi-line and multi-question queries

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multi-pdf-qna.git
cd multi-pdf-qna
