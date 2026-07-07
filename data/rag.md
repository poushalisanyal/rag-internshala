# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language models. Instead of relying only on information learned during training, a RAG system retrieves relevant documents and uses them as context for generating answers.

## Why RAG is Needed

Large language models may produce hallucinations or provide outdated information. RAG helps reduce these issues by supplying external knowledge during inference.

Benefits include:

* Improved factual accuracy
* Access to up-to-date information
* Reduced hallucinations
* Domain-specific question answering

## RAG Pipeline

A typical RAG system consists of:

1. Document Ingestion
2. Text Chunking
3. Embedding Generation
4. Vector Storage
5. Retrieval
6. Answer Generation

## Chunking

Chunking is the process of splitting large documents into smaller sections. Common parameters include chunk size and chunk overlap.

Smaller chunks improve retrieval precision, while larger chunks preserve more context.

## Embeddings

Embeddings are dense vector representations of text. Similar pieces of text produce similar vectors in embedding space.

Popular embedding models include:

* all-MiniLM-L6-v2
* BGE models
* OpenAI embedding models

## Vector Databases

Vector databases store embeddings and allow similarity search.

Examples include:

* FAISS
* ChromaDB
* Qdrant
* LanceDB
* Pinecone

## Similarity Search

When a user asks a question, the system converts the question into an embedding and searches for the most similar document chunks.

Common similarity metrics:

* Cosine Similarity
* Euclidean Distance
* Dot Product

## FAISS

FAISS (Facebook AI Similarity Search) is an efficient library for similarity search and vector retrieval. It is widely used because it is fast, lightweight, and can run locally without managed infrastructure costs.

## Metadata Filtering

Metadata such as source file, page number, category, or date can be stored alongside vectors. This enables filtered retrieval.

## Prompt Construction

Retrieved chunks are inserted into a prompt along with the user's question. The language model then generates an answer grounded in the retrieved context.

## Conclusion

RAG combines retrieval and generation to create more reliable and factual AI systems. It is commonly used in enterprise search, document assistants, and knowledge management applications.
