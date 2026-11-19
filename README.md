# Amharic Legal Chatbot Using RAG

An AI-powered legal assistant that answers questions in **Amharic** using Ethiopian legal documents. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant legal information and generate context-grounded responses.

## Overview

Accessing and understanding Ethiopian legal documents can be challenging, especially when documents are lengthy, difficult to search, or available only in PDF format.

This project addresses this challenge by developing an Amharic legal chatbot that uses a specialized collection of Ethiopian legal documents to provide relevant and context-based answers.

The system combines multilingual semantic search with an Amharic language model to improve access to legal information.

## Key Features

- Amharic question answering based on Ethiopian legal documents.
- PDF text extraction and preprocessing.
- Amharic-specific text cleaning and sentence segmentation.
- Semantic search using multilingual embeddings.
- FAISS-based vector retrieval.
- Context-grounded response generation using RAG.
- Source document and page-level traceability.

## Technical Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

1. **Data Collection:** Collects Ethiopian legal documents, including proclamations, regulations, and Federal Supreme Court Cassation Decisions.
2. **Text Extraction:** Extracts text from PDF documents using PyMuPDF.
3. **Preprocessing:** Cleans Amharic text, removes unnecessary characters, and performs sentence segmentation.
4. **Chunking:** Splits documents into 200-token chunks with an 80-token overlap.
5. **Embedding Generation:** Generates multilingual embeddings using `paraphrase-multilingual-MiniLM-L12-v2`.
6. **Vector Indexing:** Stores embeddings in FAISS using the IndexIVFFlat index.
7. **Retrieval:** Retrieves the most relevant legal text based on semantic similarity.
8. **Answer Generation:** Uses an Amharic Llama model to generate responses based on retrieved legal context.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Language Model | Llama 3.2 1B Amharic |
| Model | `rasyosef/Llama-3.2-1B-Amharic` |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Database | FAISS |
| PDF Processing | PyMuPDF |
| NLP & Text Processing | Regular Expressions |
| Model Framework | Hugging Face Transformers |
| Development Environment | Google Colab, VS Code |

## Data Sources

The knowledge base includes:

**Primary Sources**
- Federal Negarit Gazette (Proclamations and Regulations).
- Federal Supreme Court Cassation Decisions.
- Ministry of Justice: https://justice.gov.et
- Federal Supreme Court: https://fsc.gov.et
- National Bank of Ethiopia: https://nbe.gov.et

**Secondary Sources**
- lawethiopia.com
- chilot.wordpress.com

## Evaluation and Results

The system was evaluated using both automatic metrics and human evaluation.

**Automatic Evaluation**

A dataset of 123 questions was used to evaluate answer quality.

| Metric | Average Score |
|---|---:|
| Faithfulness | 0.594 |
| BERTScore | 0.753 |

**Human Evaluation**

Three legal experts evaluated 20 question-answer pairs using five criteria.

| Evaluation Criteria | Average Score |
|---|---:|
| Relevance | 3.67/5 |
| Informativeness | 3.33/5 |
| Clarity | 3.67/5 |
| Legal Accuracy | 4.00/5 |
| Completeness | 2.67/5 |

The evaluation showed promising legal accuracy and relevance, while identifying response completeness and depth as areas for improvement.

## Challenges

- Processing older scanned government documents and maintaining data quality.
- Handling the morphological complexity of Amharic.
- Overcoming limitations of keyword-based search through dense vector embeddings.
- Managing computational and language model access limitations.

## Future Improvements

- Improve answer completeness and response depth.
- Expand the Ethiopian legal document collection.
- Improve handling of complex and multi-part legal questions.
- Further improve response quality through model fine-tuning and prompt optimization.


