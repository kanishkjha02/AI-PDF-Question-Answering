# AI-PDF-Question-Answering

## Author
Kanishk Jha  
Roll No.: 2K21/CO/219  
Email: kanishkjha_co21a4_45@dtu.ac.in  

## Project Description
The **AI-PDF-Question-Answering** project focuses on creating an interactive application that allows users to interact with the content of a PDF document. By leveraging advanced natural language processing and machine learning technologies, the project enables users to upload PDFs, extract content, convert it into embeddings, and query the document's content through a question-answering system. This is achieved using tools like Streamlit for the user interface, libraries such as PyPDF2 and Langchain for PDF content extraction, and machine learning models like Gemini or Hugging Face transformers for embeddings and question-answering.

### Problem Statement
The application addresses the challenge of extracting meaningful insights from PDF documents, which are often difficult to query directly. By converting the document's text into embeddings and using a large language model, users can ask context-aware questions and receive precise answers. The solution also supports efficient handling of large documents by splitting them into manageable chunks, ensuring a seamless and responsive experience.

### Features
- PDF content extraction and display.
- Embedding generation for semantic understanding.
- Context-aware question-answering using an LLM.
- Scalable and efficient handling of large PDFs.

## Hardware Dependencies
- **NVIDIA L4 GPU**: x 1

## Setup Instructions
### Installing Requirements
To install the required dependencies, use the provided `requirements.txt` file. The following bash script can be used to install them:

```bash
# Script to install required dependencies
pip install -r requirements.txt
```

### Running the Application Locally
To run the application, follow these steps:

1. Ensure all dependencies are installed using the script above.
2. Use the following bash script to execute the test script (`test1.py`):

```bash
# Script to run the application locally
$ streamlit run streamlit_app.py
```

## Demonstration Videos
For a detailed walkthrough of the application’s functionality, refer to the videos available in the following Google Drive link:  
[Demonstration Videos](https://drive.google.com/drive/folders/1RcvPF4iXc6iMOavduRYr79ZI7oMGKNOd?usp=sharing)

