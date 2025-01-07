import time
from torch import cuda, bfloat16
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import os


# Cache model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cuda" if cuda.is_available() else "cpu",
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


# Prepare pipeline
def prepare_pipeline(model, tokenizer):
    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [torch.LongTensor(tokenizer(x)['input_ids']).cuda() for x in stop_list]

    class StopOnTokens(transformers.StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        # device=0 if cuda.is_available() else -1,
        stopping_criteria=stopping_criteria,
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=generate_text)


# Extract text from PDF
def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = [page for page in loader.lazy_load()]
    return pages


# Create vectorstore
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    return vectorstore


# Handle query
def handle_query(chain, query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']


# Streamlit App
st.title("AI PDF📄 Question Answering")

st.write("Upload a PDF file, and ask questions about its content. Answers are generated using an AI model.")

# Load model and tokenizer
model_id = "meta-llama/Llama-2-7b-chat-hf"
model, tokenizer = load_model_and_tokenizer(model_id)
llm = prepare_pipeline(model, tokenizer)

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    temp_file_path = f"./uploaded_{int(time.time())}.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF file uploaded successfully!")
    st.write("Extracting text from the PDF...")

    try:
        pdf_text = extract_text_from_pdf(temp_file_path)
        st.write("Extraction complete! You can now ask questions.")

        user_query = st.text_input("Enter your question about the PDF:")
        ask_button = st.button("Ask")

        if ask_button and user_query:
            st.write("Generating answer...")
            try:
                vectorstore = create_vectorstore(pdf_text)
                chain = ConversationalRetrievalChain.from_llm(
                    llm, vectorstore.as_retriever(), return_source_documents=True
                )
                chat_history = []
                answer = handle_query(chain, user_query, chat_history)
                st.success("Answer generated successfully!")
                st.markdown(f"*Answer:* {answer}")
            except Exception as e:
                st.error(f"Failed to generate an answer: {e}")

    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")

else:
    st.info("Please upload a PDF file to proceed.")