from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from InstructorEmbedding import INSTRUCTOR
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import create_extraction_chain
from langchain_mistralai import ChatMistralAI
from langchain.chains import LLMChain
from typing import Optional

OPENAI_API_KEY = "pWV8pKnCKvnmQjyG9FXtQaZP2fIdBcEH"

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

SYSTEM_PROMPT = """
Extract the following information about a person from the given text in JSON format with appropriate tags:
- Full name
- Email address
- Phone number
- Job title
- Company
- Address
""".strip()

EXTRACTION_QUERY = "Extract the email signature information."


class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default=None, description="The full name of the person")
    email: Optional[str] = Field(default=None, description="The email address of the person")
    phone: Optional[str] = Field(default=None, description="The phone number of the person")
    job_title: Optional[str] = Field(default=None, description="The job title of the person")
    company: Optional[str] = Field(default=None, description="The company of the person")
    address: Optional[str] = Field(default=None, description="The address of the person")


def load_pdfs(directory: str):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()


def create_embeddings(model_name: str):
    return HuggingFaceInstructEmbeddings(model_name=model_name)


def split_texts(docs, chunk_size=1024, chunk_overlap=64):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


def create_vector_store(texts, embeddings, persist_directory="db"):
    return Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()


def create_qa_chain(llm, db, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def create_runnable(prompt, llm, schema, db):
    return prompt | llm.with_structured_output(schema=Person) | db.as_retriever()


def extract_email_signature_info():
    # Load and process PDFs
    docs = load_pdfs("pdfs")
    print(f"Number of documents loaded: {len(docs)}")

    # Create embeddings
    embeddings = create_embeddings("hkunlp/instructor-large")

    # Split texts into chunks
    texts = split_texts(docs)
    print(f"Number of text chunks: {len(texts)}")

    # Create vector store
    db = create_vector_store(texts, embeddings)

    # Generate prompt
    template = generate_prompt(
        """
{context}

Question: {question}
""",
        system_prompt=SYSTEM_PROMPT,
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initialize the LLM
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=OPENAI_API_KEY)

    # Create QA chain
    qa_chain = create_qa_chain(llm, db, prompt)

    # Define the schema
    schema = {
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "job_title": {"type": "string"},
            "company": {"type": "string"},
            "address": {"type": "string"},
        }
    }

    # Create runnable
    runnable = create_runnable(prompt, llm, schema, db)

    # Run query
    result = qa_chain(EXTRACTION_QUERY)
    print(result)


if __name__ == "__main__":
    extract_email_signature_info()
