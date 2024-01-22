import os
from dotenv import load_dotenv
from os.path import join, dirname
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

HUGGINGFACEHUB_API_TOKEN =os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Load Documents
loader = TextLoader("state_of_the_union.txt", encoding="utf-8")
documents = loader.load()

# Text Splitting - Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

# Open-source Embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])

# Vector Store - PGVector
CONNECTING_STRING = os.getenv('CONNECTING_STRING')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTING_STRING,
    distance_strategy = DistanceStrategy.COSINE
)

# Question answering with Retrieval Augmented Generation
llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.1, "max_length": 64,"max_new_tokens": 512}
)

retriever = db.as_retriever(search_kwargs={"k": 2})

query = "What is the State of the Union meant for?"

prompt = """You are an AI assistant that follows instruction extremely well. 
Please be truthful and give direct answers, keep it short and simple.
If you don't know the answer, just say so without trying to make up an answer.
{context}

Question: {question}
Answer the questions in an american bro style while being polite."""

PROMPT = PromptTemplate(
    template=prompt, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
)

response = qa_stuff.run(query)
print(response)