from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.vectorstores import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# print(PINECONE_API_KEY)

extracted_data = load_pdf("./data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "langchain-test-index" 

docsearch=Pinecone.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)