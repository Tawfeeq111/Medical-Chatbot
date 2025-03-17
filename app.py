from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index" 

embeddings = download_hugging_face_embeddings()

from langchain.vectorstores import Pinecone

docsearch = Pinecone.from_existing_index(index_name, embeddings)

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                      model_type="llama",
                      config={'max_new_tokens':512,
                              'temperature':0.8})

prompt_template = PromptTemplate(
    input_variables=["context", "question"],  
    template="""
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
"""
)

chain_type_kwargs = {"prompt": prompt_template}

qa_chain = load_qa_chain(llm, chain_type="stuff", **chain_type_kwargs)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs 
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)