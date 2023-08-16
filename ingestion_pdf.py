import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# import openai

load_dotenv()
# openai.api_key =  os.environ["OPENAI_API_KEY"]

# print ( os.environ["OPENAI_API_KEY"])


def docs_pdf():
    loader = PyPDFLoader(
        "/Users/ravirame/Documents/Appd_doc_assitant/appd-doc-assistant/pdf/AppDynamics for OpenTelemetry™-08_16_2023.pdf"
    )
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    persist_dir = (
        "/Users/ravirame/Documents/Appd_doc_assitant/appd-doc-assistant/Doc_embeds"
    )
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="/Users/ravirame/Documents/Appd_doc_assitant/appd-doc-assistant/Doc_embeds",
    )
    vectorstore.persist()

    # new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    # qa = RetrievalQA.from_chain_type(
    #     llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    # )
    # res = qa.run("Give me the gist of AppDynamics Open Telemetry in 3 sentences")
    # return qa
    # print(res)


if __name__ == "__main__":
    docs_pdf()
