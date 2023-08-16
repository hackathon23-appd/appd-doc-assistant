import os
from dotenv import load_dotenv
from typing import Any


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

IndexName = "faiss_index_react"


def run_llm(query: str) -> Any:
    persist_dir = (
        "/Users/ravirame/Documents/Appd_doc_assitant/appd-doc-assistant/Doc_embeds"
    )
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function= embedding)
    # vectorstore.save_local("faiss_index_react")
    # new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
    )
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=chat, retriever=new_vectorstore.as_retriever()
    # )

    # res = qa.run("Give me the gist of AppDynamics Open Telemetry in 3 sentences")
    return qa({"query": query})


if __name__ == "__main__":
    print(
        run_llm(
            query="Give me the gist of AppDynamics Open Telemetry in 3 sentences?"
        )
    )
