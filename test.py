import os
import openai
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import ingestion_pdf


load_dotenv()

# Load the API key from the environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")
# print ( os.environ["OPENAI_API_KEY"])
embeddings = ingestion_pdf.docs_pdf
store = FAISS.load_local("faiss_index_react")
print(store)


# Now you can make API calls using the openai library
# embeddings_model = OpenAIEmbeddings()
# embeddings = embeddings_model.embed_documents(
#     [
#         "Hi there!",
#         "Oh, hello!",
#         "What's your name?",
#         "My friends call me World",
#         "Hello World!"
#     ]
# )
# len(embeddings), len(embeddings[0])
