from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return embeddings