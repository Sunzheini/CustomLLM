import os
import ssl
import certifi
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, tavily_map

from core.command_menu import CommandMenu


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')

# --------------------------------------------------------------------------------
"""
Secure HTTPS Connections - It ensures that your LangChain application can 
securely connect to external APIs and services (like OpenAI, Hugging Face, 
etc.) without SSL certificate verification errors. Avoiding SSL: 
CERTIFICATE_VERIFY_FAILED errors
"""
ssl_context = ssl.create_default_context(cafile=certifi.where())    # Creates SSL context using certifi's certificate bundle
os.environ['SSL_CERT_FILE'] = certifi.where()                       # Sets environment variable for SSL certificates
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()                  # Specifically configures the requests library to use certifi

# --------------------------------------------------------------------------------
def function_1():
    pass



























if __name__ == "__main__":
    menu = CommandMenu({
        '1': function_1,
    })
    menu.run()
