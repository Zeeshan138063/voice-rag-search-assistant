import os

from dotenv import load_dotenv
from pinecone import Pinecone

# ----- CONFIGURATION -----
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create a dense index with integrated embedding
index_name = "dense-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(name=index_name, cloud="aws", region="us-east-1",
                              embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}})

dense_index = pc.Index(index_name)

namespace=""