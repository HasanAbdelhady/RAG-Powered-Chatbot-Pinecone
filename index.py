from pdfminer.high_level import extract_text
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    return extract_text(pdf_path)

def split_text_into_chunks(text, max_chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Add overlap to preserve context
    return chunks

# --- Example usage ---
async def setup_index():
    pdf_path = "Hasan Abdelhady Resume.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text, max_chunk_size=1000, overlap=100)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks]
    
    metadata = [{"text": chunk} for chunk in chunks]
    ids = [str(uuid4()) for _ in range(len(embeddings))]

    pc = Pinecone()
    if not pc.Index("hasan-data"):
        pc.create_index(name="hasan-data", dimension=384, spec=ServerlessSpec(cloud="aws", region="us-east-1"), metric="cosine")
        print("Index created successfully ✅")
    else:
        print("Index already exists ✅")
    index = pc.Index("hasan-data")
    print("Index created successfully ✅")

    index.upsert(vectors=zip(ids, embeddings, metadata), namespace="hasan-resume")
    print("Data uploaded successfully ✅")

async def check_index_exists():
    """Check if the index exists and has data"""
    try:
        pc = Pinecone()
        index = pc.Index("hasan-data")
        stats = index.describe_index_stats()
        return stats['total_vector_count'] > 0
    except:
        return False

if __name__ == "__main__":
    setup_index()